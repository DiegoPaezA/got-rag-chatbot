import json
import os
import random
import logging
import math
from typing import List, Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.config_manager import ConfigManager

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# Distribución objetivo de tipos de nodos para el dataset
TARGET_DISTRIBUTION = {
    "Character": 0.30,
    "House": 0.25,
    "Battle": 0.15,
    "Location": 0.15,
    "Object": 0.10,
    "Organization": 0.05
}

class QAItem(BaseModel):
    question: str = Field(description="A fully self-contained natural language question.")
    ground_truth: str = Field(description="The precise answer found in the graph.")
    type: str = Field(description="Category (e.g., Relationship, Attribute).")
    difficulty: str = Field(description="Easy, Medium, or Hard.")
    evidence_source: str = Field(description="'Graph' or 'Properties'.")

class QABatch(BaseModel):
    qa_pairs: List[QAItem]

class Neo4jDatasetGenerator:
    """Generates a Ground Truth QA dataset using live data from Neo4j."""

    def __init__(self):
        """Initialize Neo4j connection and LLM based on configuration."""
        self.config = ConfigManager()
        
        # Rutas desde config
        self.output_path = ConfigManager.get("paths", "eval", "golden_dataset")
        if not self.output_path:
            self.output_path = "data/eval/golden_dataset_neo4j.jsonl"
            logger.warning(f"⚠️ Output path not found in config. Using default: {self.output_path}")

        # Configuración Neo4j
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("✅ Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"❌ Error connecting to Neo4j: {e}")
            raise e

        # Configuración LLM (Generator)
        llm_config = ConfigManager.get_llm_config("generator")
        model_name = llm_config.get("model_name", "gemini-2.5-flash")
        
        # Temperatura BAJA para asegurar JSON válido y seguimiento de reglas
        temperature = 0.1 
        
        logger.info(f"🤖 Initializing LLM Generator: {model_name} (Temp: {temperature})")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=3
        )

    def close(self):
        self.driver.close()

    def _get_stratified_node_ids(self, total_questions: int) -> List[Dict]:
        """Fetch candidate node IDs from Neo4j, balanced by type."""
        selected_nodes = []
        target_per_node = 2 
        nodes_needed = math.ceil(total_questions / target_per_node)

        logger.info("📊 Selecting candidate nodes from Neo4j...")
        
        with self.driver.session() as session:
            for label, ratio in TARGET_DISTRIBUTION.items():
                count = math.ceil(nodes_needed * ratio)
                
                # Query compatible con Neo4j 5+ (COUNT {})
                # Seleccionamos nodos que tengan al menos 1 relación para preguntas interesantes
                query = f"""
                MATCH (n:{label})
                WHERE COUNT {{ (n)--() }} > 0 
                WITH n, rand() AS r
                ORDER BY r
                LIMIT $count
                RETURN n.id AS id, labels(n) AS labels
                """
                
                try:
                    result = session.run(query, count=count)
                    fetched = [{"id": record["id"], "type": label} for record in result]
                    selected_nodes.extend(fetched)
                    logger.info(f"   - {label}: {len(fetched)} nodes selected.")
                except Exception as e:
                    logger.warning(f"⚠️ Error querying label {label}: {e}")

        random.shuffle(selected_nodes)
        return selected_nodes

    def _fetch_node_context(self, node_id: str) -> Optional[Dict]:
        """Retrieve the 'Ego Graph' (node + immediate neighbors) for LLM context."""
        query = """
        MATCH (n {id: $id})
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN 
            properties(n) as props,
            type(r) as rel_type,
            startNode(r) = n as is_outgoing,
            m.id as neighbor_id,
            m.name as neighbor_name
        """
        
        with self.driver.session() as session:
            result = session.run(query, id=node_id)
            records = list(result)

        if not records:
            return None

        core_props = records[0]["props"]
        
        # Construir representación textual de las relaciones
        connections = []
        for rec in records:
            if rec["rel_type"]:
                direction = "-->" if rec["is_outgoing"] else "<--"
                neighbor = rec["neighbor_name"] or rec["neighbor_id"]
                # Limpiar comillas para evitar romper el JSON del prompt
                neighbor_clean = str(neighbor).replace('"', "'")
                rel_str = f"(This Entity) {direction} [{rec['rel_type']}] {direction} ({neighbor_clean})"
                connections.append(rel_str)
        
        return {
            "id": node_id,
            "properties": core_props,
            "connections": connections
        }

    def _create_chain(self):
        """Construct the LangChain pipeline with strict 'Self-Contained' rules."""
        
        # IMPORTANTE: Usamos dobles llaves {{ }} para los ejemplos JSON dentro del prompt
        system_instruction = """
        You are an expert Game of Thrones Trivia Master evaluating a GraphRAG system.
        
        INPUT: A subgraph from a Neo4j database representing a specific entity.
        
        TASK: Generate 2 Question-Answer pairs to test the system's retrieval capabilities.
        
        CRITICAL RULES FOR QUESTIONS (Do NOT ignore):
        1. **Self-Contained:** The question MUST explicitly name the entity found in the properties.
           - ❌ BAD: "What region is *this house* located in?"
           - ❌ BAD: "Who owns *this item*?"
           - ✅ GOOD: "What region is *House Stark* located in?"
           - ✅ GOOD: "Who owns *Longclaw*?"
        2. **No Meta-References:** NEVER mention "this node", "the provided data", "the graph", or "the JSON". Treat it as general trivia.
        3. **Ground Truth:** Must be exact strings found in the input JSON (properties or connections).
        
        STRICT OUTPUT FORMAT:
        Return ONLY a valid JSON object. No Markdown. No extra text.
        
        JSON STRUCTURE:
        {{
            "qa_pairs": [
                {{
                    "question": "What is the seat of House Stark?",
                    "ground_truth": "Winterfell",
                    "type": "Relationship",
                    "difficulty": "Easy",
                    "evidence_source": "Graph"
                }}
            ]
        }}
        """

        human_instruction = """
        Here is the Entity Data:
        {node_json}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", human_instruction)
        ])
        
        parser = JsonOutputParser(pydantic_object=QABatch)
        return prompt | self.llm | parser

    def generate(self, num_questions=150):
        """Main execution loop."""
        nodes = self._get_stratified_node_ids(num_questions)
        if not nodes:
            logger.error("❌ No nodes found in Neo4j.")
            return

        chain = self._create_chain()
        dataset = []
        
        logger.info(f"🚀 Generating questions using Neo4j data. Output: {self.output_path}")
        pbar = tqdm(total=num_questions)
        
        for node_meta in nodes:
            if len(dataset) >= num_questions: break
            
            context = self._fetch_node_context(node_meta["id"])
            if not context: continue

            try:
                # Pasar el contexto como JSON string
                response = chain.invoke({"node_json": json.dumps(context)})
                
                if "qa_pairs" in response:
                    for item in response["qa_pairs"]:
                        if len(dataset) >= num_questions: break
                        
                        # Validar que la pregunta no contenga "this node" (filtro de seguridad extra)
                        q_lower = item["question"].lower()
                        if "this node" in q_lower or "this entity" in q_lower or "this house" in q_lower:
                            continue

                        item["question_id"] = f"neo4j_{len(dataset)}"
                        item["source_type"] = node_meta["type"]
                        item["origin_node"] = node_meta["id"]
                        
                        dataset.append(item)
                        pbar.update(1)
            except Exception as e:
                # Log level debug para no ensuciar, salvo errores críticos
                logger.debug(f"⚠️ Error generating for {node_meta['id']}: {e}")
                continue
        
        pbar.close()
        self.close()
        
        if dataset:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as f:
                for item in dataset:
                    record = item if isinstance(item, dict) else item.dict()
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"✅ Dataset successfully generated with {len(dataset)} items.")
        else:
            logger.error("❌ No questions were generated.")

if __name__ == "__main__":
    generator = Neo4jDatasetGenerator()
    generator.generate(num_questions=150)