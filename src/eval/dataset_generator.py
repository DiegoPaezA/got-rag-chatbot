import json
import os
import random
import logging
import math
from collections import defaultdict
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Configuración de Logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

load_dotenv()

# --- CONFIGURACIÓN DE ESTRATIFICACIÓN ---
TARGET_DISTRIBUTION = {
    "Character": 0.30,
    "House": 0.25,
    "Battle": 0.15,      
    "Location": 0.15,    
    "Object": 0.10,
    "Organization": 0.05
}

class QAItem(BaseModel):
    question: str = Field(description="A natural language question.")
    ground_truth: str = Field(description="The precise answer.")
    type: str = Field(description="The category (e.g., Battle, Lineage).")
    difficulty: str = Field(description="Easy, Medium, or Hard.")
    evidence_source: str = Field(description="Where did you get the answer from? Options: 'Graph' (Edges/Properties), 'Text' (Narrative), or 'Hybrid' (Both).")

class QABatch(BaseModel):
    qa_pairs: List[QAItem]

class StratifiedDatasetGenerator:
    def __init__(self, data_dir: str, model_name: str = "gemini-2.5-flash"):
        self.data_dir = data_dir
        self.nodes_path = os.path.join(data_dir, "nodes_cleaned.jsonl")
        self.edges_path = os.path.join(data_dir, "edges.jsonl")
        self.docs_path = os.path.join(data_dir, "wiki_dump.jsonl") 
        self.output_path = os.path.join("data", "eval", "golden_dataset_150.jsonl")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.8,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_retries=3
        )
        
        self.text_cache = {}
        self.graph_context = defaultdict(list) # map[node_id] -> list of edges

    def _load_auxiliary_data(self):
        """Carga texto y GRAFO en memoria."""
        
        # 1. Cargar Texto (Cache)
        if os.path.exists(self.docs_path):
            logger.info("📖 Cargando textos narrativos...")
            with open(self.docs_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        did = doc.get("id") or doc.get("title")
                        txt = doc.get("text") or doc.get("content", "")
                        if did and txt: self.text_cache[did] = txt[:2000]
                    except: continue
        
        # 2. Cargar Aristas (Grafo)
        if os.path.exists(self.edges_path):
            logger.info("🕸️ Cargando estructura del grafo (Edges)...")
            with open(self.edges_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        edge = json.loads(line)
                        src = edge['source']
                        rel = edge['relation']
                        tgt = edge['target']
                        
                        # Guardamos aristas SALIENTES (Lo que el nodo hace)
                        self.graph_context[src].append(f"(This Node) --[{rel}]--> {tgt}")
                        
                        # Guardamos aristas ENTRANTES (Lo que le hacen al nodo)
                        # CRÍTICO para Battles/Locations que son 'objetivos'
                        self.graph_context[tgt].append(f"{src} --[{rel}]--> (This Node)")
                    except: continue
            logger.info(f"🕸️ Grafo cargado con contexto para {len(self.graph_context)} nodos.")

    def _load_and_stratify_nodes(self, total_questions: int) -> List[Dict]:
        nodes_by_type: Dict[str, List[Dict]] = {k: [] for k in TARGET_DISTRIBUTION.keys()}
        nodes_by_type["Other"] = [] 

        if not os.path.exists(self.nodes_path):
            return []

        self._load_auxiliary_data() # Carga Edges y Texto

        logger.info("📦 Clasificando nodos...")
        with open(self.nodes_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    n = json.loads(line)
                    nid = n.get("id")
                    if not nid: continue
                    
                    ntype = n.get("type", "Other")
                    
                    has_edges = len(self.graph_context.get(nid, [])) > 0
                    has_props = len(n.get("properties", {})) > 0
                    has_text = nid in self.text_cache
                    
                    # Inyectamos contexto enriquecido
                    if has_edges:
                        # Limitamos a 15 conexiones aleatorias para no saturar el prompt
                        edges = self.graph_context[nid]
                        if len(edges) > 15:
                            edges = random.sample(edges, 15)
                        n["graph_connections"] = edges
                    
                    if has_text:
                        n["narrative_text"] = self.text_cache[nid]

                    if has_edges or has_props or has_text:
                        if ntype in nodes_by_type:
                            nodes_by_type[ntype].append(n)
                        else:
                            nodes_by_type["Other"].append(n)
                except: continue

        # Selección Estratificada (Igual que antes)
        selected_nodes = []
        target_nodes_count = math.ceil(total_questions / 2.5)
        
        for ntype, ratio in TARGET_DISTRIBUTION.items():
            count_needed = math.ceil(target_nodes_count * ratio)
            available = nodes_by_type.get(ntype, [])
            if not available:
                logger.warning(f"⚠️ {ntype}: 0 nodos disponibles.")
                continue
            sample = random.sample(available, min(count_needed, len(available)))
            selected_nodes.extend(sample)

        # Relleno si falta
        if len(selected_nodes) < target_nodes_count:
            deficit = target_nodes_count - len(selected_nodes)
            extras = random.sample(nodes_by_type.get("Character", []), min(deficit, len(nodes_by_type.get("Character", []))))
            selected_nodes.extend(extras)

        random.shuffle(selected_nodes)
        return selected_nodes

    def _create_chain(self):
        # 1. Instrucciones del Sistema (Estáticas)
        system_instruction = """
        You are an expert Game of Thrones trivia master.
        
        Task: Generate 2 DISTINCT Question-Answer pairs based strictly on the provided JSON data.
        
        Strategy:
        - If 'graph_connections' exist, ask about relationships (e.g., "Who participated in X?"). Label evidence_source='Graph'.
        - If 'narrative_text' exists, ask about history/reasons. Label evidence_source='Text'.
        - Ground Truth must be exact strings found in the JSON.
        
        Return ONLY a JSON object with a list 'qa_pairs'.
        """

        # 2. Entrada del Usuario (Dinámica)
        human_instruction = """
        Here is the Input Node data:
        {node_json}
        """

        # 3. Construcción del Prompt con separación clara
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", human_instruction)
        ])
        
        parser = JsonOutputParser(pydantic_object=QABatch)
        
        return prompt | self.llm | parser

    def generate(self, num_questions=150):
        nodes = self._load_and_stratify_nodes(num_questions)
        if not nodes: 
            logger.error("❌ No hay nodos para procesar.")
            return

        chain = self._create_chain()
        dataset = []
        
        logger.info(f"🚀 Iniciando generación con LLM sobre {len(nodes)} nodos...")
        
        # Barra de progreso manual
        pbar = tqdm(total=num_questions, desc="Generando preguntas")
        
        for node in nodes:
            if len(dataset) >= num_questions: break
            
            try:
                node_context = {
                    "id": node.get("id"),
                    "type": node.get("type"),
                    "graph_connections": node.get("graph_connections", []),
                    "properties": node.get("properties"),
                    "narrative_text": node.get("narrative_text", "")
                }
                
                response = chain.invoke({"node_json": json.dumps(node_context)})
                
                if "qa_pairs" in response:
                    for item in response["qa_pairs"]:
                        if len(dataset) >= num_questions: break
                        if "ground_truth" not in item:
                            for alias in ["answer", "correct_answer", "result", "Answer"]:
                                if alias in item:
                                    item["ground_truth"] = item.pop(alias)
                                    break
                        if "ground_truth" not in item:
                            logger.warning(f"⚠️ Pregunta descartada por falta de ground_truth: {item}")
                            continue
                        item["question_id"] = f"gen_{len(dataset)}"
                        item["source_type"] = node.get("type")
                        dataset.append(item)
                        pbar.update(1)
                else:
                    logger.warning(f"⚠️ Respuesta vacía del LLM para nodo: {node.get('id')}")

            except Exception as e:
                logger.error(f"❌ Error generando para '{node.get('id')}': {str(e)}")
                continue
        
        pbar.close()
        
        if dataset:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as f:
                for item in dataset:
                    record = item if isinstance(item, dict) else item.dict()
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"✅ Dataset Generado: {self.output_path} ({len(dataset)} items)")
        else:
            logger.error("❌ No se generó ninguna pregunta. Revisa los errores arriba.")

if __name__ == "__main__":
    generator = StratifiedDatasetGenerator(data_dir="data/processed")
    generator.generate(num_questions=150)