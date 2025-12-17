import os
import json
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship

logger = logging.getLogger("GraphSearch")

class GraphSearcher:
    """Generate Cypher from natural language and query Neo4j."""

    def __init__(self, config_path: str = "cfg/config.json"):
        load_dotenv()
        
        # 1. Carga de Configuración
        self.config = self._load_config(config_path)
        llm_settings = self.config.get("llm_settings", {})
        model_name = llm_settings.get("model_name", "gemini-2.5-flash")
        temperature = llm_settings.get("temperature", 0.0)

        logger.info(f"⚙️ Initializing Graph Searcher with model: {model_name} (T={temperature})")

        # 2. Configuración Neo4j
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None 

        # 3. Configuración LLM
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("❌ GOOGLE_API_KEY not found in .env")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # 4. Cache del Esquema (CRÍTICO PARA PERFORMANCE)
        self._schema_cache: Optional[str] = None

        # 5. Prompt Robusto (Fallback)
        # Si no hay config, usamos este prompt experto por defecto
        default_prompt = """
        You are an expert Neo4j Developer translating user questions into Cypher queries.
        
        SCHEMA:
        {schema}

        GOLDEN RULES:
        1. **Case Insensitivity:** Always use `toLower(n.prop) CONTAINS toLower('value')` for textual searches. Never use `=`.
        2. **Direction:** If unsure about direction, use undirected paths: `(a)-[:REL]-(b)`.
        3. **Attributes:** The main property is usually `name` or `Title`.
        4. **Output:** Return ONLY the Cypher query. No markdown, no explanations.
        5. **Limit:** Always add `LIMIT 20` to prevent huge responses.
        6. **Current Date:** If asked about "current" status, assume end of series state.

        Question: {question}
        Cypher:
        """
        
        self.cypher_prompt_template = self.config.get("prompts", {}).get(
            "cypher_generation", 
            default_prompt
        )

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Safely load JSON config; return empty dict on failure."""
        # Hacemos la ruta absoluta respecto al archivo actual para evitar errores
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Asumiendo que cfg está en la raíz del proyecto (dos niveles arriba de src/rag)
        project_root = os.path.dirname(os.path.dirname(base_dir))
        full_path = os.path.join(project_root, path)

        if not os.path.exists(full_path):
            # Intento secundario directo (por si corres desde root)
            if os.path.exists(path):
                full_path = path
            else:
                logger.warning(f"⚠️ Config file not found at {path}. Using defaults.")
                return {}
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            return {}

    def _get_driver(self):
        if self.driver is None:
            try:
                self.driver = GraphDatabase.driver(
                    self.neo4j_uri, 
                    auth=(self.neo4j_user, self.neo4j_password)
                )
            except Exception as e:
                logger.error(f"❌ Failed to connect to Neo4j: {e}")
                raise e
        return self.driver

    def close(self):
        if self.driver:
            self.driver.close()

    def _serialize(self, data: Any) -> Any:
        if isinstance(data, Node):
            node_dict = dict(data.items())
            # Priorizamos 'name' o 'Title' para la visualización
            label = list(data.labels)[0] if data.labels else "Node"
            return {
                "type": label,
                "properties": node_dict,
                "id": data.element_id
            }
        elif isinstance(data, Relationship):
            return {
                "type": data.type,
                "start": data.start_node.element_id,
                "end": data.end_node.element_id,
                "properties": dict(data.items())
            }
        elif isinstance(data, list):
            return [self._serialize(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._serialize(v) for k, v in data.items()}
        else:
            return data

    def get_schema(self) -> str:
        """Fetch schema with caching mechanism."""
        # 1. Retornar Caché si existe
        if self._schema_cache:
            return self._schema_cache

        MAX_PATTERN_SAMPLES = 1000   
        PER_REL_LIMIT = 50          

        try:
            driver = self._get_driver()
            with driver.session() as session:
                # 1) Labels + Rel Types
                labels_res = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
                labels = [r["label"] for r in labels_res]

                rels_res = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
                rel_types = [r["relationshipType"] for r in rels_res]

                schema_lines = [
                    f"Node Labels: {', '.join(labels)}",
                    f"Relationship Types: {', '.join(rel_types)}",
                    ""
                ]

                # 2) Properties (Simplificado para velocidad)
                # Solo traemos propiedades de Character y House que son las importantes
                important_labels = ["Character", "House", "Location", "Battle"]
                for label in important_labels:
                    if label in labels:
                        q = f"MATCH (n:`{label}`) RETURN keys(n) AS k LIMIT 10"
                        res = session.run(q)
                        all_keys = set()
                        for r in res:
                            all_keys.update(r["k"])
                        if all_keys:
                            schema_lines.append(f"Properties for :{label} -> {', '.join(sorted(list(all_keys)))}")

                # 3) Pattern Sampling (Tu lógica experta, mantenida)
                schema_lines.append("\nValid Relationships (Sampled):")
                if rel_types:
                    for rt in rel_types:
                        q = f"""
                        MATCH (a)-[r:`{rt}`]->(b)
                        RETURN labels(a)[0] AS la, labels(b)[0] AS lb, count(*) as cnt
                        ORDER BY cnt DESC LIMIT 5
                        """
                        try:
                            res = session.run(q)
                            for row in res:
                                schema_lines.append(f"(:{row['la']})-[:{rt}]->(:{row['lb']})")
                        except Exception:
                            continue

                self._schema_cache = "\n".join(schema_lines)
                return self._schema_cache

        except Exception as e:
            logger.warning(f"⚠️ Could not fetch live schema ({e}). Using SAFE fallback.")
            return """
                Node Labels: Character, House, Location, Battle
                Relationship Types: CHILD_OF, FATHER, MOTHER, BELONGS_TO, KILLED
            """

    def generate_cypher(self, question: str) -> str:
        schema = self.get_schema()
        
        prompt = PromptTemplate(
            template=self.cypher_prompt_template,
            input_variables=["schema", "question"]
        )
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({"schema": schema, "question": question})
            # Limpieza robusta de Markdown
            content = response.content
            content = content.replace("```cypher", "").replace("```", "").strip()
            # Eliminar punto y coma final si existe (a veces causa error en driver python)
            if content.endswith(";"):
                content = content[:-1]
            return content
        except Exception as e:
            logger.error(f"❌ Error generating Cypher: {e}")
            return ""

    def run_query(self, question: str) -> List[Dict]:
        cypher_query = self.generate_cypher(question)
        
        if not cypher_query:
            return []
            
        logger.info(f"📝 Generated Cypher: {cypher_query}")
        
        try:
            driver = self._get_driver()
            with driver.session() as session:
                result = session.run(cypher_query)
                # Serialize immediately to prevent cursor expiration
                data = [record.data() for record in result]
                return self._serialize(data)

        except Exception as e:
            # Si el error es de sintaxis, es útil loguearlo para debuggear
            logger.error(f"❌ Neo4j Execution Error: {e}")
            return []

if __name__ == "__main__":
    # Test Rápido
    searcher = GraphSearcher()
    q = "Who is the father of Jon Snow?"
    print(f"Q: {q}")
    print(f"Cypher: {searcher.generate_cypher(q)}")
    # results = searcher.run_query(q)
    # print(json.dumps(results, indent=2))