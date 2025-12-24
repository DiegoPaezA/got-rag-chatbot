import os
import json
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship
# from neo4j.time import DateTime  # Uncomment if using native time types explicitly
from src.config_manager import ConfigManager
from src.utils.llm_factory import LLMFactory

logger = logging.getLogger("GraphSearch")

class GraphSearcher:
    """Generate Cypher from natural language and query Neo4j."""

    def __init__(self, config_path: str = "cfg/config.json", llm: BaseChatModel | None = None):
        """Initialize graph searcher with LLM and Neo4j settings from config."""
        load_dotenv()
        
        # Load configuration
        config_manager = ConfigManager()
        config_manager.load(config_path)
        llm_config = config_manager.get_llm_config("graph_search")
        model_name = llm_config.get("model_name", llm_config.get("model", "gemini-2.5-flash"))
        temperature = llm_config.get("temperature", 0.0)

        logger.info(f"⚙️ Initializing Graph Searcher with model: {model_name} (T={temperature})")

        # Neo4j settings
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None 

        if llm is not None:
            self.llm = llm
        else:
            provider = llm_config.get("provider", "google")
            self.llm = LLMFactory.create_llm(llm_config, provider=provider)

        self._schema_cache: Optional[str] = None

        # Processing config
        processing_config = config_manager.get_processing_config("graph_search")
        self.cypher_limit = processing_config.get("cypher_limit", 20)

        # --- MEJORA: Prompt por defecto robusto (con reglas de búsqueda difusa) ---
        default_prompt = """
        You are an expert Neo4j Developer translating user questions into Cypher queries.
        
        SCHEMA:
        {schema}

        GOLDEN RULES:
        1. **Fuzzy Matching:** NEVER use exact match (`=`) for names. Always use `toLower(n.prop) CONTAINS toLower('value')`.
        2. **Keyword Splitting:** If searching for a complex entity (e.g., "Valyrian Dagger"), split keywords using AND to capture partial matches:
           `toLower(n.name) CONTAINS 'valyrian' AND toLower(n.name) CONTAINS 'dagger'`
        3. **Entity Discovery:** If unsure of the label, use generic matching or multiple labels: `MATCH (n:House|Location|Object)`.
        4. **Attributes:** Return specific properties like `n.name`, `n.id` explicitly.
        5. **Limit:** Always add `LIMIT 20`.

        Question: {question}
        Cypher:
        """
        
        # Intentamos cargar desde config, si falla usamos el default mejorado
        self.cypher_prompt_template = config_manager.get("prompts", "cypher_generation", default=default_prompt)

    def _get_driver(self):
        """Return or initialize the Neo4j driver connection."""
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
        """Close the driver if it was opened."""
        if self.driver:
            self.driver.close()

    def _serialize(self, data: Any) -> Any:
        """Serialize Neo4j driver results into JSON-serializable objects."""
        if isinstance(data, Node):
            node_dict = dict(data.items())
            label = list(data.labels)[0] if data.labels else "Node"
            node_dict["_label"] = label
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
        elif hasattr(data, 'iso_format'):
             return data.iso_format()
        else:
            return data

    def get_schema(self) -> str:
        """Fetch schema with caching mechanism."""
        if self._schema_cache:
            return self._schema_cache

        try:
            driver = self._get_driver()
            with driver.session() as session:
                labels_res = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
                labels = [r["label"] for r in labels_res]

                rels_res = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
                rel_types = [r["relationshipType"] for r in rels_res]

                schema_lines = [
                    f"Node Labels: {', '.join(labels)}",
                    f"Relationship Types: {', '.join(rel_types)}",
                    ""
                ]

                # Propiedades de muestra para contexto
                important_labels = ["Character", "House", "Location", "Battle", "Object"]
                for label in important_labels:
                    if label in labels:
                        q = f"MATCH (n:`{label}`) RETURN keys(n) AS k LIMIT 5"
                        res = session.run(q)
                        all_keys = set()
                        for r in res:
                            all_keys.update(r["k"])
                        if all_keys:
                            schema_lines.append(f"Properties for :{label} -> {', '.join(sorted(list(all_keys)))}")

                self._schema_cache = "\n".join(schema_lines)
                return self._schema_cache

        except Exception as e:
            logger.warning(f"⚠️ Could not fetch live schema ({e}). Using SAFE fallback.")
            return """
                Node Labels: Character, House, Location, Battle, Object
                Relationship Types: CHILD_OF, FATHER, MOTHER, BELONGS_TO, KILLED, OWNED_BY, SEATED_AT
            """

    def generate_cypher(self, question: str) -> str:
        """Generate a Cypher query from a natural-language question using the LLM."""
        schema = self.get_schema()
        
        prompt = PromptTemplate(
            template=self.cypher_prompt_template,
            input_variables=["schema", "question"]
        )
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({"schema": schema, "question": question})
            content = response.content
            content = content.replace("```cypher", "").replace("```", "").strip()
            if content.endswith(";"):
                content = content[:-1]
            return content
        except Exception as e:
            logger.error(f"❌ Error generating Cypher: {e}")
            return ""

    def run_query(self, question: str) -> List[Dict]:
        """Generate and execute Cypher for the question, returning serialized results."""
        cypher_query = self.generate_cypher(question)
        
        if not cypher_query:
            return []
            
        logger.info(f"📝 Generated Cypher: {cypher_query}")
        
        try:
            driver = self._get_driver()
            with driver.session() as session:
                result = session.run(cypher_query)
                data = [record.data() for record in result]
                return self._serialize(data)

        except Exception as e:
            logger.error(f"❌ Neo4j Execution Error: {e}")
            return []

if __name__ == "__main__":
    # Prueba rápida
    searcher = GraphSearcher()
    q = "Who owns the Valyrian Dagger?"
    print(f"Q: {q}")
    print(f"Cypher: {searcher.generate_cypher(q)}")