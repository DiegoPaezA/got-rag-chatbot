import os
import json
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from neo4j import GraphDatabase

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GraphSearch")

class GraphSearcher:
    """Generate Cypher from natural language and query Neo4j."""

    def __init__(self, config_path: str = "cfg/config.json"):
        """Initialize prompt, LLM client, and Neo4j connection settings.

        Args:
            config_path: Path to a JSON config file containing LLM settings and prompts.
        """
        load_dotenv()

        self.config = self._load_config(config_path)

        llm_settings = self.config.get("llm_settings", {})
        model_name = llm_settings.get("model_name", "gemini-2.5-flash")
        temperature = llm_settings.get("temperature", 0.0)
        
        logger.info(f"⚙️ Initializing Graph Searcher with model: {model_name} (T={temperature})")

        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        self.driver = None 

        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("❌ GOOGLE_API_KEY not found in .env")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        self.cypher_prompt_template = self.config.get("prompts", {}).get(
            "cypher_generation", 
            # Fallback if missing in config
            "Task: Generate Cypher query for: {question} with schema {schema}"
        )

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Safely load JSON config; return empty dict on failure."""
        if not os.path.exists(path):
            logger.warning(f"⚠️ Config file not found at {path}. Using defaults.")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            return {}

    def _get_driver(self):
        """Get or create the Neo4j driver lazily."""
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
        """Close the Neo4j driver if initialized."""
        if self.driver:
            self.driver.close()

    def get_schema(self) -> str:
        """Return the graph schema string; fallback to static schema if unavailable."""
        try:
            driver = self._get_driver()
            query = "CALL db.schema.visualization() YIELD nodes, relationships RETURN nodes, relationships"

            # Placeholder: convert live schema to a formatted string if needed.
            return """
            Node labels: Character, House, Organization, Battle, Location, Object, Creature, Lore
            
            Relationship types:
            (:Character)-[:CHILD_OF]->(:Character)
            (:Character)-[:MARRIED_TO]->(:Character)
            (:Character)-[:LOVER_OF]->(:Character)
            (:Character)-[:KILLED]->(:Character)
            (:Character)-[:KILLED_BY]->(:Character)
            (:Character)-[:BELONGS_TO]->(:House)
            (:Character)-[:BELONGS_TO]->(:Organization)
            (:Character)-[:OWNS_WEAPON]->(:Object)
            (:Character)-[:PARTICIPATED_IN]->(:Battle)
            (:Character)-[:BORN_IN]->(:Location)
            (:House)-[:LOCATED_IN]->(:Location)
            (:Battle)-[:TOOK_PLACE_AT]->(:Location)
            """
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch live schema: {e}")
            return "Schema not available."

    def generate_cypher(self, question: str) -> str:
        """Convert a natural language question into Cypher using the configured prompt."""
        
        schema = self.get_schema()
        
        prompt = PromptTemplate(
            template=self.cypher_prompt_template,
            input_variables=["schema", "question"]
        )
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({"schema": schema, "question": question})
            query = response.content.replace("```cypher", "").replace("```", "").strip()
            return query
        except Exception as e:
            logger.error(f"❌ Error generating Cypher: {e}")
            return ""

    def run_query(self, question: str) -> List[Dict]:
        """Generate a Cypher query from a question, execute it, and return records."""
        cypher_query = self.generate_cypher(question)
        
        if not cypher_query:
            return []
            
        logger.info(f"📝 Generated Cypher: {cypher_query}")
        
        try:
            driver = self._get_driver()
            with driver.session() as session:
                result = session.run(cypher_query)
                records = [dict(record) for record in result]
                return records
        except Exception as e:
            logger.error(f"❌ Error executing Cypher in Neo4j: {e}")
            return []

# --- PRUEBA INDIVIDUAL ---
if __name__ == "__main__":
    searcher = GraphSearcher()
    # Quick test without requiring Neo4j up (generation only)
    q = "Who is the father of Jon Snow?"
    print(f"Question: {q}")
    cypher = searcher.generate_cypher(q)
    print(f"Generated Cypher: {cypher}")