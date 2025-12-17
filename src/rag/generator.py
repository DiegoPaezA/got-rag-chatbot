import os
import json
import logging
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

logger = logging.getLogger("RAGGenerator")

class RAGGenerator:
    """Generate grounded answers by combining vector search and graph results."""

    def __init__(self, config_path: str = "cfg/config.json"):
        """Initialize LLM client and load configuration.

        Args:
            config_path: Path to JSON config with `llm_settings` and `prompts`.
        """
        load_dotenv()

        # 1. Cargar Configuración
        self.config = self._load_config(config_path)
        llm_settings = self.config.get("llm_settings", {})
        prompts = self.config.get("prompts", {})

        # 2. Configurar LLM dinámicamente
        model_name = llm_settings.get("model_name", "gemini-2.5-flash")
        temperature = llm_settings.get("temperature", 0.3)

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        logger.info(f"⚡ Generator initialized with model: {model_name} (T={temperature})")

        # 3. Cargar Prompt Robusto (Default Maester AI)
        default_prompt = """
        You are Maester AI, a keeper of the Citadel's knowledge.
        
        INSTRUCTIONS:
        1. **Graph Context Authority:** The 'Graph' section lists entities DIRECTLY connected to the subject of the user's question. TRUST THIS DATA implicitly for verified relationships (parents, spouses, seats).
        2. **Synthesis:** Combine the structural facts from the Graph with the narrative details from the Text.
        3. **Missing Info:** If the answer is in the Graph but not the Text, answer using the Graph.
        4. **Uncertainty:** If neither source has the answer, respond with "Apologies, the archives hold no answer to that."
        5. **Cite Sources:** Mention if the info came from the Graph ("The Great Ledger"), Text ("Ancient Scrolls"), or both.
        
        --------------
        GRAPH CONNECTIONS (Verified Facts):
        {graph_context}
        --------------
        ANCIENT SCROLLS (Text Fragments):
        {vector_context}
        --------------
        
        QUESTION: {question}
        ANSWER:
        """
        
        self.prompt_template = prompts.get("rag_response", default_prompt)

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load JSON config file; return empty dict if missing or invalid."""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ Error parsing config file: {e}")
                return {}
        else:
            logger.warning(f"⚠️ Config file not found at {path}. Using defaults.")
            return {}

    def _format_graph_context(self, graph_data: List[Dict[str, Any]]) -> str:
        """
        Convierte el grafo en hechos estructurados.
        INCLUYE FILTRO para descartar resultados vacíos (nulls) de OPTIONAL MATCH.
        """
        if not graph_data:
            return "No direct relationships found."

        sentences = []
        REL_KEYS = {"RelType", "rel_type", "relationship", "type(r)", "r", "REL"}

        for node in graph_data:
            # 1. Extraer Relación
            raw_rel = "RELATED_TO"
            for key in node.keys():
                if key in REL_KEYS and node[key]: # Verificamos que no sea None
                    raw_rel = node[key]
                    break
            
            clean_rel = str(raw_rel).replace("_", " ").title()

            # 2. Recopilar Detalles
            details = []
            for key, val in node.items():
                if key not in REL_KEYS and val: # Solo agregamos si val existe (no es None)
                    clean_key = key.replace("n.", "").replace("m.", "").replace("_", " ")
                    details.append(f"{clean_key}: '{val}'")
            
            # --- FILTRO ANTI-RUIDO (NUEVO) ---
            # Si no hay detalles (nombres/ids) Y la relación es genérica o nula, 
            # descartamos este registro. Es un "fantasma" de Neo4j.
            if not details:
                continue
            # ---------------------------------

            details_str = ", ".join(details)
            sentences.append(f"• FACT: Relationship type is '{clean_rel}'. Details: {details_str}.")

        if not sentences:
            return "No direct relationships found."

        return "\n".join(sentences)

    def generate_answer(self, question: str, context: Dict[str, Any]) -> str:
        """Build the final grounded answer from vector and graph context.

        Args:
            question: User's natural language question.
            context: Dict containing `vector_context` and `graph_context`.

        Returns:
            Model-generated answer string.
        """

        # Formatear Contexto Vectorial
        v_text = "\n\n".join(context.get("vector_context", []))

        # Formatear Contexto de Grafo (USANDO LA NUEVA FUNCIÓN)
        g_data = context.get("graph_context", [])
        g_text = self._format_graph_context(g_data)

        print("=== Formatted Graph Context ===")
        print(g_text)
        print("=== End of Graph Context ===")
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["vector_context", "graph_context", "question"]
        )

        chain = prompt | self.llm

        try:
            response = chain.invoke({
                "vector_context": v_text,
                "graph_context": g_text,
                "question": question
            })
            return response.content
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Apologies, the ravens failed to deliver the message."