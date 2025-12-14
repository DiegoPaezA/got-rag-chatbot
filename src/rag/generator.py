import os
import json
import logging
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
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
        # Usamos la temperatura del config, o 0.3 por defecto si no existe
        temperature = llm_settings.get("temperature", 0.3)

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        logger.info(f"⚡ Generator initialized with model: {model_name} (T={temperature})")

        # 3. Cargar Prompt desde Configuración (con Fallback por seguridad)
        default_prompt = """
        You are an AI assistant for Game of Thrones. Answer based on context.
        Context: {vector_context}
        Graph: {graph_context}
        Question: {question}
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

    def generate_answer(self, question: str, context: Dict[str, Any]) -> str:
        """Build the final grounded answer from vector and graph context.

        Args:
            question: User's natural language question.
            context: Dict containing `vector_context` (list of strings) and
                `graph_context` (list of dicts or strings).

        Returns:
            Model-generated answer string based on supplied context.
        """

        # Formatear Contexto Vectorial
        v_text = "\n\n".join(context.get("vector_context", []))

        # Formatear Contexto de Grafo (Manejo robusto de JSON)
        g_data = context.get("graph_context", [])
        if g_data:
            # Aseguramos que sea un string bonito
            g_text = json.dumps(g_data, indent=2, ensure_ascii=False)
        else:
            g_text = "No direct relationships found."

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