import logging
import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config_manager import ConfigManager
from src.utils.llm_factory import LLMFactory

logger = logging.getLogger("ContextAugmenter")

class ContextAugmenter:
    """Autonomous component that converts raw graph/vector data into a coherent narrative.

    Produces a concise Intelligence Report for the generator step by summarizing
    graph records and formatting vector snippets.
    """

    def __init__(self, config_path: str = "cfg/config.json", llm: BaseChatModel | None = None):
        """Initialize augmenter and its internal LLM.

        Args:
            config_path: Path to configuration file for LLM settings.
            llm: Optional pre-instantiated chat model (for dependency injection/testing).
        """
        load_dotenv()
        
        # Load configuration using ConfigManager
        config_manager = ConfigManager()
        config_manager.load(config_path)
        llm_config = config_manager.get_llm_config("augmenter")
        
        # Internal LLM instance (temperature 0 for factual fidelity)
        if llm is not None:
            self.llm = llm
        else:
            provider = llm_config.get("provider", "google")
            self.llm = LLMFactory.create_llm(llm_config, provider=provider)
        
        logger.info(f"🧠 Augmenter initialized with internal model: {llm_config.get('model_name', llm_config.get('model', 'gemini-2.5-flash'))}")

        # Data Analyst prompt from config
        prompt_template = config_manager.get("prompts", "augmenter", "narrator", default="""
            Task: You are a Data Analyst for the Citadel. Convert raw Graph Database records into clear, concise, natural language facts relevant to the User's Inquiry.
            
            USER INQUIRY: "{query}"
            
            RAW GRAPH DATA (JSON):
            {graph_data}
            
            INSTRUCTIONS:
            1. **Filter Noise:** Ignore records that are empty, null, or generic placeholders.
            2. **Interpret Context:** Use the query to understand specific keys (e.g., interpret 'EpisodeName' as the episode where an event occurred).
            3. **Clarity:** Output readable sentences (e.g., "Fact: Rickon Stark died in the episode 'The Black Queen'.").
            4. **Strictness:** Do not invent information. Only describe the provided JSON.
            
            OUTPUT (Bulleted list of facts):
            """)
        
        self.narrator_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query", "graph_data"]
        )

    def _narrate_graph_data(self, query: str, graph_data: List[Dict[str, Any]]) -> str:
        """Use the internal LLM to narrate graph JSON into readable facts."""
        if not graph_data:
            return ""

        # Serialize JSON for the prompt
        json_str = json.dumps(graph_data, indent=2, ensure_ascii=False)

        narrator_chain = self.narrator_prompt | self.llm | StrOutputParser()

        try:
            narrative = narrator_chain.invoke({
                "query": query,
                "graph_data": json_str
            })
            return f"### 🕸️ Great Ledger (Verified Facts from Graph):\n{narrative}"
        except Exception as e:
            logger.error(f"Error narrating graph data: {e}")
            return f"### 🕸️ Great Ledger (Raw Data):\n{json_str}"

    def _format_vector_context(self, docs: List[str]) -> str:
        """Format vector-retrieved text without using an LLM."""
        if not docs:
            return ""
        formatted = ["### 📜 Ancient Scrolls (Narrative Context):"]
        for i, doc in enumerate(docs, 1):
            clean_doc = " ".join(doc.split())
            formatted.append(f"{i}. {clean_doc}")
        return "\n".join(formatted)

    def build_context(self, query: str, vector_context: List[str], graph_context: List[Dict[str, Any]]) -> str:
        """Orchestrate the final Intelligence Report assembly."""
        # 1) Graph narration (LLM)
        g_text = self._narrate_graph_data(query, graph_context)
        
        # 2) Vector formatting (string ops)
        v_text = self._format_vector_context(vector_context)
        
        # 3) Assemble
        header = f"# 🏰 Citadel Intelligence Report\n**Subject of Inquiry:** '{query}'\n"
        
        combined = [header]
        if g_text: combined.append(g_text)
        if v_text: combined.append(v_text)
        
        if not g_text and not v_text:
            return f"No records found in the archives regarding '{query}'."
            
        return "\n\n".join(combined)