import os
import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from src.config_manager import ConfigManager
from src.utils.llm_factory import LLMFactory

logger = logging.getLogger("RAGGenerator")

class RAGGenerator:
    """
    Pure Generator: Receives a pre-processed context string and generates an answer.
    """

    def __init__(self, config_path: str = "cfg/config.json", llm: BaseChatModel | None = None):
        """Initialize the generator and its LLM backend.

        Args:
            config_path: Path to configuration file.
            llm: Optional pre-instantiated chat model (for dependency injection/testing).
        """
        load_dotenv()
        
        # Load configuration
        config_manager = ConfigManager()
        config_manager.load(config_path)
        llm_config = config_manager.get_llm_config("generator")

        # Init or inject LLM
        if llm is not None:
            self.llm = llm
        else:
            provider = llm_config.get("provider", "google")
            self.llm = LLMFactory.create_llm(llm_config, provider=provider)

        # Get prompt template from config
        self.default_prompt = config_manager.get("prompts", "generator", "system", default="""
        You are Maester AI, a keeper of the Citadel's knowledge.
        
        INSTRUCTIONS:
        1. Answer the user's question using ONLY the provided Context below.
        2. Trust the 'Graph Connections' for factual relationships (family, seats).
        3. Use 'Ancient Scrolls' for narrative details.
        4. If the answer is not in the context, say "The archives are silent on this."
        
        CONTEXT FROM THE ARCHIVES:
        {context}
        
        USER QUESTION:
        {question}
        
        ANSWER:
        """)

    def generate_answer(self, question: str, formatted_context: str) -> str:
        """Generate an answer using the provided pre-formatted context.

        Args:
            question: The user's question.
            formatted_context: The already formatted context string from the augmenter.

        Returns:
            The model-generated answer as a string.
        """
        prompt = PromptTemplate(
            template=self.default_prompt,
            input_variables=["context", "question"]
        )

        chain = prompt | self.llm

        try:
            response = chain.invoke({
                "context": formatted_context,
                "question": question
            })
            return response.content
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Apologies, the ravens failed to deliver the message."