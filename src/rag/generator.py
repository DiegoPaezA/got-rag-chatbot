import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from src.config_manager import ConfigManager

logger = logging.getLogger("RAGGenerator")

class RAGGenerator:
    """
    Pure Generator: Receives a pre-processed context string and generates an answer.
    """

    def __init__(self, config_path: str = "cfg/config.json"):
        """Initialize the generator and its LLM backend.

        Args:
            config_path: Optional future use for custom prompts/settings.
        """
        load_dotenv()
        
        # Load configuration
        config_manager = ConfigManager()
        llm_config = config_manager.get_llm_config("generator")
        
        # Init LLM
        self.llm = ChatGoogleGenerativeAI(
            model=llm_config.get("model", "gemini-2.5-flash"),
            temperature=llm_config.get("temperature", 0.3),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

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