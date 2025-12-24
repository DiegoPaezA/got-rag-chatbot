import logging
import os
import json
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from src.rag.graph_search import GraphSearcher
from src.config_manager import ConfigManager
from src.utils.llm_factory import LLMFactory

logger = logging.getLogger("HybridRetriever")

class HybridRetriever:
    def __init__(self, data_dir: str = "data/processed", config_path: str = "cfg/config.json", llm: BaseChatModel | None = None):
        """Initialize retriever with graph and vector backends plus query rewriter.

        Args:
            data_dir: Base path for processed data; used to locate Chroma DB.
            config_path: Path to configuration with LLM and embedding settings.
        """
        load_dotenv()
        
        # Load configuration using ConfigManager
        config_manager = ConfigManager()
        config_manager.load(config_path)
        llm_config = config_manager.get_llm_config("retriever")
        embedding_config = config_manager.get("embedding_settings", default={})
        processing_config = config_manager.get_processing_config("retriever")
        
        # Initialize GraphSearcher (pure Cypher-based retrieval)
        self.graph_searcher = GraphSearcher(config_path=config_path)

        # Initialize vector store
        data_root = os.path.dirname(data_dir.rstrip('/'))
        self.chroma_path = os.path.join(data_root, "chromadb")
        
        if os.path.exists(self.chroma_path):
            embeddings = GoogleGenerativeAIEmbeddings(
                model=embedding_config.get("model_name", "models/text-embedding-004"),
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            self.vector_store = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=embeddings,
                collection_name=embedding_config.get("collection_name", "got_knowledge_base")
            )
        else:
            self.vector_store = None

        if llm is not None:
            self.llm = llm
        else:
            provider = llm_config.get("provider", "google")
            self.llm = LLMFactory.create_llm(llm_config, provider=provider)
        
        # Store search parameters from config
        self.vector_search_k = processing_config.get("vector_search_k", 5)
        self.chat_history_limit = processing_config.get("chat_history_limit", 3)

    def contextualize_query(self, question: str, history: List[Tuple[str, str]]) -> str:
        """Rewrite the question using chat history for standalone retrieval."""
        if not history:
            return question
        
        # Limit history to most recent entries
        recent_history = history[-self.chat_history_limit:] if len(history) > self.chat_history_limit else history
            
        template = """
        Given a chat history and a latest user question which might reference context in the chat history, 
        formulate a standalone question which can be understood without the chat history. 
        Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
        
        Chat History:
        {history}
        
        Latest Question: {question}
        
        Standalone Question:
        """
        
        history_str = "\n".join([f"{role}: {text}" for role, text in recent_history])
        prompt = PromptTemplate(template=template, input_variables=["history", "question"])
        chain = prompt | self.llm
        
        try:
            # Fast LLM invocation
            response = chain.invoke({"history": history_str, "question": question})
            reformulated = response.content.strip()
            if reformulated != question:
                logger.info(f"🔄 Contextualized: '{question}' -> '{reformulated}'")
            return reformulated
        except Exception as e:
            logger.error(f"⚠️ Contextualization failed: {e}")
            return question

    def retrieve(self, query: str, chat_history: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Orchestrate retrieval across rewriter, vector DB, and graph search.

        Steps:
        1) Contextualize the query from chat history
        2) Retrieve similar chunks from the vector store
        3) Generate and execute Cypher against Neo4j graph
        """
        # STEP 1: Contextualize
        refined_query = self.contextualize_query(query, chat_history)
        
        logger.info(f"🔎 Retrieval Strategy: '{query}' -> Using: '{refined_query}'")
        
        context = {
            "refined_query": refined_query,  # Return for UI visibility
            "vector_context": [],
            "graph_context": []
        }

        # STEP 2: Vector search (using refined_query and k from config)
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(refined_query, k=self.vector_search_k)
                context["vector_context"] = [doc.page_content for doc in docs]
            except Exception as e:
                logger.error(f"❌ Vector search failed: {e}")

        # STEP 3: Graph search (using refined_query)
        try:
            # GraphSearcher receives the cleaned query and executes Cypher
            graph_data = self.graph_searcher.run_query(refined_query)
            context["graph_context"] = graph_data
        except Exception as e:
            logger.error(f"❌ Graph search failed: {e}")

        return context