import logging
import os
import json
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from src.rag.graph_search import GraphSearcher

logger = logging.getLogger("HybridRetriever")

class HybridRetriever:
    def __init__(self, data_dir: str = "data/processed", config_path: str = "cfg/config.json"):
        """Initialize retriever with graph and vector backends plus query rewriter.

        Args:
            data_dir: Base path for processed data; used to locate Chroma DB.
            config_path: Path to configuration with LLM and embedding settings.
        """
        load_dotenv()
        
        # 1) Load configuration for internal LLM
        self.config = self._load_config(config_path)
        llm_settings = self.config.get("llm_settings", {})
        
        # 2) Initialize GraphSearcher (pure Cypher-based retrieval)
        self.graph_searcher = GraphSearcher(config_path=config_path)

        # 3) Initialize vector store
        data_root = os.path.dirname(data_dir.rstrip('/'))
        self.chroma_path = os.path.join(data_root, "chromadb")
        
        if os.path.exists(self.chroma_path):
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            self.vector_store = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=embeddings,
                collection_name="got_knowledge_base"
            )
        else:
            self.vector_store = None

        self.llm = ChatGoogleGenerativeAI(
            model=llm_settings.get("model_name", "gemini-2.5-flash"),
            temperature=0,  # Deterministic rewriting
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load JSON config safely; return empty dict on failure."""
        if not os.path.exists(path): return {}
        with open(path, "r") as f: return json.load(f)

    def contextualize_query(self, question: str, history: List[Tuple[str, str]]) -> str:
        """Rewrite the question using chat history for standalone retrieval."""
        if not history:
            return question
            
        template = """
        Given a chat history and a latest user question which might reference context in the chat history, 
        formulate a standalone question which can be understood without the chat history. 
        Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
        
        Chat History:
        {history}
        
        Latest Question: {question}
        
        Standalone Question:
        """
        
        history_str = "\n".join([f"{role}: {text}" for role, text in history])
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

        # STEP 2: Vector search (using refined_query)
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(refined_query, k=4)
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