import logging
from typing import List, Dict, Any
import os

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Graph searcher for Neo4j-backed retrieval
from src.rag.graph_search import GraphSearcher

logger = logging.getLogger("HybridRetriever")

class HybridRetriever:
    """Combine vector search with graph search to gather contextual evidence."""

    def __init__(self, data_dir: str = "data/processed", config_path: str = "cfg/config.json"):
        """Initialize vector store and graph searcher.

        Args:
            data_dir: Base directory containing processed data and Chroma DB.
            config_path: Path to config used by the graph searcher and LLM.
        """
        load_dotenv()
        self.config_path = config_path

        self.graph_searcher = GraphSearcher(config_path=config_path)

        data_root = os.path.dirname(data_dir.rstrip('/'))
        self.chroma_path = os.path.join(data_root, "chromadb")

        if not os.path.exists(self.chroma_path):
            logger.warning(f"⚠️ Vector DB not found at {self.chroma_path}. Vector search will be empty.")
            self.vector_store = None
        else:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            self.vector_store = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=embeddings,
                collection_name="got_knowledge_base"
            )

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Run hybrid retrieval using both vector and graph sources.

        Vector search: narrative context (descriptions, lore).
        Graph search: exact facts (relationships, affiliations).
        """
        logger.info(f"🔎 Processing query: {query}")
        
        context = {
            "vector_context": [],
            "graph_context": []
        }

        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(query, k=4)
                context["vector_context"] = [doc.page_content for doc in docs]
                logger.info(f"   📄 Retrieved {len(docs)} vector documents.")
            except Exception as e:
                logger.error(f"   ❌ Vector search failed: {e}")

        try:
            graph_data = self.graph_searcher.run_query(query)
            context["graph_context"] = graph_data
            logger.info(f"   🕸️  Retrieved {len(graph_data)} graph records.")
        except Exception as e:
            logger.error(f"   ❌ Graph search failed: {e}")

        return context