import os
import json
import logging
import shutil
from typing import List, Dict, Any

from tqdm import tqdm
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("VectorStore")

class VectorDBBuilder:
    """Build and manage a vector database for document embeddings.
    
    Handles document loading, text chunking, embedding generation, and
    ingestion into ChromaDB. Supports multiple embedding providers (Google, HuggingFace)
    with configurable chunking strategies.
    """
    
    def __init__(self, data_dir: str, config_path: str = "cfg/config.json"):
        self.data_dir = data_dir
        self.docs_path = os.path.join(data_dir, "documents.jsonl")
        
        data_root = os.path.dirname(data_dir.rstrip('/')) 
        self.persist_dir = os.path.join(data_root, "chromadb")
        
        self.config = self._load_config(config_path)
        emb_settings = self.config.get("embedding_settings", {})
        
        self.provider = emb_settings.get("provider", "google").lower()
        self.model_name = emb_settings.get("model_name", "models/text-embedding-004")
        self.device = emb_settings.get("device", "cpu")

        # Text chunking configuration
        chunk_size = emb_settings.get("chunk_size", 1000)
        chunk_overlap = emb_settings.get("chunk_overlap", 200)
        separators = emb_settings.get("separators", ["\n\n", "\n", ". ", " ", ""])

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

        logger.info(f"⚙️ Provider: {self.provider.upper()} | Model: {self.model_name}")
        logger.info(f"✂️ Chunking Strategy: Size={chunk_size}, Overlap={chunk_overlap}")

        self.embedding_model = self._initialize_embedding_model()
        self.collection_name = "got_knowledge_base"

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not os.path.exists(path): return {}
        with open(path, "r", encoding="utf-8") as f: return json.load(f)

    def _initialize_embedding_model(self):
        """Initialize the embedding model based on configured provider.
        
        Returns:
            Embedding model instance (Google or HuggingFace).
            
        Raises:
            ValueError: If provider is unsupported or API key is missing.
        """
        load_dotenv()
        if self.provider == "google":
            if not os.getenv("GOOGLE_API_KEY"): raise ValueError("❌ GOOGLE_API_KEY missing")
            return GoogleGenerativeAIEmbeddings(model=self.model_name, google_api_key=os.getenv("GOOGLE_API_KEY"))
        elif self.provider == "huggingface":
            return HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs={'device': self.device})
        else:
            raise ValueError(f"❌ Provider '{self.provider}' not supported.")

    def build(self, reset: bool = False):
        """Build the vector database by loading, chunking, and embedding documents.
        
        Args:
            reset: If True, deletes existing database before building.
            
        Pipeline:
            1. Load documents from documents.jsonl
            2. Split documents into chunks using configured strategy
            3. Generate embeddings and store in ChromaDB
        """
        if reset and os.path.exists(self.persist_dir):
            logger.warning(f"🗑️ Deleting DB at {self.persist_dir}")
            try: shutil.rmtree(self.persist_dir)
            except Exception as e: logger.error(f"Error deleting: {e}")

        raw_docs = self._load_documents()
        if not raw_docs:
            logger.error("❌ No documents to ingest.")
            return

        logger.info(f"✂️ Splitting {len(raw_docs)} articles...")
        splitted_docs = self.text_splitter.split_documents(raw_docs)
        
        logger.info(f"📊 Total chunks generated: {len(splitted_docs)}")

        logger.info(f"🧠 Initializing ChromaDB...")
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_dir
        )

        batch_size = 100
        for i in tqdm(range(0, len(splitted_docs), batch_size), desc="Embedding Chunks", unit="batch"):
            batch = splitted_docs[i : i + batch_size]
            vector_store.add_documents(documents=batch)

        logger.info(f"✅ Vector Database built at: {self.persist_dir}")

    def _load_documents(self) -> List[Document]:
        """Load documents from JSONL file.
        
        Returns:
            List of LangChain Document objects with page content and metadata.
        """
        if not os.path.exists(self.docs_path):
            return []
        documents = []
        with open(self.docs_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text_content = data.get("text", "")
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "original_id": data.get("id"),
                            "type": data.get("metadata", {}).get("type", "Unknown"),
                            "source": "wiki"
                        }
                    )
                    documents.append(doc)
                except: continue
        return documents

if __name__ == "__main__":
    builder = VectorDBBuilder(data_dir="data/processed")
    builder.build(reset=True)