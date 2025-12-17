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
        load_dotenv()
        
        # 1. Cargar Configuración para el LLM interno
        self.config = self._load_config(config_path)
        llm_settings = self.config.get("llm_settings", {})
        
        # 2. Inicializar GraphSearcher (Pura búsqueda)
        self.graph_searcher = GraphSearcher(config_path=config_path)

        # 3. Inicializar Vector Store
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
            temperature=0, # Queremos precisión al reescribir
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def _load_config(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path): return {}
        with open(path, "r") as f: return json.load(f)

    def contextualize_query(self, question: str, history: List[Tuple[str, str]]) -> str:
        """Reescribe la pregunta basándose en el historial antes de buscar."""
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
            # Invocación rápida al LLM
            response = chain.invoke({"history": history_str, "question": question})
            reformulated = response.content.strip()
            if reformulated != question:
                logger.info(f"🔄 Contextualized: '{question}' -> '{reformulated}'")
            return reformulated
        except Exception as e:
            logger.error(f"⚠️ Contextualization failed: {e}")
            return question

    def retrieve(self, query: str, chat_history: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Orquesta la búsqueda:
        1. Limpia la pregunta (Contextualize)
        2. Busca en Vectores (Similarity)
        3. Busca en Grafo (Cypher)
        """
        # PASO 1: Contextualizar
        refined_query = self.contextualize_query(query, chat_history)
        
        logger.info(f"🔎 Retrieval Strategy: '{query}' -> Using: '{refined_query}'")
        
        context = {
            "refined_query": refined_query, # Devolvemos esto para que la UI lo sepa
            "vector_context": [],
            "graph_context": []
        }

        # PASO 2: Vectores (Usando refined_query)
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(refined_query, k=4)
                context["vector_context"] = [doc.page_content for doc in docs]
            except Exception as e:
                logger.error(f"❌ Vector search failed: {e}")

        # PASO 3: Grafo (Usando refined_query)
        try:
            # El graph_searcher ahora solo recibe la query limpia y ejecuta Cypher
            graph_data = self.graph_searcher.run_query(refined_query)
            context["graph_context"] = graph_data
        except Exception as e:
            logger.error(f"❌ Graph search failed: {e}")

        return context