"""Dependency wiring for FastAPI routes.

Provide shared singletons like retrievers, generators, and DB clients.
"""

from functools import lru_cache
from src.rag.retriever import HybridRetriever
from src.rag.generator import RAGGenerator


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
	"""Return a cached `HybridRetriever` singleton instance."""
	return HybridRetriever()


@lru_cache(maxsize=1)
def get_generator() -> RAGGenerator:
	"""Return a cached `RAGGenerator` singleton instance."""
	return RAGGenerator()
