"""Pydantic models for document-related data structures."""

from pydantic import BaseModel
from typing import Optional, Dict, Any


class DocumentSchema(BaseModel):
	"""Represents a document stored in the vector database."""
	id: str
	text: str
	metadata: Optional[Dict[str, Any]] = None
