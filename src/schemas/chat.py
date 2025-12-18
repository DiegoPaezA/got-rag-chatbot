"""Pydantic models for chat request/response payloads."""

from pydantic import BaseModel
from typing import List, Tuple, Optional


class ChatMessage(BaseModel):
	"""Single chat message in history."""
	role: str  # "Human" or "AI"
	content: str


class ChatRequest(BaseModel):
	"""Payload for a chat request."""
	question: str
	history: Optional[List[Tuple[str, str]]] = None


class ChatResponse(BaseModel):
	"""Model for a chat response including optional context."""
	answer: str
	refined_query: Optional[str] = None
