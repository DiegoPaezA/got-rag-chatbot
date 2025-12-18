"""API routes for the GoT RAG service."""

from fastapi import APIRouter, Depends
from .dependencies import get_retriever, get_generator
from src.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=ChatResponse)
def ask(req: ChatRequest, retriever = Depends(get_retriever), generator = Depends(get_generator)) -> ChatResponse:
	"""Answer a question using hybrid retrieval and generation."""
	ctx = retriever.retrieve(req.question)
	# In this minimal stub, pass raw context to generator; production should use augmenter.
	answer = generator.generate_answer(req.question, str(ctx))
	refined_query = ctx.get("refined_query") if isinstance(ctx, dict) else None
	return ChatResponse(answer=answer, refined_query=refined_query)
