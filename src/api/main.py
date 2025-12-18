"""FastAPI application entrypoint stub for the GoT RAG API.

Define app, include routers, and wire dependencies here. Currently empty.
"""

from fastapi import FastAPI

app = FastAPI(title="GoT RAG API", version="0.1.0")


@app.get("/health")
def health_check() -> dict:
	"""Basic liveness probe used by monitors and CI."""
	return {"status": "ok"}
