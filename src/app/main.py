"""DocRAG FastAPI Application Entry Point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="DocRAG API",
    description="RAG-based PDF Summarizer + QnA Generator",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "docrag-api"}
