"""DocRAG FastAPI Application Entry Point."""

from fastapi import FastAPI

app = FastAPI(
    title="DocRAG API",
    description="RAG-based PDF Summarizer + QnA Generator",
    version="0.1.0",
)
