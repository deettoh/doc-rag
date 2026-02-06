"""DocRAG FastAPI Application Entry Point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.exception_handlers import register_exception_handlers
from app.middleware import configure_logging, register_middleware


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    configure_logging()
    logger.info(
        "Application starting",
        environment=settings.environment,
        debug=settings.debug,
    )
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="DocRAG API",
    description="RAG-based PDF Summarizer + QnA Generator",
    version="0.1.0",
    lifespan=lifespan,
)

register_middleware(app)

register_exception_handlers(app)

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
