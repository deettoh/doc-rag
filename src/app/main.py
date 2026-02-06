"""DocRAG FastAPI Application Entry Point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.exception_handlers import register_exception_handlers
from app.middleware import configure_logging, register_middleware
from app.routers import documents_router
from app.schemas import HealthResponse


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

# Register routers
app.include_router(documents_router, prefix="/api")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
)
async def health_check() -> HealthResponse:
    """Health check endpoint for container orchestration."""
    return HealthResponse(
        status="healthy",
        service="docrag-api",
        version="0.1.0",
    )
