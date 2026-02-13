"""Pydantic schemas for retrieval endpoints."""

from pydantic import BaseModel, Field

from app.config import settings


class RetrievalRequest(BaseModel):
    """Request model for similarity search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language query to search for",
    )
    top_k: int = Field(
        # 1 <= top_k <= 50
        default=settings.top_k_retrieval,
        ge=1,
        le=50,
        description="Number of results to return",
    )


class ChunkResult(BaseModel):
    """A single chunk result from similarity search."""

    chunk_id: int = Field(..., description="Chunk database ID")
    chunk_index: int = Field(..., description="Position of chunk in original document")
    content: str = Field(..., description="Chunk text content")
    page_start: int = Field(..., description="Starting page number")
    page_end: int = Field(..., description="Ending page number")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score (1.0 = most similar)",
    )


class RetrievalResponse(BaseModel):
    """Response model for similarity search."""

    document_id: int = Field(..., description="Document that was searched")
    query: str = Field(..., description="Original query string")
    results: list[ChunkResult] = Field(
        default_factory=list,
        description="Ranked chunk results",
    )
