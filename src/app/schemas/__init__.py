"""Pydantic schemas for DocRAG API."""

from app.schemas.common import ErrorResponse, ErrorResponseWithDetails, HealthResponse

__all__ = [
    "ErrorResponse",
    "ErrorResponseWithDetails",
    "HealthResponse",
]
