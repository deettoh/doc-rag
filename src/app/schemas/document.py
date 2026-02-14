"""Pydantic schemas for document endpoints."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.document import DocumentStatus


class DocumentResponse(BaseModel):
    """Response model for document operations."""

    id: int = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    file_size_bytes: int = Field(..., description="File size in bytes")
    status: DocumentStatus = Field(..., description="Processing status")
    created_at: datetime = Field(..., description="Upload timestamp")

    model_config = {"from_attributes": True}


class DocumentStatusResponse(BaseModel):
    """Response model for document status polling."""

    id: int = Field(..., description="Document ID")
    status: DocumentStatus = Field(..., description="Current processing status")
    page_count: int | None = Field(None, description="Number of pages in PDF")
    error_message: str | None = Field(None, description="Error details if failed")
    updated_at: datetime = Field(..., description="Last status update timestamp")

    model_config = {"from_attributes": True}
