"""Pydantic schemas for summary endpoints."""

import json
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class SummaryResponse(BaseModel):
    """Response model for document summary."""

    id: int = Field(..., description="Summary ID")
    document_id: int = Field(..., description="Associated document ID")
    content: str = Field(..., description="Generated summary text")
    page_citations: list[int] = Field(
        default_factory=list,
        description="Page numbers referenced in the summary",
    )
    created_at: datetime = Field(..., description="Generation timestamp")

    model_config = {"from_attributes": True}

    @field_validator("page_citations", mode="before")
    @classmethod
    def deserialize_page_citations(cls, v: str | list[int]) -> list[int]:
        """Deserialize page_citations from JSON string if needed."""
        if isinstance(v, str):
            return json.loads(v)
        return v
