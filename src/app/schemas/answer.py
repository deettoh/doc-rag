"""Pydantic schemas for answer submission endpoints."""

from datetime import datetime

from pydantic import BaseModel, Field


class AnswerSubmissionRequest(BaseModel):
    """Request model for submitting an answer to a generated question."""

    user_answer: str = Field(
        ...,
        min_length=1,
        description="User-submitted answer text",
    )


class AnswerResponse(BaseModel):
    """Response model for an evaluated and stored answer."""

    id: int = Field(..., description="Answer ID")
    question_id: int = Field(..., description="Associated question ID")
    user_answer: str = Field(..., description="Submitted answer text")
    score: float | None = Field(None, description="Evaluation score in range 0.0-1.0")
    feedback: str | None = Field(None, description="Evaluation feedback")
    created_at: datetime = Field(..., description="Submission timestamp")

    model_config = {"from_attributes": True}
