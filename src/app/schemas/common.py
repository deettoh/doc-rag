"""Common Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., examples=["healthy"])
    service: str = Field(..., examples=["docrag-api"])
    version: str = Field(..., examples=["0.1.0"])


class ErrorResponse(BaseModel):
    """Standard error response model."""

    detail: str = Field(..., description="Human-readable error message")
    error_code: str = Field(..., description="Machine-readable error code")
    request_id: str | None = Field(None, description="Request tracking ID")


class ErrorResponseWithDetails(ErrorResponse):
    """Error response with additional details."""

    errors: list[dict] | None = Field(
        None,
        description="Validation error details",
    )
