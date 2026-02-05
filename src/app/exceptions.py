"""Custom exceptions for DocRAG application."""

from typing import Any


class DocRAGException(Exception):
    """Base exception for all DocRAG errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class NotFoundError(DocRAGException):
    """Resource not found."""

    def __init__(
        self,
        resource: str,
        resource_id: Any,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=f"{resource} with id '{resource_id}' not found",
            error_code="NOT_FOUND",
            details={
                "resource": resource,
                "resource_id": str(resource_id),
                **(details or {}),
            },
        )


class DomainValidationError(DocRAGException):
    """Input validation failed."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={"field": field, **(details or {})} if field else details,
        )


class ProcessingError(DocRAGException):
    """Document processing failed."""

    def __init__(
        self,
        message: str,
        document_id: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="PROCESSING_ERROR",
            details={"document_id": document_id, **(details or {})}
            if document_id
            else details,
        )


class ExternalServiceError(DocRAGException):
    """External service (OpenAI, etc.) failed."""

    def __init__(
        self,
        service: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=f"{service} error: {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service": service, **(details or {})},
        )
