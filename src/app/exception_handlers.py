"""Exception handlers for FastAPI application."""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.exceptions import DocRAGException, DomainValidationError, NotFoundError


def get_request_id(request: Request) -> str | None:
    """Extract request_id from request state if available."""
    return getattr(request.state, "request_id", None)


async def docrag_exception_handler(
    request: Request,
    exc: DocRAGException,
) -> JSONResponse:
    """Handle custom DocRAG exceptions."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    if isinstance(exc, NotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, DomainValidationError):
        status_code = status.HTTP_400_BAD_REQUEST

    return JSONResponse(
        status_code=status_code,
        content={
            "detail": exc.message,
            "error_code": exc.error_code,
            "request_id": get_request_id(request),
            **exc.details,
        },
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "error_code": "VALIDATION_ERROR",
            "request_id": get_request_id(request),
            "errors": exc.errors(),
        },
    )


async def generic_exception_handler(
    request: Request,
    _exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "request_id": get_request_id(request),
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the app."""
    app.add_exception_handler(DocRAGException, docrag_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
