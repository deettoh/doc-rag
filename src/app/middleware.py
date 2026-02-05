"""Middleware for FastAPI application."""

import time
import uuid

from fastapi import FastAPI, Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests with request_id tracking."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request with logging and request_id."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        # Process request
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Request completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        response.headers["X-Request-ID"] = request_id

        return response


def configure_logging() -> None:
    """Configure loguru for structured logging."""
    logger.remove()  # Avoid duplicate logs
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "{extra} | "
            "<level>{message}</level>"
        ),
        level="INFO",
        serialize=False,
    )


def register_middleware(app: FastAPI) -> None:
    """Register all middleware with the app."""
    app.add_middleware(RequestLoggingMiddleware)
