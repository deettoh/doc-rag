"""Tests for FastAPI application endpoints and middleware."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client for API testing."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_healthy(self, client: TestClient) -> None:
        """Verify /health returns 200 with healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "docrag-api"

    def test_health_includes_version(self, client: TestClient) -> None:
        """Verify /health response includes version field."""
        response = client.get("/health")

        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_request_id_header(self, client: TestClient) -> None:
        """Verify X-Request-ID header is present in response."""
        response = client.get("/health")

        assert "X-Request-ID" in response.headers
        # Verify it's a valid UUID format (36 chars with hyphens)
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) == 36
        assert request_id.count("-") == 4


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_returns_error_response(self, client: TestClient) -> None:
        """Verify unknown route returns proper 404 error."""
        response = client.get("/nonexistent")

        assert response.status_code == 404
        # Note: FastAPI's default 404 handler returns {"detail": "Not Found"}
        data = response.json()
        assert "detail" in data

    def test_request_id_in_error_response(self, client: TestClient) -> None:
        """Verify X-Request-ID header present even on errors."""
        response = client.get("/nonexistent")

        assert "X-Request-ID" in response.headers
