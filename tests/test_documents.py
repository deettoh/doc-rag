"""Tests for document upload endpoint."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.db import get_db
from app.exceptions import DomainValidationError
from app.main import app
from app.models.document import DocumentStatus
from app.services.storage import FileStorageService


@pytest.fixture
def mock_db_session() -> MagicMock:
    """Create a mock async database session."""
    return MagicMock()


@pytest.fixture
def client(mock_db_session: MagicMock) -> TestClient:
    """Create test client with mocked DB dependency."""
    app.dependency_overrides[get_db] = lambda: mock_db_session
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Generate minimal valid PDF bytes."""
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


@pytest.fixture
def sample_pdf_file(sample_pdf_content: bytes) -> tuple[str, io.BytesIO, str]:
    """Create sample PDF file tuple for upload."""
    return ("file", io.BytesIO(sample_pdf_content), "application/pdf")


class TestDocumentUpload:
    """Tests for POST /api/documents/upload endpoint."""

    def test_upload_pdf_success(
        self, client: TestClient, sample_pdf_content: bytes
    ) -> None:
        """Verify successful PDF upload returns document data."""
        with (
            patch("app.routers.documents.FileStorageService") as mock_storage_class,
            patch("app.routers.documents.DocumentRepository") as mock_repo_class,
            patch("app.routers.documents.DocumentProcessingService"),
        ):
            mock_storage = mock_storage_class.return_value
            mock_storage.save_pdf = AsyncMock(
                return_value=("/path/to/file.pdf", 100)
            )  # 100 bytes file size

            mock_document = AsyncMock()
            mock_document.id = 1
            mock_document.filename = "test.pdf"
            mock_document.file_size_bytes = 100
            mock_document.status = DocumentStatus.UPLOADED
            mock_document.created_at = "2026-01-01T00:00:00"
            mock_repo_class.create = AsyncMock(return_value=mock_document)

            response = client.post(
                "/api/documents/upload",
                files={
                    "file": (
                        "test.pdf",
                        io.BytesIO(sample_pdf_content),
                        "application/pdf",
                    )
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 1
            assert data["filename"] == "test.pdf"
            assert data["status"] == "uploaded"

    def test_upload_invalid_file_type(self, client: TestClient) -> None:
        """Verify non-PDF upload returns 400 error."""
        with patch("app.routers.documents.FileStorageService") as mock_storage_class:
            mock_storage = mock_storage_class.return_value
            mock_storage.save_pdf = AsyncMock(
                side_effect=DomainValidationError(
                    message="Only PDF files are allowed",
                    field="file",
                )
            )

            response = client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")},
            )

            assert response.status_code == 400
            data = response.json()
            assert data["error_code"] == "VALIDATION_ERROR"

    def test_upload_exceeds_size_limit(self, client: TestClient) -> None:
        """Verify oversized file returns 400 error."""
        with patch("app.routers.documents.FileStorageService") as mock_storage_class:
            mock_storage = mock_storage_class.return_value
            mock_storage.save_pdf = AsyncMock(
                side_effect=DomainValidationError(
                    message="File size exceeds maximum of 10MB",
                    field="file",
                )
            )

            response = client.post(
                "/api/documents/upload",
                files={"file": ("large.pdf", io.BytesIO(b"%PDF-"), "application/pdf")},
            )

            assert response.status_code == 400
            data = response.json()
            assert data["error_code"] == "VALIDATION_ERROR"

    def test_upload_missing_file(self, client: TestClient) -> None:
        """Verify missing file returns 422 error."""
        response = client.post("/api/documents/upload")

        assert response.status_code == 422

    def test_request_id_in_upload_response(
        self, client: TestClient, sample_pdf_content: bytes
    ) -> None:
        """Verify X-Request-ID header present in upload response."""
        with (
            patch("app.routers.documents.FileStorageService") as mock_storage_class,
            patch("app.routers.documents.DocumentRepository") as mock_repo_class,
            patch("app.routers.documents.DocumentProcessingService"),
        ):
            mock_storage = mock_storage_class.return_value
            mock_storage.save_pdf = AsyncMock(return_value=("/path/to/file.pdf", 100))

            mock_document = AsyncMock()
            mock_document.id = 1
            mock_document.filename = "test.pdf"
            mock_document.file_size_bytes = 100
            mock_document.status = DocumentStatus.UPLOADED
            mock_document.created_at = "2026-01-01T00:00:00"
            mock_repo_class.create = AsyncMock(return_value=mock_document)

            response = client.post(
                "/api/documents/upload",
                files={
                    "file": (
                        "test.pdf",
                        io.BytesIO(sample_pdf_content),
                        "application/pdf",
                    )
                },
            )

            assert "X-Request-ID" in response.headers

    def test_upload_enqueues_background_task(
        self, client: TestClient, sample_pdf_content: bytes
    ) -> None:
        """Verify upload endpoint enqueues a background processing task."""
        with (
            patch("app.routers.documents.FileStorageService") as mock_storage_class,
            patch("app.routers.documents.DocumentRepository") as mock_repo_class,
            patch("app.routers.documents.DocumentProcessingService") as mock_proc_class,
        ):
            mock_storage = mock_storage_class.return_value
            mock_storage.save_pdf = AsyncMock(return_value=("/path/to/file.pdf", 100))

            mock_document = AsyncMock()
            mock_document.id = 42
            mock_document.filename = "test.pdf"
            mock_document.file_size_bytes = 100
            mock_document.status = DocumentStatus.UPLOADED
            mock_document.created_at = "2026-01-01T00:00:00"
            mock_repo_class.create = AsyncMock(return_value=mock_document)

            response = client.post(
                "/api/documents/upload",
                files={
                    "file": (
                        "test.pdf",
                        io.BytesIO(sample_pdf_content),
                        "application/pdf",
                    )
                },
            )

            assert response.status_code == 200
            # Processing service should have been instantiated
            mock_proc_class.assert_called_once()


class TestFileStorageService:
    """Unit tests for FileStorageService."""

    def test_validate_file_type_rejects_non_pdf(self) -> None:
        """Verify validation rejects non-PDF files."""
        service = FileStorageService(upload_dir="/tmp/test")
        mock_file = MagicMock()
        mock_file.content_type = "text/plain"
        mock_file.filename = "test.txt"

        with pytest.raises(DomainValidationError) as exc_info:
            service._validate_file_type(mock_file)

        assert "PDF" in str(exc_info.value.message)

    def test_validate_file_type_accepts_pdf_by_content_type(self) -> None:
        """Verify validation accepts PDF by content type."""
        service = FileStorageService(upload_dir="/tmp/test")
        mock_file = MagicMock()
        mock_file.content_type = "application/pdf"
        mock_file.filename = "test.pdf"

        # Should not raise
        service._validate_file_type(mock_file)

    def test_validate_file_type_accepts_pdf_by_extension(self) -> None:
        """Verify validation accepts PDF by extension even with wrong content type."""
        service = FileStorageService(upload_dir="/tmp/test")
        mock_file = MagicMock()
        mock_file.content_type = "application/octet-stream"
        mock_file.filename = "document.PDF"

        # Should not raise (extension check is case-insensitive)
        service._validate_file_type(mock_file)

    def test_generate_unique_filename_preserves_extension(self) -> None:
        """Verify unique filename generation preserves .pdf extension."""
        service = FileStorageService(upload_dir="/tmp/test")
        unique_name = service._generate_unique_filename("my_document.pdf")

        assert unique_name.endswith(".pdf")
        assert unique_name != "my_document.pdf"  # Should be unique


class TestDocumentStatus:
    """Tests for GET /api/documents/{id}/status endpoint."""

    def test_get_status_returns_document_status(self, client: TestClient) -> None:
        """Verify status endpoint returns current document status."""
        with patch("app.routers.documents.DocumentRepository") as mock_repo_class:
            mock_document = MagicMock()
            mock_document.id = 1
            mock_document.status = DocumentStatus.PROCESSING
            mock_document.page_count = 5
            mock_document.error_message = None
            mock_document.updated_at = "2026-01-01T00:00:00"
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)

            response = client.get("/api/documents/1/status")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 1
            assert data["status"] == "processing"
            assert data["page_count"] == 5
            assert data["error_message"] is None

    def test_get_status_completed_with_all_fields(self, client: TestClient) -> None:
        """Verify completed status returns all expected fields."""
        with patch("app.routers.documents.DocumentRepository") as mock_repo_class:
            mock_document = MagicMock()
            mock_document.id = 2
            mock_document.status = DocumentStatus.COMPLETED
            mock_document.page_count = 10
            mock_document.error_message = None
            mock_document.updated_at = "2026-01-01T12:00:00"
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)

            response = client.get("/api/documents/2/status")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["page_count"] == 10

    def test_get_status_failed_includes_error(self, client: TestClient) -> None:
        """Verify failed status includes error message."""
        with patch("app.routers.documents.DocumentRepository") as mock_repo_class:
            mock_document = MagicMock()
            mock_document.id = 3
            mock_document.status = DocumentStatus.FAILED
            mock_document.page_count = None
            mock_document.error_message = "RuntimeError: Corrupt PDF"
            mock_document.updated_at = "2026-01-01T12:00:00"
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)

            response = client.get("/api/documents/3/status")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "failed"
            assert "Corrupt PDF" in data["error_message"]

    def test_get_status_not_found(self, client: TestClient) -> None:
        """Verify non-existent document returns 404."""
        with patch("app.routers.documents.DocumentRepository") as mock_repo_class:
            mock_repo_class.get_by_id = AsyncMock(return_value=None)

            response = client.get("/api/documents/999/status")

            assert response.status_code == 404
            data = response.json()
            assert data["error_code"] == "NOT_FOUND"
