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
def mock_db_session() -> AsyncMock:
    """Create a mock async database session."""
    return AsyncMock()


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


class TestQuestionGenerationEndpoint:
    """Tests for POST /api/documents/{id}/questions endpoint."""

    def test_generate_questions_success(self, client: TestClient) -> None:
        """Completed document should return generated questions."""
        with (
            patch("app.routers.documents.DocumentRepository") as mock_repo_class,
            patch("app.routers.documents.EmbeddingService"),
            patch("app.routers.documents.RetrievalService"),
            patch("app.routers.documents.LLMService"),
            patch("app.routers.documents.QnAGenerationService") as mock_qna_class,
        ):
            mock_document = MagicMock()
            mock_document.id = 1
            mock_document.status = DocumentStatus.COMPLETED
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)

            mock_question_1 = MagicMock()
            mock_question_1.id = 10
            mock_question_1.document_id = 1
            mock_question_1.content = "What is RAG?"
            mock_question_1.expected_answer = "Retrieval-augmented generation."
            mock_question_1.created_at = "2026-02-16T00:00:00"

            mock_question_2 = MagicMock()
            mock_question_2.id = 11
            mock_question_2.document_id = 1
            mock_question_2.content = "Why use embeddings?"
            mock_question_2.expected_answer = "For semantic retrieval."
            mock_question_2.created_at = "2026-02-16T00:00:01"

            mock_qna_service = mock_qna_class.return_value
            mock_qna_service.generate_and_store_questions = AsyncMock(
                return_value=[mock_question_1, mock_question_2]
            )

            response = client.post(
                "/api/documents/1/questions",
                json={"num_questions": 2},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["document_id"] == 1
            assert data["requested_count"] == 2
            assert data["generated_count"] == 2
            assert len(data["questions"]) == 2
            assert data["questions"][0]["content"] == "What is RAG?"

    def test_generate_questions_requires_completed_status(
        self, client: TestClient
    ) -> None:
        """Non-completed document should return validation error."""
        with patch("app.routers.documents.DocumentRepository") as mock_repo_class:
            mock_document = MagicMock()
            mock_document.id = 1
            mock_document.status = DocumentStatus.PROCESSING
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)

            response = client.post(
                "/api/documents/1/questions",
                json={"num_questions": 3},
            )

            assert response.status_code == 400
            data = response.json()
            assert data["error_code"] == "VALIDATION_ERROR"

    def test_generate_questions_document_not_found(self, client: TestClient) -> None:
        """Missing document should return NOT_FOUND."""
        with patch("app.routers.documents.DocumentRepository") as mock_repo_class:
            mock_repo_class.get_by_id = AsyncMock(return_value=None)

            response = client.post(
                "/api/documents/999/questions",
                json={"num_questions": 3},
            )

            assert response.status_code == 404
            data = response.json()
            assert data["error_code"] == "NOT_FOUND"


class TestAnswerSubmissionEndpoint:
    """Tests for POST /api/documents/{id}/questions/{qid}/answer endpoint."""

    def test_submit_answer_success(self, client: TestClient) -> None:
        """Valid submission should return persisted evaluated answer."""
        with (
            patch("app.routers.documents.DocumentRepository") as mock_repo_class,
            patch("app.routers.documents.LLMService"),
            patch("app.routers.documents.AnswerEvaluationService") as mock_eval_class,
        ):
            mock_document = MagicMock()
            mock_document.id = 1
            mock_document.status = DocumentStatus.COMPLETED
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)

            mock_answer = MagicMock()
            mock_answer.id = 50
            mock_answer.question_id = 9
            mock_answer.user_answer = "My answer text."
            mock_answer.score = 0.92
            mock_answer.feedback = "Accurate and concise."
            mock_answer.created_at = "2026-02-16T10:30:00"

            mock_eval_service = mock_eval_class.return_value
            mock_eval_service.submit_and_evaluate_answer = AsyncMock(
                return_value=mock_answer
            )

            response = client.post(
                "/api/documents/1/questions/9/answer",
                json={"user_answer": "My answer text."},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 50
            assert data["question_id"] == 9
            assert data["score"] == 0.92
            assert data["feedback"] == "Accurate and concise."

    def test_submit_answer_document_not_found(self, client: TestClient) -> None:
        """Missing document should return NOT_FOUND."""
        with patch("app.routers.documents.DocumentRepository") as mock_repo_class:
            mock_repo_class.get_by_id = AsyncMock(return_value=None)

            response = client.post(
                "/api/documents/999/questions/9/answer",
                json={"user_answer": "My answer text."},
            )

            assert response.status_code == 404
            data = response.json()
            assert data["error_code"] == "NOT_FOUND"

    def test_submit_answer_question_not_found(self, client: TestClient) -> None:
        """Missing question should return NOT_FOUND from service layer."""
        from app.exceptions import NotFoundError

        with (
            patch("app.routers.documents.DocumentRepository") as mock_repo_class,
            patch("app.routers.documents.LLMService"),
            patch("app.routers.documents.AnswerEvaluationService") as mock_eval_class,
        ):
            mock_document = MagicMock()
            mock_document.id = 1
            mock_document.status = DocumentStatus.COMPLETED
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)

            mock_eval_service = mock_eval_class.return_value
            mock_eval_service.submit_and_evaluate_answer = AsyncMock(
                side_effect=NotFoundError(resource="Question", resource_id=9)
            )

            response = client.post(
                "/api/documents/1/questions/9/answer",
                json={"user_answer": "My answer text."},
            )

            assert response.status_code == 404
            data = response.json()
            assert data["error_code"] == "NOT_FOUND"


class TestListDocumentsEndpoint:
    """Tests for GET /api/documents/ endpoint."""

    def test_list_documents_returns_documents(self, client: TestClient) -> None:
        """Should return a list of documents."""
        with patch("app.routers.documents.DocumentRepository") as mock_repo_class:
            mock_doc = MagicMock()
            mock_doc.id = 1
            mock_doc.filename = "test.pdf"
            mock_doc.file_size_bytes = 1024
            mock_doc.status = DocumentStatus.COMPLETED
            mock_doc.created_at = "2026-01-01T00:00:00"
            mock_repo_class.get_all = AsyncMock(return_value=[mock_doc])

            response = client.get("/api/documents/")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["id"] == 1
            assert data[0]["filename"] == "test.pdf"

    def test_list_documents_empty(self, client: TestClient) -> None:
        """Should return empty list when no documents exist."""
        with patch("app.routers.documents.DocumentRepository") as mock_repo_class:
            mock_repo_class.get_all = AsyncMock(return_value=[])

            response = client.get("/api/documents/")

            assert response.status_code == 200
            assert response.json() == []


class TestSummarizationEndpoint:
    """Tests for summarization endpoints."""

    def test_summarize_document_success(self, client: TestClient) -> None:
        """Completed document should return generated summary."""
        with (
            patch("app.routers.documents.DocumentRepository") as mock_repo_class,
            patch("app.routers.documents.EmbeddingService"),
            patch("app.routers.documents.RetrievalService"),
            patch("app.routers.documents.LLMService"),
            patch("app.routers.documents.SummarizationService") as mock_summ_class,
        ):
            mock_document = MagicMock()
            mock_document.id = 1
            mock_document.status = DocumentStatus.COMPLETED
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)

            mock_summary = MagicMock()
            mock_summary.id = 5
            mock_summary.document_id = 1
            mock_summary.content = "This is a summary."
            mock_summary.page_citations = [1, 2, 3]
            mock_summary.created_at = "2026-02-20T00:00:00"

            mock_service = mock_summ_class.return_value
            mock_service.generate_and_store_summary = AsyncMock(
                return_value=mock_summary
            )

            response = client.post("/api/documents/1/summarize")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 5
            assert data["document_id"] == 1
            assert data["content"] == "This is a summary."
            assert data["page_citations"] == [1, 2, 3]

    def test_summarize_requires_completed_status(self, client: TestClient) -> None:
        """Non-completed document should return validation error."""
        with patch("app.routers.documents.DocumentRepository") as mock_repo_class:
            mock_document = MagicMock()
            mock_document.id = 1
            mock_document.status = DocumentStatus.PROCESSING
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)

            response = client.post("/api/documents/1/summarize")

            assert response.status_code == 400
            data = response.json()
            assert data["error_code"] == "VALIDATION_ERROR"

    def test_summarize_document_not_found(self, client: TestClient) -> None:
        """Missing document should return NOT_FOUND."""
        with patch("app.routers.documents.DocumentRepository") as mock_repo_class:
            mock_repo_class.get_by_id = AsyncMock(return_value=None)

            response = client.post("/api/documents/999/summarize")

            assert response.status_code == 404
            data = response.json()
            assert data["error_code"] == "NOT_FOUND"

    def test_get_summary_success(self, client: TestClient) -> None:
        """Should return existing summary for a document."""
        with (
            patch("app.routers.documents.DocumentRepository") as mock_repo_class,
            patch("app.routers.documents.SummaryRepository") as mock_summ_repo,
        ):
            mock_document = MagicMock()
            mock_document.id = 1
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)

            mock_summary = MagicMock()
            mock_summary.id = 5
            mock_summary.document_id = 1
            mock_summary.content = "Persisted summary."
            mock_summary.page_citations = [1]
            mock_summary.created_at = "2026-02-20T00:00:00"
            mock_summ_repo.get_by_document_id = AsyncMock(return_value=mock_summary)

            response = client.get("/api/documents/1/summary")

            assert response.status_code == 200
            data = response.json()
            assert data["content"] == "Persisted summary."

    def test_get_summary_not_found(self, client: TestClient) -> None:
        """Should return NOT_FOUND when no summary exists."""
        with (
            patch("app.routers.documents.DocumentRepository") as mock_repo_class,
            patch("app.routers.documents.SummaryRepository") as mock_summ_repo,
        ):
            mock_document = MagicMock()
            mock_document.id = 1
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)
            mock_summ_repo.get_by_document_id = AsyncMock(return_value=None)

            response = client.get("/api/documents/1/summary")

            assert response.status_code == 404
            data = response.json()
            assert data["error_code"] == "NOT_FOUND"


class TestSearchEndpoint:
    """Tests for POST /api/documents/{id}/search endpoint."""

    def test_search_success(self, client: TestClient) -> None:
        """Valid search should return chunk results."""
        with (
            patch("app.routers.documents.DocumentRepository") as mock_repo_class,
            patch("app.routers.documents.EmbeddingService"),
            patch("app.routers.documents.RetrievalService") as mock_retrieval_class,
        ):
            mock_document = MagicMock()
            mock_document.id = 1
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)

            mock_result = MagicMock()
            mock_result.chunk_id = 10
            mock_result.chunk_index = 0
            mock_result.content = "Relevant chunk text."
            mock_result.page_start = 1
            mock_result.page_end = 1
            mock_result.score = 0.92

            mock_service = mock_retrieval_class.return_value
            mock_service.search_similar_chunks = AsyncMock(return_value=[mock_result])

            response = client.post(
                "/api/documents/1/search",
                json={"query": "test query"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["document_id"] == 1
            assert data["query"] == "test query"
            assert len(data["results"]) == 1
            assert data["results"][0]["content"] == "Relevant chunk text."
            assert data["results"][0]["score"] == 0.92

    def test_search_document_not_found(self, client: TestClient) -> None:
        """Missing document should return NOT_FOUND."""
        with patch("app.routers.documents.DocumentRepository") as mock_repo_class:
            mock_repo_class.get_by_id = AsyncMock(return_value=None)

            response = client.post(
                "/api/documents/999/search",
                json={"query": "test query"},
            )

            assert response.status_code == 404
            data = response.json()
            assert data["error_code"] == "NOT_FOUND"


class TestGetQuestionsEndpoint:
    """Tests for GET /api/documents/{id}/questions endpoint."""

    def test_get_questions_success(self, client: TestClient) -> None:
        """Should return persisted questions for a document."""
        with (
            patch("app.routers.documents.DocumentRepository") as mock_repo_class,
            patch("app.routers.documents.QuestionRepository") as mock_q_repo,
        ):
            mock_document = MagicMock()
            mock_document.id = 1
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)

            mock_question = MagicMock()
            mock_question.id = 10
            mock_question.document_id = 1
            mock_question.content = "What is RAG?"
            mock_question.expected_answer = "Retrieval-augmented generation."
            mock_question.created_at = "2026-02-16T00:00:00"
            mock_q_repo.get_by_document_id = AsyncMock(return_value=[mock_question])

            response = client.get("/api/documents/1/questions")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["content"] == "What is RAG?"

    def test_get_questions_document_not_found(self, client: TestClient) -> None:
        """Missing document should return NOT_FOUND."""
        with patch("app.routers.documents.DocumentRepository") as mock_repo_class:
            mock_repo_class.get_by_id = AsyncMock(return_value=None)

            response = client.get("/api/documents/999/questions")

            assert response.status_code == 404
            data = response.json()
            assert data["error_code"] == "NOT_FOUND"


class TestGetAnswerEndpoint:
    """Tests for GET /api/documents/{id}/questions/{qid}/answer endpoint."""

    def test_get_answer_success(self, client: TestClient) -> None:
        """Should return the latest evaluation for a question."""
        with (
            patch("app.routers.documents.DocumentRepository") as mock_repo_class,
            patch("app.routers.documents.AnswerRepository") as mock_answer_repo,
        ):
            mock_document = MagicMock()
            mock_document.id = 1
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)

            mock_answer = MagicMock()
            mock_answer.id = 50
            mock_answer.question_id = 9
            mock_answer.user_answer = "My answer."
            mock_answer.score = 0.85
            mock_answer.feedback = "Good."
            mock_answer.created_at = "2026-02-20T00:00:00"
            mock_answer_repo.get_latest_by_question_id = AsyncMock(
                return_value=mock_answer
            )

            response = client.get("/api/documents/1/questions/9/answer")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 50
            assert data["score"] == 0.85
            assert data["feedback"] == "Good."

    def test_get_answer_not_found(self, client: TestClient) -> None:
        """Should return NOT_FOUND when no answer exists."""
        with (
            patch("app.routers.documents.DocumentRepository") as mock_repo_class,
            patch("app.routers.documents.AnswerRepository") as mock_answer_repo,
        ):
            mock_document = MagicMock()
            mock_document.id = 1
            mock_repo_class.get_by_id = AsyncMock(return_value=mock_document)
            mock_answer_repo.get_latest_by_question_id = AsyncMock(return_value=None)

            response = client.get("/api/documents/1/questions/9/answer")

            assert response.status_code == 404
            data = response.json()
            assert data["error_code"] == "NOT_FOUND"
