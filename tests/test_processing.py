"""Tests for DocumentProcessingService background pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.document import DocumentStatus
from app.services.processing import DocumentProcessingService


@pytest.fixture
def mock_pdf_extractor() -> MagicMock:
    """Create a mock PDFExtractorService."""
    extractor = MagicMock()
    page1 = MagicMock()
    page1.page_number = 1
    page1.text = "First page content for testing the document processing pipeline."
    page2 = MagicMock()
    page2.page_number = 2
    page2.text = "Second page content with additional information."
    extractor.extract_text.return_value = [page1, page2]
    return extractor


@pytest.fixture
def mock_chunking_service() -> MagicMock:
    """Create a mock ChunkingService."""
    service = MagicMock()
    chunk1 = MagicMock()
    chunk1.content = "First page content for testing"
    chunk2 = MagicMock()
    chunk2.content = "the document processing pipeline."
    service.chunk_pages.return_value = [chunk1, chunk2]
    return service


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Create a mock EmbeddingService."""
    service = MagicMock()
    service.embed_chunks.return_value = [[0.1] * 768, [0.2] * 768]
    return service


@pytest.fixture
def processing_service(
    mock_pdf_extractor: MagicMock,
    mock_chunking_service: MagicMock,
    mock_embedding_service: MagicMock,
) -> DocumentProcessingService:
    """Create a DocumentProcessingService with all dependencies mocked."""
    return DocumentProcessingService(
        pdf_extractor=mock_pdf_extractor,
        chunking_service=mock_chunking_service,
        embedding_service=mock_embedding_service,
    )


def _make_mock_session() -> MagicMock:
    """Create a mock async session with commit/rollback/flush."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    return session


class TestDocumentProcessingService:
    """Tests for the background document processing pipeline."""

    @pytest.mark.asyncio
    async def test_happy_path_processes_full_pipeline(
        self,
        processing_service: DocumentProcessingService,
        mock_pdf_extractor: MagicMock,
        mock_chunking_service: MagicMock,
        mock_embedding_service: MagicMock,
    ) -> None:
        """Verify full pipeline executes: extract -> chunk -> embed with correct status transitions."""
        mock_session = _make_mock_session()
        mock_db_chunks = [MagicMock(id=1), MagicMock(id=2)]

        with (
            patch("app.services.processing.async_session_factory") as mock_factory,
            patch("app.repositories.document.DocumentRepository") as mock_doc_repo,
            patch("app.repositories.chunk.ChunkRepository") as mock_chunk_repo,
        ):
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_doc_repo.update_status = AsyncMock()
            mock_doc_repo.update_page_count = AsyncMock()
            mock_chunk_repo.create_bulk = AsyncMock(return_value=mock_db_chunks)
            mock_chunk_repo.update_embeddings = AsyncMock()

            await processing_service.process_document(1, "/path/to/doc.pdf")

            # Verify status transitions
            status_calls = mock_doc_repo.update_status.call_args_list
            assert len(status_calls) == 2
            assert status_calls[0].args[1:] == (1, DocumentStatus.PROCESSING)
            assert status_calls[1].args[1:] == (1, DocumentStatus.COMPLETED)

            # Verify pipeline steps were called
            mock_pdf_extractor.extract_text.assert_called_once_with("/path/to/doc.pdf")
            mock_doc_repo.update_page_count.assert_called_once()
            mock_chunking_service.chunk_pages.assert_called_once()
            mock_chunk_repo.create_bulk.assert_called_once()
            mock_embedding_service.embed_chunks.assert_called_once()
            mock_chunk_repo.update_embeddings.assert_called_once_with(
                mock_session, [1, 2], mock_embedding_service.embed_chunks.return_value
            )

    @pytest.mark.asyncio
    async def test_extraction_failure_sets_failed_status(
        self,
        processing_service: DocumentProcessingService,
        mock_pdf_extractor: MagicMock,
        mock_chunking_service: MagicMock,
    ) -> None:
        """Verify extraction failure transitions status to FAILED with error message."""
        mock_session = _make_mock_session()
        mock_pdf_extractor.extract_text.side_effect = RuntimeError("Corrupt PDF")

        with (
            patch("app.services.processing.async_session_factory") as mock_factory,
            patch("app.repositories.document.DocumentRepository") as mock_doc_repo,
        ):
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_doc_repo.update_status = AsyncMock()

            await processing_service.process_document(1, "/path/to/bad.pdf")

            final_status = mock_doc_repo.update_status.call_args_list[-1]
            assert final_status.args[2] == DocumentStatus.FAILED
            assert "Corrupt PDF" in final_status.kwargs.get(
                "error_message", ""
            ) or "Corrupt PDF" in str(final_status)

            mock_chunking_service.chunk_pages.assert_not_called()

    @pytest.mark.asyncio
    async def test_embedding_failure_sets_failed_status(
        self,
        processing_service: DocumentProcessingService,
        mock_embedding_service: MagicMock,
    ) -> None:
        """Verify embedding failure transitions status to FAILED."""
        mock_session = _make_mock_session()
        mock_embedding_service.embed_chunks.side_effect = RuntimeError("Model error")

        mock_db_chunks = [MagicMock(id=1), MagicMock(id=2)]

        with (
            patch("app.services.processing.async_session_factory") as mock_factory,
            patch("app.repositories.document.DocumentRepository") as mock_doc_repo,
            patch("app.repositories.chunk.ChunkRepository") as mock_chunk_repo,
        ):
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_doc_repo.update_status = AsyncMock()
            mock_doc_repo.update_page_count = AsyncMock()
            mock_chunk_repo.create_bulk = AsyncMock(return_value=mock_db_chunks)

            await processing_service.process_document(1, "/path/to/doc.pdf")

            final_status = mock_doc_repo.update_status.call_args_list[-1]
            assert final_status.args[2] == DocumentStatus.FAILED
            assert "Model error" in str(final_status)

    @pytest.mark.asyncio
    async def test_empty_chunks_skips_embedding(
        self,
        mock_pdf_extractor: MagicMock,
        mock_embedding_service: MagicMock,
    ) -> None:
        """Verify that when no chunks are produced, embedding step is skipped."""
        mock_session = _make_mock_session()
        mock_chunking = MagicMock()
        mock_chunking.chunk_pages.return_value = []

        service = DocumentProcessingService(
            pdf_extractor=mock_pdf_extractor,
            chunking_service=mock_chunking,
            embedding_service=mock_embedding_service,
        )

        with (
            patch("app.services.processing.async_session_factory") as mock_factory,
            patch("app.repositories.document.DocumentRepository") as mock_doc_repo,
            patch("app.repositories.chunk.ChunkRepository") as mock_chunk_repo,
        ):
            mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_doc_repo.update_status = AsyncMock()
            mock_doc_repo.update_page_count = AsyncMock()
            mock_chunk_repo.create_bulk = AsyncMock(return_value=[])

            await service.process_document(1, "/path/to/doc.pdf")

            mock_embedding_service.embed_chunks.assert_not_called()

            # Should still complete successfully
            final_status = mock_doc_repo.update_status.call_args_list[-1]
            assert final_status.args[2] == DocumentStatus.COMPLETED
