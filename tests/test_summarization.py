"""Integration tests for the summarization feature."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.models.answer import Answer  # noqa: F401
from app.models.base import Base
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.question import Question  # noqa: F401
from app.models.summary import Summary  # noqa: F401
from app.repositories.summary import SummaryRepository
from app.services.llm import LLMService, SummaryResult
from app.services.retrieval import ChunkSearchResult, RetrievalService
from app.services.summarization import SummarizationService

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def test_session(test_engine):
    """Create a test session."""
    session_factory = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session


@pytest.fixture
async def completed_document(test_session: AsyncSession) -> Document:
    """Create a completed document for testing."""
    doc = Document(
        filename="summary_test.pdf",
        file_path="/tmp/summary_test.pdf",
        file_size_bytes=2048,
        page_count=5,
        status=DocumentStatus.COMPLETED,
    )
    test_session.add(doc)
    await test_session.flush()
    await test_session.refresh(doc)
    return doc


@pytest.fixture
async def processing_document(test_session: AsyncSession) -> Document:
    """Create a document still being processed."""
    doc = Document(
        filename="processing.pdf",
        file_path="/tmp/processing.pdf",
        file_size_bytes=1024,
        page_count=2,
        status=DocumentStatus.PROCESSING,
    )
    test_session.add(doc)
    await test_session.flush()
    await test_session.refresh(doc)
    return doc


def _make_fake_embedding(seed: int) -> list[float]:
    """Generate a deterministic fake embedding vector for testing."""
    base = [0.0] * settings.embedding_dimension
    base[seed % settings.embedding_dimension] = 1.0
    return base


@pytest.fixture
async def embedded_chunks(
    test_session: AsyncSession, completed_document: Document
) -> list[Chunk]:
    """Create chunks with fake embeddings for the completed document."""
    chunk_data = [
        ("Machine learning is a subset of AI.", 1, 1),
        ("Neural networks have layers of nodes.", 2, 2),
        ("RAG combines retrieval and generation.", 3, 3),
    ]
    chunks = []
    for i, (content, page_start, page_end) in enumerate(chunk_data):
        chunk = Chunk(
            document_id=completed_document.id,
            chunk_index=i,
            content=content,
            page_start=page_start,
            page_end=page_end,
            char_start=i * 50,
            char_end=(i + 1) * 50,
            embedding=_make_fake_embedding(i),
            is_embedded=True,
        )
        chunks.append(chunk)
    test_session.add_all(chunks)
    await test_session.flush()
    for chunk in chunks:
        await test_session.refresh(chunk)
    return chunks


def _mock_search_results(chunks: list[Chunk]) -> list[ChunkSearchResult]:
    """Convert Chunk models into ChunkSearchResult objects."""
    return [
        ChunkSearchResult(
            chunk_id=c.id,
            chunk_index=c.chunk_index,
            content=c.content,
            page_start=c.page_start,
            page_end=c.page_end,
            score=round(0.9 - i * 0.1, 6),
        )
        for i, c in enumerate(chunks)
    ]


class TestSummarizationService:
    """Tests for the SummarizationService."""

    async def test_generate_and_store_summary(
        self,
        test_session: AsyncSession,
        completed_document: Document,
        embedded_chunks: list[Chunk],
    ) -> None:
        """Full flow: retrieve chunks -> LLM -> DB insert -> correct response."""
        search_results = _mock_search_results(embedded_chunks)

        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search_similar_chunks = AsyncMock(return_value=search_results)

        mock_llm = MagicMock(spec=LLMService)
        mock_llm.generate_summary.return_value = SummaryResult(
            summary="ML, neural networks, and RAG explained.",
            page_citations=[1, 2, 3],
        )

        service = SummarizationService(mock_retrieval, mock_llm)

        summary = await service.generate_and_store_summary(
            session=test_session,
            document_id=completed_document.id,
        )

        assert summary.content == "ML, neural networks, and RAG explained."
        assert json.loads(summary.page_citations) == [1, 2, 3]
        assert summary.document_id == completed_document.id
        assert summary.id is not None

    async def test_summary_context_includes_chunks(
        self,
        test_session: AsyncSession,
        completed_document: Document,
        embedded_chunks: list[Chunk],
    ) -> None:
        """The context string passed to LLM contains chunk text and page markers."""
        search_results = _mock_search_results(embedded_chunks)

        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search_similar_chunks = AsyncMock(return_value=search_results)

        mock_llm = MagicMock(spec=LLMService)
        mock_llm.generate_summary.return_value = SummaryResult(
            summary="Test.", page_citations=[]
        )

        service = SummarizationService(mock_retrieval, mock_llm)

        await service.generate_and_store_summary(
            session=test_session,
            document_id=completed_document.id,
        )

        call_args = mock_llm.generate_summary.call_args
        context = call_args[0][0]

        assert "[Page 1-1]" in context
        assert "Machine learning is a subset of AI." in context
        assert "[Page 3-3]" in context
        assert "RAG combines retrieval and generation." in context

    async def test_summary_stored_in_db(
        self,
        test_session: AsyncSession,
        completed_document: Document,
        embedded_chunks: list[Chunk],
    ) -> None:
        """After the call, the summary is persisted and retrievable."""
        search_results = _mock_search_results(embedded_chunks)

        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search_similar_chunks = AsyncMock(return_value=search_results)

        mock_llm = MagicMock(spec=LLMService)
        mock_llm.generate_summary.return_value = SummaryResult(
            summary="Stored summary.", page_citations=[1]
        )

        service = SummarizationService(mock_retrieval, mock_llm)

        await service.generate_and_store_summary(
            session=test_session,
            document_id=completed_document.id,
        )

        stored = await SummaryRepository.get_by_document_id(
            test_session, completed_document.id
        )
        assert stored is not None
        assert stored.content == "Stored summary."
        assert json.loads(stored.page_citations) == [1]

    async def test_empty_chunks_raises_validation_error(
        self,
        test_session: AsyncSession,
        completed_document: Document,
    ) -> None:
        """Raises DomainValidationError when no embedded chunks exist."""
        from app.exceptions import DomainValidationError

        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search_similar_chunks = AsyncMock(return_value=[])

        mock_llm = MagicMock(spec=LLMService)

        service = SummarizationService(mock_retrieval, mock_llm)

        with pytest.raises(DomainValidationError):
            await service.generate_and_store_summary(
                session=test_session,
                document_id=completed_document.id,
            )

        mock_llm.generate_summary.assert_not_called()

    async def test_uses_dedicated_summarization_top_k_default(
        self,
        test_session: AsyncSession,
        completed_document: Document,
        embedded_chunks: list[Chunk],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When top_k is omitted, service should use settings.summarization_top_k."""
        search_results = _mock_search_results(embedded_chunks)

        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search_similar_chunks = AsyncMock(return_value=search_results)

        mock_llm = MagicMock(spec=LLMService)
        mock_llm.generate_summary.return_value = SummaryResult(
            summary="Uses summary top_k default.",
            page_citations=[1],
        )

        monkeypatch.setattr(settings, "summarization_top_k", 21)
        service = SummarizationService(mock_retrieval, mock_llm)

        await service.generate_and_store_summary(
            session=test_session,
            document_id=completed_document.id,
        )

        call_kwargs = mock_retrieval.search_similar_chunks.call_args.kwargs
        assert call_kwargs["top_k"] == 21


class TestBuildContext:
    """Tests for context building logic."""

    def test_context_respects_max_size(self) -> None:
        """Context is truncated to respect max_llm_context_size."""
        chunks = [
            ChunkSearchResult(
                chunk_id=i,
                chunk_index=i,
                content="x" * 2000,
                page_start=i + 1,
                page_end=i + 1,
                score=0.9,
            )
            for i in range(5)
        ]

        context = SummarizationService._build_context(chunks)

        assert len(context) <= settings.max_llm_context_size

    def test_context_contains_page_markers(self) -> None:
        """Each chunk segment has a [Page X-Y] marker."""
        chunks = [
            ChunkSearchResult(
                chunk_id=1,
                chunk_index=0,
                content="Some content.",
                page_start=3,
                page_end=4,
                score=0.9,
            )
        ]

        context = SummarizationService._build_context(chunks)

        assert "[Page 3-4]" in context
        assert "Some content." in context


class TestSummaryRepository:
    """Tests for SummaryRepository persistence."""

    async def test_create_and_retrieve(
        self,
        test_session: AsyncSession,
        completed_document: Document,
    ) -> None:
        """Summary can be created and retrieved by document_id."""
        summary = await SummaryRepository.create(
            test_session,
            document_id=completed_document.id,
            content="Test summary.",
            page_citations=[1, 3],
        )

        assert summary.id is not None
        assert summary.content == "Test summary."

        retrieved = await SummaryRepository.get_by_document_id(
            test_session, completed_document.id
        )
        assert retrieved is not None
        assert retrieved.id == summary.id
        assert json.loads(retrieved.page_citations) == [1, 3]

    async def test_get_returns_none_when_no_summary(
        self,
        test_session: AsyncSession,
        completed_document: Document,
    ) -> None:
        """Returns None when no summary exists for the document."""
        result = await SummaryRepository.get_by_document_id(
            test_session, completed_document.id
        )
        assert result is None
