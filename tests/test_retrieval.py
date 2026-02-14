"""Integration tests for retrieval (similarity search) functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Imported all models to ensure Document relationship initializes correctly
from app.config import settings
from app.models.answer import Answer  # noqa: F401
from app.models.base import Base
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.question import Question  # noqa: F401
from app.models.summary import Summary  # noqa: F401
from app.repositories.chunk import ChunkRepository
from app.services.retrieval import ChunkSearchResult, RetrievalService

# Use in-memory SQLite
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
async def sample_document(test_session: AsyncSession) -> Document:
    """Create a sample document for testing."""
    doc = Document(
        filename="retrieval_test.pdf",
        file_path="/tmp/retrieval_test.pdf",
        file_size_bytes=2048,
        page_count=3,
        status=DocumentStatus.UPLOADED,
    )
    test_session.add(doc)
    await test_session.flush()
    await test_session.refresh(doc)
    return doc


@pytest.fixture
async def second_document(test_session: AsyncSession) -> Document:
    """Create a second document for cross-document filtering tests."""
    doc = Document(
        filename="other_doc.pdf",
        file_path="/tmp/other_doc.pdf",
        file_size_bytes=1024,
        page_count=2,
        status=DocumentStatus.UPLOADED,
    )
    test_session.add(doc)
    await test_session.flush()
    await test_session.refresh(doc)
    return doc


@pytest.fixture
async def embedded_chunks(
    test_session: AsyncSession, sample_document: Document
) -> list[Chunk]:
    """Create chunks with fake embeddings stored in the database."""
    chunk_contents = [
        "Nasi Lemak is the national dish of Malaysia.",
        "Jay does BJJ.",
        "Kuala Lumpur is better than Penang",
        "RAG is lowkey a glorified search and retrieval.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    chunks = []
    for i, content in enumerate(chunk_contents):
        chunk = Chunk(
            document_id=sample_document.id,
            chunk_index=i,
            content=content,
            page_start=i + 1,
            page_end=i + 1,
            char_start=i * 60,
            char_end=(i + 1) * 60,
            embedding=_make_fake_embedding(i),
            is_embedded=True,
        )
        chunks.append(chunk)
    test_session.add_all(chunks)
    await test_session.flush()
    for chunk in chunks:
        await test_session.refresh(chunk)
    return chunks


def _make_fake_embedding(seed: int) -> list[float]:
    """Generate a deterministic fake embedding vector for testing."""
    base = [0.0] * settings.embedding_dimension
    # Set a few distinguishing values so vectors differ
    base[seed % settings.embedding_dimension] = 1.0
    base[(seed * 7) % settings.embedding_dimension] = 0.5
    return base


class TestRetrievalService:
    """Tests for RetrievalService using mocked dependencies."""

    async def test_search_returns_expected_results(
        self, test_session: AsyncSession, sample_document: Document
    ) -> None:
        """Known query returns expected chunks via mocked similarity_search."""
        expected_chunk = Chunk(
            document_id=sample_document.id,
            chunk_index=0,
            content="Machine learning overview",
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=25,
            embedding=_make_fake_embedding(0),
            is_embedded=True,
        )
        test_session.add(expected_chunk)
        await test_session.flush()
        await test_session.refresh(expected_chunk)

        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embeddings.return_value = [
            [0.1] * settings.embedding_dimension
        ]

        mock_search_results = [(expected_chunk, 0.15)]

        service = RetrievalService(mock_embedding_service)

        with patch.object(
            ChunkRepository, "similarity_search", new_callable=AsyncMock
        ) as mock_sim:
            mock_sim.return_value = mock_search_results

            results = await service.search_similar_chunks(
                session=test_session,
                document_id=sample_document.id,
                query="What is machine learning?",
                top_k=3,
            )

        assert len(results) == 1
        assert results[0].chunk_id == expected_chunk.id
        assert results[0].content == "Machine learning overview"
        assert results[0].score == pytest.approx(0.85, abs=0.001)

        # Verify args were passed through
        mock_embedding_service.generate_embeddings.assert_called_once_with(
            ["What is machine learning?"]
        )
        mock_sim.assert_called_once_with(
            session=test_session,
            document_id=sample_document.id,
            query_embedding=[0.1] * settings.embedding_dimension,
            top_k=3,
        )

    async def test_search_filters_by_document_id(
        self,
        test_session: AsyncSession,
        sample_document: Document,
        second_document: Document,
    ) -> None:
        """Similarity search only returns chunks from the requested document."""
        chunk_doc1 = Chunk(
            document_id=sample_document.id,
            chunk_index=0,
            content="From document one",
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=18,
            embedding=_make_fake_embedding(0),
            is_embedded=True,
        )
        chunk_doc2 = Chunk(
            document_id=second_document.id,
            chunk_index=0,
            content="From document two",
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=17,
            embedding=_make_fake_embedding(1),
            is_embedded=True,
        )
        test_session.add_all([chunk_doc1, chunk_doc2])
        await test_session.flush()
        await test_session.refresh(chunk_doc1)

        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embeddings.return_value = [
            [0.1] * settings.embedding_dimension
        ]

        # Only doc1's chunk should be returned
        mock_search_results = [(chunk_doc1, 0.2)]

        service = RetrievalService(mock_embedding_service)

        with patch.object(
            ChunkRepository, "similarity_search", new_callable=AsyncMock
        ) as mock_sim:
            mock_sim.return_value = mock_search_results

            results = await service.search_similar_chunks(
                session=test_session,
                document_id=sample_document.id,
                query="test query",
            )

        assert len(results) == 1
        assert results[0].content == "From document one"

        # Verify repository was called with the correct document_id
        call_kwargs = mock_sim.call_args.kwargs
        assert call_kwargs["document_id"] == sample_document.id

    async def test_search_respects_top_k(
        self,
        test_session: AsyncSession,
        embedded_chunks: list[Chunk],
        sample_document: Document,
    ) -> None:
        """Only top_k results are returned even when more chunks exist."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embeddings.return_value = [
            [0.1] * settings.embedding_dimension
        ]

        top_k = 2
        mock_search_results = [
            (embedded_chunks[0], 0.1),
            (embedded_chunks[3], 0.2),
        ]

        service = RetrievalService(mock_embedding_service)

        with patch.object(
            ChunkRepository, "similarity_search", new_callable=AsyncMock
        ) as mock_sim:
            mock_sim.return_value = mock_search_results

            results = await service.search_similar_chunks(
                session=test_session,
                document_id=sample_document.id,
                query="RAG systems",
                top_k=top_k,
            )

        assert len(results) == top_k
        # Verify top_k was passed through to the repository
        call_kwargs = mock_sim.call_args.kwargs
        assert call_kwargs["top_k"] == top_k

    async def test_search_empty_results(
        self,
        test_session: AsyncSession,
        sample_document: Document,
    ) -> None:
        """Search returns empty list when no embedded chunks exist."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embeddings.return_value = [
            [0.1] * settings.embedding_dimension
        ]

        service = RetrievalService(mock_embedding_service)

        with patch.object(
            ChunkRepository, "similarity_search", new_callable=AsyncMock
        ) as mock_sim:
            mock_sim.return_value = []

            results = await service.search_similar_chunks(
                session=test_session,
                document_id=sample_document.id,
                query="anything",
            )

        assert len(results) == 0
        assert results == []

    async def test_search_uses_default_top_k_from_config(
        self,
        test_session: AsyncSession,
        sample_document: Document,
    ) -> None:
        """When top_k is not specified, the config default is used."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embeddings.return_value = [
            [0.1] * settings.embedding_dimension
        ]

        service = RetrievalService(mock_embedding_service)

        with patch.object(
            ChunkRepository, "similarity_search", new_callable=AsyncMock
        ) as mock_sim:
            mock_sim.return_value = []

            await service.search_similar_chunks(
                session=test_session,
                document_id=sample_document.id,
                query="test",
            )

        call_kwargs = mock_sim.call_args.kwargs
        assert call_kwargs["top_k"] == settings.top_k_retrieval

    async def test_score_is_one_minus_distance(
        self,
        test_session: AsyncSession,
        sample_document: Document,
    ) -> None:
        """ChunkSearchResult.score should be 1 - cosine_distance."""
        chunk = Chunk(
            document_id=sample_document.id,
            chunk_index=0,
            content="Score test chunk",
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=16,
            embedding=_make_fake_embedding(0),
            is_embedded=True,
        )
        test_session.add(chunk)
        await test_session.flush()
        await test_session.refresh(chunk)

        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embeddings.return_value = [
            [0.1] * settings.embedding_dimension
        ]

        distance = 0.35
        expected_score = round(1.0 - distance, 6)

        service = RetrievalService(mock_embedding_service)

        with patch.object(
            ChunkRepository, "similarity_search", new_callable=AsyncMock
        ) as mock_sim:
            mock_sim.return_value = [(chunk, distance)]

            results = await service.search_similar_chunks(
                session=test_session,
                document_id=sample_document.id,
                query="score test",
            )

        assert len(results) == 1
        assert results[0].score == pytest.approx(expected_score, abs=0.0001)

    async def test_results_preserve_chunk_metadata(
        self,
        test_session: AsyncSession,
        embedded_chunks: list[Chunk],
        sample_document: Document,
    ) -> None:
        """Returned results include correct chunk metadata fields."""
        target_chunk = embedded_chunks[2]

        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embeddings.return_value = [
            [0.1] * settings.embedding_dimension
        ]

        service = RetrievalService(mock_embedding_service)

        with patch.object(
            ChunkRepository, "similarity_search", new_callable=AsyncMock
        ) as mock_sim:
            mock_sim.return_value = [(target_chunk, 0.1)]

            results = await service.search_similar_chunks(
                session=test_session,
                document_id=sample_document.id,
                query="Kuala Lumpur",
                top_k=1,
            )

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ChunkSearchResult)
        assert result.chunk_id == target_chunk.id
        assert result.chunk_index == target_chunk.chunk_index
        assert result.content == target_chunk.content
        assert result.page_start == target_chunk.page_start
        assert result.page_end == target_chunk.page_end
