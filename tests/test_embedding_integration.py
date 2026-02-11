"""Integration tests for the embedding flow with database."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.models import Base, Chunk, Document, DocumentStatus
from app.repositories import ChunkRepository
from app.services import EmbeddingService
from app.services.chunking import ChunkData

# Use in-memory SQLite for integration tests
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
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        file_size_bytes=1024,
        page_count=5,
        status=DocumentStatus.UPLOADED,
    )
    test_session.add(doc)
    await test_session.flush()
    await test_session.refresh(doc)
    return doc


@pytest.fixture
async def sample_chunks(
    test_session: AsyncSession, sample_document: Document
) -> list[Chunk]:
    """Create sample chunks for testing."""
    chunks = [
        Chunk(
            document_id=sample_document.id,
            chunk_index=i,
            content=f"Test chunk {i}",
            page_start=1,
            page_end=1,
            char_start=i * 10,
            char_end=(i + 1) * 10,
        )
        for i in range(3)
    ]
    test_session.add_all(chunks)
    await test_session.flush()
    for chunk in chunks:
        await test_session.refresh(chunk)
    return chunks


class TestEmbeddingIntegration:
    """Integration tests for embedding + storage flow."""

    async def test_update_embeddings_sets_is_embedded_flag(
        self, test_session: AsyncSession, sample_chunks: list[Chunk]
    ) -> None:
        """update_embeddings should set is_embedded to True."""
        for chunk in sample_chunks:
            assert chunk.is_embedded is False
            assert chunk.embedding is None

        fake_embeddings = [
            [float(i) for _ in range(settings.embedding_dimension)]
            for i in range(len(sample_chunks))
        ]

        chunk_ids = [chunk.id for chunk in sample_chunks]
        await ChunkRepository.update_embeddings(
            test_session, chunk_ids, fake_embeddings
        )
        await test_session.commit()

        result = await test_session.execute(
            select(Chunk).where(Chunk.id.in_(chunk_ids)).order_by(Chunk.chunk_index)
        )
        updated_chunks = list(result.scalars().all())

        for chunk in updated_chunks:
            assert chunk.is_embedded is True
            assert chunk.embedding is not None
            assert len(chunk.embedding) == settings.embedding_dimension

    async def test_get_unembedded_chunks_filters_correctly(
        self, test_session: AsyncSession, sample_chunks: list[Chunk]
    ) -> None:
        """get_unembedded_chunks should return only chunks without embeddings."""
        fake_embedding = [[1.0] * settings.embedding_dimension]
        await ChunkRepository.update_embeddings(
            test_session, [sample_chunks[0].id], fake_embedding
        )
        await test_session.commit()

        document_id = sample_chunks[0].document_id
        unembedded = await ChunkRepository.get_unembedded_chunks(
            test_session, document_id
        )

        assert len(unembedded) == 2
        assert unembedded[0].chunk_index == 1
        assert unembedded[1].chunk_index == 2
        assert all(chunk.is_embedded is False for chunk in unembedded)

    async def test_get_unembedded_chunks_returns_empty_when_all_embedded(
        self, test_session: AsyncSession, sample_chunks: list[Chunk]
    ) -> None:
        """get_unembedded_chunks should return empty list when all chunks are embedded."""
        fake_embeddings = [
            [float(i)] * settings.embedding_dimension for i in range(len(sample_chunks))
        ]
        chunk_ids = [chunk.id for chunk in sample_chunks]
        await ChunkRepository.update_embeddings(
            test_session, chunk_ids, fake_embeddings
        )
        await test_session.commit()

        document_id = sample_chunks[0].document_id
        unembedded = await ChunkRepository.get_unembedded_chunks(
            test_session, document_id
        )

        assert len(unembedded) == 0

    @patch("app.services.embedding.SentenceTransformer")
    async def test_embed_chunks_generates_correct_vectors(
        self, mock_transformer_cls: MagicMock
    ) -> None:
        """EmbeddingService should generate correct-length vectors from chunks."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(2, settings.embedding_dimension)
        mock_transformer_cls.return_value = mock_model

        chunk_data = [
            ChunkData(
                document_id=1,
                chunk_index=0,
                content="First chunk",
                page_start=1,
                page_end=1,
                char_start=0,
                char_end=11,
            ),
            ChunkData(
                document_id=1,
                chunk_index=1,
                content="Second chunk",
                page_start=1,
                page_end=1,
                char_start=12,
                char_end=24,
            ),
        ]

        service = EmbeddingService()
        embeddings = service.embed_chunks(chunk_data)

        assert len(embeddings) == 2
        assert all(len(emb) == settings.embedding_dimension for emb in embeddings)
        mock_model.encode.assert_called_once_with(
            ["First chunk", "Second chunk"], convert_to_numpy=True
        )

    async def test_full_embed_and_store_workflow(
        self, test_session: AsyncSession, sample_document: Document
    ) -> None:
        """Test the complete flow: create chunks -> embed -> store -> verify."""
        chunk_data = [
            ChunkData(
                document_id=sample_document.id,
                chunk_index=i,
                content=f"Workflow test chunk {i}",
                page_start=1,
                page_end=1,
                char_start=i * 20,
                char_end=(i + 1) * 20,
            )
            for i in range(2)
        ]

        db_chunks = await ChunkRepository.create_bulk(test_session, chunk_data)
        await test_session.commit()

        for chunk in db_chunks:
            assert chunk.is_embedded is False

        fake_embeddings = [
            [float(i)] * settings.embedding_dimension for i in range(len(db_chunks))
        ]

        chunk_ids = [chunk.id for chunk in db_chunks]
        await ChunkRepository.update_embeddings(
            test_session, chunk_ids, fake_embeddings
        )
        await test_session.commit()

        all_chunks = await ChunkRepository.get_by_document_id(
            test_session, sample_document.id
        )
        assert len(all_chunks) == 2
        assert all(chunk.is_embedded is True for chunk in all_chunks)
        assert all(chunk.embedding is not None for chunk in all_chunks)

        unembedded = await ChunkRepository.get_unembedded_chunks(
            test_session, sample_document.id
        )
        assert len(unembedded) == 0
