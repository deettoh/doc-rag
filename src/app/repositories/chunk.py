"""Repository for chunk database operations."""

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.services.chunking import ChunkData


class ChunkRepository:
    """Handle chunk persistence operations."""

    @staticmethod
    async def create_bulk(
        session: AsyncSession,
        chunks: list[ChunkData],
    ) -> list[Chunk]:
        """Batch insert chunks into the database."""
        db_chunks = [
            Chunk(
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
            )
            for chunk in chunks
        ]
        session.add_all(db_chunks)
        await session.flush()
        return db_chunks

    @staticmethod
    async def get_by_document_id(
        session: AsyncSession,
        document_id: int,
    ) -> list[Chunk]:
        """Retrieve all chunks for a document, ordered by chunk_index."""
        result = await session.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_unembedded_chunks(
        session: AsyncSession,
        document_id: int,
    ) -> list[Chunk]:
        """Retrieve chunks that have not been embedded yet."""
        result = await session.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id, Chunk.is_embedded.is_(False))
            .order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())

    @staticmethod
    async def update_embeddings(
        session: AsyncSession,
        chunk_ids: list[int],
        embeddings: list[list[float]],
    ) -> None:
        """Bulk update embedding vectors for the given chunk IDs."""
        for chunk_id, embedding in zip(chunk_ids, embeddings, strict=True):
            await session.execute(
                update(Chunk)
                .where(Chunk.id == chunk_id)
                .values(embedding=embedding, is_embedded=True)
            )
        await session.flush()
