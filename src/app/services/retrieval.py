"""Retrieval service for similarity search over document chunks."""

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.services.embedding import EmbeddingService


@dataclass
class ChunkSearchResult:
    """A single chunk result from similarity search."""

    chunk_id: int
    chunk_index: int
    content: str
    page_start: int
    page_end: int
    score: float  # cosine similarity (1 - distance)


class RetrievalService:
    """Retrieve relevant chunks for a query using vector similarity."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    async def search_similar_chunks(
        self,
        session: AsyncSession,
        document_id: int,
        query: str,
        top_k: int | None = None,
    ) -> list[ChunkSearchResult]:
        """Search for chunks most similar to the query text.

        Args:
            session: Database session
            document_id: Document to search within
            query: Natural language query string
            top_k: Number of results (defaults to config value)

        Returns:
            List of ChunkSearchResult ordered by descending similarity
        """
        if top_k is None:
            top_k = settings.top_k_retrieval

        query_embedding = self.embedding_service.generate_embeddings([query])[0]

        # Deferred import to avoid circular dependency
        from app.repositories.chunk import ChunkRepository

        results = await ChunkRepository.similarity_search(
            session=session,
            document_id=document_id,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return [
            ChunkSearchResult(
                chunk_id=chunk.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                score=round(1.0 - distance, 6),
            )
            for chunk, distance in results
        ]
