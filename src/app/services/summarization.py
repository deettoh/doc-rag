"""Summarization service orchestrating retrieval, LLM, and persistence."""

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.exceptions import DomainValidationError
from app.models.summary import Summary
from app.services.llm import LLMService
from app.services.retrieval import RetrievalService


class SummarizationService:
    """Generate and store document summaries using retrieved context."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service

    async def generate_and_store_summary(
        self,
        session: AsyncSession,
        document_id: int,
        top_k: int | None = None,
    ) -> Summary:
        """Retrieve top chunks, generate a summary via LLM, and persist it.

        Args:
            session: Database session.
            document_id: Document to summarize.
            top_k: Number of chunks to use as context
                   (defaults to config value).

        Returns:
            The persisted Summary model instance.

        Raises:
            DomainValidationError: If no embedded chunks are found.
        """
        if top_k is None:
            top_k = settings.top_k_retrieval

        # Deferred import to avoid circular dependency
        from app.repositories.summary import SummaryRepository

        logger.info(
            "Starting summary generation",
            document_id=document_id,
            top_k=top_k,
        )

        chunks = await self.retrieval_service.search_similar_chunks(
            session=session,
            document_id=document_id,
            query="main topics and key points",
            top_k=top_k,
        )

        if not chunks:
            raise DomainValidationError(
                "No embedded chunks found for this document. "
                "Ensure the document has been fully processed.",
                field="document_id",
            )

        context = self._build_context(chunks)

        logger.info(
            "Sending context to LLM for summarization",
            document_id=document_id,
            chunk_count=len(chunks),
            context_length=len(context),
        )

        result = self.llm_service.generate_summary(context)

        summary = await SummaryRepository.create(
            session=session,
            document_id=document_id,
            content=result.summary,
            page_citations=result.page_citations,
        )

        logger.info(
            "Summary generated and stored",
            document_id=document_id,
            summary_id=summary.id,
            citation_count=len(result.page_citations),
        )

        return summary

    @staticmethod
    def _build_context(chunks: list) -> str:
        """Concatenate chunk text with page markers, respecting max context size."""
        parts = []
        current_length = 0

        for chunk in chunks:
            marker = f"[Page {chunk.page_start}-{chunk.page_end}]"
            segment = f"{marker}\n{chunk.content}\n"

            if current_length + len(segment) > settings.max_llm_context_size:
                break

            parts.append(segment)
            current_length += len(segment)

        return "\n".join(parts)
