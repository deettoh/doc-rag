"""Question generation service orchestrating retrieval, LLM, and persistence."""

import re
from difflib import SequenceMatcher

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.exceptions import DomainValidationError
from app.models.question import Question
from app.services.llm import LLMService, QuestionResult
from app.services.retrieval import RetrievalService


class QnAGenerationService:
    """Generate and store document questions using retrieved context."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service

    async def generate_and_store_questions(
        self,
        session: AsyncSession,
        document_id: int,
        num_questions: int | None = None,
        top_k: int | None = None,
    ) -> list[Question]:
        """Generate, deduplicate, and persist questions for a document."""
        if num_questions is None:
            num_questions = settings.default_questions_per_document

        if top_k is None:
            top_k = settings.top_k_retrieval

        # Deferred import to avoid circular dependency
        from app.repositories.question import QuestionRepository

        logger.info(
            "Starting question generation",
            document_id=document_id,
            num_questions=num_questions,
            top_k=top_k,
        )

        chunks = await self.retrieval_service.search_similar_chunks(
            session=session,
            document_id=document_id,
            query="key concepts and testable details",
            top_k=top_k,
        )

        if not chunks:
            raise DomainValidationError(
                "No embedded chunks found for this document. "
                "Ensure the document has been fully processed.",
                field="document_id",
            )

        context = self._build_context(chunks)

        unique_questions: list[QuestionResult] = []
        for attempt in range(1, settings.question_generation_max_attempts + 1):
            remaining = num_questions - len(unique_questions)
            if remaining <= 0:
                break

            logger.info(
                "Requesting questions from LLM",
                document_id=document_id,
                attempt=attempt,
                remaining=remaining,
                context_length=len(context),
            )

            generated = self.llm_service.generate_questions(
                context=context,
                num_questions=remaining,
            )
            unique_questions = self._merge_unique_questions(unique_questions, generated)

        if len(unique_questions) < num_questions:
            raise DomainValidationError(
                "Unable to generate enough unique questions for this document.",
                field="num_questions",
                details={
                    "requested": num_questions,
                    "generated": len(unique_questions),
                },
            )

        selected_questions = unique_questions[:num_questions]
        persisted_questions = await QuestionRepository.create_bulk(
            session=session,
            document_id=document_id,
            questions=[
                (question.question, question.expected_answer)
                for question in selected_questions
            ],
        )

        logger.info(
            "Questions generated and stored",
            document_id=document_id,
            question_count=len(persisted_questions),
        )

        return persisted_questions

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

    @classmethod
    def _merge_unique_questions(
        cls,
        existing: list[QuestionResult],
        candidates: list[QuestionResult],
    ) -> list[QuestionResult]:
        """Merge candidate questions while removing semantically similar entries."""
        merged = list(existing)

        for candidate in candidates:
            if any(
                cls._are_similar_questions(candidate.question, saved.question)
                for saved in merged
            ):
                continue
            merged.append(candidate)

        return merged

    @staticmethod
    def _normalize_question(question: str) -> str:
        """Normalize question text for similarity comparison."""
        cleaned = re.sub(r"[^a-z0-9\s]", " ", question.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _are_similar_questions(cls, first: str, second: str) -> bool:
        """Check if two questions are similar enough to be treated as duplicates."""
        normalized_first = cls._normalize_question(first)
        normalized_second = cls._normalize_question(second)

        if not normalized_first or not normalized_second:
            return False

        if normalized_first == normalized_second:
            return True

        threshold = settings.question_dedup_similarity_threshold

        # Character level
        sequence_score = SequenceMatcher(
            a=normalized_first,
            b=normalized_second,
        ).ratio()
        token_set_first = set(normalized_first.split())
        token_set_second = set(normalized_second.split())

        # Jaccard similarity, disregards order
        union = token_set_first | token_set_second
        token_overlap = (
            0.0 if not union else len(token_set_first & token_set_second) / len(union)
        )

        return sequence_score >= threshold or token_overlap >= threshold
