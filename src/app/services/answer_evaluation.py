"""Answer evaluation service orchestrating validation, LLM scoring, and storage."""

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.exceptions import DomainValidationError, NotFoundError
from app.models.answer import Answer
from app.repositories.answer import AnswerRepository
from app.repositories.question import QuestionRepository
from app.services.llm import LLMService


class AnswerEvaluationService:
    """Submit and evaluate user answers against generated questions."""

    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    async def submit_and_evaluate_answer(
        self,
        session: AsyncSession,
        document_id: int,
        question_id: int,
        user_answer: str,
    ) -> Answer:
        """Validate question ownership, evaluate answer, and persist result."""
        question = await QuestionRepository.get_by_id_and_document_id(
            session=session,
            question_id=question_id,
            document_id=document_id,
        )
        if question is None:
            raise NotFoundError(resource="Question", resource_id=question_id)

        if settings.prevent_duplicate_answers:
            existing_answer = await AnswerRepository.get_latest_by_question_id(
                session=session,
                question_id=question_id,
            )
            if existing_answer is not None:
                raise DomainValidationError(
                    "An answer has already been submitted for this question.",
                    field="question_id",
                )

        logger.info(
            "Evaluating submitted answer",
            document_id=document_id,
            question_id=question_id,
        )

        expected_answer = question.expected_answer or ""
        if expected_answer.strip():
            evaluation = self.llm_service.evaluate_answer(
                question=question.content,
                expected_answer=expected_answer,
                user_answer=user_answer,
            )
            score = evaluation.score
            feedback = evaluation.feedback
        else:
            score = 0.0
            feedback = (
                "No expected answer is available for this question. "
                "Stored submission without LLM judging."
            )

        answer = await AnswerRepository.create(
            session=session,
            question_id=question_id,
            user_answer=user_answer,
            score=score,
            feedback=feedback,
        )

        logger.info(
            "Answer evaluated and stored",
            document_id=document_id,
            question_id=question_id,
            answer_id=answer.id,
            score=score,
        )

        return answer
