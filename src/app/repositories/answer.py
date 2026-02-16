"""Repository for answer database operations."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.answer import Answer


class AnswerRepository:
    """Handle answer persistence operations."""

    @staticmethod
    async def create(
        session: AsyncSession,
        question_id: int,
        user_answer: str,
        score: float,
        feedback: str,
    ) -> Answer:
        """Create a new evaluated answer record."""
        answer = Answer(
            question_id=question_id,
            user_answer=user_answer,
            score=score,
            feedback=feedback,
        )
        session.add(answer)
        await session.flush()
        await session.refresh(answer)
        return answer

    @staticmethod
    async def get_latest_by_question_id(
        session: AsyncSession,
        question_id: int,
    ) -> Answer | None:
        """Retrieve the most recent answer for a question if it exists."""
        result = await session.execute(
            select(Answer)
            .where(Answer.question_id == question_id)
            .order_by(Answer.created_at.desc(), Answer.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
