"""Repository for question database operations."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.question import Question


class QuestionRepository:
    """Handle question persistence operations."""

    @staticmethod
    async def create_bulk(
        session: AsyncSession,
        document_id: int,
        questions: list[tuple[str, str]],
    ) -> list[Question]:
        """Batch insert generated questions for a document."""
        db_questions = [
            Question(
                document_id=document_id,
                content=question,
                expected_answer=expected_answer,
            )
            for question, expected_answer in questions
        ]
        session.add_all(db_questions)
        await session.flush()
        for question in db_questions:
            await session.refresh(question)
        return db_questions

    @staticmethod
    async def get_by_document_id(
        session: AsyncSession,
        document_id: int,
    ) -> list[Question]:
        """Retrieve all generated questions for a document."""
        result = await session.execute(
            select(Question)
            .where(Question.document_id == document_id)
            .order_by(Question.created_at.asc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_by_id_and_document_id(
        session: AsyncSession,
        question_id: int,
        document_id: int,
    ) -> Question | None:
        """Retrieve a question by id scoped to a specific document."""
        result = await session.execute(
            select(Question).where(
                Question.id == question_id,
                Question.document_id == document_id,
            )
        )
        return result.scalar_one_or_none()
