"""Integration tests for answer submission and evaluation."""

from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.exceptions import DomainValidationError, NotFoundError
from app.models.answer import Answer  # noqa: F401
from app.models.base import Base
from app.models.chunk import Chunk  # noqa: F401
from app.models.document import Document, DocumentStatus
from app.models.question import Question
from app.models.summary import Summary  # noqa: F401
from app.repositories.answer import AnswerRepository
from app.services.answer_evaluation import AnswerEvaluationService
from app.services.llm import EvaluationResult, LLMService

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
    """Create a completed document for answer evaluation tests."""
    doc = Document(
        filename="answers_test.pdf",
        file_path="/tmp/answers_test.pdf",
        file_size_bytes=1024,
        page_count=2,
        status=DocumentStatus.COMPLETED,
    )
    test_session.add(doc)
    await test_session.flush()
    await test_session.refresh(doc)
    return doc


@pytest.fixture
async def generated_question(
    test_session: AsyncSession, completed_document: Document
) -> Question:
    """Create a generated question with an expected answer."""
    question = Question(
        document_id=completed_document.id,
        content="What is RAG?",
        expected_answer="Retrieval-augmented generation.",
    )
    test_session.add(question)
    await test_session.flush()
    await test_session.refresh(question)
    return question


@pytest.fixture
async def question_without_expected_answer(
    test_session: AsyncSession, completed_document: Document
) -> Question:
    """Create a generated question without expected answer text."""
    question = Question(
        document_id=completed_document.id,
        content="What is this question about?",
        expected_answer=None,
    )
    test_session.add(question)
    await test_session.flush()
    await test_session.refresh(question)
    return question


class TestAnswerEvaluationService:
    """Tests for answer submission, evaluation, and duplicate prevention."""

    async def test_submit_and_evaluate_answer_success(
        self,
        test_session: AsyncSession,
        completed_document: Document,
        generated_question: Question,
    ) -> None:
        """LLM evaluation result should be stored with the submitted answer."""
        mock_llm = MagicMock(spec=LLMService)
        mock_llm.evaluate_answer.return_value = EvaluationResult(
            score=0.85,
            feedback="Good explanation with correct key terms.",
        )

        service = AnswerEvaluationService(mock_llm)
        answer = await service.submit_and_evaluate_answer(
            session=test_session,
            document_id=completed_document.id,
            question_id=generated_question.id,
            user_answer="RAG combines retrieval and generation.",
        )

        assert answer.id is not None
        assert answer.question_id == generated_question.id
        assert answer.score == pytest.approx(0.85)
        assert answer.feedback == "Good explanation with correct key terms."
        mock_llm.evaluate_answer.assert_called_once()

    async def test_submit_with_missing_expected_answer_uses_heuristic(
        self,
        test_session: AsyncSession,
        completed_document: Document,
        question_without_expected_answer: Question,
    ) -> None:
        """When expected answer is missing, service stores heuristic result."""
        mock_llm = MagicMock(spec=LLMService)
        service = AnswerEvaluationService(mock_llm)

        answer = await service.submit_and_evaluate_answer(
            session=test_session,
            document_id=completed_document.id,
            question_id=question_without_expected_answer.id,
            user_answer="My response text.",
        )

        assert answer.score == 0.0
        assert "Stored submission without LLM judging" in (answer.feedback or "")
        mock_llm.evaluate_answer.assert_not_called()

    async def test_duplicate_submission_rejected(
        self,
        test_session: AsyncSession,
        completed_document: Document,
        generated_question: Question,
    ) -> None:
        """Second answer for same question should be rejected when enabled."""
        mock_llm = MagicMock(spec=LLMService)
        mock_llm.evaluate_answer.return_value = EvaluationResult(
            score=0.9,
            feedback="Correct.",
        )
        service = AnswerEvaluationService(mock_llm)

        await service.submit_and_evaluate_answer(
            session=test_session,
            document_id=completed_document.id,
            question_id=generated_question.id,
            user_answer="First answer.",
        )

        with pytest.raises(DomainValidationError):
            await service.submit_and_evaluate_answer(
                session=test_session,
                document_id=completed_document.id,
                question_id=generated_question.id,
                user_answer="Second answer.",
            )

    async def test_question_not_found_raises_not_found(
        self,
        test_session: AsyncSession,
        completed_document: Document,
    ) -> None:
        """Submitting for a missing question should raise NotFoundError."""
        mock_llm = MagicMock(spec=LLMService)
        service = AnswerEvaluationService(mock_llm)

        with pytest.raises(NotFoundError):
            await service.submit_and_evaluate_answer(
                session=test_session,
                document_id=completed_document.id,
                question_id=9999,
                user_answer="Any answer",
            )


class TestAnswerRepository:
    """Tests for AnswerRepository operations used by the service."""

    async def test_create_and_get_latest_by_question_id(
        self,
        test_session: AsyncSession,
        generated_question: Question,
    ) -> None:
        """Latest answer lookup should return most recently created record."""
        first = await AnswerRepository.create(
            session=test_session,
            question_id=generated_question.id,
            user_answer="First",
            score=0.2,
            feedback="Weak",
        )
        second = await AnswerRepository.create(
            session=test_session,
            question_id=generated_question.id,
            user_answer="Second",
            score=0.8,
            feedback="Better",
        )

        assert first.id is not None
        latest = await AnswerRepository.get_latest_by_question_id(
            session=test_session,
            question_id=generated_question.id,
        )
        assert latest is not None
        assert latest.id == second.id
