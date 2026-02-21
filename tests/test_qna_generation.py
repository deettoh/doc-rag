"""Integration tests for the question generation feature."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.exceptions import DomainValidationError
from app.models.answer import Answer  # noqa: F401
from app.models.base import Base
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.question import Question  # noqa: F401
from app.models.summary import Summary  # noqa: F401
from app.repositories.question import QuestionRepository
from app.services.llm import LLMService, QuestionResult
from app.services.question_generation import QnAGenerationService
from app.services.retrieval import ChunkSearchResult, RetrievalService

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
    """Create a completed document for question generation tests."""
    doc = Document(
        filename="questions_test.pdf",
        file_path="/tmp/questions_test.pdf",
        file_size_bytes=2048,
        page_count=4,
        status=DocumentStatus.COMPLETED,
    )
    test_session.add(doc)
    await test_session.flush()
    await test_session.refresh(doc)
    return doc


def _make_fake_embedding(seed: int) -> list[float]:
    """Generate a deterministic fake embedding vector for testing."""
    base = [0.0] * settings.embedding_dimension
    base[seed % settings.embedding_dimension] = 1.0
    return base


@pytest.fixture
async def embedded_chunks(
    test_session: AsyncSession, completed_document: Document
) -> list[Chunk]:
    """Create chunks with fake embeddings for the completed document."""
    chunk_data = [
        ("Nasi Lemak is the national dish of Malaysia.", 1, 1),
        ("Jay does BJJ.", 2, 2),
        ("Kuala Lumpur is better than Penang", 3, 3),
    ]
    chunks = []
    for i, (content, page_start, page_end) in enumerate(chunk_data):
        chunk = Chunk(
            document_id=completed_document.id,
            chunk_index=i,
            content=content,
            page_start=page_start,
            page_end=page_end,
            char_start=i * 50,
            char_end=(i + 1) * 50,
            embedding=_make_fake_embedding(i),
            is_embedded=True,
        )
        chunks.append(chunk)
    test_session.add_all(chunks)
    await test_session.flush()
    for chunk in chunks:
        await test_session.refresh(chunk)
    return chunks


def _mock_search_results(chunks: list[Chunk]) -> list[ChunkSearchResult]:
    """Convert Chunk models into ChunkSearchResult objects."""
    return [
        ChunkSearchResult(
            chunk_id=c.id,
            chunk_index=c.chunk_index,
            content=c.content,
            page_start=c.page_start,
            page_end=c.page_end,
            score=round(0.9 - i * 0.1, 6),
        )
        for i, c in enumerate(chunks)
    ]


class TestQnAGenerationService:
    """Tests for question generation, deduplication, and persistence."""

    async def test_generate_and_store_questions(
        self,
        test_session: AsyncSession,
        completed_document: Document,
        embedded_chunks: list[Chunk],
    ) -> None:
        """Generate N questions and persist them for the document."""
        search_results = _mock_search_results(embedded_chunks)

        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search_similar_chunks = AsyncMock(return_value=search_results)

        mock_llm = MagicMock(spec=LLMService)
        mock_llm.generate_questions.return_value = [
            QuestionResult(
                question="What is Brazilian Jiu Jitsu?",
                expected_answer=("A ground based grappling combat sport."),
            ),
            QuestionResult(
                question="Who is Jay?",
                expected_answer=("Hu Jay."),
            ),
        ]

        service = QnAGenerationService(mock_retrieval, mock_llm)
        questions = await service.generate_and_store_questions(
            session=test_session,
            document_id=completed_document.id,
            num_questions=2,
        )

        assert len(questions) == 2
        assert questions[0].document_id == completed_document.id
        assert questions[0].content == "What is Brazilian Jiu Jitsu?"
        assert questions[0].expected_answer is not None

    async def test_context_includes_chunk_content_and_markers(
        self,
        test_session: AsyncSession,
        completed_document: Document,
        embedded_chunks: list[Chunk],
    ) -> None:
        """Context passed to LLM includes page markers and chunk text."""
        search_results = _mock_search_results(embedded_chunks)

        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search_similar_chunks = AsyncMock(return_value=search_results)

        mock_llm = MagicMock(spec=LLMService)
        mock_llm.generate_questions.return_value = [
            QuestionResult(question="Q1?", expected_answer="A1"),
        ]

        service = QnAGenerationService(mock_retrieval, mock_llm)
        await service.generate_and_store_questions(
            session=test_session,
            document_id=completed_document.id,
            num_questions=1,
        )

        context = mock_llm.generate_questions.call_args.kwargs["context"]
        assert "[Page 1-1]" in context
        assert "Nasi Lemak is the national dish of Malaysia." in context

    async def test_deduplicates_similar_questions(
        self,
        test_session: AsyncSession,
        completed_document: Document,
        embedded_chunks: list[Chunk],
    ) -> None:
        """Similar questions are deduplicated and generation retries fill gaps."""
        search_results = _mock_search_results(embedded_chunks)

        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search_similar_chunks = AsyncMock(return_value=search_results)

        mock_llm = MagicMock(spec=LLMService)
        mock_llm.generate_questions.side_effect = [
            [
                QuestionResult(
                    question="What is Kuala Lumpur?",
                    expected_answer=("The capital of Malaysia."),
                ),
                QuestionResult(
                    question="What is Kuala Lumpur?",
                    expected_answer=("Malaysia's capital city."),
                ),
            ],
            [
                QuestionResult(
                    question="What does RAG stand for?",
                    expected_answer=("Retrieval-Augmented Generation."),
                )
            ],
        ]

        service = QnAGenerationService(mock_retrieval, mock_llm)
        questions = await service.generate_and_store_questions(
            session=test_session,
            document_id=completed_document.id,
            num_questions=2,
        )

        assert len(questions) == 2
        assert mock_llm.generate_questions.call_count == 2
        assert questions[0].content == "What is Kuala Lumpur?"
        assert questions[1].content == "What does RAG stand for?"

    async def test_empty_chunks_raises_validation_error(
        self,
        test_session: AsyncSession,
        completed_document: Document,
    ) -> None:
        """Raises DomainValidationError when no embedded chunks are found."""
        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search_similar_chunks = AsyncMock(return_value=[])

        mock_llm = MagicMock(spec=LLMService)
        service = QnAGenerationService(mock_retrieval, mock_llm)

        with pytest.raises(DomainValidationError):
            await service.generate_and_store_questions(
                session=test_session,
                document_id=completed_document.id,
                num_questions=2,
            )

        mock_llm.generate_questions.assert_not_called()

    async def test_llm_recovery_on_invalid_output(
        self,
        test_session: AsyncSession,
        completed_document: Document,
        embedded_chunks: list[Chunk],
    ) -> None:
        """LLM returns invalid schema first, valid on retry; QnA generation succeeds."""
        search_results = _mock_search_results(embedded_chunks)

        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.search_similar_chunks = AsyncMock(return_value=search_results)

        with patch("app.services.llm.OpenAI"):
            llm_service = LLMService(api_key="test-key")

        bad_json = json.dumps({"questions": [{"question": "Missing expected answer"}]})
        good_json = json.dumps(
            {
                "questions": [
                    {
                        "question": "What is Nasi Lemak?",
                        "expected_answer": "A Malaysian dish.",
                    }
                ]
            }
        )
        choice_bad = MagicMock()
        choice_bad.message.content = bad_json
        resp_bad = MagicMock()
        resp_bad.choices = [choice_bad]

        choice_good = MagicMock()
        choice_good.message.content = good_json
        resp_good = MagicMock()
        resp_good.choices = [choice_good]

        llm_service.client.chat.completions.create.side_effect = [
            resp_bad,
            resp_good,
        ]

        service = QnAGenerationService(mock_retrieval, llm_service)
        questions = await service.generate_and_store_questions(
            session=test_session,
            document_id=completed_document.id,
            num_questions=1,
        )

        assert len(questions) == 1
        assert questions[0].content == "What is Nasi Lemak?"
        assert llm_service.client.chat.completions.create.call_count == 2


class TestQuestionRepository:
    """Tests for QuestionRepository persistence helpers."""

    async def test_create_bulk_and_retrieve(
        self,
        test_session: AsyncSession,
        completed_document: Document,
    ) -> None:
        """Questions can be batch created and fetched by document_id."""
        created = await QuestionRepository.create_bulk(
            session=test_session,
            document_id=completed_document.id,
            questions=[
                ("What is RAG?", "A retrieval-augmented generation technique."),
                ("Why is KL better than penang?", "Tastier food."),
            ],
        )

        assert len(created) == 2
        assert created[0].id is not None

        fetched = await QuestionRepository.get_by_document_id(
            session=test_session,
            document_id=completed_document.id,
        )

        assert len(fetched) == 2
        assert fetched[0].content == "What is RAG?"
