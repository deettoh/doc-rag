"""End-to-end happy path test for the full DocRAG flow.

Flow: Upload -> Process -> Summarize -> Generate Questions -> Answer
using TestClient with an in-memory SQLite database. Only external
dependencies (LLM API, embedding model, file storage) are mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import get_db
from app.main import app
from app.models.answer import Answer  # noqa: F401
from app.models.base import Base
from app.models.chunk import Chunk  # noqa: F401
from app.models.document import Document  # noqa: F401
from app.models.question import Question  # noqa: F401
from app.models.summary import Summary  # noqa: F401
from app.services.chunking import ChunkingService
from app.services.processing import DocumentProcessingService

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def e2e_engine():
    """Create an async SQLite engine for E2E testing."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def e2e_session_factory(e2e_engine):
    """Create the session factory."""
    return async_sessionmaker(e2e_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
def e2e_session_holder(e2e_session_factory):
    """Track sessions created for dependency injection."""
    sessions: list[AsyncSession] = []

    async def _get_db():
        async with e2e_session_factory() as session:
            try:
                sessions.append(session)
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    return _get_db, sessions


class TestE2EHappyPath:
    """Full end-to-end happy path through the DocRAG application."""

    @pytest.mark.asyncio
    async def test_full_flow(
        self,
        e2e_session_factory,
        e2e_session_holder,
    ) -> None:
        """Upload -> Process -> Summarize -> Questions -> Answer -> Evaluate."""
        get_db_override, _ = e2e_session_holder
        app.dependency_overrides[get_db] = get_db_override

        try:
            mock_pdf_extractor = MagicMock()
            page1 = MagicMock()
            page1.page_number = 1
            page1.text = "Machine learning is a subset of AI."
            page2 = MagicMock()
            page2.page_number = 2
            page2.text = "Neural networks have multiple layers."
            mock_pdf_extractor.extract_text.return_value = [page1, page2]

            mock_embedding = MagicMock()
            mock_embedding.embed_chunks.return_value = [
                [0.1] * 768,
                [0.2] * 768,
            ]
            mock_embedding.generate_embeddings.return_value = [[0.15] * 768]

            # 1. Upload
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                with (
                    patch(
                        "app.routers.documents.FileStorageService"
                    ) as mock_storage_cls,
                    patch("app.routers.documents.DocumentProcessingService"),
                ):
                    mock_storage = mock_storage_cls.return_value
                    mock_storage.save_pdf = AsyncMock(
                        return_value=("/tmp/test_e2e.pdf", 2048)
                    )

                    upload_resp = await client.post(
                        "/api/documents/upload",
                        files={
                            "file": (
                                "test.pdf",
                                b"%PDF-1.4 test",
                                "application/pdf",
                            )
                        },
                    )

                assert upload_resp.status_code == 200
                doc_data = upload_resp.json()
                doc_id = doc_data["id"]
                assert doc_data["status"] == "uploaded"

                # 2. Process
                processing_service = DocumentProcessingService(
                    pdf_extractor=mock_pdf_extractor,
                    chunking_service=ChunkingService(),
                    embedding_service=mock_embedding,
                )

                with (
                    patch(
                        "app.services.processing.async_session_factory",
                        e2e_session_factory,
                    ),
                    patch(
                        "app.repositories.chunk.ChunkRepository.update_embeddings",
                        new=AsyncMock(),
                    ),
                ):
                    await processing_service.process_document(
                        doc_id, "/tmp/test_e2e.pdf"
                    )

                status_resp = await client.get(f"/api/documents/{doc_id}/status")
                assert status_resp.status_code == 200
                assert status_resp.json()["status"] == "completed"

                # 3. Summarize
                with (
                    patch("app.routers.documents.EmbeddingService") as mock_emb_cls,
                    patch("app.routers.documents.LLMService") as mock_llm_cls,
                    patch("app.routers.documents.RetrievalService") as mock_ret_cls,
                ):
                    mock_ret = mock_ret_cls.return_value
                    mock_chunk = MagicMock()
                    mock_chunk.content = "Relevant text."
                    mock_chunk.page_start = 1
                    mock_ret.search_similar_chunks = AsyncMock(
                        return_value=[mock_chunk]
                    )

                    mock_emb_cls.return_value = mock_embedding

                    mock_llm = mock_llm_cls.return_value
                    mock_llm.generate_summary.return_value = MagicMock(
                        summary="Text overview.",
                        page_citations=[1, 2],
                    )

                    summ_resp = await client.post(f"/api/documents/{doc_id}/summarize")

                assert summ_resp.status_code == 200
                summ_data = summ_resp.json()
                assert summ_data["content"] == "Text overview."
                assert summ_data["page_citations"] == [1, 2]
                summary_id = summ_data["id"]

                # 4. Get persisted summary
                get_summ_resp = await client.get(f"/api/documents/{doc_id}/summary")
                assert get_summ_resp.status_code == 200
                assert get_summ_resp.json()["id"] == summary_id

                # 5. Generate questions
                with (
                    patch("app.routers.documents.EmbeddingService") as mock_emb_cls,
                    patch("app.routers.documents.LLMService") as mock_llm_cls,
                    patch("app.routers.documents.RetrievalService") as mock_ret_cls,
                ):
                    mock_ret = mock_ret_cls.return_value
                    mock_chunk = MagicMock()
                    mock_chunk.content = "Relevant text."
                    mock_chunk.page_start = 1
                    mock_ret.search_similar_chunks = AsyncMock(
                        return_value=[mock_chunk]
                    )

                    mock_emb_cls.return_value = mock_embedding

                    mock_llm = mock_llm_cls.return_value
                    mock_llm.generate_questions.return_value = [
                        MagicMock(
                            question="What is text?",
                            expected_answer="Text answer.",
                        ),
                    ]

                    q_resp = await client.post(
                        f"/api/documents/{doc_id}/questions",
                        json={"num_questions": 1},
                    )

                assert q_resp.status_code == 200
                q_data = q_resp.json()
                assert q_data["generated_count"] == 1
                question_id = q_data["questions"][0]["id"]

                # 6. Get persisted questions
                get_q_resp = await client.get(f"/api/documents/{doc_id}/questions")
                assert get_q_resp.status_code == 200
                assert len(get_q_resp.json()) >= 1

                # 7. Submit and evaluate answer
                with patch("app.routers.documents.LLMService") as mock_llm_cls:
                    mock_llm = mock_llm_cls.return_value
                    mock_eval = MagicMock()
                    mock_eval.score = 0.9
                    mock_eval.feedback = "Excellent."
                    mock_llm.evaluate_answer.return_value = mock_eval

                    answer_resp = await client.post(
                        f"/api/documents/{doc_id}/questions/{question_id}/answer",
                        json={"user_answer": "Answer is text."},
                    )

                assert answer_resp.status_code == 200
                ans_data = answer_resp.json()
                assert ans_data["score"] == 0.9
                assert ans_data["feedback"] == "Excellent."
                answer_id = ans_data["id"]

                # 8. Get persisted answer evaluation
                get_ans_resp = await client.get(
                    f"/api/documents/{doc_id}/questions/{question_id}/answer"
                )
                assert get_ans_resp.status_code == 200
                assert get_ans_resp.json()["id"] == answer_id

        finally:
            app.dependency_overrides.clear()
