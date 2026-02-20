"""Document upload and management endpoints."""

from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.exceptions import DomainValidationError, NotFoundError
from app.models.document import DocumentStatus
from app.repositories.answer import AnswerRepository
from app.repositories.document import DocumentRepository
from app.repositories.question import QuestionRepository
from app.repositories.summary import SummaryRepository
from app.schemas.answer import AnswerResponse, AnswerSubmissionRequest
from app.schemas.document import DocumentResponse, DocumentStatusResponse
from app.schemas.question import (
    QuestionGenerationRequest,
    QuestionGenerationResponse,
    QuestionResponse,
)
from app.schemas.retrieval import ChunkResult, RetrievalRequest, RetrievalResponse
from app.schemas.summary import SummaryResponse
from app.services.answer_evaluation import AnswerEvaluationService
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.services.processing import DocumentProcessingService
from app.services.question_generation import QnAGenerationService
from app.services.retrieval import RetrievalService
from app.services.storage import FileStorageService
from app.services.summarization import SummarizationService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get(
    "/",
    response_model=list[DocumentResponse],
    summary="List all documents",
    description="Return all uploaded documents ordered by most recent first.",
)
async def list_documents(
    session: AsyncSession = Depends(get_db),
) -> list[DocumentResponse]:
    """List all uploaded documents."""
    documents = await DocumentRepository.get_all(session)
    return [DocumentResponse.model_validate(doc) for doc in documents]


@router.post(
    "/upload",
    response_model=DocumentResponse,
    summary="Upload a PDF document",
    description="Upload a PDF file for processing. Returns document metadata.",
)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Upload a PDF document and start background processing."""
    storage_service = FileStorageService()

    logger.info("Processing document upload", filename=file.filename)

    file_path, file_size = await storage_service.save_pdf(file)

    document = await DocumentRepository.create(
        session=session,
        filename=file.filename or "upload.pdf",
        file_path=file_path,
        file_size_bytes=file_size,
    )
    # Ensure the document row is committed before background processing starts.
    # BackgroundTasks may execute before dependency teardown commit.
    await session.commit()
    await session.refresh(document)

    logger.info(
        "Document uploaded successfully",
        document_id=document.id,
        filename=document.filename,
        file_size_bytes=file_size,
    )

    processing_service = DocumentProcessingService()
    background_tasks.add_task(
        processing_service.process_document,
        document.id,
        file_path,
    )

    logger.info(
        "Background processing enqueued",
        document_id=document.id,
    )

    return DocumentResponse.model_validate(document)


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get document processing status",
    description="Poll the current processing status of an uploaded document.",
)
async def get_document_status(
    document_id: int,
    session: AsyncSession = Depends(get_db),
) -> DocumentStatusResponse:
    """Return the current processing status of a document."""
    document = await DocumentRepository.get_by_id(session, document_id)
    if document is None:
        raise NotFoundError(resource="Document", resource_id=document_id)

    return DocumentStatusResponse.model_validate(document)


@router.post(
    "/{document_id}/search",
    response_model=RetrievalResponse,
    summary="Search document chunks",
    description="Find the most relevant chunks in a document for a given query.",
)
async def search_document(
    document_id: int,
    request: RetrievalRequest,
    session: AsyncSession = Depends(get_db),
) -> RetrievalResponse:
    """Search for relevant chunks within a document using similarity search."""
    document = await DocumentRepository.get_by_id(session, document_id)
    if document is None:
        raise NotFoundError(resource="Document", resource_id=document_id)

    logger.info(
        "Executing similarity search",
        document_id=document_id,
        query_length=len(request.query),
        top_k=request.top_k,
    )

    embedding_service = EmbeddingService()
    retrieval_service = RetrievalService(embedding_service)

    results = await retrieval_service.search_similar_chunks(
        session=session,
        document_id=document_id,
        query=request.query,
        top_k=request.top_k,
    )

    logger.info(
        "Similarity search completed",
        document_id=document_id,
        results_count=len(results),
    )

    return RetrievalResponse(
        document_id=document_id,
        query=request.query,
        results=[
            ChunkResult(
                chunk_id=r.chunk_id,
                chunk_index=r.chunk_index,
                content=r.content,
                page_start=r.page_start,
                page_end=r.page_end,
                score=r.score,
            )
            for r in results
        ],
    )


@router.get(
    "/{document_id}/summary",
    response_model=SummaryResponse,
    summary="Get existing document summary",
    description="Retrieve the most recent summary for a document if it exists.",
)
async def get_document_summary(
    document_id: int,
    session: AsyncSession = Depends(get_db),
) -> SummaryResponse:
    """Retrieve an existing summary for a document."""
    document = await DocumentRepository.get_by_id(session, document_id)
    if document is None:
        raise NotFoundError(resource="Document", resource_id=document_id)

    summary = await SummaryRepository.get_by_document_id(session, document_id)
    if summary is None:
        raise NotFoundError(resource="Summary", resource_id=document_id)

    return SummaryResponse.model_validate(summary)


@router.post(
    "/{document_id}/summarize",
    response_model=SummaryResponse,
    summary="Generate document summary",
    description=(
        "Generate a summary for a fully processed document using "
        "retrieved chunks and LLM."
    ),
)
async def summarize_document(
    document_id: int,
    session: AsyncSession = Depends(get_db),
) -> SummaryResponse:
    """Generate and store a summary for a completed document."""
    document = await DocumentRepository.get_by_id(session, document_id)
    if document is None:
        raise NotFoundError(resource="Document", resource_id=document_id)

    if document.status != DocumentStatus.COMPLETED:
        raise DomainValidationError(
            f"Document must be fully processed before summarization. "
            f"Current status: {document.status}",
            field="status",
        )

    logger.info(
        "Summarization requested",
        document_id=document_id,
    )

    embedding_service = EmbeddingService()
    retrieval_service = RetrievalService(embedding_service)
    llm_service = LLMService()
    summarization_service = SummarizationService(retrieval_service, llm_service)

    summary = await summarization_service.generate_and_store_summary(
        session=session,
        document_id=document_id,
    )

    logger.info(
        "Summarization completed",
        document_id=document_id,
        summary_id=summary.id,
    )

    return SummaryResponse.model_validate(summary)


@router.get(
    "/{document_id}/questions",
    response_model=list[QuestionResponse],
    summary="Get existing document questions",
    description="Retrieve all previously generated questions for a document.",
)
async def get_document_questions(
    document_id: int,
    session: AsyncSession = Depends(get_db),
) -> list[QuestionResponse]:
    """Retrieve all existing questions for a document."""
    document = await DocumentRepository.get_by_id(session, document_id)
    if document is None:
        raise NotFoundError(resource="Document", resource_id=document_id)

    questions = await QuestionRepository.get_by_document_id(session, document_id)
    return [QuestionResponse.model_validate(q) for q in questions]


@router.post(
    "/{document_id}/questions",
    response_model=QuestionGenerationResponse,
    summary="Generate document questions",
    description=(
        "Generate and store study questions for a fully processed document "
        "using retrieved chunks and LLM."
    ),
)
async def generate_document_questions(
    document_id: int,
    request: QuestionGenerationRequest | None = None,
    session: AsyncSession = Depends(get_db),
) -> QuestionGenerationResponse:
    """Generate and store questions for a completed document."""
    if request is None:
        request = QuestionGenerationRequest()

    document = await DocumentRepository.get_by_id(session, document_id)
    if document is None:
        raise NotFoundError(resource="Document", resource_id=document_id)

    if document.status != DocumentStatus.COMPLETED:
        raise DomainValidationError(
            f"Document must be fully processed before question generation. "
            f"Current status: {document.status}",
            field="status",
        )

    logger.info(
        "Question generation requested",
        document_id=document_id,
        requested_count=request.num_questions,
    )

    embedding_service = EmbeddingService()
    retrieval_service = RetrievalService(embedding_service)
    llm_service = LLMService()
    question_service = QnAGenerationService(retrieval_service, llm_service)

    questions = await question_service.generate_and_store_questions(
        session=session,
        document_id=document_id,
        num_questions=request.num_questions,
    )

    logger.info(
        "Question generation completed",
        document_id=document_id,
        generated_count=len(questions),
    )

    return QuestionGenerationResponse(
        document_id=document_id,
        requested_count=request.num_questions,
        generated_count=len(questions),
        questions=[QuestionResponse.model_validate(question) for question in questions],
    )


@router.get(
    "/{document_id}/questions/{question_id}/answer",
    response_model=AnswerResponse,
    summary="Get latest evaluation for a question",
    description="Retrieve the most recent evaluation and feedback for a specific question.",
)
async def get_question_answer(
    document_id: int,
    question_id: int,
    session: AsyncSession = Depends(get_db),
) -> AnswerResponse:
    """Retrieve the latest evaluation for a question."""
    document = await DocumentRepository.get_by_id(session, document_id)
    if document is None:
        raise NotFoundError(resource="Document", resource_id=document_id)

    answer = await AnswerRepository.get_latest_by_question_id(session, question_id)
    if answer is None:
        raise NotFoundError(resource="Answer", resource_id=question_id)

    return AnswerResponse.model_validate(answer)


@router.post(
    "/{document_id}/questions/{question_id}/answer",
    response_model=AnswerResponse,
    summary="Submit answer for a generated question",
    description=(
        "Submit a user answer for a generated question, evaluate it, "
        "and store score and feedback."
    ),
)
async def submit_question_answer(
    document_id: int,
    question_id: int,
    request: AnswerSubmissionRequest,
    session: AsyncSession = Depends(get_db),
) -> AnswerResponse:
    """Submit an answer for a generated question and persist evaluation."""
    document = await DocumentRepository.get_by_id(session, document_id)
    if document is None:
        raise NotFoundError(resource="Document", resource_id=document_id)

    logger.info(
        "Answer submission requested",
        document_id=document_id,
        question_id=question_id,
    )

    llm_service = LLMService()
    answer_service = AnswerEvaluationService(llm_service)
    answer = await answer_service.submit_and_evaluate_answer(
        session=session,
        document_id=document_id,
        question_id=question_id,
        user_answer=request.user_answer,
    )

    logger.info(
        "Answer submission completed",
        document_id=document_id,
        question_id=question_id,
        answer_id=answer.id,
    )

    return AnswerResponse.model_validate(answer)
