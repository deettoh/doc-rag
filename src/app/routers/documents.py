"""Document upload and management endpoints."""

from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.exceptions import NotFoundError
from app.repositories.document import DocumentRepository
from app.schemas.document import DocumentResponse, DocumentStatusResponse
from app.schemas.retrieval import ChunkResult, RetrievalRequest, RetrievalResponse
from app.services.embedding import EmbeddingService
from app.services.processing import DocumentProcessingService
from app.services.retrieval import RetrievalService
from app.services.storage import FileStorageService

router = APIRouter(prefix="/documents", tags=["Documents"])


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
