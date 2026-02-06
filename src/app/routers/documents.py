"""Document upload and management endpoints."""

from fastapi import APIRouter, Depends, UploadFile
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.repositories.document import DocumentRepository
from app.schemas.document import DocumentResponse
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
    session: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Upload a PDF document for processing."""
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

    return DocumentResponse.model_validate(document)
