"""Background document processing service.

Orchestrates the full pipeline: extract -> chunk -> embed,
tracking document status transitions throughout.
"""

from loguru import logger

from app.db import async_session_factory
from app.models.document import DocumentStatus
from app.services.chunking import ChunkingService
from app.services.embedding import EmbeddingService
from app.services.pdf_extractor import PDFExtractorService


class DocumentProcessingService:
    """Orchestrate PDF extraction, chunking, and embedding as a background task."""

    def __init__(
        self,
        pdf_extractor: PDFExtractorService | None = None,
        chunking_service: ChunkingService | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.pdf_extractor = pdf_extractor or PDFExtractorService()
        self.chunking_service = chunking_service or ChunkingService()
        self.embedding_service = embedding_service or EmbeddingService()

    async def process_document(self, document_id: int, file_path: str) -> None:
        """Run the full processing pipeline for an uploaded document.

        Uses its own database session because BackgroundTasks run after
        the request session has already been closed.
        """
        # Deferred imports to avoid circular dependency
        from app.repositories.chunk import ChunkRepository
        from app.repositories.document import DocumentRepository

        async with async_session_factory() as session:
            try:
                await DocumentRepository.update_status(
                    session, document_id, DocumentStatus.PROCESSING
                )
                await session.commit()

                logger.info(
                    "Starting document processing",
                    document_id=document_id,
                )

                pages = self.pdf_extractor.extract_text(file_path)

                await DocumentRepository.update_page_count(
                    session, document_id, len(pages)
                )
                await session.commit()

                logger.info(
                    "PDF extraction complete",
                    document_id=document_id,
                    page_count=len(pages),
                )

                chunks = self.chunking_service.chunk_pages(pages, document_id)

                db_chunks = await ChunkRepository.create_bulk(session, chunks)
                await session.commit()

                logger.info(
                    "Chunking complete",
                    document_id=document_id,
                    chunk_count=len(db_chunks),
                )

                if db_chunks:
                    batch_size = 100
                    for i in range(0, len(db_chunks), batch_size):
                        batch_db_chunks = db_chunks[i : i + batch_size]
                        batch_chunks = chunks[i : i + batch_size]

                        logger.info(
                            "Processing embedding batch",
                            document_id=document_id,
                            batch_start=i,
                            batch_end=i + len(batch_db_chunks),
                            total=len(db_chunks),
                        )

                        batch_embeddings = self.embedding_service.embed_chunks(
                            batch_chunks
                        )
                        batch_ids = [c.id for c in batch_db_chunks]

                        await ChunkRepository.update_embeddings(
                            session, batch_ids, batch_embeddings
                        )
                        await session.commit()

                        logger.info(
                            "Batch embedding complete",
                            document_id=document_id,
                            batch_end=i + len(batch_db_chunks),
                        )

                await DocumentRepository.update_status(
                    session, document_id, DocumentStatus.COMPLETED
                )
                await session.commit()

                logger.info(
                    "Document processing completed successfully",
                    document_id=document_id,
                )

            except Exception as exc:
                await session.rollback()
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "Document processing failed",
                    document_id=document_id,
                    error=error_msg,
                )
                try:
                    await DocumentRepository.update_status(
                        session,
                        document_id,
                        DocumentStatus.FAILED,
                        error_message=error_msg,
                    )
                    await session.commit()
                except Exception as status_exc:
                    logger.error(
                        "Failed to update document status to FAILED",
                        document_id=document_id,
                        error=str(status_exc),
                    )
