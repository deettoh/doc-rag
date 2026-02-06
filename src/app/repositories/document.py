"""Repository for document database operations."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentStatus


class DocumentRepository:
    """Handle document persistence operations."""

    @staticmethod
    async def create(
        session: AsyncSession,
        filename: str,
        file_path: str,
        file_size_bytes: int,
    ) -> Document:
        """Create a new document record with UPLOADED status."""
        document = Document(
            filename=filename,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            status=DocumentStatus.UPLOADED,
        )
        session.add(document)
        await session.flush()
        await session.refresh(document)
        return document

    @staticmethod
    async def get_by_id(session: AsyncSession, document_id: int) -> Document | None:
        """Retrieve a document by its ID."""
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()
