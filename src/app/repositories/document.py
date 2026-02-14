"""Repository for document database operations."""

from sqlalchemy import select, update
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

    @staticmethod
    async def update_status(
        session: AsyncSession,
        document_id: int,
        status: DocumentStatus,
        error_message: str | None = None,
    ) -> None:
        """Transition a document to a new processing status."""
        values: dict = {"status": status}
        if error_message is not None:
            values["error_message"] = error_message
        await session.execute(
            update(Document).where(Document.id == document_id).values(**values)
        )
        await session.flush()

    @staticmethod
    async def update_page_count(
        session: AsyncSession,
        document_id: int,
        page_count: int,
    ) -> None:
        """Set the page count after PDF extraction."""
        await session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(page_count=page_count)
        )
        await session.flush()
