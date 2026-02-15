"""Repository for summary database operations."""

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.summary import Summary


class SummaryRepository:
    """Handle summary persistence operations."""

    @staticmethod
    async def create(
        session: AsyncSession,
        document_id: int,
        content: str,
        page_citations: list[int],
    ) -> Summary:
        """Create a new summary record for a document."""
        summary = Summary(
            document_id=document_id,
            content=content,
            page_citations=json.dumps(page_citations),
        )
        session.add(summary)
        await session.flush()
        await session.refresh(summary)
        return summary

    @staticmethod
    async def get_by_document_id(
        session: AsyncSession,
        document_id: int,
    ) -> Summary | None:
        """Retrieve the most recent summary for a document."""
        result = await session.execute(
            select(Summary)
            .where(Summary.document_id == document_id)
            .order_by(Summary.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
