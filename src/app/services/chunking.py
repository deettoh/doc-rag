"""Text chunking service using LangChain RecursiveCharacterTextSplitter."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services.pdf_extractor import PageContent


@dataclass
class ChunkData:
    """Intermediate representation of a chunk before DB persistence."""

    document_id: int
    chunk_index: int
    content: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int


class ChunkingService:
    """Split extracted PDF text into chunks with page-aware metadata."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_pages(
        self,
        pages: list[PageContent],
        document_id: int,
    ) -> list[ChunkData]:
        """
        Split page-wise text into overlapping chunks with metadata.

        Args:
            pages: List of PageContent from PDFExtractorService
            document_id: ID of the parent document

        Returns:
            List of ChunkData with page ranges and character offsets
        """
        if not pages:
            return []

        page_boundaries = self._build_page_boundaries(pages)
        full_text = self._join_pages(pages)

        if not full_text.strip():
            return []

        raw_chunks = self.splitter.split_text(full_text)

        chunks: list[ChunkData] = []
        search_start = 0

        for chunk_index, chunk_text in enumerate(raw_chunks):
            char_start = full_text.find(chunk_text, search_start)
            if char_start == -1:
                char_start = full_text.find(chunk_text)
            char_end = char_start + len(chunk_text)

            page_start = self._resolve_page(char_start, page_boundaries)
            page_end = self._resolve_page(char_end - 1, page_boundaries)

            chunks.append(
                ChunkData(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=chunk_text,
                    page_start=page_start,
                    page_end=page_end,
                    char_start=char_start,
                    char_end=char_end,
                )
            )

            # Advance the search to allow overlap but avoid false matches
            search_start = char_start + 1

        return chunks

    def _build_page_boundaries(
        self, pages: list[PageContent]
    ) -> list[tuple[int, int, int]]:
        """
        Build a list of (page_number, char_start, char_end) tuples.

        Uses the same joining strategy as _join_pages so offsets align.
        """
        boundaries: list[tuple[int, int, int]] = []
        offset = 0
        separator = "\n\n"

        for i, page in enumerate(pages):
            start = offset
            end = offset + len(page.text)
            boundaries.append((page.page_number, start, end))
            offset = end
            if i < len(pages) - 1:
                offset += len(separator)

        return boundaries

    def _join_pages(self, pages: list[PageContent]) -> str:
        """Join page texts with double newline separator."""
        return "\n\n".join(page.text for page in pages)

    def _resolve_page(
        self,
        char_offset: int,
        boundaries: list[tuple[int, int, int]],
    ) -> int:
        """Resolve which page a character offset belongs to."""
        for page_num, start, end in boundaries:
            if start <= char_offset <= end:
                return page_num

        # If offset falls in separator between pages, return the next page
        for i, (_page_num, _start, end) in enumerate(boundaries[:-1]):
            next_start = boundaries[i + 1][1]
            if end < char_offset < next_start:
                return boundaries[i + 1][0]

        # Fallback: return last page
        return boundaries[-1][0]
