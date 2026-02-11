"""Tests for chunking service."""

import pytest

from app.services.chunking import ChunkData, ChunkingService
from app.services.pdf_extractor import PageContent


@pytest.fixture
def chunking_service() -> ChunkingService:
    """Create chunking service with small sizes for testing."""
    return ChunkingService(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def sample_pages() -> list[PageContent]:
    """Create sample multi-page content for chunking tests."""
    return [
        PageContent(page_number=1, text="This is the first page. " * 10),
        PageContent(page_number=2, text="This is the second page. " * 10),
        PageContent(page_number=3, text="This is the third page. " * 10),
    ]


@pytest.fixture
def short_page() -> list[PageContent]:
    """Create a single short page that fits in one chunk."""
    return [PageContent(page_number=1, text="Short text.")]


class TestChunkingService:
    """Tests for ChunkingService."""

    def test_chunks_are_created(
        self,
        chunking_service: ChunkingService,
        sample_pages: list[PageContent],
    ) -> None:
        """Chunking should produce at least one chunk."""
        chunks = chunking_service.chunk_pages(sample_pages, document_id=1)

        assert len(chunks) > 0
        assert all(isinstance(c, ChunkData) for c in chunks)

    def test_chunk_index_sequential(
        self,
        chunking_service: ChunkingService,
        sample_pages: list[PageContent],
    ) -> None:
        """Chunk indices should be sequential starting from 0."""
        chunks = chunking_service.chunk_pages(sample_pages, document_id=1)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_document_id_preserved(
        self,
        chunking_service: ChunkingService,
        sample_pages: list[PageContent],
    ) -> None:
        """Document ID should be set on all chunks."""
        chunks = chunking_service.chunk_pages(sample_pages, document_id=42)

        assert all(c.document_id == 42 for c in chunks)

    def test_page_start_end_valid(
        self,
        chunking_service: ChunkingService,
        sample_pages: list[PageContent],
    ) -> None:
        """Page start should be <= page end, and within valid range."""
        chunks = chunking_service.chunk_pages(sample_pages, document_id=1)

        for chunk in chunks:
            assert chunk.page_start >= 1
            assert chunk.page_end <= 3
            assert chunk.page_start <= chunk.page_end

    def test_char_offsets_valid(
        self,
        chunking_service: ChunkingService,
        sample_pages: list[PageContent],
    ) -> None:
        """Character offsets should be non-negative and start < end."""
        chunks = chunking_service.chunk_pages(sample_pages, document_id=1)

        for chunk in chunks:
            assert chunk.char_start >= 0
            assert chunk.char_end > chunk.char_start

    def test_char_offsets_match_content(
        self,
        chunking_service: ChunkingService,
        sample_pages: list[PageContent],
    ) -> None:
        """Char offsets should correctly map to content in the full text."""
        chunks = chunking_service.chunk_pages(sample_pages, document_id=1)
        full_text = "\n\n".join(p.text for p in sample_pages)

        for chunk in chunks:
            extracted = full_text[chunk.char_start : chunk.char_end]
            assert extracted == chunk.content

    def test_deterministic_output(
        self,
        chunking_service: ChunkingService,
        sample_pages: list[PageContent],
    ) -> None:
        """Same input should produce identical output."""
        chunks_a = chunking_service.chunk_pages(sample_pages, document_id=1)
        chunks_b = chunking_service.chunk_pages(sample_pages, document_id=1)

        assert len(chunks_a) == len(chunks_b)
        for a, b in zip(chunks_a, chunks_b, strict=True):
            assert a.content == b.content
            assert a.char_start == b.char_start
            assert a.char_end == b.char_end
            assert a.page_start == b.page_start
            assert a.page_end == b.page_end

    def test_different_chunk_size_changes_count(
        self, sample_pages: list[PageContent]
    ) -> None:
        """Smaller chunk size should produce more chunks."""
        small = ChunkingService(chunk_size=50, chunk_overlap=10)
        large = ChunkingService(chunk_size=200, chunk_overlap=20)

        small_chunks = small.chunk_pages(sample_pages, document_id=1)
        large_chunks = large.chunk_pages(sample_pages, document_id=1)

        assert len(small_chunks) > len(large_chunks)

    def test_single_short_page(
        self,
        chunking_service: ChunkingService,
        short_page: list[PageContent],
    ) -> None:
        """Short text should produce exactly one chunk."""
        chunks = chunking_service.chunk_pages(short_page, document_id=1)

        assert len(chunks) == 1
        assert chunks[0].content == "Short text."
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 1

    def test_empty_pages_returns_empty(self, chunking_service: ChunkingService) -> None:
        """Empty page list should return no chunks."""
        chunks = chunking_service.chunk_pages([], document_id=1)
        assert chunks == []

    def test_blank_pages_returns_empty(self, chunking_service: ChunkingService) -> None:
        """Pages with only whitespace should return no chunks."""
        pages = [PageContent(page_number=1, text="   \n\n  ")]
        chunks = chunking_service.chunk_pages(pages, document_id=1)
        assert chunks == []

    def test_multi_page_chunk_spans_pages(self) -> None:
        """A chunk crossing page boundaries should have different page_start/end."""
        # chunk_size=500 is larger than page size (200 chars)
        service = ChunkingService(chunk_size=500, chunk_overlap=50)
        pages = [
            PageContent(page_number=1, text="A" * 200),
            PageContent(page_number=2, text="B" * 200),
        ]
        chunks = service.chunk_pages(pages, document_id=1)

        spanning = [c for c in chunks if c.page_start != c.page_end]
        assert len(spanning) > 0, "At least one chunk should span two pages"
