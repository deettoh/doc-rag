"""Tests for PDF extraction service."""

import tempfile
from pathlib import Path

import fitz
import pytest

from app.exceptions import DomainValidationError
from app.services.pdf_extractor import (
    PDFExtractorService,
    normalize_whitespace,
    remove_hyphenation,
    trim_headers_footers,
)


@pytest.fixture
def pdf_extractor() -> PDFExtractorService:
    """Create PDF extractor service instance."""
    return PDFExtractorService()


@pytest.fixture
def sample_pdf_path() -> str:
    """Create a sample PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello World!\n\nThis is a test document.")
        doc.save(f.name)
        doc.close()
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def multi_page_pdf_path() -> str:
    """Create a multi-page PDF for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        doc = fitz.open()

        page1 = doc.new_page()
        page1.insert_text((72, 72), "Page 1 content\nFirst page of the document.")

        page2 = doc.new_page()
        page2.insert_text((72, 72), "Page 2 content\nSecond page of the document.")

        page3 = doc.new_page()
        page3.insert_text((72, 72), "Page 3 content\nThird page of the document.")

        doc.save(f.name)
        doc.close()
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def hyphenated_pdf_path() -> str:
    """Create a PDF with hyphenated words across lines."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (72, 72), "This is a docu-\nment with hyphen-\nation across lines."
        )
        doc.save(f.name)
        doc.close()
        yield f.name
    Path(f.name).unlink(missing_ok=True)


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""

    def test_collapse_multiple_spaces(self) -> None:
        """Multiple spaces should collapse to single space."""
        text = "Hello    World"
        result = normalize_whitespace(text)
        assert result == "Hello World"

    def test_collapse_multiple_newlines(self) -> None:
        """Multiple newlines should collapse to single paragraph break."""
        text = "Para 1\n\n\n\n\nPara 2"
        result = normalize_whitespace(text)
        assert result == "Para 1\n\nPara 2"

    def test_strip_line_whitespace(self) -> None:
        """Leading/trailing whitespace per line should be stripped."""
        text = "  Line 1  \n  Line 2  "
        result = normalize_whitespace(text)
        assert result == "Line 1\nLine 2"

    def test_replace_tabs(self) -> None:
        """Tabs should be replaced with spaces."""
        text = "Hello\tWorld"
        result = normalize_whitespace(text)
        assert result == "Hello World"


class TestRemoveHyphenation:
    """Tests for de-hyphenation logic."""

    def test_join_hyphenated_words(self) -> None:
        """Hyphenated words at line ends should be joined."""
        text = "docu-\nment"
        result = remove_hyphenation(text)
        assert result == "document"

    def test_preserve_real_hyphens(self) -> None:
        """Real hyphens (not at line end) should be preserved."""
        text = "self-contained text"
        result = remove_hyphenation(text)
        assert result == "self-contained text"

    def test_multiple_hyphenations(self) -> None:
        """Multiple hyphenations in same text."""
        text = "This docu-\nment has multi-\nple examples."
        result = remove_hyphenation(text)
        assert result == "This document has multiple examples."


class TestTrimHeadersFooters:
    """Tests for header/footer trimming."""

    def test_remove_page_number_only(self) -> None:
        """Just the page number should be removed."""
        text = "1\nActual content here\n1"
        result = trim_headers_footers(text, page_num=1, total_pages=10)
        assert "Actual content here" in result
        assert result.strip() == "Actual content here"

    def test_remove_page_x_format(self) -> None:
        """'Page N' format should be removed."""
        text = "Page 1\nActual content"
        result = trim_headers_footers(text, page_num=1, total_pages=10)
        assert result.strip() == "Actual content"

    def test_remove_page_x_of_y_format(self) -> None:
        """'Page N of M' format should be removed."""
        text = "Page 1 of 10\nActual content"
        result = trim_headers_footers(text, page_num=1, total_pages=10)
        assert result.strip() == "Actual content"

    def test_remove_separator_lines(self) -> None:
        """Separator lines should be removed."""
        text = "----------------\nActual content\n----------------"
        result = trim_headers_footers(text, page_num=1, total_pages=10)
        assert result.strip() == "Actual content"

    def test_preserve_middle_content(self) -> None:
        """Content in middle of page should be preserved."""
        text = "Line 1\nLine 2\nLine 3\nPage 5\nLine 5\nLine 6\nLine 7"
        result = trim_headers_footers(text, page_num=5, total_pages=10)
        # Page 5 is in middle, should be preserved
        assert "Page 5" in result


class TestPDFExtractorService:
    """Tests for PDFExtractorService class."""

    def test_extract_text_returns_pages(
        self, pdf_extractor: PDFExtractorService, sample_pdf_path: str
    ) -> None:
        """Extract text should return list of PageContent."""
        pages = pdf_extractor.extract_text(sample_pdf_path)

        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert "Hello World" in pages[0].text

    def test_extract_text_multi_page(
        self, pdf_extractor: PDFExtractorService, multi_page_pdf_path: str
    ) -> None:
        """Multi-page PDF should return content for each page."""
        pages = pdf_extractor.extract_text(multi_page_pdf_path)

        assert len(pages) == 3
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2
        assert pages[2].page_number == 3
        assert "Page 1 content" in pages[0].text
        assert "Page 2 content" in pages[1].text
        assert "Page 3 content" in pages[2].text

    def test_extract_full_text(
        self, pdf_extractor: PDFExtractorService, multi_page_pdf_path: str
    ) -> None:
        """Extract full text combines all pages."""
        full_text = pdf_extractor.extract_full_text(multi_page_pdf_path)

        assert "Page 1 content" in full_text
        assert "Page 2 content" in full_text
        assert "Page 3 content" in full_text

    def test_get_page_count(
        self, pdf_extractor: PDFExtractorService, multi_page_pdf_path: str
    ) -> None:
        """Get page count returns correct number."""
        count = pdf_extractor.get_page_count(multi_page_pdf_path)
        assert count == 3

    def test_extract_text_file_not_found(
        self, pdf_extractor: PDFExtractorService
    ) -> None:
        """Non-existent file should raise DomainValidationError."""
        with pytest.raises(DomainValidationError) as exc_info:
            pdf_extractor.extract_text("/nonexistent/path.pdf")

        assert "not found" in exc_info.value.message.lower()

    def test_extract_text_invalid_pdf(self, pdf_extractor: PDFExtractorService) -> None:
        """Invalid PDF should raise DomainValidationError."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"not a pdf")
            f.flush()

            with pytest.raises(DomainValidationError) as exc_info:
                pdf_extractor.extract_text(f.name)

            assert "Failed to open PDF" in exc_info.value.message

            Path(f.name).unlink(missing_ok=True)

    def test_get_page_count_file_not_found(
        self, pdf_extractor: PDFExtractorService
    ) -> None:
        """Get page count with non-existent file should raise error."""
        with pytest.raises(DomainValidationError):
            pdf_extractor.get_page_count("/nonexistent/path.pdf")
