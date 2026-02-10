"""PDF text extraction and cleanup service using PyMuPDF."""

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from app.exceptions import DomainValidationError


@dataclass
class PageContent:
    """Represents extracted text content from a single PDF page."""

    page_number: int  # 1-indexed
    text: str


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in extracted text.

    - Collapse multiple spaces to single space
    - Collapse multiple newlines to double newline (paragraph break)
    - Strip leading/trailing whitespace from lines
    """
    text = text.replace("\t", " ")
    text = re.sub(r" +", " ", text)

    lines = [line.strip() for line in text.split("\n")]

    result_lines: list[str] = []
    empty_count = 0
    for line in lines:
        if line == "":
            empty_count += 1
            if empty_count <= 1:
                result_lines.append(line)
        else:
            empty_count = 0
            result_lines.append(line)

    return "\n".join(result_lines).strip()


def remove_hyphenation(text: str) -> str:
    """
    Join words that are hyphenated across line breaks.

    Handles patterns like "docu-\nment" -> "document"
    """
    pattern = r"(\w)-\n(\w)"
    return re.sub(pattern, r"\1\2", text)


def trim_headers_footers(text: str, page_num: int, total_pages: int) -> str:
    """
    Remove common header/footer patterns using simple heuristics.

    Removes:
    - Page numbers at start/end of text (e.g., "Page 1", "1", "1 of 10")
    - Common header/footer separators (lines of dashes, etc.)
    """
    lines = text.split("\n")

    # Guard clause
    if not lines:
        return text

    page_patterns = [
        rf"^\s*{page_num}\s*$",
        rf"^\s*Page\s+{page_num}\s*$",
        rf"^\s*-\s*{page_num}\s*-\s*$",
        rf"^\s*{page_num}\s+of\s+{total_pages}\s*$",
        rf"^\s*Page\s+{page_num}\s+of\s+{total_pages}\s*$",
        r"^\s*[-_=]{10,}\s*$",  # Separator lines (10+ chars)
    ]

    def is_header_footer(line: str) -> bool:
        return any(re.match(p, line, re.IGNORECASE) for p in page_patterns)

    # Only check first few and last few lines for header/footer patterns
    filtered_lines: list[str] = []
    for i, line in enumerate(lines):
        if (i < 3 or i >= len(lines) - 3) and is_header_footer(line):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


class PDFExtractorService:
    """Extract and clean text content from PDF files."""

    def extract_text(self, file_path: str) -> list[PageContent]:
        """
        Extract text from all pages of a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of PageContent objects with cleaned text per page

        Raises:
            DomainValidationError: If file doesn't exist or is not a valid PDF
        """
        path = Path(file_path)
        if not path.exists():
            raise DomainValidationError(
                message=f"PDF file not found: {file_path}",
                field="file_path",
            )

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise DomainValidationError(
                message=f"Failed to open PDF: {e!s}",
                field="file_path",
                details={"error": str(e)},
            ) from e

        try:
            total_pages = len(doc)
            pages: list[PageContent] = []

            for page_num in range(total_pages):
                page = doc[page_num]
                raw_text = page.get_text()

                cleaned_text = self._clean_text(
                    raw_text,
                    page_num=page_num + 1,
                    total_pages=total_pages,
                )

                pages.append(
                    PageContent(
                        page_number=page_num + 1,
                        text=cleaned_text,
                    )
                )

            return pages
        finally:
            doc.close()

    def extract_full_text(self, file_path: str) -> str:
        """
        Extract and combine text from all pages.

        Args:
            file_path: Path to the PDF file

        Returns:
            Combined cleaned text from all pages
        """
        pages = self.extract_text(file_path)
        return "\n\n".join(page.text for page in pages if page.text.strip())

    def get_page_count(self, file_path: str) -> int:
        """
        Get the number of pages in a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Number of pages
        """
        path = Path(file_path)
        if not path.exists():
            raise DomainValidationError(
                message=f"PDF file not found: {file_path}",
                field="file_path",
            )

        try:
            doc = fitz.open(file_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            raise DomainValidationError(
                message=f"Failed to open PDF: {e!s}",
                field="file_path",
                details={"error": str(e)},
            ) from e

    def _clean_text(self, text: str, page_num: int, total_pages: int) -> str:
        """Apply the full text cleanup pipeline."""
        text = remove_hyphenation(text)
        text = normalize_whitespace(text)
        text = trim_headers_footers(text, page_num, total_pages)
        return text
