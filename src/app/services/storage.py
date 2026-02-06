"""File storage service for PDF uploads."""

import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from app.config import settings
from app.exceptions import DomainValidationError

ALLOWED_CONTENT_TYPES = {"application/pdf"}
ALLOWED_EXTENSIONS = {".pdf"}


class FileStorageService:
    """Handle PDF file storage operations."""

    def __init__(self, upload_dir: str | None = None) -> None:
        self.upload_dir = Path(upload_dir or settings.upload_dir)

    async def save_pdf(self, file: UploadFile) -> tuple[str, int]:
        """
        Save uploaded PDF file to storage.

        Returns:
            Tuple of (file_path, file_size_bytes)

        Raises:
            DomainValidationError: If file type or size is invalid
        """
        self._validate_file_type(file)

        self.upload_dir.mkdir(parents=True, exist_ok=True)

        unique_filename = self._generate_unique_filename(file.filename or "upload.pdf")
        file_path = self.upload_dir / unique_filename

        file_size = await self._write_file(file, file_path)
        self._validate_file_size(file_size)

        return str(file_path), file_size

    def delete_pdf(self, file_path: str) -> None:
        """Delete a PDF file from storage."""
        path = Path(file_path)
        if path.exists():
            path.unlink()

    def _validate_file_type(self, file: UploadFile) -> None:
        """Validate that the uploaded file is a PDF."""
        content_type = file.content_type or ""
        filename = file.filename or ""
        extension = Path(filename).suffix.lower()

        if (
            content_type not in ALLOWED_CONTENT_TYPES
            and extension not in ALLOWED_EXTENSIONS
        ):
            raise DomainValidationError(
                message="Only PDF files are allowed",
                field="file",
                details={"content_type": content_type, "extension": extension},
            )

    def _validate_file_size(self, size_bytes: int) -> None:
        """Validate file size is within limits."""
        if size_bytes > settings.max_upload_size_bytes:
            raise DomainValidationError(
                message=f"File size exceeds maximum of {settings.max_upload_size_mb}MB",
                field="file",
                details={
                    "size_bytes": size_bytes,
                    "max_bytes": settings.max_upload_size_bytes,
                },
            )

    def _generate_unique_filename(self, original_filename: str) -> str:
        """Generate a unique filename preserving original extension."""
        extension = Path(original_filename).suffix.lower() or ".pdf"
        return f"{uuid.uuid4()}{extension}"

    async def _write_file(self, file: UploadFile, file_path: Path) -> int:
        """Write uploaded file to disk and return size."""
        total_size = 0
        async with aiofiles.open(file_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks at a time
                await out_file.write(chunk)
                total_size += len(chunk)
        return total_size
