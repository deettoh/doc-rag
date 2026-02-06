"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str = ""

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/docrag"

    # Application Settings
    max_upload_size_mb: int = 10
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    max_llm_context_size: int = 4000
    request_timeout_seconds: int = 30
    job_timeout_seconds: int = 300

    # Storage Settings
    upload_dir: str = "/app/storage/pdfs"

    # Environment Settings
    environment: str = "development"
    debug: bool = True

    @property
    def max_upload_size_bytes(self) -> int:
        """Return max upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
