"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the API."""

    database_url: str
    document_storage_path: Path = Path("/data/uploads")
    max_upload_size_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    chunk_size_tokens: int = Field(default=500, gt=0)
    chunk_overlap_tokens: int = Field(default=75, ge=0)
    ollama_url: str = "http://ollama:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = Field(default=768, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0, le=256)
    ollama_timeout_seconds: float = Field(default=30.0, gt=0)
    search_top_k: int = Field(default=5, gt=0, le=100)
    search_similarity_threshold: float = Field(default=0.5, ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> Self:
        """Ensure overlap cannot prevent the chunker from progressing."""
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()  # type: ignore[call-arg]
