"""FastAPI dependency providers for application services."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends

from garden_center.config import Settings, get_settings
from garden_center.embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from garden_center.file_storage import FileStorage, LocalFileStorage


def get_file_storage(
    settings: Annotated[Settings, Depends(get_settings)],
) -> FileStorage:
    """Build local file storage from application settings."""
    return LocalFileStorage(settings.document_storage_path)


async def get_embedding_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[EmbeddingProvider]:
    """Build the configured embedding provider for a request."""
    provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    try:
        yield provider
    finally:
        await provider.aclose()
