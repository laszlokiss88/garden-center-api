"""FastAPI dependency providers for application services."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends

from garden_center.config import Settings, get_settings
from garden_center.embeddings import EmbeddingProvider, OllamaEmbeddingProvider


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
