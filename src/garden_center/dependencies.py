"""FastAPI dependency providers for application services."""

from typing import Annotated

from fastapi import Depends

from garden_center.config import Settings, get_settings
from garden_center.embeddings import EmbeddingProvider, OllamaEmbeddingProvider


def get_embedding_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingProvider:
    """Build the configured embedding provider for a request."""
    return OllamaEmbeddingProvider(
        base_url=settings.ollama_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
