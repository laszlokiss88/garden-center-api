"""FastAPI dependency providers for application services."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends

from garden_center.config import Settings, get_settings
from garden_center.embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from garden_center.file_storage import FileStorage, LocalFileStorage
from garden_center.llm import AnthropicLLMProvider, LLMProvider, OllamaLLMProvider


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


async def get_llm_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[LLMProvider]:
    """Build the configured answer-generation provider for a request."""
    if settings.llm_provider == "anthropic":
        provider: AnthropicLLMProvider | OllamaLLMProvider = AnthropicLLMProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_tokens=settings.llm_max_tokens,
            max_concurrent_generations=(
                settings.anthropic_llm_max_concurrent_generations
            ),
        )
    else:
        provider = OllamaLLMProvider(
            base_url=settings.ollama_url,
            model=settings.ollama_llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_tokens=settings.llm_max_tokens,
            max_concurrent_generations=settings.ollama_llm_max_concurrent_generations,
        )
    try:
        yield provider
    finally:
        await provider.aclose()
