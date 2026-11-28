"""FastAPI dependency providers for application services."""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from garden_center.ai.embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from garden_center.ai.llm import (
    AnthropicLLMProvider,
    LLMProvider,
    OllamaLLMProvider,
)
from garden_center.ai.types import LLMProviderName
from garden_center.core.config import Settings, get_settings
from garden_center.database.engine import get_database_session
from garden_center.storage.file_storage import FileStorage
from garden_center.storage.local_file_storage import LocalFileStorage


def get_file_storage(
    settings: Annotated[Settings, Depends(get_settings)],
) -> FileStorage:
    """Build local file storage from application settings."""
    return LocalFileStorage(settings.document_storage_path)


async def get_embedding_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncGenerator[EmbeddingProvider]:
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
) -> AsyncGenerator[LLMProvider]:
    """Build the configured answer-generation provider for a request."""
    provider: LLMProvider
    match settings.llm_provider:
        case LLMProviderName.ANTHROPIC:
            provider = AnthropicLLMProvider(
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_model,
                timeout_seconds=settings.llm_timeout_seconds,
                max_tokens=settings.llm_max_tokens,
                max_concurrent_generations=(
                    settings.anthropic_llm_max_concurrent_generations
                ),
            )
        case LLMProviderName.OLLAMA:
            provider = OllamaLLMProvider(
                base_url=settings.ollama_url,
                model=settings.ollama_llm_model,
                timeout_seconds=settings.llm_timeout_seconds,
                max_tokens=settings.llm_max_tokens,
                max_concurrent_generations=(
                    settings.ollama_llm_max_concurrent_generations
                ),
            )
        case unsupported_provider:
            raise ValueError(f"Unsupported LLM provider: {unsupported_provider}")
    try:
        yield provider
    finally:
        await provider.aclose()


SessionDependency = Annotated[AsyncSession, Depends(get_database_session)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]
EmbeddingDependency = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]
LLMDependency = Annotated[LLMProvider, Depends(get_llm_provider)]
StorageDependency = Annotated[FileStorage, Depends(get_file_storage)]
