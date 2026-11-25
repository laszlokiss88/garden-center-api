"""Embedding provider abstraction and Ollama implementation."""

import logging
import math
from collections.abc import Sequence
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class EmbeddingProviderError(RuntimeError):
    """Base error for embedding provider failures safe to expose to clients."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class EmbeddingProvider(Protocol):
    """Provider-independent interface used by indexing and search services."""

    provider_name: str
    model: str
    dimensions: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding for each supplied text."""
        ...


class OllamaEmbeddingProvider:
    """Generate embeddings through Ollama's batch embedding endpoint."""

    provider_name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        dimensions: int = 768,
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=transport,
        )

    async def aclose(self) -> None:
        """Release connections held by the provider."""
        await self._client.aclose()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate and validate a batch of Ollama embeddings."""
        if not texts:
            return []

        try:
            response = await self._client.post(
                "/api/embed",
                json={"model": self.model, "input": list(texts)},
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning("Ollama embedding request failed: %s", exc)
            raise EmbeddingProviderError(
                "ollama_unavailable",
                "Ollama is unavailable or the embedding request timed out.",
            ) from exc
        except httpx.HTTPError as exc:
            logger.warning("Ollama embedding HTTP error: %s", exc)
            raise EmbeddingProviderError(
                "ollama_request_failed", "The Ollama embedding request failed."
            ) from exc

        if response.status_code == 404:
            raise EmbeddingProviderError(
                "embedding_model_missing",
                f"Ollama model '{self.model}' is not installed.",
            )
        if response.is_error:
            logger.warning(
                "Ollama returned HTTP %d: %s", response.status_code, response.text
            )
            raise EmbeddingProviderError(
                "ollama_request_failed",
                f"Ollama returned HTTP {response.status_code}.",
            )

        try:
            payload = response.json()
            embeddings = payload["embeddings"]
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise ValueError("unexpected embedding count")
            validated = [self._validate_vector(vector) for vector in embeddings]
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Ollama returned an invalid embedding response")
            raise EmbeddingProviderError(
                "invalid_embedding_response",
                "Ollama returned an invalid embedding response.",
            ) from exc
        return validated

    def _validate_vector(self, vector: object) -> list[float]:
        if not isinstance(vector, list) or len(vector) != self.dimensions:
            raise ValueError("unexpected embedding dimensions")
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("embedding contains non-finite values")
        return values
