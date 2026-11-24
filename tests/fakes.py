"""Small deterministic test doubles shared by embedding integration tests."""

import uuid
from collections.abc import Sequence

from garden_center.embeddings import EmbeddingProviderError


class FakeEmbeddingProvider:
    """Deterministic three-dimensional embedding provider."""

    provider_name = "test"
    model = f"test-model-{uuid.uuid4()}"
    dimensions = 3

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "rose" in text.lower():
                vectors.append([1.0, 0.0, 0.0])
            elif "soil" in text.lower():
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class FailingEmbeddingProvider:
    """Provider that simulates an unavailable Ollama service."""

    provider_name = "ollama"
    model = "nomic-embed-text"
    dimensions = 768

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingProviderError(
            "ollama_unavailable",
            "Ollama is unavailable or the embedding request timed out.",
        )
