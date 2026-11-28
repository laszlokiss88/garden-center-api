"""Focused tests for process-local endpoint rate limits."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from garden_center.api.dependencies import get_embedding_provider, get_llm_provider
from garden_center.api.rate_limiting import limiter
from garden_center.main import app
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider


@pytest.fixture(autouse=True)
def reset_rate_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter.reset()
    settings = SimpleNamespace(
        questions_rate_limit_per_minute=1,
        document_upload_rate_limit_per_minute=1,
    )
    monkeypatch.setattr(
        "garden_center.api.rate_limiting.get_settings", lambda: settings
    )


def test_question_rate_limit_returns_429() -> None:
    async def scenario() -> tuple[int, dict[str, object]]:
        async def no_results(*_: object, **__: object) -> list[object]:
            return []

        app.dependency_overrides[get_embedding_provider] = FakeEmbeddingProvider
        app.dependency_overrides[get_llm_provider] = FakeLLMProvider
        try:
            with patch("garden_center.rag.service.semantic_search", no_results):
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://testserver",
                ) as client:
                    first = await client.post(
                        "/api/v1/questions", json={"question": "Rose care?"}
                    )
                    second = await client.post(
                        "/api/v1/questions",
                        json={"question": "Soil care?"},
                        headers={"X-Forwarded-For": "203.0.113.10"},
                    )
        finally:
            app.dependency_overrides.pop(get_embedding_provider, None)
            app.dependency_overrides.pop(get_llm_provider, None)
        assert first.status_code == 200
        return second.status_code, second.json()

    status_code, body = asyncio.run(scenario())

    assert status_code == 429
    assert body["detail"]["code"] == "rate_limit_exceeded"
