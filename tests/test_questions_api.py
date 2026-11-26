"""Focused tests for the LangGraph question-answering endpoint."""

import asyncio
import uuid
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient, Response

from garden_center.dependencies import get_embedding_provider, get_llm_provider
from garden_center.main import app
from garden_center.search_service import SearchResult
from tests.fakes import FailingLLMProvider, FakeEmbeddingProvider, FakeLLMProvider


def _post_question(llm_provider: object, results: list[SearchResult]) -> Response:
    async def scenario() -> Response:
        async def fake_search(*_args: object, **_kwargs: object) -> list[SearchResult]:
            return results

        app.dependency_overrides[get_embedding_provider] = FakeEmbeddingProvider
        app.dependency_overrides[get_llm_provider] = lambda: llm_provider
        try:
            with patch("garden_center.rag_service.semantic_search", fake_search):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://testserver"
                ) as client:
                    return await client.post(
                        "/api/v1/questions",
                        json={"question": "How should I care for roses?"},
                    )
        finally:
            app.dependency_overrides.pop(get_embedding_provider, None)
            app.dependency_overrides.pop(get_llm_provider, None)

    return asyncio.run(scenario())


def _source() -> SearchResult:
    return SearchResult(
        document_id=uuid.uuid4(),
        filename="rose-guide.pdf",
        page_number=3,
        chunk_index=2,
        text="Roses need regular watering.",
        similarity=0.92,
    )


def test_question_returns_grounded_answer_and_sources() -> None:
    source = _source()

    response = _post_question(FakeLLMProvider(), [source])

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Roses need regular watering.",
        "provider": "test",
        "model": "test-generation-model",
        "sources": [
            {
                "document_id": str(source.document_id),
                "filename": "rose-guide.pdf",
                "page_number": 3,
                "chunk_index": 2,
                "text": "Roses need regular watering.",
                "similarity": 0.92,
            }
        ],
    }


def test_question_without_relevant_documents_skips_generation() -> None:
    provider = FakeLLMProvider()

    response = _post_question(provider, [])

    assert response.status_code == 200
    assert response.json()["answer"].startswith("The available documents")
    assert response.json()["sources"] == []
    assert response.json()["provider"] is None
    assert provider.calls == 0


@pytest.mark.parametrize(
    ("code", "message", "status_code"),
    [
        ("generation_timeout", "Answer generation timed out.", 504),
        ("generation_provider_unavailable", "Provider unavailable.", 503),
        ("invalid_generation_response", "Invalid provider response.", 502),
    ],
)
def test_question_provider_failures_return_appropriate_errors(
    code: str, message: str, status_code: int
) -> None:
    response = _post_question(FailingLLMProvider(code, message), [_source()])

    assert response.status_code == status_code
    assert response.json()["detail"] == {
        "code": code,
        "message": message,
    }
