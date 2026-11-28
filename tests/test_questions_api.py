"""Focused tests for the LangGraph question-answering endpoint."""

import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import replace
from unittest.mock import patch

import pytest
import tiktoken
from httpx import ASGITransport, AsyncClient, Response
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

import garden_center.rag.service as rag_service
from garden_center.ai.llm import LLMGenerationResult
from garden_center.api.dependencies import (
    get_embedding_provider,
    get_llm_provider,
)
from garden_center.main import app
from garden_center.rag.search import SearchResult
from garden_center.rag.service import _select_context_results
from tests.fakes import FailingLLMProvider, FakeEmbeddingProvider, FakeLLMProvider


@pytest.fixture
def trace_exporter() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(
        tracer_provider=provider,
        config=TraceConfig(
            hide_inputs=True,
            hide_outputs=True,
            hide_llm_invocation_parameters=True,
        ),
    )
    yield exporter
    instrumentor.uninstrument()
    provider.shutdown()


def _post_question(llm_provider: object, results: list[SearchResult]) -> Response:
    async def scenario() -> Response:
        async def fake_search(*_args: object, **_kwargs: object) -> list[SearchResult]:
            return results

        app.dependency_overrides[get_embedding_provider] = FakeEmbeddingProvider
        app.dependency_overrides[get_llm_provider] = lambda: llm_provider
        try:
            with patch("garden_center.rag.service.semantic_search", fake_search):
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
        token_count=6,
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


def test_invalid_question_uses_standard_error_response() -> None:
    async def scenario() -> Response:
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/api/v1/questions",
                json={"question": "   "},
            )

    response = asyncio.run(scenario())

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "invalid_request",
            "message": "The request contains invalid or missing data.",
        }
    }


def test_context_budget_excludes_oversized_chunks() -> None:
    encoding = tiktoken.get_encoding("cl100k_base")
    lower_text = "soil " * 100
    higher_text = "rose " * 100
    lower_similarity = replace(
        _source(),
        text=lower_text,
        token_count=len(encoding.encode(lower_text)),
        similarity=0.7,
    )
    higher_similarity = replace(
        _source(),
        text=higher_text,
        token_count=len(encoding.encode(higher_text)),
        similarity=0.95,
    )

    selection = _select_context_results(
        "How should I care for roses?",
        [lower_similarity, higher_similarity],
        token_budget=150,
        output_token_allowance=20,
    )

    selected = selection["results"]
    assert selected == []
    assert selection["truncated"] is False


def test_shared_graph_keeps_concurrent_runtime_contexts_independent() -> None:
    async def scenario() -> tuple[
        list[tuple[object, object, str, int, float]],
        FakeLLMProvider,
        FakeLLMProvider,
        int,
    ]:
        calls: list[tuple[object, object, str, int, float]] = []
        both_started = asyncio.Event()
        started_count = 0

        async def fake_search(
            session: object,
            embedding_provider: object,
            question: str,
            top_k: int,
            similarity_threshold: float,
        ) -> list[SearchResult]:
            nonlocal started_count
            calls.append(
                (
                    session,
                    embedding_provider,
                    question,
                    top_k,
                    similarity_threshold,
                )
            )
            started_count += 1
            if started_count == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)
            return [_source()]

        session_a = object()
        session_b = object()
        embedding_a = FakeEmbeddingProvider()
        embedding_b = FakeEmbeddingProvider()
        llm_a = FakeLLMProvider()
        llm_b = FakeLLMProvider()
        graph_id = id(rag_service._RAG_GRAPH)

        with patch("garden_center.rag.service.semantic_search", fake_search):
            await asyncio.gather(
                rag_service.answer_question(
                    session_a,  # type: ignore[arg-type]
                    embedding_a,
                    llm_a,
                    "question-a",
                    3,
                    0.3,
                    4096,
                    512,
                ),
                rag_service.answer_question(
                    session_b,  # type: ignore[arg-type]
                    embedding_b,
                    llm_b,
                    "question-b",
                    7,
                    0.7,
                    2048,
                    256,
                ),
            )

        assert id(rag_service._RAG_GRAPH) == graph_id
        assert next(call for call in calls if call[0] is session_a) == (
            session_a,
            embedding_a,
            "question-a",
            3,
            0.3,
        )
        assert next(call for call in calls if call[0] is session_b) == (
            session_b,
            embedding_b,
            "question-b",
            7,
            0.7,
        )
        return calls, llm_a, llm_b, graph_id

    calls, llm_a, llm_b, _ = asyncio.run(scenario())

    assert len(calls) == 2
    assert llm_a.calls == 1
    assert llm_b.calls == 1


@pytest.mark.parametrize(
    ("code", "message", "status_code"),
    [
        ("generation_timeout", "Answer generation timed out.", 504),
        ("generation_provider_unavailable", "Provider unavailable.", 503),
        ("invalid_generation_response", "Invalid provider response.", 502),
    ],
)
def test_question_provider_failures_return_appropriate_errors(
    code: str,
    message: str,
    status_code: int,
) -> None:
    response = _post_question(FailingLLMProvider(code, message), [_source()])

    assert response.status_code == status_code
    assert response.json()["detail"] == {
        "code": code,
        "message": message,
    }


def test_question_traces_workflow_context_and_generation(
    trace_exporter: InMemorySpanExporter,
) -> None:
    response = _post_question(FakeLLMProvider(), [_source()])

    assert response.status_code == 200
    spans = {span.name: span for span in trace_exporter.get_finished_spans()}
    workflow = spans["garden_center_rag"]
    for child_name in (
        "retrieve",
        "check_relevance",
        "generate",
    ):
        assert spans[child_name].parent == workflow.context

    context_attributes = spans["retrieve"].attributes
    assert context_attributes["rag.retrieval.retrieved_chunks"] == 1
    assert context_attributes["rag.context.retrieved_chunks"] == 1
    assert context_attributes["rag.context.included_chunks"] == 1
    assert context_attributes["rag.context.token_budget"] == 4096
    assert context_attributes["rag.context.chunks_excluded"] is False

    generation_attributes = spans["generate"].attributes
    assert generation_attributes["gen_ai.provider.name"] == "test"
    assert generation_attributes["gen_ai.request.model"] == "test-generation-model"
    assert generation_attributes["gen_ai.usage.input_tokens"] == 42
    assert generation_attributes["gen_ai.usage.output_tokens"] == 8
    serialized_attributes = repr([span.attributes for span in spans.values()])
    assert "How should I care for roses?" not in serialized_attributes
    assert "Roses need regular watering." not in serialized_attributes


def test_generation_failure_is_recorded_without_changing_response(
    trace_exporter: InMemorySpanExporter,
) -> None:
    response = _post_question(
        FailingLLMProvider("generation_provider_unavailable", "Unavailable."),
        [_source()],
    )

    assert response.status_code == 503
    generation_span = next(
        span for span in trace_exporter.get_finished_spans() if span.name == "generate"
    )
    assert generation_span.status.status_code is StatusCode.ERROR


def test_generation_capacity_rejects_without_queueing_and_releases_slot() -> None:
    async def scenario() -> tuple[Response, Response, Response]:
        started = asyncio.Event()
        release = asyncio.Event()

        class BlockingLLMProvider(FakeLLMProvider):
            def __init__(self) -> None:
                super().__init__()
                self.max_concurrent_generations = 1
                self.attempts = 0

            async def generate(
                self, question: str, sources: Sequence[str]
            ) -> LLMGenerationResult:
                self.attempts += 1
                if self.attempts == 1:
                    started.set()
                    await release.wait()
                return await super().generate(question, sources)

        async def fake_search(*_: object, **__: object) -> list[SearchResult]:
            return [_source()]

        provider = BlockingLLMProvider()
        app.dependency_overrides[get_embedding_provider] = FakeEmbeddingProvider
        app.dependency_overrides[get_llm_provider] = lambda: provider
        try:
            with patch("garden_center.rag.service.semantic_search", fake_search):
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://testserver",
                ) as client:
                    first_task = asyncio.create_task(
                        client.post(
                            "/api/v1/questions",
                            json={"question": "First question"},
                        )
                    )
                    await asyncio.wait_for(started.wait(), timeout=1)
                    second = await client.post(
                        "/api/v1/questions",
                        json={"question": "Second question"},
                    )
                    release.set()
                    first = await first_task
                    third = await client.post(
                        "/api/v1/questions",
                        json={"question": "Third question"},
                    )
        finally:
            release.set()
            app.dependency_overrides.pop(get_embedding_provider, None)
            app.dependency_overrides.pop(get_llm_provider, None)
        return first, second, third

    first, second, third = asyncio.run(scenario())

    assert first.status_code == 200
    assert second.status_code == 503
    assert second.json()["detail"]["code"] == "generation_capacity_unavailable"
    assert third.status_code == 200
