"""Focused tests for Ollama failures, indexing, and semantic search."""

import asyncio
import hashlib
import uuid
from typing import NoReturn

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from garden_center.database import async_session_factory
from garden_center.dependencies import get_embedding_provider
from garden_center.embeddings import EmbeddingProviderError, OllamaEmbeddingProvider
from garden_center.main import app
from garden_center.models import (
    ChunkEmbedding,
    Document,
    DocumentChunk,
    ProcessingStatus,
)
from garden_center.search_service import semantic_search
from tests.fakes import FakeEmbeddingProvider


def _missing_model(_: httpx.Request) -> httpx.Response:
    return httpx.Response(404, json={"error": "model not found"})


def _invalid_response(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"embeddings": [[0.1]]})


def _connection_failure(request: httpx.Request) -> NoReturn:
    raise httpx.ConnectError("connection refused", request=request)


@pytest.mark.integration
def test_search_returns_source_metadata() -> None:
    async def scenario() -> None:
        document_id = uuid.uuid4()
        provider = FakeEmbeddingProvider()
        document = Document(
            id=document_id,
            filename="search.pdf",
            content_type="application/pdf",
            file_size=100,
            content_hash=hashlib.sha256(document_id.bytes).hexdigest(),
            storage_key=f"documents/{document_id}.pdf",
            processing_status=ProcessingStatus.COMPLETED,
            chunks=[
                DocumentChunk(
                    id=uuid.uuid4(),
                    chunk_index=0,
                    text_content="Roses need regular watering.",
                    source_page=1,
                    token_count=6,
                    embeddings=[
                        ChunkEmbedding(
                            embedding_provider=provider.provider_name,
                            embedding_model=provider.model,
                            embedding_dimensions=provider.dimensions,
                            vector_data=[1.0, 0.0, 0.0],
                        )
                    ],
                ),
                DocumentChunk(
                    id=uuid.uuid4(),
                    chunk_index=1,
                    text_content="Healthy soil contains organic matter.",
                    source_page=2,
                    token_count=6,
                    embeddings=[
                        ChunkEmbedding(
                            embedding_provider=provider.provider_name,
                            embedding_model=provider.model,
                            embedding_dimensions=provider.dimensions,
                            vector_data=[0.0, 1.0, 0.0],
                        )
                    ],
                ),
            ],
        )
        async with async_session_factory() as session:
            session.add(document)
            await session.commit()

        app.dependency_overrides[get_embedding_provider] = lambda: provider
        transport = ASGITransport(app=app)
        try:
            async with AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                search = await client.post(
                    "/api/v1/search",
                    json={"query": "rose care", "similarity_threshold": 0.5},
                )
                assert search.status_code == 200, search.text
                results = search.json()["results"]
                assert len(results) == 1
                assert results[0] == {
                    "document_id": str(document_id),
                    "filename": "search.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                    "text": "Roses need regular watering.",
                    "similarity": pytest.approx(1.0),
                }

                empty = await client.post(
                    "/api/v1/search",
                    json={"query": "unrelated", "similarity_threshold": 0.9},
                )
                assert empty.status_code == 200
                assert empty.json()["results"] == []
        finally:
            app.dependency_overrides.pop(get_embedding_provider, None)
            async with async_session_factory() as session:
                persisted = await session.get(Document, document_id)
                if persisted is not None:
                    await session.delete(persisted)
                    await session.commit()

    asyncio.run(scenario())


@pytest.mark.integration
def test_search_excludes_processing_documents() -> None:
    async def scenario() -> None:
        document_id = uuid.uuid4()
        provider = FakeEmbeddingProvider()
        document = Document(
            id=document_id,
            filename="processing.pdf",
            content_type="application/pdf",
            file_size=100,
            content_hash=hashlib.sha256(document_id.bytes).hexdigest(),
            storage_key=f"documents/{document_id}.pdf",
            processing_status=ProcessingStatus.PROCESSING,
            chunks=[
                DocumentChunk(
                    id=uuid.uuid4(),
                    chunk_index=0,
                    text_content="Roses need regular watering.",
                    source_page=1,
                    token_count=6,
                    embeddings=[
                        ChunkEmbedding(
                            embedding_provider=provider.provider_name,
                            embedding_model=provider.model,
                            embedding_dimensions=provider.dimensions,
                            vector_data=[1.0, 0.0, 0.0],
                        )
                    ],
                )
            ],
        )
        async with async_session_factory() as session:
            session.add(document)
            await session.commit()
            try:
                results = await semantic_search(session, provider, "rose care", 5, 0.5)
                assert results == []
            finally:
                await session.delete(document)
                await session.commit()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("handler", "expected_code"),
    [
        (_missing_model, "embedding_model_missing"),
        (_invalid_response, "invalid_embedding_response"),
        (_connection_failure, "ollama_unavailable"),
    ],
)
def test_ollama_provider_failures(
    handler: object,
    expected_code: str,
) -> None:
    async def scenario() -> None:
        provider = OllamaEmbeddingProvider(
            "http://ollama.test",
            "nomic-embed-text",
            dimensions=3,
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(EmbeddingProviderError) as error:
                await provider.embed(["text"])
            assert error.value.code == expected_code
        finally:
            await provider.aclose()

    asyncio.run(scenario())
