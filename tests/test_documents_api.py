"""Integration tests for document persistence and lifecycle."""

import asyncio
import hashlib
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from garden_center.config import get_settings
from garden_center.database import async_session_factory
from garden_center.dependencies import get_embedding_provider
from garden_center.main import app
from garden_center.models import ChunkEmbedding, Document, DocumentChunk
from tests.fakes import FailingEmbeddingProvider, FakeEmbeddingProvider
from tests.pdf_factory import make_pdf


@pytest.mark.integration
def test_document_upload_retrieval_duplicate_and_cascade_delete() -> None:
    async def scenario() -> None:
        unique_text = f"Garden care integration test {uuid.uuid4()}"
        pdf = make_pdf([unique_text, "A second page about healthy plants."])
        app.dependency_overrides[get_embedding_provider] = FakeEmbeddingProvider
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            concurrent_uploads = await asyncio.gather(
                client.post(
                    "/api/v1/documents",
                    files={"file": ("garden.pdf", pdf, "application/pdf")},
                ),
                client.post(
                    "/api/v1/documents",
                    files={"file": ("garden-copy.pdf", pdf, "application/pdf")},
                ),
            )
            assert sorted(response.status_code for response in concurrent_uploads) == [
                201,
                409,
            ]
            created = next(
                response
                for response in concurrent_uploads
                if response.status_code == 201
            )
            assert created.status_code == 201, created.text
            body = created.json()
            document_id = body["id"]
            assert body["processing_status"] == "completed"
            assert body["chunk_count"] >= 2
            assert "chunks" not in body
            assert created.headers["location"] == f"/api/v1/documents/{document_id}"

            stored_path = get_settings().document_storage_path / f"{document_id}.pdf"
            assert stored_path.exists()

            duplicate = await client.post(
                "/api/v1/documents",
                files={"file": ("copy.pdf", pdf, "application/pdf")},
            )
            assert duplicate.status_code == 409
            assert duplicate.json()["detail"]["code"] == "duplicate_document"

            listing = await client.get("/api/v1/documents")
            assert listing.status_code == 200
            listing_body = listing.json()
            assert listing_body["limit"] == 20
            assert listing_body["offset"] == 0
            assert listing_body["total"] >= 1
            listed = next(
                item for item in listing_body["items"] if item["id"] == document_id
            )
            assert listed["chunk_count"] == body["chunk_count"]

            first_page = await client.get(
                "/api/v1/documents", params={"limit": 1, "offset": 0}
            )
            second_page = await client.get(
                "/api/v1/documents", params={"limit": 1, "offset": 1}
            )
            assert len(first_page.json()["items"]) == 1
            assert first_page.json()["total"] == second_page.json()["total"]
            if second_page.json()["items"]:
                assert (
                    first_page.json()["items"][0]["id"]
                    != second_page.json()["items"][0]["id"]
                )

            invalid_page = await client.get(
                "/api/v1/documents", params={"limit": 101, "offset": -1}
            )
            assert invalid_page.status_code == 422

            detail = await client.get(f"/api/v1/documents/{document_id}")
            assert detail.status_code == 200
            assert detail.json()["filename"] == "garden.pdf"
            assert [chunk["source_page"] for chunk in detail.json()["chunks"]] == [1, 2]

            async with async_session_factory() as session:
                chunk_id = await session.scalar(
                    select(DocumentChunk.id).where(
                        DocumentChunk.document_id == uuid.UUID(document_id)
                    )
                )
                assert chunk_id is not None
                automatic_embedding_count = await session.scalar(
                    select(func.count())
                    .select_from(ChunkEmbedding)
                    .where(ChunkEmbedding.chunk_id == chunk_id)
                )
                assert automatic_embedding_count == 1
                session.add(
                    ChunkEmbedding(
                        chunk_id=chunk_id,
                        embedding_provider="test",
                        embedding_model="test-model",
                        embedding_dimensions=3,
                        vector_data=[0.1, 0.2, 0.3],
                    )
                )
                await session.commit()

            deleted = await client.delete(f"/api/v1/documents/{document_id}")
            assert deleted.status_code == 204
            assert not stored_path.exists()

            missing = await client.get(f"/api/v1/documents/{document_id}")
            assert missing.status_code == 404

            async with async_session_factory() as session:
                chunk_count = await session.scalar(
                    select(func.count())
                    .select_from(DocumentChunk)
                    .where(DocumentChunk.document_id == uuid.UUID(document_id))
                )
                embedding_count = await session.scalar(
                    select(func.count())
                    .select_from(ChunkEmbedding)
                    .where(ChunkEmbedding.chunk_id == chunk_id)
                )
                document_count = await session.scalar(
                    select(func.count())
                    .select_from(Document)
                    .where(Document.id == uuid.UUID(document_id))
                )
                assert (document_count, chunk_count, embedding_count) == (0, 0, 0)

        app.dependency_overrides.pop(get_embedding_provider, None)

    asyncio.run(scenario())


@pytest.mark.integration
def test_failed_upload_is_cleaned_up_and_can_be_retried() -> None:
    async def scenario() -> None:
        pdf = make_pdf([f"Indexing failure retry {uuid.uuid4()}"])
        content_hash = hashlib.sha256(pdf).hexdigest()
        storage_path = get_settings().document_storage_path
        files_before = set(storage_path.glob("*.pdf"))
        transport = ASGITransport(app=app)
        app.dependency_overrides[get_embedding_provider] = FailingEmbeddingProvider

        try:
            async with AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                failed = await client.post(
                    "/api/v1/documents",
                    files={"file": ("recovery.pdf", pdf, "application/pdf")},
                )
                assert failed.status_code == 503
                assert failed.json()["detail"]["code"] == "ollama_unavailable"

                async with async_session_factory() as session:
                    document_count = await session.scalar(
                        select(func.count())
                        .select_from(Document)
                        .where(Document.content_hash == content_hash)
                    )
                    chunk_count = await session.scalar(
                        select(func.count())
                        .select_from(DocumentChunk)
                        .join(Document)
                        .where(Document.content_hash == content_hash)
                    )
                    embedding_count = await session.scalar(
                        select(func.count())
                        .select_from(ChunkEmbedding)
                        .join(DocumentChunk)
                        .join(Document)
                        .where(Document.content_hash == content_hash)
                    )
                    assert (document_count, chunk_count, embedding_count) == (0, 0, 0)
                assert set(storage_path.glob("*.pdf")) == files_before

                app.dependency_overrides[get_embedding_provider] = FakeEmbeddingProvider
                retried = await client.post(
                    "/api/v1/documents",
                    files={"file": ("retry.pdf", pdf, "application/pdf")},
                )
                assert retried.status_code == 201, retried.text
                assert retried.json()["processing_status"] == "completed"

                async with async_session_factory() as session:
                    persisted = await session.scalar(
                        select(Document).where(Document.content_hash == content_hash)
                    )
                    assert persisted is not None

                deleted = await client.delete(
                    f"/api/v1/documents/{retried.json()['id']}"
                )
                assert deleted.status_code == 204
        finally:
            app.dependency_overrides.pop(get_embedding_provider, None)

    asyncio.run(scenario())


@pytest.mark.integration
def test_document_validation_error_response() -> None:
    async def scenario() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/documents",
                files={"file": ("broken.pdf", b"%PDF-broken", "application/pdf")},
            )
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "malformed_pdf"

    asyncio.run(scenario())
