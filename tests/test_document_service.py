"""Integration tests for persistence failure cleanup."""

import asyncio
import hashlib
import io
import uuid
from pathlib import Path

import pytest
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from garden_center.core.config import Settings, get_settings
from garden_center.database.engine import async_session_factory
from garden_center.database.models import Document
from garden_center.documents.service import create_document
from garden_center.storage.local_file_storage import LocalFileStorage
from tests.pdf_factory import make_pdf


@pytest.mark.integration
@pytest.mark.parametrize(
    "commit_error",
    [
        IntegrityError("forced integrity failure", {}, Exception("failure")),
        SQLAlchemyError("forced database failure"),
    ],
    ids=["integrity-error", "general-database-error"],
)
def test_database_failure_removes_stored_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commit_error: SQLAlchemyError,
) -> None:
    async def scenario() -> None:
        current = get_settings()
        settings = Settings(
            database_url=current.database_url,
            document_storage_path=tmp_path,
            max_upload_size_bytes=current.max_upload_size_bytes,
            chunk_size_tokens=current.chunk_size_tokens,
            chunk_overlap_tokens=current.chunk_overlap_tokens,
        )
        storage = LocalFileStorage(tmp_path)
        upload = UploadFile(
            filename="failure.pdf",
            file=io.BytesIO(make_pdf([f"Database failure cleanup {uuid.uuid4()}"])),
            headers={"content-type": "application/pdf"},
        )

        async with async_session_factory() as session:

            async def fail_commit() -> None:
                raise commit_error

            monkeypatch.setattr(session, "commit", fail_commit)
            with pytest.raises(type(commit_error)):
                await create_document(session, upload, settings, storage)

        assert not any(path.is_file() for path in tmp_path.rglob("*"))

    asyncio.run(scenario())


@pytest.mark.integration
def test_refresh_failure_preserves_committed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        current = get_settings()
        settings = Settings(
            database_url=current.database_url,
            document_storage_path=tmp_path,
            max_upload_size_bytes=current.max_upload_size_bytes,
            chunk_size_tokens=current.chunk_size_tokens,
            chunk_overlap_tokens=current.chunk_overlap_tokens,
        )
        storage = LocalFileStorage(tmp_path)
        content = make_pdf([f"Refresh failure {uuid.uuid4()}"])
        upload = UploadFile(
            filename="refresh-failure.pdf",
            file=io.BytesIO(content),
            headers={"content-type": "application/pdf"},
        )

        async with async_session_factory() as session:

            async def fail_refresh(_: Document) -> None:
                raise SQLAlchemyError("forced refresh failure")

            monkeypatch.setattr(session, "refresh", fail_refresh)
            with pytest.raises(SQLAlchemyError, match="forced refresh failure"):
                await create_document(session, upload, settings, storage)

            document = await session.scalar(
                select(Document).where(
                    Document.content_hash == hashlib.sha256(content).hexdigest()
                )
            )
            assert document is not None
            assert (tmp_path / document.storage_key).is_file()

            storage_key = document.storage_key
            await session.delete(document)
            await session.commit()
            await storage.delete(storage_key)

    asyncio.run(scenario())
