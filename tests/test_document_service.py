"""Integration tests for persistence failure cleanup."""

import asyncio
import io
import uuid
from pathlib import Path

import pytest
from fastapi import UploadFile
from sqlalchemy.exc import SQLAlchemyError

from garden_center.config import Settings, get_settings
from garden_center.database import async_session_factory
from garden_center.document_service import create_document
from tests.pdf_factory import make_pdf


@pytest.mark.integration
def test_database_failure_removes_stored_file(
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
        upload = UploadFile(
            filename="failure.pdf",
            file=io.BytesIO(make_pdf([f"Database failure cleanup {uuid.uuid4()}"])),
            headers={"content-type": "application/pdf"},
        )

        async with async_session_factory() as session:

            async def fail_commit() -> None:
                raise SQLAlchemyError("forced commit failure")

            monkeypatch.setattr(session, "commit", fail_commit)
            with pytest.raises(SQLAlchemyError):
                await create_document(session, upload, settings)

        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())
