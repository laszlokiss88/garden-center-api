"""Focused tests for local file storage key safety."""

import asyncio
from pathlib import Path

import pytest

from garden_center.file_storage import LocalFileStorage


@pytest.mark.parametrize(
    "directory",
    ["../outside", "..\\outside", "/outside", "C:\\outside"],
)
def test_local_storage_rejects_keys_outside_root(
    tmp_path: Path, directory: str
) -> None:
    async def scenario() -> None:
        storage = LocalFileStorage(tmp_path)

        with pytest.raises(ValueError):
            await storage.upload(b"content", "text/plain", directory, ".txt")
        with pytest.raises(ValueError):
            await storage.delete(f"{directory}/file.txt")

    asyncio.run(scenario())


def test_local_storage_generates_unique_keys(tmp_path: Path) -> None:
    async def scenario() -> None:
        storage = LocalFileStorage(tmp_path)

        first = await storage.upload(b"first", "application/pdf", "documents", ".pdf")
        second = await storage.upload(b"second", "application/pdf", "documents", "pdf")

        assert first.storage_key != second.storage_key
        assert first.storage_key.startswith("documents/")
        assert first.storage_key.endswith(".pdf")
        assert (tmp_path / first.storage_key).read_bytes() == b"first"
        assert (tmp_path / second.storage_key).read_bytes() == b"second"

    asyncio.run(scenario())
