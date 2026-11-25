"""Generic file storage abstraction and local implementation."""

import asyncio
import os
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol


@dataclass(frozen=True, slots=True)
class StoredFile:
    """Reference to content persisted by a storage provider."""

    storage_key: str


class FileStorage(Protocol):
    """Upload and delete binary content by provider-generated storage key."""

    async def upload(
        self,
        content: bytes,
        content_type: str,
        directory: str,
        extension: str,
    ) -> StoredFile:
        """Persist content and return its provider-generated storage key."""
        ...

    async def delete(self, storage_key: str) -> None:
        """Delete content when present."""
        ...


class LocalFileStorage:
    """Store binary files below a configured local directory."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path.resolve()

    async def upload(
        self,
        content: bytes,
        content_type: str,
        directory: str,
        extension: str,
    ) -> StoredFile:
        """Write content to a file."""
        return await asyncio.to_thread(
            self._upload, content, content_type, directory, extension
        )

    async def delete(self, storage_key: str) -> None:
        """Delete a stored file if it exists."""
        await asyncio.to_thread(self._delete, storage_key)

    def _path(self, key: str, create_parent: bool = False) -> Path:
        if not key or "\x00" in key:
            raise ValueError("Storage key must be a non-empty relative path.")

        posix_key = PurePosixPath(key.replace("\\", "/"))
        windows_key = PureWindowsPath(key)
        if (
            not posix_key.parts
            or posix_key.is_absolute()
            or windows_key.is_absolute()
            or windows_key.drive
            or ".." in posix_key.parts
        ):
            raise ValueError("Storage key must stay within the storage directory.")

        path = (self.storage_path / Path(*posix_key.parts)).resolve()
        if not path.is_relative_to(self.storage_path):
            raise ValueError("Storage key must stay within the storage directory.")
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _upload(
        self,
        content: bytes,
        content_type: str,
        directory: str,
        extension: str,
    ) -> StoredFile:
        del content_type  # Local storage does not persist media-type metadata.
        if (
            not extension
            or "\x00" in extension
            or "/" in extension
            or "\\" in extension
        ):
            raise ValueError("Extension must be a filename extension.")

        suffix = extension if extension.startswith(".") else f".{extension}"
        if suffix == ".":
            raise ValueError("Extension must be a filename extension.")
        filename = f"{uuid.uuid4()}{suffix}"
        key = f"{directory}/{filename}" if directory else filename
        destination = self._path(key, create_parent=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(content)
            os.replace(temporary, destination)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise
        return StoredFile(storage_key=key)

    def _delete(self, storage_key: str) -> None:
        self._path(storage_key).unlink(missing_ok=True)
