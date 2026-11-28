"""Generic binary file storage abstraction."""

from dataclasses import dataclass
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
