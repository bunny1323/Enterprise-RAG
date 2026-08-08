"""
Storage Service — thin dependency-injection wrapper over StorageClient.
Keeps infrastructure details out of pipeline steps and agents.
"""
from app.config.logging import get_logger
from app.infrastructure.storage.client import StorageClient

logger = get_logger(__name__)


class StorageService:
    """
    Thin stateless wrapper over StorageClient for use as a FastAPI dependency.

    All actual I/O is delegated to StorageClient; this layer exists only to
    enforce the services/infrastructure boundary in clean architecture.
    """

    def __init__(self, client: StorageClient) -> None:
        self._client = client

    def save_upload(self, file_bytes: bytes, filename: str) -> str:
        """Save raw upload bytes. Returns absolute path."""
        return self._client.save_upload(file_bytes, filename)

    def save_processed(self, document_id: str, data: dict) -> str:
        """Save processed document JSON. Returns absolute path."""
        return self._client.save_processed(document_id, data)

    def save_chunk(self, document_id: str, chunk_id: str, content: str) -> str:
        """Save individual chunk text. Returns absolute path."""
        return self._client.save_chunk(document_id, chunk_id, content)

    def load_processed(self, document_id: str) -> dict:
        """Load a saved processed document JSON."""
        return self._client.load_processed(document_id)
