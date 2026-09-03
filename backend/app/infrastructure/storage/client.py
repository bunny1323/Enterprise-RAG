"""
Local filesystem storage client.
Handles raw uploads, processed JSON, and chunk text file persistence.
"""
import json
import uuid
from pathlib import Path

from app.config.logging import get_logger

logger = get_logger(__name__)


class StorageClient:
    """File system storage for raw uploads, processed docs, and chunk text."""

    def __init__(
        self,
        raw_path: str,
        processed_path: str,
        chunks_path: str,
    ) -> None:
        self._raw = Path(raw_path)
        self._processed = Path(processed_path)
        self._chunks = Path(chunks_path)

        # Ensure directories exist at initialization time
        for directory in (self._raw, self._processed, self._chunks):
            directory.mkdir(parents=True, exist_ok=True)

    # ── Upload storage ─────────────────────────────────────────────────────────

    def save_upload(self, file_bytes: bytes, filename: str) -> str:
        """
        Save raw uploaded file bytes to the raw storage directory.

        A UUID prefix is prepended to prevent filename collisions.

        Args:
            file_bytes: Binary content of the uploaded file.
            filename: Original filename (used as suffix).

        Returns:
            Absolute path string to the saved file.
        """
        # Path traversal protection
        filename = Path(filename).name
        if not filename:
            raise ValueError("Invalid filename")

        safe_name = f"{uuid.uuid4().hex}_{filename}"
        dest = (self._raw / safe_name).resolve()
        
        # Verify it stays within the raw directory
        if not str(dest).startswith(str(self._raw.resolve())):
            raise ValueError("Path traversal detected")

        dest.write_bytes(file_bytes)
        logger.info("storage.upload_saved", path=str(dest), size=len(file_bytes))
        return str(dest)

    # ── Processed storage ──────────────────────────────────────────────────────

    def save_processed(self, document_id: str, data: dict) -> str:
        """
        Persist parsed document JSON to the processed storage directory.

        Args:
            document_id: Document UUID string (used as filename stem).
            data: Parsed document dictionary from DocumentParserService.

        Returns:
            Absolute path string to the saved JSON file.
        """
        dest = self._processed / f"{document_id}.json"
        dest.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("storage.processed_saved", path=str(dest))
        return str(dest)

    # ── Chunk storage ──────────────────────────────────────────────────────────

    def save_chunk(
        self,
        document_id: str,
        chunk_id: str,
        content: str,
    ) -> str:
        """
        Save individual chunk text to a per-document subdirectory.

        Args:
            document_id: Parent document UUID string (subdirectory name).
            chunk_id: Unique chunk identifier (filename stem).
            content: Raw text content of the chunk.

        Returns:
            Absolute path string to the saved .txt file.
        """
        chunk_dir = self._chunks / document_id
        chunk_dir.mkdir(parents=True, exist_ok=True)

        dest = chunk_dir / f"{chunk_id}.txt"
        dest.write_text(content, encoding="utf-8")
        return str(dest)

    # ── Read helpers ───────────────────────────────────────────────────────────

    def load_processed(self, document_id: str) -> dict:
        """Load a previously saved processed document JSON."""
        src = self._processed / f"{document_id}.json"
        return json.loads(src.read_text(encoding="utf-8"))

    def raw_file_exists(self, filename: str) -> bool:
        """Return True if a raw file with the given name exists."""
        return (self._raw / filename).exists()
