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
        self._raw = Path(raw_path).resolve()
        self._processed = Path(processed_path).resolve()
        self._chunks = Path(chunks_path).resolve()

        # Ensure directories exist at initialization time
        for directory in (self._raw, self._processed, self._chunks):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def raw_dir(self) -> Path:
        """Return the authoritative raw storage directory."""
        return self._raw

    # ── Upload storage ─────────────────────────────────────────────────────────

    def save_upload(self, file_bytes: bytes, filename: str) -> str:
        """
        Save raw uploaded file bytes to the raw storage directory.

        A UUID prefix is prepended to prevent filename collisions.
        Guarantees that the file is verified on disk before returning the path.

        Args:
            file_bytes: Binary content of the uploaded file.
            filename: Original filename (used as suffix).

        Returns:
            Absolute canonical path string to the saved file.

        Raises:
            ValueError: If file_bytes is empty or filename is invalid.
            IOError: If disk write or post-write verification fails.
        """
        if not file_bytes:
            raise ValueError("Uploaded file content is empty (0 bytes)")

        # Path traversal protection
        filename = Path(filename).name
        if not filename:
            raise ValueError("Invalid filename")

        safe_name = f"{uuid.uuid4().hex}_{filename}"
        dest = (self._raw / safe_name).resolve()

        # Verify it stays within the raw directory
        if not dest.is_relative_to(self._raw):
            raise ValueError("Path traversal detected")

        # Ensure destination directory exists immediately before writing
        dest.parent.mkdir(parents=True, exist_ok=True)

        dest.write_bytes(file_bytes)

        # Immediate verification that the file was written completely
        if not dest.is_file():
            raise IOError(f"Storage write failed: file does not exist at {dest}")

        actual_size = dest.stat().st_size
        if actual_size != len(file_bytes):
            raise IOError(
                f"Storage write mismatch: expected {len(file_bytes)} bytes, wrote {actual_size}"
            )

        logger.info("storage.upload_saved", path=str(dest), size_bytes=actual_size)
        return str(dest)

    def get_raw_path(self, storage_path_or_filename: str) -> Path:
        """
        Resolve a storage path or filename against the authoritative raw storage directory.

        If the path is already an existing file, returns its canonical Path.
        If an absolute path from a previous container or environment is given but missing,
        checks if the basename exists within the authoritative raw directory.
        """
        p = Path(storage_path_or_filename)
        if p.is_file():
            return p.resolve()

        candidate = (self._raw / p.name).resolve()
        if candidate.is_file():
            return candidate

        return p.resolve() if p.is_absolute() else candidate

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
