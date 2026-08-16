"""
SHA-256 hashing utilities for file deduplication.
"""
import hashlib


def compute_sha256(file_path: str) -> str:
    """
    Compute the SHA-256 hex digest of a file without loading it entirely into memory.

    Reads in 8 KiB chunks to handle arbitrarily large PDFs efficiently.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        Lowercase hexadecimal SHA-256 digest string.

    Raises:
        FileNotFoundError: If the path does not point to an existing file.
        OSError: On any other I/O error.
    """
    sha256 = hashlib.sha256()
    chunk_size = 8192  # 8 KiB — sweet spot for I/O throughput vs. memory

    with open(file_path, "rb") as fh:
        while True:
            data = fh.read(chunk_size)
            if not data:
                break
            sha256.update(data)

    return sha256.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """
    Compute SHA-256 of an in-memory bytes object.

    Args:
        data: Raw bytes to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest string.
    """
    return hashlib.sha256(data).hexdigest()


def compute_content_hash(text: str) -> str:
    """
    Compute SHA-256 of normalized text content (Level 2 deduplication).
    Collapses whitespace and lowercases text to ignore formatting differences.
    """
    import re
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_chunk_hash(content: str) -> str:
    """
    Compute SHA-256 of raw chunk content for chunk-level deduplication (Level 3).
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

