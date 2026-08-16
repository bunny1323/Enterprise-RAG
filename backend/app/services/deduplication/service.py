"""
Multi-Level Deduplication & Version Classification Service.

Hierarchy of deduplication:
Level 1: SHA-256 raw file hash (exact file match)
Level 2: Content hash (normalized text match)
Level 3: Chunk-level content hash comparison (exact chunk match)
Level 4: MinHash / LSH candidate detection (near-duplicate text)
Level 5: Semantic embedding similarity (configurable cosine threshold)
Level 6: Version classification decision (EXACT_DUP, CONTENT_DUP, NEAR_DUP, UPDATED_VERSION, PARTIAL_CHANGE, NEW)
"""
from uuid import UUID

from pydantic import BaseModel

from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
from app.models.chunk import Chunk
from app.services.deduplication.minhash import LSHIndex, MinHasher
from app.utils.hashing import compute_content_hash

logger = get_logger(__name__)


class DeduplicationResult(BaseModel):
    """Result of multi-level deduplication check."""

    classification: str  # "EXACT_DUP" | "CONTENT_DUP" | "NEAR_DUP" | "UPDATED_VERSION" | "PARTIAL_CHANGE" | "NEW"
    canonical_document_id: UUID | None = None
    existing_sha256: str | None = None
    content_hash: str | None = None
    reusable_chunk_ids: list[str] = []
    new_chunk_ids: list[str] = []
    version_number: int = 1
    supersedes_document_id: UUID | None = None


class DeduplicationService:
    """
    Service for executing multi-level deduplication without expensive LLM calls.
    """

    def __init__(self) -> None:
        self._hasher = MinHasher(num_perm=64)

    # ── Level 1: File Hash (SHA-256) ──────────────────────────────────────────

    async def check_file_hash(
        self, sha256: str, tenant_id: str, current_doc_id: UUID, postgres: PostgresClient
    ) -> dict | None:
        """Query PostgreSQL for an existing completed document with the same SHA-256."""
        row = await postgres.fetchrow(
            """
            SELECT id, sha256, file_name, version, content_hash
            FROM documents
            WHERE sha256 = $1 AND tenant_id = $2 AND id != $3 AND status != 'FAILED'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            sha256,
            tenant_id,
            current_doc_id,
        )
        return row

    # ── Level 2: Content Hash (Normalized Text) ──────────────────────────────

    async def check_content_hash(
        self, content_hash: str, tenant_id: str, current_doc_id: UUID, postgres: PostgresClient
    ) -> dict | None:
        """Query PostgreSQL for an existing completed document with the same content_hash."""
        row = await postgres.fetchrow(
            """
            SELECT id, sha256, file_name, version, content_hash
            FROM documents
            WHERE content_hash = $1 AND tenant_id = $2 AND id != $3 AND status != 'FAILED'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            content_hash,
            tenant_id,
            current_doc_id,
        )
        return row

    # ── Level 3: Chunk-level Hash Comparison ─────────────────────────────────

    async def partition_chunks_by_hash(
        self, chunks: list[Chunk], tenant_id: str, postgres: PostgresClient
    ) -> tuple[list[str], list[str]]:
        """
        Compare chunk content_hashes against existing indexed chunks in DB.

        Returns:
            (reusable_chunk_ids, new_chunk_ids)
        """
        if not chunks:
            return [], []

        hashes = [c.content_hash for c in chunks if c.content_hash]
        if not hashes:
            return [], [c.chunk_id for c in chunks]

        # Find existing chunk_ids that match these content_hashes under tenant_id
        rows = await postgres.fetch(
            """
            SELECT chunk_id, content_hash
            FROM chunks
            WHERE content_hash = ANY($1::text[]) AND tenant_id = $2
            """,
            hashes,
            tenant_id,
        )

        existing_hashes = {r["content_hash"] for r in rows}

        reusable_ids: list[str] = []
        new_ids: list[str] = []

        for chunk in chunks:
            if chunk.content_hash and chunk.content_hash in existing_hashes:
                reusable_ids.append(chunk.chunk_id)
            else:
                new_ids.append(chunk.chunk_id)

        logger.info(
            "dedup.chunk_partition",
            total=len(chunks),
            reusable=len(reusable_ids),
            new=len(new_ids),
        )
        return reusable_ids, new_ids

    # ── Level 4: MinHash / LSH Near-Duplicate Candidate Detection ──────────────

    def find_near_duplicate_candidates(
        self, new_chunks: list[Chunk], existing_chunks: list[dict]
    ) -> list[tuple[str, str]]:
        """
        Use MinHash + LSH to find candidate pairs of near-duplicate chunks.

        Returns list of (new_chunk_id, existing_chunk_id) pairs.
        """
        index = LSHIndex(num_perm=64, bands=8)
        chunk_map: dict[str, Chunk] = {}

        # Insert existing chunks into LSH
        for ec in existing_chunks:
            cid = ec.get("chunk_id", "")
            content = ec.get("content", "")
            if cid and content:
                sig = self._hasher.compute_signature(content)
                index.insert(cid, sig)

        candidate_pairs: list[tuple[str, str]] = []

        # Query new chunks against LSH index
        for nc in new_chunks:
            sig = self._hasher.compute_signature(nc.content)
            candidates = index.query(sig)
            for candidate_id in candidates:
                candidate_pairs.append((nc.chunk_id, candidate_id))

        return candidate_pairs
