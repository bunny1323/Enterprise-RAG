"""
Step 02 — Duplicate Detection.
Computes SHA-256 and checks Supabase for existing documents with the same hash.
"""
from typing import Any

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
from app.utils.hashing import compute_sha256

logger = get_logger(__name__)


class DuplicateDocumentError(Exception):
    """Raised when a document with the same SHA-256 already exists."""

    def __init__(self, sha256: str, existing_id: str) -> None:
        self.sha256 = sha256
        self.existing_id = existing_id
        super().__init__(
            f"Duplicate document detected. SHA-256={sha256[:16]}... "
            f"already exists as document_id={existing_id}"
        )


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Compute file SHA-256 and reject duplicates before expensive processing.

    Args:
        state: Current ingestion state (storage_path must be set).
        services: Must contain 'postgres' key → PostgresClient.

    Returns:
        Updated state with sha256 set.

    Raises:
        DuplicateDocumentError: If document with same hash already exists.
    """
    logger.info("step.duplicate.start", document_id=str(state.document_id))

    postgres: PostgresClient = services["postgres"]

    # ── Compute SHA-256 ────────────────────────────────────────────────────────
    sha256 = compute_sha256(state.storage_path)
    logger.debug("step.duplicate.hash", sha256=sha256[:16] + "...", path=state.storage_path)

    # ── Query Supabase for existing document ───────────────────────────────────
    row = await postgres.fetchrow(
        "SELECT id FROM documents WHERE sha256 = $1",
        sha256,
    )

    if row is not None:
        existing_id = str(row["id"])
        logger.warning(
            "step.duplicate.found",
            sha256=sha256[:16] + "...",
            existing_id=existing_id,
        )
        raise DuplicateDocumentError(sha256=sha256, existing_id=existing_id)

    logger.info(
        "step.duplicate.unique",
        document_id=str(state.document_id),
        sha256=sha256[:16] + "...",
    )

    # ── Update sha256 in Supabase ──────────────────────────────────────────────
    await postgres.execute(
        "UPDATE documents SET sha256 = $1 WHERE id = $2",
        sha256,
        state.document_id,
    )

    return state.model_copy(update={"sha256": sha256})
