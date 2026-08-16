from typing import Any
from uuid import UUID

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
from app.models.document import DocumentStatus
from app.models.job import JobStatus
from app.services.deduplication.service import DeduplicationService
from app.utils.hashing import compute_sha256

logger = get_logger(__name__)


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Compute file SHA-256 and check for exact file duplicates (Level 1).

    If duplicate is found:
      - Sets status to DUPLICATE (never FAILED).
      - Records canonical_document_id reference to original document.
      - Short-circuits remaining pipeline execution cleanly.
    """
    logger.info("step.duplicate.start", document_id=str(state.document_id))

    postgres: PostgresClient = services["postgres"]
    dedup_service = DeduplicationService()

    # ── 1. Compute SHA-256 ───────────────────────────────────────────────────
    sha256 = compute_sha256(state.storage_path)
    logger.debug("step.duplicate.hash", sha256=sha256[:16] + "...", path=state.storage_path)

    # ── 2. Level 1 Exact File Duplicate Check ─────────────────────────────
    existing_row = await dedup_service.check_file_hash(
        sha256=sha256,
        tenant_id=state.tenant_id,
        current_doc_id=state.document_id,
        postgres=postgres,
    )

    if existing_row is not None:
        canonical_id = UUID(str(existing_row["id"]))
        logger.info(
            "step.duplicate.exact_found",
            sha256=sha256[:16] + "...",
            canonical_document_id=str(canonical_id),
        )

        # Mark document as DUPLICATE in PostgreSQL
        await postgres.execute(
            """
            UPDATE documents
            SET sha256                = $1,
                status                = $2,
                canonical_document_id = $3,
                progress_percent      = 100
            WHERE id = $4
            """,
            sha256,
            DocumentStatus.DUPLICATE.value,
            canonical_id,
            state.document_id,
        )

        if state.job_id:
            await postgres.execute(
                """
                UPDATE ingestion_jobs
                SET status           = $1,
                    progress_percent = 100
                WHERE job_id = $2
                """,
                JobStatus.DUPLICATE.value,
                state.job_id,
            )

        return state.model_copy(
            update={
                "sha256": sha256,
                "status": DocumentStatus.DUPLICATE,
                "dup_classification": "EXACT_DUP",
                "canonical_document_id": canonical_id,
                "progress_percent": 100,
            }
        )

    # ── Unique File ──────────────────────────────────────────────────────────
    logger.info(
        "step.duplicate.unique",
        document_id=str(state.document_id),
        sha256=sha256[:16] + "...",
    )

    await postgres.execute(
        "UPDATE documents SET sha256 = $1 WHERE id = $2",
        sha256,
        state.document_id,
    )

    return state.model_copy(
        update={
            "sha256": sha256,
            "status": DocumentStatus.CHECKING_DUPLICATE,
        }
    )

