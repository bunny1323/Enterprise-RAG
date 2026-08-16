import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
from app.models.document import DocumentStatus
from app.models.job import JobStatus
from app.pipelines.ingestion.steps import (
    s01_validate,
    s02_duplicate,
    s03_parse,
    s04_vision,
    s05_chunk,
    s05b_incremental,
    s06_metadata,
    s07_embed,
    s08_index,
)

logger = get_logger(__name__)

# Steps in execution order — each step is (name, coroutine_function, progress_after, job_status)
_PIPELINE_STEPS = [
    ("validate",    s01_validate, 10,  JobStatus.VALIDATING),
    ("duplicate",   s02_duplicate, 15,  JobStatus.CHECKING_DUPLICATE),
    ("parse",       s03_parse, 35,  JobStatus.PARSING),
    ("vision",      s04_vision, 50,  JobStatus.VISION),
    ("chunk",       s05_chunk, 65,  JobStatus.CHUNKING),
    ("incremental", s05b_incremental, 70, JobStatus.CHUNKING),
    ("metadata",    s06_metadata, 75,  JobStatus.METADATA),
    ("embed",       s07_embed, 90,  JobStatus.EMBEDDING),
    ("index",       s08_index, 100, JobStatus.INDEXING),
]

_PER_STAGE_TIMEOUT_SECONDS = 300  # 5 minutes per stage


class PipelineCancelledException(Exception):
    """Raised when an ingestion run is cancelled by the user."""
    pass


class IngestionPipeline:
    """
    Sequential pipeline that executes ingestion steps and tracks database progress.

    Design principles:
    - Pure execution engine with checkpointing & resumability.
    - Updates both documents table and ingestion_jobs table.
    - Supports per-stage timeouts and cancellation checks.
    """

    def __init__(self, services: dict[str, Any]) -> None:
        self._services = services

    async def run(self, state: IngestionState) -> UUID:
        """
        Execute all pipeline steps for a given ingestion state.
        Skipping already completed stages if state.stage_checkpoints is populated.
        """
        postgres: PostgresClient = self._services["postgres"]
        doc_id = state.document_id
        job_id = state.job_id

        logger.info(
            "pipeline.start",
            document_id=str(doc_id),
            job_id=str(job_id) if job_id else None,
            tenant_id=state.tenant_id,
            industry=state.industry,
        )

        for step_name, step_fn, progress_after, job_status in _PIPELINE_STEPS:
            # 1. Check for cancellation
            if state.cancelled or await self._check_is_cancelled(postgres, job_id):
                logger.warning("pipeline.cancelled", document_id=str(doc_id), job_id=str(job_id))
                await self._update_cancelled_status(postgres, doc_id, job_id)
                raise PipelineCancelledException(f"Job {job_id} was cancelled")

            # 2. Check if stage is already checkpointed (resumability)
            if state.stage_checkpoints.get(step_name):
                logger.info(
                    "pipeline.step_skipped_already_completed",
                    step=step_name,
                    document_id=str(doc_id),
                )
                continue

            try:
                logger.info(
                    "pipeline.step_start",
                    step=step_name,
                    document_id=str(doc_id),
                )

                # Update job & document status to stage status
                await self._update_stage_status(
                    postgres, doc_id, job_id, step_name, job_status.value, progress_after
                )

                # Execute step with timeout
                state = await asyncio.wait_for(
                    step_fn(state, self._services), timeout=_PER_STAGE_TIMEOUT_SECONDS
                )

                # Record checkpoint
                checkpoints = dict(state.stage_checkpoints)
                checkpoints[step_name] = True
                state = state.model_copy(
                    update={
                        "stage_checkpoints": checkpoints,
                        "last_successful_stage": step_name,
                    }
                )

                # Persist checkpoint to database
                await self._save_checkpoint(postgres, job_id, step_name, checkpoints)

                # Short-circuit if document was determined to be DUPLICATE or QUARANTINED
                if state.status in (DocumentStatus.COMPLETED, DocumentStatus.FAILED) or state.dup_classification == "EXACT_DUP":
                    logger.info("pipeline.short_circuit", step=step_name, status=state.status.value)
                    break

                logger.info(
                    "pipeline.step_complete",
                    step=step_name,
                    document_id=str(doc_id),
                    progress=progress_after,
                )

            except asyncio.TimeoutError:
                err_msg = f"Step '{step_name}' timed out after {_PER_STAGE_TIMEOUT_SECONDS}s"
                logger.error("pipeline.step_timeout", step=step_name, document_id=str(doc_id))
                await self._update_failed_status(postgres, doc_id, job_id, err_msg)
                raise TimeoutError(err_msg)

            except Exception as err:
                logger.error(
                    "pipeline.step_failed",
                    step=step_name,
                    document_id=str(doc_id),
                    error=str(err),
                    exc_info=True,
                )
                await self._update_failed_status(postgres, doc_id, job_id, str(err)[:2000])
                raise

        logger.info("pipeline.complete", document_id=str(doc_id))
        return doc_id

    async def _check_is_cancelled(self, postgres: PostgresClient, job_id: UUID | None) -> bool:
        if not job_id:
            return False
        row = await postgres.fetchrow(
            "SELECT status FROM ingestion_jobs WHERE job_id = $1", job_id
        )
        return row is not None and row["status"] == JobStatus.CANCELLED.value

    async def _update_stage_status(
        self,
        postgres: PostgresClient,
        doc_id: UUID,
        job_id: UUID | None,
        stage_name: str,
        status_value: str,
        progress_after: int,
    ) -> None:
        await postgres.execute(
            """
            UPDATE documents
            SET status           = $1,
                progress_percent = $2
            WHERE id = $3
            """,
            status_value,
            progress_after,
            doc_id,
        )
        if job_id:
            await postgres.execute(
                """
                UPDATE ingestion_jobs
                SET status           = $1,
                    current_stage    = $2,
                    progress_percent = $3
                WHERE job_id = $4
                """,
                status_value,
                stage_name,
                progress_after,
                job_id,
            )

    async def _save_checkpoint(
        self,
        postgres: PostgresClient,
        job_id: UUID | None,
        step_name: str,
        checkpoints: dict[str, bool],
    ) -> None:
        if not job_id:
            return
        await postgres.execute(
            """
            INSERT INTO pipeline_checkpoints (job_id, last_successful_stage, stage_data, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (job_id) DO UPDATE
            SET last_successful_stage = EXCLUDED.last_successful_stage,
                stage_data            = EXCLUDED.stage_data,
                updated_at            = NOW()
            """,
            job_id,
            step_name,
            json.dumps(checkpoints),
        )
        await postgres.execute(
            """
            UPDATE ingestion_jobs
            SET last_successful_stage = $1,
                stage_checkpoints     = $2
            WHERE job_id = $3
            """,
            step_name,
            json.dumps(checkpoints),
            job_id,
        )

    async def _update_failed_status(
        self, postgres: PostgresClient, doc_id: UUID, job_id: UUID | None, err_msg: str
    ) -> None:
        now = datetime.now(timezone.utc)
        await postgres.execute(
            """
            UPDATE documents
            SET status        = $1,
                error_message = $2,
                completed_at  = $3
            WHERE id = $4
            """,
            DocumentStatus.FAILED.value,
            err_msg,
            now,
            doc_id,
        )
        if job_id:
            await postgres.execute(
                """
                UPDATE ingestion_jobs
                SET status        = $1,
                    error_message = $2,
                    completed_at  = $3
                WHERE job_id = $4
                """,
                JobStatus.FAILED.value,
                err_msg,
                now,
                job_id,
            )

    async def _update_cancelled_status(
        self, postgres: PostgresClient, doc_id: UUID, job_id: UUID | None
    ) -> None:
        now = datetime.now(timezone.utc)
        await postgres.execute(
            """
            UPDATE documents
            SET status       = 'CANCELLED',
                completed_at = $1
            WHERE id = $2
            """,
            now,
            doc_id,
        )
        if job_id:
            await postgres.execute(
                """
                UPDATE ingestion_jobs
                SET status       = $1,
                    cancelled_at = $2,
                    completed_at = $2
                WHERE job_id = $3
                """,
                JobStatus.CANCELLED.value,
                now,
                job_id,
            )

