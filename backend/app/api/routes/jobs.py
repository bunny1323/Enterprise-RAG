"""
Ingestion Job API routes.
GET  /api/v1/jobs/{job_id}          — poll job status, stage checkpoints, progress
POST /api/v1/jobs/{job_id}/cancel   — cancel an active ingestion job
POST /api/v1/jobs/{job_id}/retry    — retry a failed/cancelled job from last checkpoint
"""
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import PostgresDep, SupervisorDep, TenantContextDep
from app.config.logging import get_logger
from app.models.job import IngestionJob, JobStatus

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.get(
    "/{job_id}",
    response_model=IngestionJob,
    summary="Get ingestion job status and progress",
)
async def get_job_status(
    job_id: str,
    postgres: PostgresDep,
    tenant_ctx: TenantContextDep,
) -> IngestionJob:
    """
    Poll an ingestion job's status, progress, checkpoints, and current stage.
    """
    try:
        j_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid job_id format: {job_id}",
        )

    row = await postgres.fetchrow(
        """
        SELECT job_id, document_id, tenant_id, assistant_id, knowledge_base_id,
               status, current_stage, progress_percent, retry_count,
               last_successful_stage, stage_checkpoints, error_message,
               created_at, started_at, completed_at, timeout_at, cancelled_at, metadata
        FROM ingestion_jobs
        WHERE job_id = $1 AND tenant_id = $2 AND knowledge_base_id = $3
        """,
        j_uuid,
        tenant_ctx.tenant_id,
        tenant_ctx.knowledge_base_id,
    )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job {job_id} not found",
        )

    checkpoints = row["stage_checkpoints"] if row["stage_checkpoints"] else {}
    if isinstance(checkpoints, str):
        import json
        checkpoints = json.loads(checkpoints)

    meta = row["metadata"] if row["metadata"] else {}
    if isinstance(meta, str):
        import json
        meta = json.loads(meta)

    return IngestionJob(
        job_id=row["job_id"],
        document_id=row["document_id"],
        tenant_id=row["tenant_id"],
        assistant_id=row["assistant_id"],
        knowledge_base_id=row["knowledge_base_id"],
        status=JobStatus(row["status"]) if row["status"] in JobStatus.__members__ else JobStatus.FAILED,
        current_stage=row["current_stage"],
        progress_percent=row["progress_percent"],
        retry_count=row["retry_count"],
        last_successful_stage=row["last_successful_stage"],
        stage_checkpoints=checkpoints,
        error_message=row["error_message"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        timeout_at=row["timeout_at"],
        cancelled_at=row["cancelled_at"],
        metadata=meta,
    )


@router.post(
    "/{job_id}/cancel",
    summary="Cancel an active ingestion job",
)
async def cancel_job(
    job_id: str,
    postgres: PostgresDep,
    tenant_ctx: TenantContextDep,
) -> dict:
    """Mark an active ingestion job as CANCELLED."""
    try:
        j_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid job_id format: {job_id}",
        )

    row = await postgres.fetchrow(
        """
        SELECT job_id, document_id, status FROM ingestion_jobs
        WHERE job_id = $1 AND tenant_id = $2 AND knowledge_base_id = $3
        """,
        j_uuid,
        tenant_ctx.tenant_id,
        tenant_ctx.knowledge_base_id,
    )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    now = datetime.now(timezone.utc)
    await postgres.execute(
        """
        UPDATE ingestion_jobs
        SET status = $1, cancelled_at = $2, completed_at = $2
        WHERE job_id = $3
        """,
        JobStatus.CANCELLED.value,
        now,
        j_uuid,
    )
    await postgres.execute(
        "UPDATE documents SET status = 'CANCELLED', completed_at = $1 WHERE id = $2",
        now,
        row["document_id"],
    )

    logger.info("job.cancelled", job_id=job_id, document_id=str(row["document_id"]))
    return {"job_id": job_id, "status": JobStatus.CANCELLED.value}
