"""
Ingestion Supervisor Agent.
Handles upload acceptance, transactional registration, and durable background processing.
"""
import asyncio
import uuid
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
from app.models.document import DocumentStatus
from app.pipelines.ingestion.pipeline import IngestionPipeline, PipelineCancelledException
from app.services.storage.service import StorageService

logger = get_logger(__name__)

class IngestionSupervisor:
    """
    Supervisor agent that manages the ingestion lifecycle.

    PostgreSQL is the durable work queue; each process claims jobs under a
    renewable lease so multiple instances cannot process a job concurrently.
    """

    def __init__(
        self,
        pipeline: IngestionPipeline,
        storage: StorageService,
        postgres: PostgresClient,
    ) -> None:
        self._pipeline = pipeline
        self._storage = storage
        self._postgres = postgres
        self._worker_id = uuid.uuid4().hex

    # ── Upload handler (fast path) ─────────────────────────────────────────────

    async def handle_upload(
        self,
        file: UploadFile,
        industry: str = "manufacturing",
        tenant_id: str = "default",
        assistant_id: str = "default",
        knowledge_base_id: str = "default",
    ) -> dict:
        """
        Accept an upload and atomically register its document and durable job.

        Parsing, indexing, and model inference are deferred to the worker. File
        persistence and the database transaction complete before this returns.

        Args:
            file: FastAPI UploadFile object from the multipart request.
            industry: Industry domain label (default: manufacturing).
            tenant_id: Tenant identifier.
            assistant_id: Assistant identifier.
            knowledge_base_id: Knowledge base identifier.

        Returns:
            {'document_id': str, 'job_id': str, 'status': 'RECEIVED'}
        """
        logger.info(
            "supervisor.upload_start",
            filename=file.filename,
            industry=industry,
            tenant_id=tenant_id,
            kb_id=knowledge_base_id,
        )

        # Read file bytes from multipart stream
        file_bytes = await file.read()
        if not file_bytes:
            raise ValueError(f"Uploaded file '{file.filename}' is empty (0 bytes)")

        # Enforce file-size limits before writing to disk
        from app.config.settings import get_settings
        settings = get_settings()
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise ValueError(
                f"Uploaded file '{file.filename}' ({len(file_bytes) / (1024 * 1024):.1f} MB) "
                f"exceeds maximum allowed upload limit of {settings.max_upload_size_mb} MB"
            )

        # Save to raw storage (synchronous path op with verified write)
        storage_path = self._storage.save_upload(file_bytes, file.filename or "upload.pdf")

        # Explicit verification that the file is physically present and readable
        saved_file = Path(storage_path)
        if not saved_file.is_file():
            raise IOError(f"Could not verify saved upload on disk at {storage_path}")

        logger.info(
            "supervisor.upload_saved",
            filename=file.filename,
            storage_path=storage_path,
            size_bytes=len(file_bytes),
            tenant_id=tenant_id,
        )

        registration = await self._postgres.register_ingestion(
            sha256=f"pending_{uuid.uuid4().hex}",
            file_name=file.filename or "upload.pdf",
            storage_path=storage_path,
            industry=industry,
            tenant_id=tenant_id,
            assistant_id=assistant_id,
            knowledge_base_id=knowledge_base_id,
            max_attempts=settings.ingestion_job_max_attempts,
        )
        document_id: UUID = registration["document_id"]
        job_id: UUID = registration["job_id"]

        logger.info(
            "supervisor.upload_complete",
            document_id=str(document_id),
            job_id=str(job_id),
            storage_path=storage_path,
            size_bytes=len(file_bytes),
            worker_id=self._worker_id,
        )

        return {
            "document_id": str(document_id),
            "job_id": str(job_id),
            "status": "RECEIVED",
        }

    # ── Database worker (slow path, runs as background task) ──────────────────

    async def process_queue(self) -> None:
        """
        Poll PostgreSQL for work, claiming jobs with leases safe across processes.
        Expired leases make interrupted work eligible for bounded replay.
        """
        from app.config.settings import get_settings

        settings = get_settings()
        poll_seconds = settings.ingestion_job_poll_seconds
        lease_seconds = settings.ingestion_job_lease_seconds
        heartbeat_seconds = max(1, lease_seconds // 3)
        logger.info("supervisor.database_worker_started", worker_id=self._worker_id)
        while True:
            try:
                job = await self._postgres.claim_ingestion_job(
                    self._worker_id, lease_seconds
                )
            except Exception as err:
                logger.exception(
                    "supervisor.job_claim_failed",
                    worker_id=self._worker_id,
                    error=str(err),
                )
                await asyncio.sleep(poll_seconds)
                continue

            if job is None:
                await asyncio.sleep(poll_seconds)
                continue

            try:
                await self._process_claimed_job(
                    job,
                    lease_seconds=lease_seconds,
                    heartbeat_seconds=heartbeat_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "supervisor.claimed_job_failed",
                    job_id=str(job.get("job_id")),
                    worker_id=self._worker_id,
                )
                await asyncio.sleep(poll_seconds)

    async def _process_claimed_job(
        self,
        job: dict,
        *,
        lease_seconds: int,
        heartbeat_seconds: int,
    ) -> None:
        job_id = job["job_id"]
        document_id = job["document_id"]
        raw_path = job["storage_path"] or ""
        candidate = self._storage.get_raw_path(raw_path) if raw_path else None
        if candidate is None or not candidate.is_file():
            message = (
                f"File not found on storage disk at '{raw_path}'. "
                "Attach persistent storage or re-upload the document."
            )
            await self._postgres.record_ingestion_failure(
                job_id,
                document_id,
                self._worker_id,
                message,
                force_fail=True,
            )
            return

        state = IngestionState(
            document_id=document_id,
            job_id=job_id,
            tenant_id=job["tenant_id"],
            assistant_id=job["assistant_id"],
            knowledge_base_id=job["knowledge_base_id"],
            filename=job["file_name"],
            industry=job["industry"],
            storage_path=str(candidate.resolve()),
            status=DocumentStatus.PENDING,
        )
        heartbeat = asyncio.create_task(
            self._heartbeat(job_id, lease_seconds, heartbeat_seconds)
        )
        pipeline_task = asyncio.create_task(self._pipeline.run(state))
        try:
            done, _ = await asyncio.wait(
                {heartbeat, pipeline_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if heartbeat in done:
                try:
                    still_owned = heartbeat.result()
                except Exception as err:
                    still_owned = False
                    logger.error(
                        "supervisor.lease_heartbeat_failed",
                        job_id=str(job_id),
                        error=str(err),
                    )
                if not still_owned:
                    logger.warning("supervisor.lease_lost", job_id=str(job_id))
                    pipeline_task.cancel()
                    await asyncio.gather(pipeline_task, return_exceptions=True)
                    return

            try:
                await pipeline_task
            except asyncio.CancelledError:
                raise
            except PipelineCancelledException:
                await self._postgres.release_ingestion_claim(
                    job_id, self._worker_id
                )
                return
            except Exception as err:
                logger.exception(
                    "supervisor.pipeline_failed",
                    job_id=str(job_id),
                    error=str(err),
                )
                try:
                    await self._postgres.record_ingestion_failure(
                        job_id, document_id, self._worker_id, str(err)
                    )
                except Exception:
                    logger.exception(
                        "supervisor.failure_status_write_failed",
                        job_id=str(job_id),
                    )
                return

            await self._postgres.release_ingestion_claim(job_id, self._worker_id)
        finally:
            if not heartbeat.done():
                heartbeat.cancel()
            if not pipeline_task.done():
                pipeline_task.cancel()
            await asyncio.gather(heartbeat, pipeline_task, return_exceptions=True)

    async def _heartbeat(
        self,
        job_id: UUID,
        lease_seconds: int,
        heartbeat_seconds: int,
    ) -> bool:
        while True:
            await asyncio.sleep(heartbeat_seconds)
            if not await self._postgres.renew_ingestion_lease(
                job_id, self._worker_id, lease_seconds
            ):
                return False
