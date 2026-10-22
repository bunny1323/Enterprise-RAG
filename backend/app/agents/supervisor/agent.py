"""
Ingestion Supervisor Agent.
Handles upload acceptance, immediate document registration, and async queue processing.
Returns document_id in <100ms by deferring all heavy processing to the background queue.
"""
import asyncio
import traceback
import uuid
from uuid import UUID

from fastapi import UploadFile

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
from app.models.document import DocumentStatus
from app.pipelines.ingestion.pipeline import IngestionPipeline
from app.services.storage.service import StorageService

logger = get_logger(__name__)


class IngestionSupervisor:
    """
    Supervisor agent that manages the ingestion lifecycle.

    Responsibilities:
    1. Accept file upload → save to disk → register in Supabase → return document_id.
    2. Enqueue the ingestion state to an asyncio.Queue.
    3. Process queue in a background task, running the full pipeline for each document.

    The split between handle_upload (sync-fast) and process_queue (async-slow)
    is what guarantees <100ms response times for the upload endpoint.
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
        self._queue: asyncio.Queue[IngestionState] = asyncio.Queue()

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
        Accept an upload, register the document and job, and enqueue for processing.

        This method MUST return in <100ms. All expensive work happens in process_queue.

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

        # INSERT document record immediately
        doc_row = await self._postgres.fetchrow(
            """
            INSERT INTO documents (
                sha256, file_name, storage_path, industry,
                tenant_id, assistant_id, knowledge_base_id, status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            """,
            f"pending_{uuid.uuid4().hex}",  # Unique sha256 placeholder
            file.filename or "upload.pdf",
            storage_path,
            industry,
            tenant_id,
            assistant_id,
            knowledge_base_id,
            DocumentStatus.PENDING.value,
        )

        document_id: UUID = doc_row["id"]  # type: ignore[index]

        # INSERT ingestion job record immediately
        job_row = await self._postgres.fetchrow(
            """
            INSERT INTO ingestion_jobs (
                document_id, tenant_id, assistant_id, knowledge_base_id, status, started_at
            )
            VALUES ($1, $2, $3, $4, $5, NOW())
            RETURNING job_id
            """,
            document_id,
            tenant_id,
            assistant_id,
            knowledge_base_id,
            "RECEIVED",
        )

        job_id: UUID = job_row["job_id"]  # type: ignore[index]

        # Build initial state and enqueue for background processing
        state = IngestionState(
            document_id=document_id,
            job_id=job_id,
            tenant_id=tenant_id,
            assistant_id=assistant_id,
            knowledge_base_id=knowledge_base_id,
            filename=file.filename or "upload.pdf",
            industry=industry,
            storage_path=storage_path,
            status=DocumentStatus.PENDING,
        )

        await self._queue.put(state)

        logger.info(
            "supervisor.upload_complete",
            document_id=str(document_id),
            job_id=str(job_id),
            storage_path=storage_path,
            size_bytes=len(file_bytes),
            queued=self._queue.qsize(),
        )

        return {
            "document_id": str(document_id),
            "job_id": str(job_id),
            "status": "RECEIVED",
        }

    # ── Queue processor (slow path, runs as background task) ──────────────────

    async def process_queue(self) -> None:
        """
        Continuously pull IngestionState items from the queue and run the pipeline.

        Runs as a long-lived asyncio background task started at application startup.
        Errors are caught per-document so one failure does not stop the queue.
        """
        logger.info("supervisor.queue_worker_started")

        # Recover interrupted or unstarted jobs from PostgreSQL
        try:
            pending_docs = await self._postgres.fetch(
                """
                SELECT id, file_name, storage_path, industry, tenant_id, assistant_id, knowledge_base_id 
                FROM documents 
                WHERE status IN ('PENDING', 'PROCESSING', 'EXTRACTING', 'PARSING', 'CHUNKING', 'EMBEDDING', 'WAITING_FOR_EMBEDDING_QUOTA')
                """
            )
            for doc in pending_docs:
                raw_path = doc.get("storage_path", "")
                resolved_path = None

                # Check if the file actually exists on this container's filesystem
                if raw_path and Path(raw_path).is_file():
                    resolved_path = str(Path(raw_path).resolve())
                elif raw_path:
                    # Attempt recovery via authoritative raw directory
                    candidate = self._storage.get_raw_path(raw_path)
                    if candidate.is_file():
                        resolved_path = str(candidate.resolve())

                # If file is missing (e.g. ephemeral container restart on Render)
                if not resolved_path:
                    logger.error(
                        "supervisor.recovered_file_missing",
                        document_id=str(doc["id"]),
                        storage_path=raw_path,
                        filename=doc["file_name"],
                    )
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc)
                    err_msg = (
                        f"File not found on storage disk at '{raw_path}'. "
                        "On Render, files stored on the default ephemeral filesystem do not survive restarts or redeployments. "
                        "To prevent this, attach a Render Persistent Disk (set RAW_STORAGE_PATH=/data/raw) or re-upload the document."
                    )
                    await self._postgres.execute(
                        """
                        UPDATE documents
                        SET status = 'FAILED', error_message = $1, completed_at = $2
                        WHERE id = $3
                        """,
                        err_msg,
                        now,
                        doc["id"],
                    )
                    await self._postgres.execute(
                        """
                        UPDATE ingestion_jobs
                        SET status = 'FAILED', error_message = $1, completed_at = $2
                        WHERE document_id = $3
                        """,
                        err_msg,
                        now,
                        doc["id"],
                    )
                    continue

                job_row = await self._postgres.fetchrow(
                    "SELECT job_id FROM ingestion_jobs WHERE document_id = $1 ORDER BY started_at DESC LIMIT 1",
                    doc["id"]
                )
                if job_row:
                    state = IngestionState(
                        document_id=doc["id"],
                        job_id=job_row["job_id"],
                        tenant_id=doc["tenant_id"],
                        assistant_id=doc["assistant_id"],
                        knowledge_base_id=doc["knowledge_base_id"],
                        filename=doc["file_name"],
                        industry=doc["industry"],
                        storage_path=resolved_path,
                        status=DocumentStatus.PENDING,
                    )
                    await self._queue.put(state)
            
            if pending_docs:
                logger.info("supervisor.queue_recovered", count=len(pending_docs))
        except Exception as e:
            logger.error("supervisor.queue_recovery_failed", error=str(e))

        while True:
            state: IngestionState = await self._queue.get()
            logger.info(
                "supervisor.processing",
                document_id=str(state.document_id),
                queue_remaining=self._queue.qsize(),
            )

            try:
                await self._pipeline.run(state)
            except Exception:
                logger.error(
                    "supervisor.pipeline_failed",
                    document_id=str(state.document_id),
                    traceback=traceback.format_exc(),
                )
            finally:
                self._queue.task_done()
