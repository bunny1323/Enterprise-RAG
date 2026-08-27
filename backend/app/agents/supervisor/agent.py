"""
Ingestion Supervisor Agent.
Handles upload acceptance, immediate document registration, and async queue processing.
Returns document_id in <100ms by deferring all heavy processing to the background queue.
"""
import asyncio
import traceback
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

        # Save to raw storage (synchronous path op, fast)
        storage_path = self._storage.save_upload(file_bytes, file.filename or "upload.pdf")

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
            "",  # sha256 placeholder — step 02 will compute and update this
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
                        storage_path=doc["storage_path"],
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
