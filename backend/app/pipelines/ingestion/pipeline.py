"""
Ingestion Pipeline Orchestrator.
Executes all 8 pipeline steps sequentially, updating Supabase progress after each.
Handles failures by marking the document FAILED with an error message.
"""
from typing import Any
from uuid import UUID

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
from app.models.document import DocumentStatus
from app.pipelines.ingestion.steps import (
    s01_validate,
    s02_duplicate,
    s03_parse,
    s04_vision,
    s05_chunk,
    s06_metadata,
    s07_embed,
    s08_index,
)

logger = get_logger(__name__)

# Steps in execution order — each step is (name, coroutine_function, progress_after)
_PIPELINE_STEPS = [
    ("validate",   s01_validate, 10),
    ("duplicate",  s02_duplicate, 15),
    ("parse",      s03_parse, 35),
    ("vision",     s04_vision, 50),
    ("chunk",      s05_chunk, 65),
    ("metadata",   s06_metadata, 75),
    ("embed",      s07_embed, 90),
    ("index",      s08_index, 100),
]


class IngestionPipeline:
    """
    Sequential pipeline that executes ingestion steps and tracks Supabase progress.

    Design principles:
    - No business logic: purely orchestrates step execution.
    - Agents handle decisions; pipeline handles execution order.
    - Any step failure marks the document FAILED and re-raises.
    """

    def __init__(self, services: dict[str, Any]) -> None:
        self._services = services

    async def run(self, state: IngestionState) -> UUID:
        """
        Execute all pipeline steps for a given ingestion state.

        Args:
            state: Initial IngestionState with document_id, storage_path, etc.

        Returns:
            document_id UUID on successful completion.

        Raises:
            Exception: Any step exception after marking the document FAILED.
        """
        postgres: PostgresClient = self._services["postgres"]
        doc_id = state.document_id

        logger.info("pipeline.start", document_id=str(doc_id), industry=state.industry)

        for step_name, step_fn, progress_after in _PIPELINE_STEPS:
            try:
                logger.info(
                    "pipeline.step_start",
                    step=step_name,
                    document_id=str(doc_id),
                )

                state = await step_fn(state, self._services)

                # Update progress in Supabase after each successful step
                await postgres.execute(
                    """
                    UPDATE documents
                    SET status           = $1,
                        progress_percent = $2
                    WHERE id = $3
                    """,
                    state.status.value,
                    progress_after,
                    doc_id,
                )

                logger.info(
                    "pipeline.step_complete",
                    step=step_name,
                    document_id=str(doc_id),
                    progress=progress_after,
                )

            except Exception as err:
                logger.error(
                    "pipeline.step_failed",
                    step=step_name,
                    document_id=str(doc_id),
                    error=str(err),
                    exc_info=True,
                )

                # Mark document as FAILED in Supabase
                from datetime import datetime, timezone

                await postgres.execute(
                    """
                    UPDATE documents
                    SET status        = $1,
                        error_message = $2,
                        completed_at  = $3
                    WHERE id = $4
                    """,
                    DocumentStatus.FAILED.value,
                    str(err)[:2000],  # Truncate for DB column
                    datetime.now(timezone.utc),
                    doc_id,
                )

                raise

        logger.info("pipeline.complete", document_id=str(doc_id))
        return doc_id
