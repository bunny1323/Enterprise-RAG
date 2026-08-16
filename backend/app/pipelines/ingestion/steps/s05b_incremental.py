"""
Step 05b — Incremental Deduplication.
Partition chunks into reusable (unchanged) vs new/modified chunks using Level 3 content_hash comparison.
"""
from typing import Any

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
from app.services.deduplication.service import DeduplicationService

logger = get_logger(__name__)


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Partition chunks into reusable_chunk_ids (already embedded/indexed) and new_chunk_ids.
    """
    logger.info("step.incremental.start", document_id=str(state.document_id))

    if not state.chunks:
        logger.warning("step.incremental.skipped", reason="chunks list is empty")
        return state

    postgres: PostgresClient = services["postgres"]
    dedup_svc = DeduplicationService()

    reusable_ids, new_ids = await dedup_svc.partition_chunks_by_hash(
        chunks=state.chunks,
        tenant_id=state.tenant_id,
        postgres=postgres,
    )

    logger.info(
        "step.incremental.complete",
        document_id=str(state.document_id),
        total_chunks=len(state.chunks),
        reusable_chunks=len(reusable_ids),
        new_chunks=len(new_ids),
    )

    return state.model_copy(
        update={
            "reusable_chunk_ids": reusable_ids,
            "new_chunk_ids": new_ids,
        }
    )
