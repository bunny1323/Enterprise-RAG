"""
Step 06 — Metadata Enrichment.
Applies industry-specific metadata to all chunks via MetadataService.
"""
from typing import Any

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.services.metadata.service import MetadataService

logger = get_logger(__name__)


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Enrich chunks with domain-specific metadata from the industry config.

    Args:
        state: Current ingestion state (chunks must be set).
        services: Must contain 'metadata' key → MetadataService.

    Returns:
        Updated state with enriched chunks.

    Raises:
        ValueError: If chunks are not available.
    """
    logger.info("step.metadata.start", document_id=str(state.document_id))

    if not state.chunks:
        raise ValueError("Cannot enrich metadata: chunks list is empty or None")

    metadata_svc: MetadataService = services["metadata"]

    enriched_chunks = metadata_svc.enrich_chunks(state.chunks, state.industry)

    logger.info(
        "step.metadata.complete",
        document_id=str(state.document_id),
        chunks=len(enriched_chunks),
        industry=state.industry,
    )

    return state.model_copy(update={"chunks": enriched_chunks})
