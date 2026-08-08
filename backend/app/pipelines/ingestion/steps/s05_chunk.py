"""
Step 05 — Chunk.
Invokes ChunkingService to produce hierarchical chunks from the parsed document.
"""
from typing import Any

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.models.document import DocumentStatus
from app.services.chunking.service import ChunkingService

logger = get_logger(__name__)


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Split the parsed document into hierarchical parent and child chunks.

    Args:
        state: Current ingestion state (parsed_doc must be set).
        services: Must contain 'chunker' key → ChunkingService.

    Returns:
        Updated state with chunks list and status=CHUNKING.

    Raises:
        ValueError: If parsed_doc is not available.
    """
    logger.info("step.chunk.start", document_id=str(state.document_id))

    if state.parsed_doc is None:
        raise ValueError("Cannot chunk: parsed_doc is None. Did step 03 run successfully?")

    chunker: ChunkingService = services["chunker"]

    chunks = chunker.chunk(
        parsed_doc=state.parsed_doc,
        document_id=state.document_id,
        industry=state.industry,
    )

    # Categorize chunks for logging
    from app.models.chunk import ChunkType

    type_counts = {ct.value: 0 for ct in ChunkType}
    for chunk in chunks:
        type_counts[chunk.chunk_type.value] += 1

    logger.info(
        "step.chunk.complete",
        document_id=str(state.document_id),
        total=len(chunks),
        **type_counts,
    )

    return state.model_copy(
        update={
            "chunks": chunks,
            "status": DocumentStatus.CHUNKING,
        }
    )
