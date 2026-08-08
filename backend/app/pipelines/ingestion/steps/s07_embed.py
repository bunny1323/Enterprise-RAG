"""
Step 07 — Embed.
Generates Voyage Multimodal-3 embeddings for all chunks.
Text, TABLE, and IMAGE/DIAGRAM chunks all use text content for embedding.
"""
import asyncio
from typing import Any

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.models.document import DocumentStatus
from app.services.embeddings.service import EmbeddingService

logger = get_logger(__name__)


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Embed all chunk contents using Voyage Multimodal-3 API.

    Uses the text content for all chunk types (text content for TABLE chunks
    is Markdown; for IMAGE/DIAGRAM chunks it's the vision analysis summary).

    Args:
        state: Current ingestion state (chunks must be set).
        services: Must contain 'embedder' key → EmbeddingService.

    Returns:
        Updated state with vectors list (parallel to chunks) and status=EMBEDDING.

    Raises:
        ValueError: If chunks are not available or vector count mismatches.
    """
    logger.info("step.embed.start", document_id=str(state.document_id))

    if not state.chunks:
        raise ValueError("Cannot embed: chunks list is empty or None")

    embedder: EmbeddingService = services["embedder"]

    # Extract text content from all chunks for batch embedding
    texts = [chunk.content for chunk in state.chunks]

    logger.info(
        "step.embed.submitting",
        document_id=str(state.document_id),
        chunk_count=len(texts),
    )

    # Run blocking Voyage API call in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    vectors = await loop.run_in_executor(None, embedder.embed_batch, texts)

    if len(vectors) != len(state.chunks):
        raise ValueError(
            f"Vector count mismatch: {len(vectors)} vectors for {len(state.chunks)} chunks"
        )

    # Validate embedding dimension
    if vectors and len(vectors[0]) != 1024:
        logger.warning(
            "step.embed.unexpected_dim",
            expected=1024,
            actual=len(vectors[0]),
        )

    logger.info(
        "step.embed.complete",
        document_id=str(state.document_id),
        vectors=len(vectors),
        dim=len(vectors[0]) if vectors else 0,
    )

    return state.model_copy(
        update={
            "vectors": vectors,
            "status": DocumentStatus.EMBEDDING,
        }
    )
