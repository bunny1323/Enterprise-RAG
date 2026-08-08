"""
Step 08 — Index (Atomic Dual-Write).
Writes to Weaviate, Neo4j, Supabase chunks table, and marks document COMPLETED.

Ordering:
1. Weaviate vector upsert
2. Neo4j document tree creation
3. Supabase chunks bulk insert
4. Supabase documents status update → COMPLETED
"""
import json
from datetime import datetime, timezone
from typing import Any

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.infrastructure.neo4j.client import Neo4jClient
from app.infrastructure.postgres.client import PostgresClient
from app.infrastructure.weaviate.client import WeaviateClient
from app.models.chunk import Chunk
from app.models.document import DocumentStatus

logger = get_logger(__name__)


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Atomically write chunks to all storage backends and mark document COMPLETED.

    Args:
        state: Current ingestion state (chunks and vectors must be set).
        services: Must contain 'weaviate', 'neo4j', 'postgres' keys.

    Returns:
        Updated state with status=COMPLETED.

    Raises:
        ValueError: If chunks or vectors are missing.
        Exception: Re-raises any storage error after logging.
    """
    logger.info("step.index.start", document_id=str(state.document_id))

    if not state.chunks:
        raise ValueError("Cannot index: chunks list is empty or None")
    if not state.vectors:
        raise ValueError("Cannot index: vectors list is empty or None")
    if len(state.chunks) != len(state.vectors):
        raise ValueError(
            f"Chunk/vector mismatch: {len(state.chunks)} chunks, {len(state.vectors)} vectors"
        )

    weaviate_client: WeaviateClient = services["weaviate"]
    neo4j_client: Neo4jClient = services["neo4j"]
    postgres: PostgresClient = services["postgres"]
    doc_id = str(state.document_id)

    # ── 1. Weaviate vector upsert ──────────────────────────────────────────────
    logger.debug("step.index.weaviate_start", document_id=doc_id)
    import asyncio

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: weaviate_client.upsert_chunks(state.chunks, state.vectors),
    )
    logger.info("step.index.weaviate_done", document_id=doc_id, chunks=len(state.chunks))

    # ── 2. Neo4j document tree ─────────────────────────────────────────────────
    logger.debug("step.index.neo4j_start", document_id=doc_id)
    await neo4j_client.create_document_tree(doc_id, state.chunks)
    logger.info("step.index.neo4j_done", document_id=doc_id)

    # ── 3. Supabase chunks bulk insert ─────────────────────────────────────────
    logger.debug("step.index.postgres_chunks_start", document_id=doc_id)
    await _insert_chunks_postgres(postgres, state.chunks)
    logger.info(
        "step.index.postgres_chunks_done",
        document_id=doc_id,
        chunks=len(state.chunks),
    )

    # ── 4. Supabase document status → COMPLETED ────────────────────────────────
    completed_at = datetime.now(timezone.utc)
    await postgres.execute(
        """
        UPDATE documents
        SET status           = $1,
            progress_percent = $2,
            page_count       = $3,
            completed_at     = $4
        WHERE id = $5
        """,
        DocumentStatus.COMPLETED.value,
        100,
        state.page_count or 0,
        completed_at,
        state.document_id,
    )

    logger.info(
        "step.index.complete",
        document_id=doc_id,
        completed_at=completed_at.isoformat(),
    )

    return state.model_copy(update={"status": DocumentStatus.COMPLETED})


async def _insert_chunks_postgres(
    postgres: PostgresClient,
    chunks: list[Chunk],
) -> None:
    """Bulk-insert chunks into the Supabase chunks table."""
    for chunk in chunks:
        await postgres.execute(
            """
            INSERT INTO chunks (
                chunk_id, parent_id, document_id, content,
                page_number, bounding_box, chunk_type,
                access_classification, industry_domain,
                hierarchy_path, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (chunk_id) DO UPDATE
            SET content               = EXCLUDED.content,
                metadata              = EXCLUDED.metadata
            """,
            chunk.chunk_id,
            chunk.parent_id,
            chunk.document_id,
            chunk.content,
            chunk.page_number,
            json.dumps(chunk.bounding_box) if chunk.bounding_box else None,
            chunk.chunk_type.value,
            chunk.access_classification,
            chunk.industry_domain,
            chunk.hierarchy_path,
            chunk.metadata,
        )
