import asyncio
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
from app.models.job import JobStatus

logger = get_logger(__name__)


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Atomically write chunks to all storage backends with per-backend state tracking.

    Ordering:
    1. PostgreSQL chunks bulk insert
    2. Weaviate vector upsert
    3. Neo4j document tree creation
    4. Knowledge base version update
    5. Document & Job status → COMPLETED / PARTIAL
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

    # Initialize indexing state record
    await postgres.execute(
        """
        INSERT INTO indexing_state (document_id, weaviate_status, neo4j_status, postgres_chunks_status, updated_at)
        VALUES ($1, 'PENDING', 'PENDING', 'PENDING', NOW())
        ON CONFLICT (document_id) DO NOTHING
        """,
        state.document_id,
    )

    pg_ok, weaviate_ok, neo4j_ok = False, False, False
    last_err = ""

    # ── 1. PostgreSQL chunks bulk insert ─────────────────────────────────────────
    try:
        logger.debug("step.index.postgres_chunks_start", document_id=doc_id)
        await _insert_chunks_postgres(postgres, state.chunks)
        pg_ok = True
        await postgres.execute(
            "UPDATE indexing_state SET postgres_chunks_status = 'COMPLETED' WHERE document_id = $1",
            state.document_id,
        )
        logger.info("step.index.postgres_chunks_done", document_id=doc_id, chunks=len(state.chunks))
    except Exception as err:
        logger.error("step.index.postgres_failed", error=str(err), document_id=doc_id)
        last_err += f"PostgreSQL: {str(err)}; "
        await postgres.execute(
            "UPDATE indexing_state SET postgres_chunks_status = 'FAILED', last_error = $1 WHERE document_id = $2",
            str(err)[:500],
            state.document_id,
        )

    # ── 2. Weaviate vector upsert ──────────────────────────────────────────────
    try:
        logger.debug("step.index.weaviate_start", document_id=doc_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: weaviate_client.upsert_chunks(state.chunks, state.vectors),
        )
        weaviate_ok = True
        await postgres.execute(
            """
            UPDATE indexing_state
            SET weaviate_status = 'COMPLETED', weaviate_chunk_count = $1
            WHERE document_id = $2
            """,
            len(state.chunks),
            state.document_id,
        )
        logger.info("step.index.weaviate_done", document_id=doc_id, chunks=len(state.chunks))
    except Exception as err:
        logger.error("step.index.weaviate_failed", error=str(err), document_id=doc_id)
        last_err += f"Weaviate: {str(err)}; "
        await postgres.execute(
            "UPDATE indexing_state SET weaviate_status = 'FAILED', last_error = $1 WHERE document_id = $2",
            str(err)[:500],
            state.document_id,
        )

    # ── 3. Neo4j document tree ─────────────────────────────────────────────────
    try:
        logger.debug("step.index.neo4j_start", document_id=doc_id)
        await neo4j_client.create_document_tree(doc_id, state.chunks)
        neo4j_ok = True
        await postgres.execute(
            """
            UPDATE indexing_state
            SET neo4j_status = 'COMPLETED', neo4j_node_count = $1
            WHERE document_id = $2
            """,
            len(state.chunks),
            state.document_id,
        )
        logger.info("step.index.neo4j_done", document_id=doc_id)
    except Exception as err:
        logger.error("step.index.neo4j_failed", error=str(err), document_id=doc_id)
        last_err += f"Neo4j: {str(err)}; "
        await postgres.execute(
            "UPDATE indexing_state SET neo4j_status = 'FAILED', last_error = $1 WHERE document_id = $2",
            str(err)[:500],
            state.document_id,
        )

    # Determine final document & job status
    completed_at = datetime.now(timezone.utc)

    if pg_ok and weaviate_ok and neo4j_ok:
        final_doc_status = DocumentStatus.COMPLETED
        final_job_status = JobStatus.COMPLETED
        progress = 100

        # Increment knowledge base version
        await postgres.execute(
            """
            INSERT INTO knowledge_base_versions (knowledge_base_id, tenant_id, version, updated_at)
            VALUES ($1, $2, 1, NOW())
            ON CONFLICT (knowledge_base_id, tenant_id) DO UPDATE
            SET version    = knowledge_base_versions.version + 1,
                updated_at = NOW()
            """,
            state.knowledge_base_id,
            state.tenant_id,
        )
    elif pg_ok or weaviate_ok or neo4j_ok:
        final_doc_status = DocumentStatus.PARTIAL
        final_job_status = JobStatus.PARTIAL
        progress = 95
        logger.warning("step.index.partial_success", document_id=doc_id, errors=last_err)
    else:
        raise RuntimeError(f"All indexing backends failed: {last_err}")

    # Update document status
    await postgres.execute(
        """
        UPDATE documents
        SET status           = $1,
            progress_percent = $2,
            page_count       = $3,
            completed_at     = $4,
            error_message    = $5
        WHERE id = $6
        """,
        final_doc_status.value,
        progress,
        state.page_count or 0,
        completed_at,
        last_err if last_err else None,
        state.document_id,
    )

    if state.job_id:
        await postgres.execute(
            """
            UPDATE ingestion_jobs
            SET status           = $1,
                progress_percent = $2,
                completed_at     = $3,
                error_message    = $4
            WHERE job_id = $5
            """,
            final_job_status.value,
            progress,
            completed_at,
            last_err if last_err else None,
            state.job_id,
        )

    logger.info(
        "step.index.complete",
        document_id=doc_id,
        status=final_doc_status.value,
        completed_at=completed_at.isoformat(),
    )

    return state.model_copy(update={"status": final_doc_status, "progress_percent": progress})


async def _insert_chunks_postgres(
    postgres: PostgresClient,
    chunks: list[Chunk],
) -> None:
    """Bulk-insert chunks into the Supabase/PostgreSQL chunks table."""
    for chunk in chunks:
        await postgres.execute(
            """
            INSERT INTO chunks (
                chunk_id, parent_id, document_id, tenant_id, assistant_id, knowledge_base_id,
                content, content_hash, section, subsection, context_prefix,
                embedding_representation, page_number, bounding_box, chunk_type,
                access_classification, industry_domain, hierarchy_path, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
            ON CONFLICT (chunk_id) DO UPDATE
            SET content               = EXCLUDED.content,
                content_hash          = EXCLUDED.content_hash,
                context_prefix        = EXCLUDED.context_prefix,
                metadata              = EXCLUDED.metadata
            """,
            chunk.chunk_id,
            chunk.parent_id,
            chunk.document_id,
            chunk.tenant_id,
            chunk.assistant_id,
            chunk.knowledge_base_id,
            chunk.content,
            chunk.content_hash,
            chunk.section,
            chunk.subsection,
            chunk.context_prefix,
            chunk.embedding_representation,
            chunk.page_number,
            json.dumps(chunk.bounding_box) if chunk.bounding_box else None,
            chunk.chunk_type.value,
            chunk.access_classification,
            chunk.industry_domain,
            chunk.hierarchy_path,
            chunk.metadata,
        )

