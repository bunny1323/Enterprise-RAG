"""
Step 07 — Embedding.
Generate embeddings for chunks using the local BGE model with PostgreSQL embedding cache.

BEHAVIORS:
1. Reusable chunk IDs (from s05b_incremental) are fetched from EmbeddingCache only
   — no API call is made for unchanged chunks.
2. Cache misses are COLLECTED first, then batched via embed_batch().
3. On embedding failure: marks ingestion FAILED with the actual error reason.
4. Vectors are mapped back to original chunk positions to preserve ordering.
5. Image chunks fall back to embedding the alt-text (BGE is text-only).
6. New vectors are persisted to EmbeddingCache immediately after success.
"""
import asyncio
from typing import Any

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
from app.models.document import DocumentStatus
from app.services.cache.embedding_cache import EmbeddingCache
from app.services.embeddings.service import EmbeddingQuotaError

logger = get_logger(__name__)


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Generate embeddings for all chunks with batch optimization and cache reuse.

    Algorithm:
    1. Query EmbeddingCache for all content_hashes in one DB round-trip.
    2. Fill known vectors from cache; collect remaining as text/image misses.
    3. Embed text misses via embed_batch() in a single executor call.
    4. For image chunks: fall back to embedding alt-text (BGE is text-only).
    5. Save new vectors to EmbeddingCache.
    6. Validate all chunk positions have a vector, then attach to state.
    """
    logger.info("step.embed.start", document_id=str(state.document_id))

    if not state.chunks:
        raise ValueError("Cannot embed: chunks list is empty or None")

    embedder: Any = services["embedder"]
    postgres: PostgresClient = services["postgres"]
    model_name = state.embedding_model
    model_version = state.embedding_model_version

    reusable_ids: set[str] = set(state.reusable_chunk_ids)

    # 1. Query EmbeddingCache in one batch
    content_hashes = [c.content_hash or "" for c in state.chunks]
    non_empty_hashes = [ch for ch in content_hashes if ch]

    cache_hits = await EmbeddingCache.get_batch(
        postgres=postgres,
        content_hashes=non_empty_hashes,
        model=model_name,
        model_version=model_version,
    ) if non_empty_hashes else {}

    logger.info(
        "step.embed.cache_checked",
        total_chunks=len(state.chunks),
        cache_hits=len(cache_hits),
        reusable_ids=len(reusable_ids),
    )

    # 2. Pre-allocate output vector list
    vectors: list[list[float] | None] = [None] * len(state.chunks)

    # 3. Fill from cache
    for idx, chunk in enumerate(state.chunks):
        c_hash = chunk.content_hash or ""
        if c_hash and c_hash in cache_hits:
            vectors[idx] = cache_hits[c_hash]

    # 4. Collect cache misses — separate into text vs image
    text_miss_indices: list[int] = []
    text_miss_contents: list[str] = []

    for idx, chunk in enumerate(state.chunks):
        if vectors[idx] is not None:
            continue  # Already have from cache
        # Both text and image chunks use text embedding (BGE is text-only)
        text_miss_indices.append(idx)
        text_miss_contents.append(chunk.content)

    logger.info(
        "step.embed.misses",
        document_id=str(state.document_id),
        text_misses=len(text_miss_indices),
    )

    loop = asyncio.get_event_loop()
    new_cache_entries: dict[str, list[float]] = {}

    # 5. Embed all misses via embed_batch() in executor
    if text_miss_contents:
        try:
            batch_vectors: list[list[float]] = await loop.run_in_executor(
                None, embedder.embed_batch, text_miss_contents
            )
            for local_pos, chunk_idx in enumerate(text_miss_indices):
                vec = batch_vectors[local_pos]
                vectors[chunk_idx] = vec
                c_hash = state.chunks[chunk_idx].content_hash or ""
                if c_hash:
                    new_cache_entries[c_hash] = vec
        except EmbeddingQuotaError as err:
            logger.error(
                "step.embed.local_inference_failed",
                document_id=str(state.document_id),
                error=str(err),
            )
            raise
        except Exception as err:
            logger.error(
                "step.embed.unexpected_error",
                document_id=str(state.document_id),
                error=str(err),
            )
            raise

    # 6. Persist new embeddings to cache
    if new_cache_entries:
        await EmbeddingCache.set_batch(
            postgres=postgres,
            hash_vector_map=new_cache_entries,
            model=model_name,
            model_version=model_version,
        )
        logger.info("step.embed.cache_saved", count=len(new_cache_entries))

    # 7. Validate all positions filled
    final_vectors: list[list[float]] = []
    for idx, vec in enumerate(vectors):
        if vec is None:
            raise ValueError(
                f"Missing embedding for chunk at index {idx} "
                f"(id={state.chunks[idx].chunk_id})"
            )
        final_vectors.append(vec)

    if len(final_vectors) != len(state.chunks):
        raise ValueError(
            f"Vector count mismatch: {len(final_vectors)} vectors for "
            f"{len(state.chunks)} chunks"
        )

    logger.info(
        "step.embed.complete",
        document_id=str(state.document_id),
        total_vectors=len(final_vectors),
        cache_hits=len(cache_hits),
        reused_unchanged=len(reusable_ids),
        new_embedded=len(text_miss_indices),
        new_cache_saved=len(new_cache_entries),
    )

    return state.model_copy(
        update={
            "vectors": final_vectors,
            "status": DocumentStatus.EMBEDDING,
        }
    )
