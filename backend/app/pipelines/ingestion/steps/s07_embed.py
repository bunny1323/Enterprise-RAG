"""
Step 07 — Embedding.
Generate embeddings for chunks using Voyage API with PostgreSQL embedding cache.

FIXED BEHAVIORS:
1. Reusable chunk IDs (from s05b_incremental) are fetched from EmbeddingCache only
   — no Voyage API call is made for unchanged chunks.
2. Cache misses are COLLECTED first, then batched in a single embed_batch() call
   — no more embed_batch([single_chunk]) per miss in a for-loop.
3. Vectors are mapped back to original chunk positions to preserve ordering.
4. Image embeddings remain separate (one per image path).
"""
import asyncio
from typing import Any

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
from app.models.document import DocumentStatus
from app.models.job import JobStatus
from app.services.cache.embedding_cache import EmbeddingCache
from app.services.embeddings.service import EmbeddingService, RateLimitQuotaError

logger = get_logger(__name__)


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Generate embeddings for all chunks with batch optimization and cache reuse.

    Algorithm:
    1. Build position-indexed map of all chunks.
    2. For reusable_chunk_ids (unchanged, already indexed), check EmbeddingCache only.
    3. Collect cache misses into text vs image buckets.
    4. Send ALL text cache misses to Voyage in ONE batched embed_batch() call.
    5. Embed image misses individually (embed_image).
    6. Map returned vectors back to original positions.
    7. Save new vectors to EmbeddingCache.
    """
    logger.info("step.embed.start", document_id=str(state.document_id))

    if not state.chunks:
        raise ValueError("Cannot embed: chunks list is empty or None")

    embedder: EmbeddingService = services["embedder"]
    postgres: PostgresClient = services["postgres"]
    model_name = state.embedding_model
    model_version = state.embedding_model_version

    reusable_ids: set[str] = set(state.reusable_chunk_ids)

    # ── 1. Build content_hash list and query EmbeddingCache ──────────────────
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

    # ── 2. Pre-allocate output vector list matching chunk positions ───────────
    vectors: list[list[float] | None] = [None] * len(state.chunks)

    # ── 3. Fill from cache ────────────────────────────────────────────────────
    for idx, chunk in enumerate(state.chunks):
        c_hash = chunk.content_hash or ""
        if c_hash and c_hash in cache_hits:
            vectors[idx] = cache_hits[c_hash]

    # ── 4. Collect cache misses ───────────────────────────────────────────────
    text_miss_indices: list[int] = []
    text_miss_contents: list[str] = []
    image_miss_indices: list[int] = []

    for idx, chunk in enumerate(state.chunks):
        if vectors[idx] is not None:
            continue  # Already have from cache

        image_path = chunk.metadata.get("image_path") if chunk.metadata else None
        if chunk.embedding_representation == "image" and image_path:
            image_miss_indices.append(idx)
        else:
            text_miss_indices.append(idx)
            text_miss_contents.append(chunk.content)

    logger.info(
        "step.embed.misses",
        document_id=str(state.document_id),
        text_misses=len(text_miss_indices),
        image_misses=len(image_miss_indices),
    )

    loop = asyncio.get_event_loop()
    new_cache_entries: dict[str, list[float]] = {}

    # ── 5. Batch embed ALL text misses in a single API call ───────────────────
    if text_miss_contents:
        try:
            batch_vectors = await loop.run_in_executor(
                None, embedder.embed_batch, text_miss_contents
            )
            for local_pos, chunk_idx in enumerate(text_miss_indices):
                vec = batch_vectors[local_pos]
                vectors[chunk_idx] = vec
                c_hash = state.chunks[chunk_idx].content_hash or ""
                if c_hash:
                    new_cache_entries[c_hash] = vec

        except RateLimitQuotaError as quota_err:
            logger.warning(
                "step.embed.quota_exceeded_text_batch",
                document_id=str(state.document_id),
                retry_after=quota_err.retry_after,
            )
            await _set_quota_status(postgres, state)
            await asyncio.sleep(quota_err.retry_after)

            # Retry the entire text batch after waiting
            batch_vectors = await loop.run_in_executor(
                None, embedder.embed_batch, text_miss_contents
            )
            for local_pos, chunk_idx in enumerate(text_miss_indices):
                vec = batch_vectors[local_pos]
                vectors[chunk_idx] = vec
                c_hash = state.chunks[chunk_idx].content_hash or ""
                if c_hash:
                    new_cache_entries[c_hash] = vec

    # ── 6. Embed image misses individually ────────────────────────────────────
    for chunk_idx in image_miss_indices:
        chunk = state.chunks[chunk_idx]
        image_path = chunk.metadata.get("image_path") if chunk.metadata else None
        try:
            vec = await loop.run_in_executor(None, embedder.embed_image, image_path)
            vectors[chunk_idx] = vec
        except RateLimitQuotaError as quota_err:
            logger.warning(
                "step.embed.quota_exceeded_image",
                document_id=str(state.document_id),
                retry_after=quota_err.retry_after,
            )
            await _set_quota_status(postgres, state)
            await asyncio.sleep(quota_err.retry_after)
            vec = await loop.run_in_executor(None, embedder.embed_image, image_path)
            vectors[chunk_idx] = vec
        except Exception as err:
            logger.warning(
                "step.embed.image_failed_fallback_text",
                chunk_id=chunk.chunk_id,
                error=str(err),
            )
            # Fallback: embed the text summary of the image chunk
            fallback_vecs = await loop.run_in_executor(
                None, embedder.embed_batch, [chunk.content]
            )
            vectors[chunk_idx] = fallback_vecs[0]
            c_hash = chunk.content_hash or ""
            if c_hash:
                new_cache_entries[c_hash] = fallback_vecs[0]

    # ── 7. Save new embeddings to cache ───────────────────────────────────────
    if new_cache_entries:
        await EmbeddingCache.set_batch(
            postgres=postgres,
            hash_vector_map=new_cache_entries,
            model=model_name,
            model_version=model_version,
        )

    # ── 8. Validate all positions filled ──────────────────────────────────────
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
        new_text_embedded=len(text_miss_indices),
        new_image_embedded=len(image_miss_indices),
        new_cache_saved=len(new_cache_entries),
    )

    return state.model_copy(
        update={
            "vectors": final_vectors,
            "status": DocumentStatus.EMBEDDING,
        }
    )


async def _set_quota_status(postgres: PostgresClient, state: IngestionState) -> None:
    """Update document and job status to WAITING_FOR_EMBEDDING_QUOTA."""
    await postgres.execute(
        "UPDATE documents SET status = $1 WHERE id = $2",
        DocumentStatus.WAITING_FOR_EMBEDDING_QUOTA.value,
        state.document_id,
    )
    if state.job_id:
        await postgres.execute(
            "UPDATE ingestion_jobs SET status = $1 WHERE job_id = $2",
            JobStatus.WAITING_FOR_EMBEDDING_QUOTA.value,
            state.job_id,
        )
