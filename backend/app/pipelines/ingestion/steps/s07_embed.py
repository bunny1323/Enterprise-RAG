import asyncio
import time
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
    Generate embeddings for all chunks using Voyage API with PostgreSQL embedding cache.

    Steps:
    1. Check EmbeddingCache for existing vectors by chunk content_hash.
    2. Submit cache misses to Voyage API in batches.
    3. On HTTP 429: set status to WAITING_FOR_EMBEDDING_QUOTA, sleep Retry-After, retry.
    4. Save new vectors to EmbeddingCache.
    """
    logger.info("step.embed.start", document_id=str(state.document_id))

    if not state.chunks:
        raise ValueError("Cannot embed: chunks list is empty or None")

    embedder: EmbeddingService = services["embedder"]
    postgres: PostgresClient = services["postgres"]
    model_name = state.embedding_model
    model_version = state.embedding_model_version

    content_hashes = [c.content_hash or "" for c in state.chunks if c.content_hash]

    # 1. Query EmbeddingCache for hits
    cache_hits = await EmbeddingCache.get_batch(
        postgres=postgres,
        content_hashes=content_hashes,
        model=model_name,
        model_version=model_version,
    )

    logger.info(
        "step.embed.cache_checked",
        total_chunks=len(state.chunks),
        hits=len(cache_hits),
        misses=len(state.chunks) - len(cache_hits),
    )

    vectors: list[list[float]] = []
    new_cache_entries: dict[str, list[float]] = {}
    loop = asyncio.get_event_loop()

    for idx, chunk in enumerate(state.chunks):
        c_hash = chunk.content_hash or ""

        # Hit from cache
        if c_hash in cache_hits:
            vectors.append(cache_hits[c_hash])
            continue

        # Cache Miss: call Voyage API
        try:
            image_path = chunk.metadata.get("image_path") if chunk.metadata else None

            if chunk.embedding_representation == "image" and image_path:
                vector = await loop.run_in_executor(None, embedder.embed_image, image_path)
            else:
                batch_vecs = await loop.run_in_executor(None, embedder.embed_batch, [chunk.content])
                vector = batch_vecs[0]

            vectors.append(vector)
            if c_hash:
                new_cache_entries[c_hash] = vector

        except RateLimitQuotaError as quota_err:
            logger.warning(
                "step.embed.quota_exceeded",
                document_id=str(state.document_id),
                retry_after=quota_err.retry_after,
            )

            # Update status to WAITING_FOR_EMBEDDING_QUOTA
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

            # Sleep Retry-After and retry this chunk
            await asyncio.sleep(quota_err.retry_after)

            batch_vecs = await loop.run_in_executor(None, embedder.embed_batch, [chunk.content])
            vector = batch_vecs[0]
            vectors.append(vector)
            if c_hash:
                new_cache_entries[c_hash] = vector

    # Save new embeddings to cache
    if new_cache_entries:
        await EmbeddingCache.set_batch(
            postgres=postgres,
            hash_vector_map=new_cache_entries,
            model=model_name,
            model_version=model_version,
        )

    if len(vectors) != len(state.chunks):
        raise ValueError(f"Vector count mismatch: {len(vectors)} vectors for {len(state.chunks)} chunks")

    logger.info(
        "step.embed.complete",
        document_id=str(state.document_id),
        total_vectors=len(vectors),
        cache_hits=len(cache_hits),
        new_embedded=len(new_cache_entries),
    )

    return state.model_copy(
        update={
            "vectors": vectors,
            "status": DocumentStatus.EMBEDDING,
        }
    )

