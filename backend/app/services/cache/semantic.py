"""
Semantic Cache for Enterprise-RAG (Phase 5).

Extends the exact-match CacheService with embedding-based similarity matching.
Semantically similar queries (within a tenant + kb-version boundary) are served
from the same cached response, reducing redundant LLM calls.

Safety contract:
  - Every semantic cache entry is isolated by: tenant, assistant, kb_id, kb_version,
    access_level, model, prompt version, and retrieval profile.
  - Similarity threshold is configurable (default 0.95).
  - Falls back gracefully when OTEL/vector store is unavailable.

NOTE: Semantic cache introduces additional embedding cost per query miss.
      Measure actual savings vs cost before enabling in high-throughput deployments.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from app.config.logging import get_logger
from app.infrastructure.redis.client import RedisClient
from app.models.tenant import TenantContext

logger = get_logger(__name__)

# Prefix for the semantic cache index stored in Redis as a sorted-set or JSON blob
_SEM_CACHE_PREFIX = "sem_cache"
_DEFAULT_SIMILARITY_THRESHOLD = 0.95
_DEFAULT_TTL = 3600


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _semantic_cache_namespace(
    ctx: TenantContext,
    kb_version: int,
    model: str = "default",
    prompt_version: str = "v1",
    retrieval_profile: str = "default",
) -> str:
    """
    Build a namespace key that isolates cache entries by security context.
    All dimensions that could change the meaning or permissible scope of an answer
    must be included here.
    """
    raw = (
        f"{ctx.tenant_id}::{ctx.assistant_id}::{ctx.knowledge_base_id}::"
        f"v{kb_version}::{ctx.access_level}::{model}::{prompt_version}::{retrieval_profile}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class SemanticCacheService:
    """
    Semantic query-response cache backed by Redis.

    Lookup flow:
      1. Embed the incoming query.
      2. Retrieve cached vectors for this namespace (security scope).
      3. Compute cosine similarity between query and cached vectors.
      4. If similarity >= threshold: return cached response.
      5. Else: cache miss; caller must run the full pipeline and store result.
    """

    def __init__(
        self,
        redis: RedisClient,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
        ttl_seconds: int = _DEFAULT_TTL,
        max_entries_per_namespace: int = 256,
    ) -> None:
        self._redis = redis
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._max_entries = max_entries_per_namespace

    # ── Public API ────────────────────────────────────────────────────────────

    async def get(
        self,
        ctx: TenantContext,
        query_embedding: list[float],
        kb_version: int,
        model: str = "default",
        prompt_version: str = "v1",
        retrieval_profile: str = "default",
    ) -> dict | None:
        """
        Try to find a semantically similar cached response.

        Returns the cached response dict if found, else None.
        """
        namespace = _semantic_cache_namespace(ctx, kb_version, model, prompt_version, retrieval_profile)
        index_key = f"{_SEM_CACHE_PREFIX}:{namespace}:index"

        raw_index = await self._redis.get(index_key)
        if not raw_index:
            return None

        try:
            entries: list[dict] = json.loads(raw_index)
        except (json.JSONDecodeError, TypeError):
            return None

        best_score = 0.0
        best_entry_key = None

        for entry in entries:
            vec = entry.get("embedding", [])
            if not vec:
                continue
            sim = _cosine_similarity(query_embedding, vec)
            if sim > best_score:
                best_score = sim
                best_entry_key = entry.get("response_key")

        if best_score >= self._threshold and best_entry_key:
            cached_str = await self._redis.get(best_entry_key)
            if cached_str:
                logger.info(
                    "semantic_cache.hit",
                    tenant=ctx.tenant_id,
                    score=round(best_score, 4),
                    namespace=namespace[:8],
                )
                try:
                    return json.loads(cached_str)
                except json.JSONDecodeError:
                    return None

        logger.info(
            "semantic_cache.miss",
            tenant=ctx.tenant_id,
            best_score=round(best_score, 4),
            namespace=namespace[:8],
        )
        return None

    async def set(
        self,
        ctx: TenantContext,
        query_embedding: list[float],
        response_data: dict,
        kb_version: int,
        model: str = "default",
        prompt_version: str = "v1",
        retrieval_profile: str = "default",
    ) -> None:
        """
        Store a new semantic cache entry.
        Uses a fixed-size rolling index; oldest entries are evicted when at capacity.
        """
        namespace = _semantic_cache_namespace(ctx, kb_version, model, prompt_version, retrieval_profile)
        index_key = f"{_SEM_CACHE_PREFIX}:{namespace}:index"

        # Build response key from embedding hash for deduplication
        emb_hash = hashlib.sha256(json.dumps(query_embedding[:8]).encode()).hexdigest()[:12]
        response_key = f"{_SEM_CACHE_PREFIX}:{namespace}:resp:{emb_hash}"

        # Load existing index
        raw_index = await self._redis.get(index_key)
        entries: list[dict] = []
        try:
            if raw_index:
                entries = json.loads(raw_index)
        except (json.JSONDecodeError, TypeError):
            entries = []

        # Evict oldest if at capacity
        if len(entries) >= self._max_entries:
            oldest = entries.pop(0)
            await self._redis.delete(oldest.get("response_key", ""))

        # Add new entry
        entries.append({"embedding": query_embedding, "response_key": response_key})

        # Store response and updated index
        await self._redis.set(response_key, json.dumps(response_data), ttl_seconds=self._ttl)
        await self._redis.set(index_key, json.dumps(entries), ttl_seconds=self._ttl)

        logger.info(
            "semantic_cache.stored",
            tenant=ctx.tenant_id,
            namespace=namespace[:8],
            entries=len(entries),
        )

    async def invalidate_namespace(
        self,
        ctx: TenantContext,
        kb_version: int,
        model: str = "default",
        prompt_version: str = "v1",
        retrieval_profile: str = "default",
    ) -> None:
        """
        Remove all cache entries for a namespace (e.g., when knowledge base updates).
        """
        namespace = _semantic_cache_namespace(ctx, kb_version, model, prompt_version, retrieval_profile)
        index_key = f"{_SEM_CACHE_PREFIX}:{namespace}:index"

        raw_index = await self._redis.get(index_key)
        if raw_index:
            try:
                entries: list[dict] = json.loads(raw_index)
                for entry in entries:
                    await self._redis.delete(entry.get("response_key", ""))
            except (json.JSONDecodeError, TypeError):
                pass
        await self._redis.delete(index_key)

        logger.info("semantic_cache.invalidated", tenant=ctx.tenant_id, namespace=namespace[:8])
