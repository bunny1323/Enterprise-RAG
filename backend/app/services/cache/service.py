"""
Security-aware RAG Cache Service.
Manages caching for retrieval queries, query embeddings, and LLM responses.

Cache keys incorporate tenant_id, assistant_id, knowledge_base_id,
knowledge_base_version, user access_level, and query hash to ensure
cache isolation and prevent unauthorized cross-context data leaks.
"""
import hashlib
import json
from typing import Any

from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
from app.infrastructure.redis.client import RedisClient
from app.models.tenant import TenantContext

logger = get_logger(__name__)


def make_security_cache_key(
    ctx: TenantContext,
    kb_version: int,
    query: str,
    prefix: str = "query_res",
) -> str:
    """
    Build a security-isolated cache key.
    Includes tenant, kb_version, access_level, and query hash.
    """
    query_hash = hashlib.sha256(query.lower().strip().encode("utf-8")).hexdigest()[:16]
    raw = (
        f"{prefix}::{ctx.tenant_id}::{ctx.assistant_id}::{ctx.knowledge_base_id}::"
        f"v{kb_version}::{ctx.access_level}::{query_hash}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CacheService:
    """
    Service for checking and storing security-isolated RAG responses.
    """

    def __init__(self, redis: RedisClient, postgres: PostgresClient) -> None:
        self._redis = redis
        self._postgres = postgres

    async def get_kb_version(self, tenant_id: str, kb_id: str) -> int:
        """Fetch current version for a knowledge base."""
        row = await self._postgres.fetchrow(
            """
            SELECT version FROM knowledge_base_versions
            WHERE tenant_id = $1 AND knowledge_base_id = $2
            """,
            tenant_id,
            kb_id,
        )
        return row["version"] if row else 1

    async def get_response(self, ctx: TenantContext, query: str) -> dict | None:
        """Get cached response if present for this exact security context."""
        kb_ver = await self.get_kb_version(ctx.tenant_id, ctx.knowledge_base_id)
        cache_key = make_security_cache_key(ctx, kb_ver, query, prefix="rag_resp")

        cached_str = await self._redis.get(cache_key)
        if cached_str:
            logger.info("rag_cache.hit", tenant=ctx.tenant_id, query=query[:30])
            try:
                return json.loads(cached_str)
            except json.JSONDecodeError:
                return None

        logger.info("rag_cache.miss", tenant=ctx.tenant_id, query=query[:30])
        return None

    async def set_response(
        self, ctx: TenantContext, query: str, response_data: dict, ttl_seconds: int = 3600
    ) -> None:
        """Cache response under this exact security context."""
        kb_ver = await self.get_kb_version(ctx.tenant_id, ctx.knowledge_base_id)
        cache_key = make_security_cache_key(ctx, kb_ver, query, prefix="rag_resp")

        await self._redis.set(cache_key, json.dumps(response_data), ttl_seconds=ttl_seconds)
        logger.info("rag_cache.stored", tenant=ctx.tenant_id, ttl=ttl_seconds)

    async def get_query_embedding(self, ctx: TenantContext, query: str) -> list[float] | None:
        """Get cached embedding vector for query string."""
        kb_ver = await self.get_kb_version(ctx.tenant_id, ctx.knowledge_base_id)
        cache_key = make_security_cache_key(ctx, kb_ver, query, prefix="query_vec")

        cached_str = await self._redis.get(cache_key)
        if cached_str:
            try:
                return json.loads(cached_str)
            except json.JSONDecodeError:
                return None
        return None

    async def set_query_embedding(
        self, ctx: TenantContext, query: str, vector: list[float], ttl_seconds: int = 86400
    ) -> None:
        """Cache embedding vector for query string."""
        kb_ver = await self.get_kb_version(ctx.tenant_id, ctx.knowledge_base_id)
        cache_key = make_security_cache_key(ctx, kb_ver, query, prefix="query_vec")

        await self._redis.set(cache_key, json.dumps(vector), ttl_seconds=ttl_seconds)
