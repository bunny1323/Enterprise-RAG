"""
Redis infrastructure client using redis-py async client.
Provides short-term caching for queries, query embeddings, and LLM responses.
"""
from typing import Any

import redis.asyncio as redis

from app.config.logging import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Async Redis client wrapper."""

    def __init__(self, redis_url: str) -> None:
        self._url = redis_url
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis server."""
        try:
            self._client = redis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5.0,
            )
            await self._client.ping()
            logger.info("redis.connected", url=self._url)
        except Exception as err:
            logger.warning("redis.connect_failed", error=str(err), url=self._url)
            self._client = None

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.aclose()
            logger.info("redis.closed")

    def is_connected(self) -> bool:
        return self._client is not None

    async def get(self, key: str) -> str | None:
        """Get cached string value by key."""
        if not self._client:
            return None
        try:
            val = await self._client.get(key)
            return str(val) if val is not None else None
        except Exception as err:
            logger.warning("redis.get_error", key=key, error=str(err))
            return None

    async def set(self, key: str, value: str, ttl_seconds: int = 3600) -> None:
        """Set cached string value with TTL."""
        if not self._client:
            return
        try:
            await self._client.set(key, value, ex=ttl_seconds)
        except Exception as err:
            logger.warning("redis.set_error", key=key, error=str(err))

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        if not self._client:
            return
        try:
            await self._client.delete(key)
        except Exception as err:
            logger.warning("redis.delete_error", key=key, error=str(err))
