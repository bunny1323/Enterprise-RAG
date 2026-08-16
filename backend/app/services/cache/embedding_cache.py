"""
PostgreSQL-backed Embedding Cache.
Key: SHA-256(content_hash + embedding_model + embedding_model_version)
Stores pre-computed Voyage vectors to avoid expensive duplicate API calls.
"""
import hashlib
import json
from typing import Any

from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient

logger = get_logger(__name__)


def make_cache_key(content_hash: str, model: str, model_version: str) -> str:
    """Generate deterministic embedding cache key."""
    raw = f"{content_hash}::{model}::{model_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """
    Stateless wrapper for getting and setting pre-computed chunk embeddings.
    """

    @staticmethod
    async def get_batch(
        postgres: PostgresClient,
        content_hashes: list[str],
        model: str,
        model_version: str,
    ) -> dict[str, list[float]]:
        """
        Query embedding_cache table for matching vectors.

        Returns:
            Dict mapping content_hash -> 1024-dim float vector.
        """
        if not content_hashes:
            return {}

        keys_map = {
            make_cache_key(ch, model, model_version): ch
            for ch in content_hashes
        }
        keys = list(keys_map.keys())

        rows = await postgres.fetch(
            """
            SELECT cache_key, vector
            FROM embedding_cache
            WHERE cache_key = ANY($1::text[])
            """,
            keys,
        )

        hits: dict[str, list[float]] = {}
        for row in rows:
            c_key = row["cache_key"]
            vec_raw = row["vector"]
            ch = keys_map.get(c_key)
            if ch:
                if isinstance(vec_raw, str):
                    hits[ch] = json.loads(vec_raw)
                elif isinstance(vec_raw, list):
                    hits[ch] = vec_raw

        logger.info(
            "embed_cache.get_batch",
            requested=len(content_hashes),
            hits=len(hits),
            misses=len(content_hashes) - len(hits),
        )
        return hits

    @staticmethod
    async def set_batch(
        postgres: PostgresClient,
        hash_vector_map: dict[str, list[float]],
        model: str,
        model_version: str,
    ) -> None:
        """Store newly computed embeddings into PostgreSQL cache."""
        if not hash_vector_map:
            return

        for ch, vector in hash_vector_map.items():
            cache_key = make_cache_key(ch, model, model_version)
            await postgres.execute(
                """
                INSERT INTO embedding_cache (cache_key, content_hash, embedding_model, embedding_model_version, vector, created_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (cache_key) DO NOTHING
                """,
                cache_key,
                ch,
                model,
                model_version,
                json.dumps(vector),
            )

        logger.info("embed_cache.set_batch", stored=len(hash_vector_map))
