"""
PostgreSQL infrastructure client using asyncpg.
Manages connection pool, schema initialization, and typed query helpers.
"""
import json
from typing import Any

import asyncpg
from asyncpg import Pool, Record

from app.config.logging import get_logger

logger = get_logger(__name__)

# ── DDL ────────────────────────────────────────────────────────────────────────
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    sha256           TEXT        UNIQUE NOT NULL,
    file_name        TEXT        NOT NULL,
    storage_path     TEXT        NOT NULL,
    industry         TEXT        NOT NULL DEFAULT 'manufacturing',
    page_count       INTEGER,
    status           TEXT        NOT NULL DEFAULT 'PENDING',
    progress_percent INTEGER     DEFAULT 0,
    metadata         JSONB       DEFAULT '{}',
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    completed_at     TIMESTAMPTZ,
    error_message    TEXT
);

CREATE INDEX IF NOT EXISTS idx_docs_sha256 ON documents(sha256);
CREATE INDEX IF NOT EXISTS idx_docs_status  ON documents(status);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id              TEXT        PRIMARY KEY,
    parent_id             TEXT,
    document_id           UUID        REFERENCES documents(id) ON DELETE CASCADE,
    content               TEXT        NOT NULL,
    page_number           INTEGER,
    bounding_box          JSONB,
    chunk_type            TEXT,
    access_classification TEXT        DEFAULT 'INTERNAL',
    industry_domain       TEXT        DEFAULT 'manufacturing',
    hierarchy_path        TEXT,
    metadata              JSONB       DEFAULT '{}',
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type     ON chunks(chunk_type);
"""


class PostgresClient:
    """Async PostgreSQL client backed by an asyncpg connection pool."""

    def __init__(self, database_url: str) -> None:
        # asyncpg requires postgresql:// not postgres://
        self._url = database_url.replace("postgres://", "postgresql://", 1)
        self._pool: Pool | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def init_pool(self, min_size: int = 2, max_size: int = 10) -> None:
        """Create the connection pool. Call once at application startup."""
        self._pool = await asyncpg.create_pool(
            self._url,
            min_size=min_size,
            max_size=max_size,
            command_timeout=30,
            # Use custom codec so JSONB columns return dicts, not strings
            init=self._init_connection,
        )
        logger.info("postgres.pool_created", min=min_size, max=max_size)

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        """Register JSON/JSONB codecs on each new connection."""
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )
        await conn.set_type_codec(
            "json",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

    async def init_schema(self) -> None:
        """Create tables and indexes if they do not already exist."""
        await self.execute(_SCHEMA_SQL)
        logger.info("postgres.schema_initialized")

    async def close(self) -> None:
        """Drain and close the connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("postgres.pool_closed")

    # ── Query helpers ──────────────────────────────────────────────────────────

    @property
    def pool(self) -> Pool:
        if self._pool is None:
            raise RuntimeError("PostgresClient.init_pool() must be called first")
        return self._pool

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a write query (INSERT / UPDATE / DELETE / DDL). Returns status string."""
        async with self.pool.acquire() as conn:
            result: str = await conn.execute(query, *args)
            return result

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        """Execute a SELECT and return all rows as plain dicts."""
        async with self.pool.acquire() as conn:
            rows: list[Record] = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        """Execute a SELECT and return the first row as a plain dict, or None."""
        async with self.pool.acquire() as conn:
            row: Record | None = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Execute a SELECT and return the first column of the first row."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)
