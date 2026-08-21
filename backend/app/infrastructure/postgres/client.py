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
_BASE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id                    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    sha256                TEXT        NOT NULL,
    file_name             TEXT        NOT NULL,
    storage_path          TEXT        NOT NULL,
    industry              TEXT        NOT NULL DEFAULT 'manufacturing',
    tenant_id             TEXT        NOT NULL DEFAULT 'default',
    assistant_id          TEXT        NOT NULL DEFAULT 'default',
    knowledge_base_id     TEXT        NOT NULL DEFAULT 'default',
    content_hash          TEXT,
    version               INTEGER     DEFAULT 1,
    canonical_document_id UUID        REFERENCES documents(id),
    supersedes            UUID        REFERENCES documents(id),
    parser_version        TEXT,
    embedding_model       TEXT,
    embedding_model_version TEXT,
    page_count            INTEGER,
    status                TEXT        NOT NULL DEFAULT 'PENDING',
    progress_percent      INTEGER     DEFAULT 0,
    metadata              JSONB       DEFAULT '{}',
    created_at            TIMESTAMPTZ DEFAULT NOW(),
    completed_at          TIMESTAMPTZ,
    error_message         TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id              TEXT        PRIMARY KEY,
    parent_id             TEXT,
    document_id           UUID        REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id             TEXT        NOT NULL DEFAULT 'default',
    assistant_id          TEXT        NOT NULL DEFAULT 'default',
    knowledge_base_id     TEXT        NOT NULL DEFAULT 'default',
    content               TEXT        NOT NULL,
    content_hash          TEXT,
    section               TEXT,
    subsection            TEXT,
    context_prefix        TEXT,
    embedding_representation TEXT     DEFAULT 'text',
    page_number           INTEGER,
    bounding_box          JSONB,
    chunk_type            TEXT,
    access_classification TEXT        DEFAULT 'INTERNAL',
    industry_domain       TEXT        DEFAULT 'manufacturing',
    hierarchy_path        TEXT,
    metadata              JSONB       DEFAULT '{}',
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id           UUID        REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id             TEXT        NOT NULL DEFAULT 'default',
    assistant_id          TEXT        NOT NULL DEFAULT 'default',
    knowledge_base_id     TEXT        NOT NULL DEFAULT 'default',
    status                TEXT        NOT NULL DEFAULT 'RECEIVED',
    current_stage         TEXT,
    progress_percent      INTEGER     DEFAULT 0,
    retry_count           INTEGER     DEFAULT 0,
    last_successful_stage TEXT,
    stage_checkpoints     JSONB       DEFAULT '{}',
    error_message         TEXT,
    created_at            TIMESTAMPTZ DEFAULT NOW(),
    started_at            TIMESTAMPTZ,
    completed_at          TIMESTAMPTZ,
    timeout_at            TIMESTAMPTZ,
    cancelled_at          TIMESTAMPTZ,
    metadata              JSONB       DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
    job_id                UUID        PRIMARY KEY REFERENCES ingestion_jobs(job_id) ON DELETE CASCADE,
    last_successful_stage TEXT,
    stage_data            JSONB       DEFAULT '{}',
    retry_count           INTEGER     DEFAULT 0,
    updated_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS indexing_state (
    document_id           UUID        PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    weaviate_status       TEXT        DEFAULT 'PENDING',
    neo4j_status          TEXT        DEFAULT 'PENDING',
    postgres_chunks_status TEXT       DEFAULT 'PENDING',
    weaviate_chunk_count  INTEGER     DEFAULT 0,
    neo4j_node_count      INTEGER     DEFAULT 0,
    last_error            TEXT,
    updated_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    cache_key             TEXT        PRIMARY KEY,
    content_hash          TEXT        NOT NULL,
    embedding_model       TEXT        NOT NULL,
    embedding_model_version TEXT      NOT NULL,
    vector                JSONB       NOT NULL,
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_base_versions (
    knowledge_base_id     TEXT        NOT NULL,
    tenant_id             TEXT        NOT NULL,
    version               INTEGER     NOT NULL DEFAULT 1,
    document_count        INTEGER     DEFAULT 0,
    chunk_count           INTEGER     DEFAULT 0,
    updated_at            TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (knowledge_base_id, tenant_id)
);
"""

_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_docs_sha256 ON documents(sha256);
CREATE INDEX IF NOT EXISTS idx_docs_status  ON documents(status);
CREATE INDEX IF NOT EXISTS idx_docs_tenant  ON documents(tenant_id, knowledge_base_id);
CREATE INDEX IF NOT EXISTS idx_docs_content_hash ON documents(content_hash);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type     ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant   ON chunks(tenant_id, knowledge_base_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);

CREATE INDEX IF NOT EXISTS idx_jobs_document ON ingestion_jobs(document_id);
CREATE INDEX IF NOT EXISTS idx_jobs_tenant   ON ingestion_jobs(tenant_id, knowledge_base_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status   ON ingestion_jobs(status);

CREATE INDEX IF NOT EXISTS idx_embed_cache_hash ON embedding_cache(content_hash, embedding_model);
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
        """Create tables and indexes safely, supporting older existing schemas.
        
        The execution order is critical:
        1. Initialize base tables.
        2. Apply column migrations (adding missing columns to old tables).
        3. Create indexes that may depend on the newly added columns.
        """
        await self._initialize_base_schema()
        await self._apply_column_migrations()
        await self._apply_indexes_and_constraints()
        logger.info("postgres.schema_initialized")
        
    async def _initialize_base_schema(self) -> None:
        """Create tables if they do not exist."""
        await self.execute(_BASE_TABLES_SQL)

    async def _apply_indexes_and_constraints(self) -> None:
        """Create indexes and constraints, which may depend on migrated columns."""
        await self.execute(_INDEXES_SQL)

    async def _apply_column_migrations(self) -> None:
        """
        Idempotent column migrations using ALTER TABLE ... ADD COLUMN IF NOT EXISTS.

        Runs after CREATE TABLE IF NOT EXISTS so existing tables from earlier
        schema versions get all new columns without data loss.
        """
        migrations = [
            # ── documents table ───────────────────────────────────────────────
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS assistant_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS knowledge_base_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS canonical_document_id UUID",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS supersedes UUID",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS parser_version TEXT",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_model TEXT",
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_model_version TEXT",
            # ── chunks table ─────────────────────────────────────────────────
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS assistant_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS knowledge_base_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_representation TEXT DEFAULT 'text'",
            # ── ingestion_jobs table ─────────────────────────────────────────
            "ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS assistant_id TEXT NOT NULL DEFAULT 'default'",
            "ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS knowledge_base_id TEXT NOT NULL DEFAULT 'default'",
        ]
        for sql in migrations:
            try:
                await self.execute(sql)
            except Exception as err:
                # Log but don't fail — some may already exist or have different types
                logger.warning("postgres.migration_warning", sql=sql[:80], error=str(err))
        logger.info("postgres.column_migrations_applied")

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
