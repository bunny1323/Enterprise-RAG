from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.agents.supervisor.state import IngestionState
from app.infrastructure.postgres.client import PostgresClient
from app.models.chunk import Chunk
from app.models.document import DocumentStatus
from app.models.job import JobStatus
from app.pipelines.ingestion import pipeline as pipeline_module
from app.pipelines.ingestion.pipeline import IngestionPipeline, PartialIndexingError


class _Transaction:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    async def __aenter__(self):
        self.connection.transaction_started = True

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.connection.rolled_back = exc_type is not None


class _Acquire:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class _Connection:
    def __init__(self) -> None:
        self.document_id = uuid4()
        self.job_id = uuid4()
        self.transaction_started = False
        self.rolled_back = False
        self.fail_job_insert = False
        self.calls: list[tuple[str, tuple]] = []

    def transaction(self) -> _Transaction:
        return _Transaction(self)

    async def execute(self, query: str, *args, **kwargs):
        self.calls.append((query, args))

    async def fetchrow(self, query: str, *args):
        self.calls.append((query, args))
        if "INSERT INTO documents" in query:
            return {"id": self.document_id}
        if "INSERT INTO ingestion_jobs" in query:
            if self.fail_job_insert:
                raise RuntimeError("job insert failed")
            return {"job_id": self.job_id}
        if "WITH candidate AS" in query:
            return {"job_id": self.job_id, "document_id": self.document_id}
        if "UPDATE ingestion_jobs" in query and "RETURNING status" in query:
            return {"status": "RETRY_WAIT"}
        return None


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self) -> _Acquire:
        return _Acquire(self.connection)


@pytest.mark.asyncio
async def test_document_and_job_registration_share_one_transaction():
    connection = _Connection()
    postgres = PostgresClient("postgresql://unused")
    postgres._pool = _Pool(connection)

    result = await postgres.register_ingestion(
        sha256="pending_test",
        file_name="manual.pdf",
        storage_path="/data/raw/manual.pdf",
        industry="manufacturing",
        tenant_id="tenant-a",
        assistant_id="assistant-a",
        knowledge_base_id="kb-a",
        max_attempts=4,
    )

    assert result == {
        "document_id": connection.document_id,
        "job_id": connection.job_id,
    }
    assert connection.transaction_started
    assert not connection.rolled_back
    assert len(connection.calls) == 2
    assert connection.calls[1][1][-1] == 4


@pytest.mark.asyncio
async def test_failed_job_insert_rolls_back_document_registration():
    connection = _Connection()
    connection.fail_job_insert = True
    postgres = PostgresClient("postgresql://unused")
    postgres._pool = _Pool(connection)

    with pytest.raises(RuntimeError, match="job insert failed"):
        await postgres.register_ingestion(
            sha256="pending_test",
            file_name="manual.pdf",
            storage_path="/data/raw/manual.pdf",
            industry="manufacturing",
            tenant_id="tenant-a",
            assistant_id="assistant-a",
            knowledge_base_id="kb-a",
        )

    assert connection.transaction_started
    assert connection.rolled_back


@pytest.mark.asyncio
async def test_job_claim_uses_skip_locked_and_persists_a_lease():
    connection = _Connection()
    postgres = PostgresClient("postgresql://unused")
    postgres._pool = _Pool(connection)

    job = await postgres.claim_ingestion_job("worker-a", lease_seconds=90)

    assert job["job_id"] == connection.job_id
    assert connection.transaction_started
    claim_query, args = connection.calls[-1]
    assert "FOR UPDATE SKIP LOCKED" in claim_query
    assert "lease_expires_at" in claim_query
    assert "'PARTIAL'" in claim_query
    assert "'COMPLETED', 'PARTIAL', 'DUPLICATE'" not in claim_query
    assert args == ("worker-a", 90)


@pytest.mark.asyncio
async def test_failed_job_attempt_is_requeued_with_a_bounded_retry_policy():
    connection = _Connection()
    postgres = PostgresClient("postgresql://unused")
    postgres._pool = _Pool(connection)

    retrying = await postgres.record_ingestion_failure(
        connection.job_id,
        connection.document_id,
        "worker-a",
        "temporary parser failure",
    )

    assert retrying is True
    assert connection.transaction_started
    failure_query, args = connection.calls[0]
    assert "attempt_count < max_attempts" in failure_query
    assert args[-1] is False
    assert any("UPDATE documents" in query for query, _ in connection.calls)


@pytest.mark.asyncio
async def test_partial_indexing_result_is_signaled_for_worker_retry(monkeypatch):
    async def partial_step(state, services):
        return state.model_copy(
            update={
                "status": DocumentStatus.PARTIAL,
                "error_message": "Neo4j unavailable",
            }
        )

    postgres = MagicMock()
    postgres.execute = AsyncMock()
    postgres.fetchrow = AsyncMock(return_value=None)
    monkeypatch.setattr(pipeline_module, "_PIPELINE_STEPS", [
        ("partial", partial_step, 100, JobStatus.INDEXING),
    ])
    monkeypatch.setattr(
        "app.config.settings.get_settings",
        lambda: SimpleNamespace(
            ingestion_timeout_partial_override=1,
            ingestion_timeout_partial=1,
        ),
    )
    state = IngestionState(
        document_id=uuid4(),
        job_id=uuid4(),
        filename="manual.pdf",
        storage_path="/data/raw/manual.pdf",
    )
    pipeline = IngestionPipeline(services={"postgres": postgres})

    with pytest.raises(PartialIndexingError, match="Neo4j unavailable"):
        await pipeline.run(state)


@pytest.mark.asyncio
async def test_partial_indexing_invalidates_cached_knowledge_base_answers():
    async def postgres_execute(query, *args, **kwargs):
        if "INSERT INTO chunks" in query:
            raise RuntimeError("temporary PostgreSQL write failure")
        return "OK"

    postgres = MagicMock()
    postgres.execute = AsyncMock(side_effect=postgres_execute)
    weaviate = MagicMock()
    neo4j = MagicMock()
    neo4j.create_document_tree = AsyncMock()
    document_id = uuid4()
    chunk = Chunk(
        chunk_id=f"{document_id}::p1::s0",
        document_id=document_id,
        content="Evidence",
        page_number=1,
    )
    state = IngestionState(
        document_id=document_id,
        job_id=uuid4(),
        filename="manual.pdf",
        storage_path="/data/raw/manual.pdf",
        chunks=[chunk],
        vectors=[[0.1, 0.2]],
    )
    index_step = importlib.import_module(
        "app.pipelines.ingestion.steps.s08_index"
    )

    result = await index_step.step(
        state,
        {"postgres": postgres, "weaviate": weaviate, "neo4j": neo4j},
    )

    assert result.status == DocumentStatus.PARTIAL
    version_updates = [
        call for call in postgres.execute.await_args_list
        if "knowledge_base_versions" in call.args[0]
    ]
    assert len(version_updates) == 1
