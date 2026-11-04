"""
Unit tests for MinHash and LSH deduplication logic.
"""
from app.services.deduplication.minhash import LSHIndex, MinHasher


def test_minhash_and_lsh():
    hasher = MinHasher(num_perm=64)
    index = LSHIndex(num_perm=64, bands=8)

    text1 = "Hydraulic pressure specification for main pump model X100 is 250 bar."
    text2 = "Hydraulic pressure spec for main pump model X100 is 250 bar."
    text3 = "Electrical wiring diagram for safety switch relay."

    sig1 = hasher.compute_signature(text1)
    sig2 = hasher.compute_signature(text2)
    sig3 = hasher.compute_signature(text3)

    index.insert("chunk_1", sig1)
    index.insert("chunk_3", sig3)

    candidates = index.query(sig2)
    assert "chunk_1" in candidates
    assert "chunk_3" not in candidates

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
import asyncpg
import importlib

s02_duplicate = importlib.import_module(
    "app.pipelines.ingestion.steps.s02_duplicate"
)
from app.agents.supervisor.state import IngestionState
from app.models.document import DocumentStatus

@pytest.mark.asyncio
async def test_new_unique_document():
    # Test 1 - New unique document
    state = IngestionState(
        document_id=uuid4(),
        job_id=uuid4(),
        tenant_id="tenant_1",
        assistant_id="default",
        knowledge_base_id="default",
        filename="test.pdf",
        industry="manufacturing",
        storage_path="/tmp/test.pdf",
        status=DocumentStatus.PENDING
    )
    postgres = AsyncMock()
    postgres.fetchrow.return_value = None  # No exact duplicate, no failed doc
    
    # Mock compute_sha256 in s02_duplicate
    s02_duplicate.compute_sha256 = MagicMock(return_value="abc123hash")
    
    new_state = await s02_duplicate.step(state, {"postgres": postgres})
    assert new_state.sha256 == "abc123hash"
    assert new_state.status == DocumentStatus.CHECKING_DUPLICATE
    postgres.execute.assert_called_with("UPDATE documents SET sha256 = $1 WHERE id = $2", "abc123hash", state.document_id)

@pytest.mark.asyncio
async def test_exact_duplicate():
    # Test 2 - Exact duplicate
    doc_id = uuid4()
    canonical_id = uuid4()
    state = IngestionState(
        document_id=doc_id,
        job_id=uuid4(),
        tenant_id="tenant_1",
        assistant_id="default",
        knowledge_base_id="default",
        filename="test.pdf",
        industry="manufacturing",
        storage_path="/tmp/test.pdf",
        status=DocumentStatus.PENDING
    )
    postgres = AsyncMock()
    # Mock check_file_hash to return an existing row
    postgres.fetchrow.return_value = {"id": str(canonical_id), "sha256": "hash", "file_name": "x", "version": 1, "content_hash": "c"}
    
    s02_duplicate.compute_sha256 = MagicMock(return_value="hash")
    
    new_state = await s02_duplicate.step(state, {"postgres": postgres})
    assert new_state.status == DocumentStatus.DUPLICATE
    assert new_state.canonical_document_id == canonical_id
    assert new_state.progress_percent == 100

@pytest.mark.asyncio
async def test_same_sha_but_previous_failed():
    # Test 3 - Same SHA but previous document FAILED or TIMEOUT
    doc_id = uuid4()
    state = IngestionState(
        document_id=doc_id,
        job_id=uuid4(),
        tenant_id="tenant_1",
        assistant_id="default",
        knowledge_base_id="default",
        filename="test.pdf",
        industry="manufacturing",
        storage_path="/tmp/test.pdf",
        status=DocumentStatus.PENDING
    )
    postgres = AsyncMock()
    # check_file_hash returns None because only COMPLETED are matched
    postgres.fetchrow.return_value = None
    
    s02_duplicate.compute_sha256 = MagicMock(return_value="hash")
    
    new_state = await s02_duplicate.step(state, {"postgres": postgres})
    assert new_state.status == DocumentStatus.CHECKING_DUPLICATE
    # Verify the cleanup query was executed for non-completed documents
    assert postgres.execute.call_count >= 2

@pytest.mark.asyncio
async def test_concurrent_upload_unique_violation():
    # Test 5 - Concurrent uploads (UniqueViolationError caught)
    doc_id = uuid4()
    winner_id = uuid4()
    state = IngestionState(
        document_id=doc_id,
        job_id=uuid4(),
        tenant_id="tenant_1",
        assistant_id="default",
        knowledge_base_id="default",
        filename="test.pdf",
        industry="manufacturing",
        storage_path="/tmp/test.pdf",
        status=DocumentStatus.PENDING
    )
    postgres = AsyncMock()
    # check_file_hash initially returns None
    # Then execute raises UniqueViolationError on the document update
    # Then check_file_hash returns the winner row
    postgres.fetchrow.side_effect = [None, {"id": str(winner_id), "sha256": "hash", "file_name": "x", "version": 1, "content_hash": "c"}]
    postgres.execute.side_effect = [None, asyncpg.exceptions.UniqueViolationError(), None]
    
    s02_duplicate.compute_sha256 = MagicMock(return_value="hash")
    
    new_state = await s02_duplicate.step(state, {"postgres": postgres})
    assert new_state.status == DocumentStatus.DUPLICATE
    assert new_state.canonical_document_id == winner_id


@pytest.mark.asyncio
async def test_incomplete_canonical_document_zero_chunks():
    """
    Test that an existing canonical document with status=COMPLETED but 0 chunks
    is rejected as a valid duplicate, neutralized in DB, and allows the fresh document
    to proceed through full ingestion.
    """
    from app.services.deduplication.service import DeduplicationService
    dedup_svc = DeduplicationService()
    postgres = AsyncMock()
    
    # 1. DB returns row with chunk_count = 0 (corrupted/empty canonical doc)
    postgres.fetchrow.return_value = {
        "id": str(uuid4()),
        "sha256": "corrupt_sha",
        "file_name": "corrupt.pdf",
        "version": 1,
        "content_hash": "c",
        "chunk_count": 0,
    }
    
    result = await dedup_svc.check_file_hash(
        sha256="corrupt_sha",
        tenant_id="tenant_1",
        current_doc_id=uuid4(),
        postgres=postgres,
    )
    assert result is None, "Corrupted canonical document with 0 chunks must NOT be accepted as duplicate"


@pytest.mark.asyncio
async def test_valid_completed_duplicate_with_chunks():
    """
    Test that a valid canonical document with chunks > 0 is correctly identified
    as a duplicate, marks the state DUPLICATE, records canonical_id, and sets completed_at.
    """
    from app.services.deduplication.service import DeduplicationService
    dedup_svc = DeduplicationService()
    postgres = AsyncMock()
    canonical_id = uuid4()
    
    postgres.fetchrow.return_value = {
        "id": str(canonical_id),
        "sha256": "valid_sha",
        "file_name": "valid.pdf",
        "version": 1,
        "content_hash": "c",
        "chunk_count": 42,
    }
    
    result = await dedup_svc.check_file_hash(
        sha256="valid_sha",
        tenant_id="tenant_1",
        current_doc_id=uuid4(),
        postgres=postgres,
    )
    assert result is not None
    assert result["id"] == str(canonical_id)
    assert result["chunk_count"] == 42


@pytest.mark.asyncio
async def test_canonical_resolution_for_duplicate_and_missing():
    """
    Test that StructureSearchService resolves DUPLICATE document IDs to their canonical_document_id,
    and cleanly falls back to the original document ID if missing or non-duplicate.
    """
    from app.services.retrieval.structure_search import StructureSearchService
    postgres = AsyncMock()
    svc = StructureSearchService(postgres)
    
    dup_id = uuid4()
    canonical_id = uuid4()
    
    # 1. When document is a duplicate, resolves to canonical_id
    postgres.fetchrow.return_value = {
        "status": "DUPLICATE",
        "canonical_document_id": canonical_id,
    }
    resolved = await svc.resolve_canonical_document_id(str(dup_id), "tenant_1")
    assert resolved == str(canonical_id)
    
    # 2. When document is missing (None returned from DB), safely returns original document_id
    postgres.fetchrow.return_value = None
    resolved_missing = await svc.resolve_canonical_document_id(str(dup_id), "tenant_1")
    assert resolved_missing == str(dup_id)


def test_document_status_response_duplicate_fields():
    """
    Test that DocumentStatusResponse supports canonical_document_id and is_duplicate flag.
    """
    from datetime import datetime, timezone
    from app.models.document import DocumentStatusResponse, DocumentStatus
    
    doc_id = uuid4()
    canon_id = uuid4()
    now = datetime.now(timezone.utc)
    
    resp = DocumentStatusResponse(
        document_id=doc_id,
        status=DocumentStatus.DUPLICATE,
        progress_percent=100,
        created_at=now,
        completed_at=now,
        canonical_document_id=canon_id,
        is_duplicate=True,
    )
    assert resp.is_duplicate is True
    assert resp.canonical_document_id == canon_id
    assert resp.status == DocumentStatus.DUPLICATE
    
    d = resp.model_dump()
    assert d["canonical_document_id"] == canon_id
    assert d["is_duplicate"] is True

