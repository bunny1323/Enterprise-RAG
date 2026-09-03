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
    # Test 3 - Same SHA but previous document FAILED
    doc_id = uuid4()
    failed_id = uuid4()
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
    # First call is check_file_hash (returns None because it excludes FAILED)
    # Second call is get_failed_document_by_hash (returns the failed doc)
    postgres.fetchrow.side_effect = [None, {"id": str(failed_id), "sha256": "hash"}]
    
    s02_duplicate.compute_sha256 = MagicMock(return_value="hash")
    
    new_state = await s02_duplicate.step(state, {"postgres": postgres})
    assert new_state.status == DocumentStatus.CHECKING_DUPLICATE
    # Verify the failed doc was renamed
    postgres.execute.assert_any_call("UPDATE documents SET sha256 = $1 WHERE id = $2", f"hash_failed_{failed_id}", failed_id)

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
    # check_file_hash initially returns None, get_failed returns None
    # Then execute raises UniqueViolationError
    # Then check_file_hash returns the winner row
    postgres.fetchrow.side_effect = [None, None, {"id": str(winner_id), "sha256": "hash", "file_name": "x", "version": 1, "content_hash": "c"}]
    postgres.execute.side_effect = [asyncpg.exceptions.UniqueViolationError(), None, None]
    
    s02_duplicate.compute_sha256 = MagicMock(return_value="hash")
    
    new_state = await s02_duplicate.step(state, {"postgres": postgres})
    assert new_state.status == DocumentStatus.DUPLICATE
    assert new_state.canonical_document_id == winner_id
