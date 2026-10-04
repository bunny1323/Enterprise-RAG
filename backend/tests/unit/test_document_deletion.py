"""
Unit tests for document deletion workflow.
"""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
import pytest
from fastapi import HTTPException

from app.api.routes.documents import delete_document
from app.models.tenant import TenantContext


@pytest.mark.asyncio
async def test_delete_document_not_found():
    postgres = MagicMock()
    postgres.fetchrow = AsyncMock(return_value=None)

    tenant_ctx = TenantContext(tenant_id="default")
    request = MagicMock()

    doc_id = str(uuid4())
    with pytest.raises(HTTPException) as exc_info:
        await delete_document(
            document_id=doc_id,
            postgres=postgres,
            tenant_ctx=tenant_ctx,
            request=request,
        )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_wrong_tenant_forbidden():
    postgres = MagicMock()
    doc_id = str(uuid4())
    postgres.fetchrow = AsyncMock(
        return_value={
            "id": doc_id,
            "tenant_id": "other_tenant",
            "knowledge_base_id": "default",
            "storage_path": None,
            "file_name": "test.pdf",
        }
    )

    tenant_ctx = TenantContext(tenant_id="my_tenant")
    request = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await delete_document(
            document_id=doc_id,
            postgres=postgres,
            tenant_ctx=tenant_ctx,
            request=request,
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_delete_document_success(tmp_path):
    # Create fake files to test cleanup
    raw_file = tmp_path / "test.pdf"
    raw_file.write_bytes(b"pdf content")

    proc_dir = tmp_path / "processed"
    fig_dir = proc_dir / "figures" / "test"
    fig_dir.mkdir(parents=True)
    fig_file = fig_dir / "page1.png"
    fig_file.write_bytes(b"png content")

    doc_id = str(uuid4())
    postgres = MagicMock()
    postgres.fetchrow = AsyncMock(
        return_value={
            "id": doc_id,
            "tenant_id": "default",
            "knowledge_base_id": "default",
            "storage_path": str(raw_file),
            "file_name": "test.pdf",
        }
    )
    postgres.delete_document = AsyncMock(return_value=True)

    weaviate = MagicMock()
    neo4j = MagicMock()
    neo4j.delete_document = AsyncMock()

    request = MagicMock()
    request.app.state.weaviate = weaviate
    request.app.state.neo4j = neo4j
    request.app.state.settings = MagicMock(processed_storage_path=str(proc_dir))

    tenant_ctx = TenantContext(tenant_id="default")

    res = await delete_document(
        document_id=doc_id,
        postgres=postgres,
        tenant_ctx=tenant_ctx,
        request=request,
    )

    assert res["status"] == "deleted"
    assert res["document_id"] == doc_id
    assert not raw_file.exists()
    assert not fig_dir.exists()

    weaviate.delete_by_document.assert_called_once_with(document_id=doc_id, tenant_id="default")
    neo4j.delete_document.assert_awaited_once_with(document_id=doc_id, tenant_id="default")
    postgres.delete_document.assert_awaited_once()
