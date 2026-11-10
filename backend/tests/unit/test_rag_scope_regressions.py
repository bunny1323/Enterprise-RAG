from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.agents.retrieval.agent import RetrievalAgent
from app.agents.retrieval.state import QueryState
from app.api.dependencies import require_admin_access
from app.api.routes.documents import ClassificationRequest, set_document_classification
from app.infrastructure.postgres.client import PostgresClient
from app.infrastructure.weaviate import client as weaviate_module
from app.models.retrieval import SearchResult
from app.models.tenant import TenantContext
from app.pipelines.ingestion.steps.s08_index import _insert_chunks_postgres
from app.services.metadata.service import MetadataService
from app.services.query.normalization import QueryNormalizationService
from app.services.retrieval.hierarchical import HierarchicalRetrievalService


def test_ingestion_classification_defaults_to_internal_even_if_config_says_public():
    assert MetadataService._get_default_classification(
        {"default_classification": "PUBLIC", "security_classifications": ["PUBLIC"]}
    ) == "INTERNAL"


def test_ingestion_classification_keeps_internal_default():
    metadata_service = MetadataService()
    metadata_service._cache["manufacturing"] = {
        "default_classification": "INTERNAL",
        "entity_types": [],
        "relation_types": [],
        "chunking_rules": {},
    }

    from app.models.chunk import Chunk

    chunk = Chunk(
        chunk_id="doc-1::p1::c1",
        document_id=uuid4(),
        content="Internal content",
        page_number=1,
    )
    enriched = metadata_service.enrich_chunks([chunk], "manufacturing")

    assert enriched[0].access_classification == "INTERNAL"


@pytest.mark.asyncio
async def test_public_structural_query_uses_scoped_structure_index():
    sections = [
        SearchResult(
            chunk_id="struct::section::doc-1::1::1",
            score=1,
            content="SECTION 1: Introduction",
            document_id="doc-1",
            section="SECTION 1 Introduction",
            metadata={"source": "document_structure"},
        )
    ]
    structure = MagicMock()
    structure.list_sections = AsyncMock(return_value=sections)
    structure.count_sections = AsyncMock()
    dense = MagicMock()
    dense.search = AsyncMock()

    agent = RetrievalAgent(
        embedder=MagicMock(),
        dense_svc=dense,
        bm25_svc=MagicMock(),
        graph_svc=MagicMock(),
        reranker=MagicMock(),
        normalizer=QueryNormalizationService(),
        confidence_svc=MagicMock(),
        hierarchical_svc=MagicMock(),
        structure_svc=structure,
    )
    result = await agent.retrieve(
        QueryState(
            query="How many sections are there in the manual?",
            tenant_context=TenantContext(),
        ),
        permitted_access_levels=["PUBLIC"],
    )

    assert result.strategy_used == "document_structure_count"
    assert result.reranked_results == sections
    assert result.retrieval_diagnostics["channels"]["structural"]
    structure.list_sections.assert_awaited_once_with(
        tenant_id="default",
        knowledge_base_id="default",
        permitted_access_levels=["PUBLIC"],
    )
    structure.count_sections.assert_not_awaited()
    dense.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_structural_index_falls_back_instead_of_claiming_zero_sections():
    structure = MagicMock()
    structure.list_sections = AsyncMock(return_value=[])
    dense = MagicMock()
    dense.search = AsyncMock(return_value=[])
    bm25 = MagicMock()
    bm25.search = AsyncMock(return_value=[])

    agent = RetrievalAgent(
        embedder=MagicMock(embed_batch=MagicMock(return_value=[[0.1]])),
        dense_svc=dense,
        bm25_svc=bm25,
        graph_svc=MagicMock(),
        reranker=MagicMock(),
        normalizer=QueryNormalizationService(),
        confidence_svc=MagicMock(),
        hierarchical_svc=MagicMock(),
        structure_svc=structure,
    )
    result = await agent.retrieve(
        QueryState(
            query="How many sections are there in the manual?",
            tenant_context=TenantContext(),
        ),
        permitted_access_levels=["PUBLIC"],
    )

    assert result.strategy_used != "document_structure_count"
    assert result.reranked_results == []
    assert dense.search.await_args.kwargs["permitted_access_levels"] == ["PUBLIC"]


@pytest.mark.asyncio
async def test_parent_expansion_respects_allowed_classification_and_falls_back_to_child():
    postgres = MagicMock()
    postgres.fetch = AsyncMock(return_value=[])
    service = HierarchicalRetrievalService(postgres)
    child = SearchResult(
        chunk_id="child-1",
        parent_id="internal-parent",
        content="PUBLIC child passage",
        document_id="doc-1",
    )

    results = await service.expand_to_parents(
        [child],
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        permitted_access_levels=["PUBLIC"],
    )

    assert results == [child]
    query, parent_ids, tenant_id, knowledge_base_id, access_levels = postgres.fetch.await_args.args
    assert "access_classification = ANY($4::text[])" in query
    assert parent_ids == ["internal-parent"]
    assert (tenant_id, knowledge_base_id, access_levels) == (
        "tenant-a",
        "kb-a",
        ["PUBLIC"],
    )


@pytest.mark.asyncio
async def test_vector_store_hits_are_rechecked_against_postgres_classifications():
    postgres = MagicMock()
    postgres.fetch = AsyncMock(return_value=[{"chunk_id": "public-chunk"}])
    service = HierarchicalRetrievalService(postgres)

    accessible = await service.get_accessible_chunk_ids(
        ["public-chunk", "internal-chunk", "missing-chunk"],
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        permitted_access_levels=["PUBLIC"],
    )

    assert accessible == {"public-chunk"}
    query, chunk_ids, tenant_id, knowledge_base_id, access_levels = postgres.fetch.await_args.args
    assert "access_classification = ANY($4::text[])" in query
    assert set(chunk_ids) == {"public-chunk", "internal-chunk", "missing-chunk"}
    assert (tenant_id, knowledge_base_id, access_levels) == (
        "tenant-a",
        "kb-a",
        ["PUBLIC"],
    )


@pytest.mark.asyncio
async def test_empty_access_scope_never_expands_parent_chunks():
    postgres = MagicMock()
    postgres.fetch = AsyncMock()
    service = HierarchicalRetrievalService(postgres)

    results = await service.expand_to_parents(
        [SearchResult(chunk_id="child", parent_id="parent", content="child")],
        permitted_access_levels=[],
    )

    assert results == []
    postgres.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_structure_sql_excludes_missing_or_disallowed_chunk_classification():
    postgres = PostgresClient("postgresql://unused")
    postgres.fetch = AsyncMock(return_value=[])
    postgres.fetchval = AsyncMock(return_value=0)

    await postgres.get_sections("tenant-a", "kb-a", permitted_access_levels=["PUBLIC"])
    section_sql, *_section_args = postgres.fetch.await_args.args
    assert "c.tenant_id = $1 AND c.knowledge_base_id = $2" in section_sql
    assert "c.access_classification IS NULL" in section_sql
    assert "c.access_classification = ANY($3::text[])" in section_sql
    assert postgres.fetch.await_args.args[-1] == ["PUBLIC"]

    await postgres.count_sections(
        "tenant-a",
        "kb-a",
        permitted_access_levels=["PUBLIC"],
    )
    count_sql, *_count_args = postgres.fetchval.await_args.args
    assert "COUNT(*) FILTER" in count_sql
    assert "c.access_classification IS NULL" in count_sql
    assert postgres.fetchval.await_args.args[-1] == ["PUBLIC"]

    await postgres.get_page_format_entries(
        "tenant-a",
        "kb-a",
        notation="2-3",
        permitted_access_levels=["PUBLIC"],
    )
    page_sql, *_page_args = postgres.fetch.await_args.args
    assert "c.access_classification IS NULL" in page_sql
    assert postgres.fetch.await_args.args[-1] == ["PUBLIC"]


def test_weaviate_classification_filters_use_exact_scalar_equality(monkeypatch):
    class FilterExpression:
        def __init__(self, expression):
            self.expression = expression

        def __or__(self, other):
            return FilterExpression(("or", self.expression, other.expression))

    class PropertyFilter:
        def __init__(self, property_name):
            self.property_name = property_name

        def equal(self, value):
            return FilterExpression(("equal", self.property_name, value))

    monkeypatch.setattr(
        weaviate_module.wvc.query.Filter,
        "by_property",
        lambda property_name: PropertyFilter(property_name),
    )

    result = weaviate_module._access_classification_filter(["PUBLIC", "INTERNAL"])

    assert result.expression == (
        "or",
        ("equal", "access_classification", "PUBLIC"),
        ("equal", "access_classification", "INTERNAL"),
    )


def test_reclassification_requires_verified_administrator():
    anonymous = TenantContext(
        claims={"is_authenticated": False, "is_admin": False, "api_auth_disabled_public_only": True}
    )

    with pytest.raises(HTTPException) as exc_info:
        require_admin_access(anonymous, "Document classification")

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_public_reclassification_updates_postgres_before_weaviate():
    document_id = uuid4()
    postgres = MagicMock()
    postgres.fetchrow = AsyncMock(return_value={"id": document_id, "status": "COMPLETED"})
    postgres.fetchval = AsyncMock(return_value=2)
    postgres.set_document_access_classification = AsyncMock(
        side_effect=[
            {"chunk-1": "INTERNAL", "chunk-2": "INTERNAL"},
            {"chunk-1": "PENDING", "chunk-2": "PENDING"},
        ]
    )
    postgres.restore_document_access_classification = AsyncMock()
    weaviate = MagicMock()
    weaviate.update_document_access_classification.return_value = {
        "object-1": "INTERNAL",
        "object-2": "INTERNAL",
    }
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(weaviate=weaviate))
    )
    admin = TenantContext(
        claims={"is_authenticated": True, "is_admin": True}
    )

    response = await set_document_classification(
        str(document_id),
        ClassificationRequest(access_classification="PUBLIC"),
        request,
        postgres,
        admin,
    )

    assert response["access_classification"] == "PUBLIC"
    assert postgres.set_document_access_classification.await_count == 2
    assert (
        postgres.set_document_access_classification.await_args_list[0].kwargs[
            "access_classification"
        ]
        == "PENDING"
    )
    assert (
        postgres.set_document_access_classification.await_args_list[1].kwargs[
            "access_classification"
        ]
        == "PUBLIC"
    )
    weaviate.update_document_access_classification.assert_called_once()
    postgres.restore_document_access_classification.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_public_reclassification_restores_postgres_classifications():
    document_id = uuid4()
    previous = {"chunk-1": "INTERNAL", "chunk-2": "RESTRICTED"}
    postgres = MagicMock()
    postgres.fetchrow = AsyncMock(return_value={"id": document_id, "status": "COMPLETED"})
    postgres.fetchval = AsyncMock(return_value=2)
    postgres.set_document_access_classification = AsyncMock(return_value=previous)
    postgres.restore_document_access_classification = AsyncMock()
    weaviate = MagicMock()
    weaviate.update_document_access_classification.side_effect = RuntimeError("index mismatch")
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(weaviate=weaviate))
    )
    admin = TenantContext(
        claims={"is_authenticated": True, "is_admin": True}
    )

    with pytest.raises(HTTPException) as exc_info:
        await set_document_classification(
            str(document_id),
            ClassificationRequest(access_classification="PUBLIC"),
            request,
            postgres,
            admin,
        )

    assert exc_info.value.status_code == 503
    postgres.restore_document_access_classification.assert_awaited_once_with(
        document_id=document_id,
        tenant_id="default",
        knowledge_base_id="default",
        previous_classifications=previous,
    )


@pytest.mark.asyncio
async def test_postgres_reingestion_upsert_refreshes_classification():
    from app.models.chunk import Chunk

    postgres = MagicMock()
    postgres.execute = AsyncMock()
    chunk = Chunk(
        chunk_id="doc-1::p1::c1",
        document_id=uuid4(),
        content="Public content",
        page_number=1,
        access_classification="PUBLIC",
    )

    await _insert_chunks_postgres(postgres, [chunk])

    upsert_sql = postgres.execute.await_args.args[0]
    assert "access_classification = EXCLUDED.access_classification" in upsert_sql
