from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.middleware.cors import CORSMiddleware

from app import main
from app.services.retrieval.bm25_search import BM25SearchService
from app.services.retrieval.dense_search import DenseSearchService
from app.services.retrieval.graph_search import GraphSearchService


@pytest.mark.asyncio
async def test_empty_access_policy_does_not_fall_back_to_public_results():
    weaviate = MagicMock()
    dense = DenseSearchService(weaviate)
    bm25 = BM25SearchService(weaviate)

    assert await dense.search([0.1, 0.2], permitted_access_levels=[]) == []
    assert await bm25.search("query", permitted_access_levels=[]) == []
    weaviate.vector_search.assert_not_called()
    weaviate.bm25_search.assert_not_called()


@pytest.mark.asyncio
async def test_graph_search_scopes_chunk_lookup_by_tenant_kb_and_access_level():
    neo4j = MagicMock()
    neo4j.graph_search = AsyncMock(return_value=[{"chunk_id": "chunk-1"}])
    postgres = MagicMock()
    postgres.fetch = AsyncMock(
        return_value=[
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "content": "Permitted graph evidence.",
                "page_number": 2,
                "chunk_type": "TEXT",
                "context_prefix": "",
                "section": "SECTION 1",
            }
        ]
    )
    service = GraphSearchService(neo4j, postgres)

    results = await service.search(
        "pump",
        tenant_id="tenant-a",
        knowledge_base_id="manuals",
        permitted_access_levels=["PUBLIC"],
    )

    assert len(results) == 1
    query, chunk_ids, tenant_id, knowledge_base_id, access_levels = postgres.fetch.await_args.args
    assert "knowledge_base_id = $3" in query
    assert "access_classification = ANY($4::text[])" in query
    assert chunk_ids == ["chunk-1"]
    assert tenant_id == "tenant-a"
    assert knowledge_base_id == "manuals"
    assert access_levels == ["PUBLIC"]


@pytest.mark.asyncio
async def test_empty_graph_access_policy_skips_graph_and_postgres_queries():
    neo4j = MagicMock()
    neo4j.graph_search = AsyncMock()
    postgres = MagicMock()
    postgres.fetch = AsyncMock()
    service = GraphSearchService(neo4j, postgres)

    assert await service.search("pump", permitted_access_levels=[]) == []
    neo4j.graph_search.assert_not_awaited()
    postgres.fetch.assert_not_awaited()


def test_production_cors_only_allows_configured_frontend_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://frontend.example")
    settings = MagicMock()
    settings.app_name = "EnterpriseRAG"
    settings.is_production = True
    settings.processed_storage_path = str(tmp_path)
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    app = main.create_app()
    cors = next(item for item in app.user_middleware if item.cls is CORSMiddleware)

    assert cors.kwargs["allow_origins"] == ["https://frontend.example"]
    assert cors.kwargs["allow_origin_regex"] is None
