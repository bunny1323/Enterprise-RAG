import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.query_workflow.graph import build_query_graph
from app.agents.query_workflow.nodes import QueryNodes
from app.agents.retrieval.agent import RetrievalAgent
from app.agents.retrieval.state import QueryState
from app.agents.supervisor.agent import IngestionSupervisor
from app.models.retrieval import SearchResult
from app.models.tenant import TenantContext
from app.services.generation.citations import CitationService
from app.services.generation.hallucination import GroundednessVerificationService
from app.services.query.normalization import QueryNormalizationService


class _RepeatedlyPartialNodes:
    def __init__(self) -> None:
        self.retrieve_count = 0
        self.refine_count = 0
        self.verify_count = 0

    async def evaluate_policy(self, state):
        return {"permitted_access_levels": ["INTERNAL"]}

    async def check_cache(self, state):
        return {"cache_hit": False}

    async def retrieve_evidence(self, state):
        self.retrieve_count += 1
        return {}

    async def check_answerability(self, state):
        return {"answerable": True, "confidence_level": "HIGH"}

    async def refine_query(self, state):
        self.refine_count += 1
        return {"retries": state["retries"] + 1}

    async def generate_response(self, state):
        return {"answer": "A partially grounded response."}

    async def verify_citations(self, state):
        self.verify_count += 1
        return {"verification_status": "PARTIALLY_SUPPORTED"}


@pytest.mark.asyncio
async def test_partial_verification_retries_are_bounded():
    nodes = _RepeatedlyPartialNodes()
    graph = build_query_graph(nodes)

    result = await graph.ainvoke(
        {
            "query": "Explain the system",
            "tenant_context": TenantContext(),
            "top_k": 5,
            "intent": "GENERAL_QA",
            "intent_confidence": 1.0,
            "permitted_access_levels": [],
            "cache_hit": False,
            "retrieval_strategy": "",
            "dense_results": [],
            "bm25_results": [],
            "graph_results": [],
            "fused_results": [],
            "reranked_results": [],
            "confidence_level": "HIGH",
            "confidence_score": 1.0,
            "retries": 0,
            "answer": "",
            "citations": [],
            "sources": [],
            "verification_status": "UNSUPPORTED",
            "evidence_coverage": "COMPLETE",
            "answerable": True,
            "retrieval_trace": {},
            "trace_id": "test",
            "latency_ms": 0,
            "error_message": None,
        }
    )

    assert nodes.retrieve_count == 3
    assert nodes.refine_count == 2
    assert nodes.verify_count == 3
    assert result["retries"] == 2
    assert result["verification_status"] == "PARTIALLY_SUPPORTED"


@pytest.mark.asyncio
async def test_insufficient_evidence_refusal_remains_unsupported():
    nodes = QueryNodes(
        retrieval_agent=MagicMock(),
        llm_provider=MagicMock(),
        citation_svc=CitationService(),
        verifier_svc=GroundednessVerificationService(),
        opa_client=MagicMock(),
        cache_svc=None,
    )
    state = {
        "intent": "GENERAL_QA",
        "answerable": False,
        "answer": "The provided evidence does not contain enough information to answer this question.",
        "verification_status": "UNSUPPORTED",
        "citations": ["stale citation"],
        "sources": ["stale source"],
        "reranked_results": [
            SearchResult(
                chunk_id="evidence-1",
                content="This evidence is unrelated to the requested fact.",
                score=0.5,
            )
        ],
    }

    result = await nodes.verify_citations(state)

    assert result["verification_status"] == "UNSUPPORTED"
    assert result["citations"] == []
    assert result["sources"] == []


@pytest.mark.asyncio
async def test_query_refinement_keeps_original_query_stable():
    nodes = QueryNodes(
        retrieval_agent=MagicMock(),
        llm_provider=MagicMock(),
        citation_svc=CitationService(),
        verifier_svc=GroundednessVerificationService(),
        opa_client=MagicMock(),
        cache_svc=None,
    )
    state = {
        "query": "How do I reset the pump?",
        "original_query": "How do I reset the pump?",
        "normalized_query": "reset pump",
        "retries": 0,
    }

    result = await nodes.refine_query(state)

    assert result["query"] == "How do I reset the pump?"
    assert result["original_query"] == "How do I reset the pump?"
    assert result["refined_query"] == "reset pump explain details"
    assert result["retries"] == 1


@pytest.mark.asyncio
async def test_cache_failure_degrades_to_uncached_query():
    cache = MagicMock()
    cache.get_response = AsyncMock(side_effect=RuntimeError("Redis unavailable"))
    nodes = QueryNodes(
        retrieval_agent=MagicMock(),
        llm_provider=MagicMock(),
        citation_svc=CitationService(),
        verifier_svc=GroundednessVerificationService(),
        opa_client=MagicMock(),
        cache_svc=cache,
    )
    state = {
        "query": "How do I reset the pump?",
        "tenant_context": TenantContext(),
        "permitted_access_levels": ["PUBLIC", "INTERNAL"],
        "cache_hit": False,
        "error_message": None,
    }

    result = await nodes.check_cache(state)

    assert result["cache_hit"] is False
    cache.get_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_hit_populates_intent():
    cache = MagicMock()
    cache.get_response = AsyncMock(return_value={
        "answer": "Cached answer.",
        "citations": [],
        "sources": [],
        "confidence": "HIGH",
        "confidence_score": 1.0,
        "verification_status": "SUPPORTED",
        "retrieval_strategy": "exact_cache",
    })
    nodes = QueryNodes(
        retrieval_agent=MagicMock(),
        llm_provider=MagicMock(),
        citation_svc=CitationService(),
        verifier_svc=GroundednessVerificationService(),
        opa_client=MagicMock(),
        cache_svc=cache,
        normalizer=QueryNormalizationService(),
    )
    state = {
        "query": "What is the hydraulic oil capacity?",
        "original_query": "What is the hydraulic oil capacity?",
        "tenant_context": TenantContext(),
        "permitted_access_levels": ["PUBLIC", "INTERNAL"],
        "cache_hit": False,
        "intent": "",
        "intent_confidence": 0.0,
        "error_message": None,
    }

    result = await nodes.check_cache(state)

    assert result["cache_hit"] is True
    assert result["intent"] == "SPECIFICATION"
    assert result["intent_confidence"] > 0.0


@pytest.mark.asyncio
async def test_query_embedding_cache_failures_degrade_to_live_embedding():
    embedder = MagicMock()
    embedder.embed_batch.return_value = [[0.1, 0.2]]
    cache = MagicMock()
    cache.get_query_embedding = AsyncMock(side_effect=RuntimeError("Redis unavailable"))
    cache.set_query_embedding = AsyncMock(side_effect=RuntimeError("Redis unavailable"))
    dense = MagicMock()
    dense.search = AsyncMock(return_value=[])
    bm25 = MagicMock()
    bm25.search = AsyncMock(return_value=[])
    agent = RetrievalAgent(
        embedder=embedder,
        dense_svc=dense,
        bm25_svc=bm25,
        graph_svc=MagicMock(),
        reranker=MagicMock(),
        normalizer=QueryNormalizationService(),
        confidence_svc=MagicMock(),
        hierarchical_svc=MagicMock(),
        cache_svc=cache,
    )

    result = await agent.retrieve(
        QueryState(query="Pump flow rate", tenant_context=TenantContext()),
        permitted_access_levels=["PUBLIC"],
    )

    assert result.strategy_used == "failed"
    embedder.embed_batch.assert_called_once()
    cache.get_query_embedding.assert_awaited_once()
    cache.set_query_embedding.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_claims_jobs_from_durable_postgres_queue(monkeypatch):
    postgres = MagicMock()
    postgres.claim_ingestion_job = AsyncMock(return_value=None)
    settings = MagicMock()
    settings.ingestion_job_poll_seconds = 0.001
    settings.ingestion_job_lease_seconds = 30
    monkeypatch.setattr("app.config.settings.get_settings", lambda: settings)
    supervisor = IngestionSupervisor(
        pipeline=MagicMock(),
        storage=MagicMock(),
        postgres=postgres,
    )
    worker = asyncio.create_task(supervisor.process_queue())
    try:
        while not postgres.claim_ingestion_job.await_count:
            await asyncio.sleep(0)
    finally:
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker

    worker_id, lease_seconds = postgres.claim_ingestion_job.await_args.args
    assert worker_id == supervisor._worker_id
    assert lease_seconds == 30
