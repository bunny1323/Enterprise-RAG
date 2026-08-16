"""
Query and Chat API Routes.
POST /api/v1/chat   — Grounded RAG conversational endpoint (Retrieval + Generation + Citations)
POST /api/v1/search — Raw search endpoint (Retrieval only)
"""
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.agents.retrieval.agent import RetrievalAgent
from app.agents.retrieval.state import QueryState
from app.api.dependencies import (
    PostgresDep,
    SettingsDep,
    TenantContextDep,
    get_opa,
    get_redis,
)
from app.config.logging import get_logger
from app.infrastructure.opa.client import OPAClient
from app.infrastructure.redis.client import RedisClient
from app.models.query import (
    Citation,
    EvidenceSnippet,
    QueryRequest,
    QueryResponse,
    SourceRef,
)
from app.models.tenant import TenantContext
from app.services.cache.service import CacheService
from app.services.confidence.service import ConfidenceScoringService
from app.services.embeddings.service import EmbeddingService
from app.services.generation.citations import CitationService
from app.services.generation.hallucination import GroundednessVerificationService
from app.services.generation.llm import OllamaProvider
from app.services.query.normalization import QueryNormalizationService
from app.services.retrieval.bm25_search import BM25SearchService
from app.services.retrieval.dense_search import DenseSearchService
from app.services.retrieval.graph_search import GraphSearchService
from app.services.retrieval.reranking import VoyageRerankService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post(
    "/chat",
    response_model=QueryResponse,
    summary="Grounded conversational RAG query",
)
async def chat_query(
    request: Request,
    body: QueryRequest,
    tenant_ctx: TenantContextDep,
    settings: SettingsDep,
    postgres: PostgresDep,
) -> QueryResponse:
    """
    Full Enterprise RAG query execution pipeline:
    1. Authenticate identity / TenantContext
    2. Evaluate OPA policy decision for permitted filters
    3. Check security-aware cache
    4. Execute RetrievalAgent (Dense + BM25 + Graph + RRF + Rerank)
    5. Score confidence & generate answer via LLM
    6. Map citations & verify groundedness
    7. Cache response and return QueryResponse
    """
    start_time = time.time()
    trace_id = f"tr_{uuid.uuid4().hex[:12]}"

    logger.info("chat.request_received", trace_id=trace_id, query=body.query[:50], tenant=tenant_ctx.tenant_id)

    # 1. OPA Policy Evaluation
    opa_client: OPAClient = getattr(request.app.state, "opa", OPAClient())
    policy_decision = await opa_client.evaluate_retrieval_policy(tenant_ctx)
    if not policy_decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied by policy: {policy_decision.denied_reason or 'Unauthorized'}",
        )

    # 2. Check Security-Aware Cache
    redis_client: RedisClient | None = getattr(request.app.state, "redis", None)
    if redis_client:
        cache_svc = CacheService(redis_client, postgres)
        cached_resp = await cache_svc.get_response(tenant_ctx, body.query)
        if cached_resp:
            cached_resp["latency_ms"] = int((time.time() - start_time) * 1000)
            return QueryResponse(**cached_resp)

    # 3. Initialize Retrieval Agent components from app state
    embedder: EmbeddingService = request.app.state.embedder
    dense_svc = DenseSearchService(request.app.state.weaviate)
    bm25_svc = BM25SearchService(request.app.state.weaviate)
    graph_svc = GraphSearchService(request.app.state.neo4j, postgres)
    reranker = VoyageRerankService(settings.voyage_api_key)
    normalizer = QueryNormalizationService()
    confidence_svc = ConfidenceScoringService()

    retrieval_agent = RetrievalAgent(
        embedder=embedder,
        dense_svc=dense_svc,
        bm25_svc=bm25_svc,
        graph_svc=graph_svc,
        reranker=reranker,
        normalizer=normalizer,
        confidence_svc=confidence_svc,
    )

    # 4. Execute Retrieval
    initial_qstate = QueryState(
        query=body.query,
        tenant_context=tenant_ctx,
        top_k=body.top_k,
    )

    retrieved_state = await retrieval_agent.retrieve(
        query_state=initial_qstate,
        permitted_access_levels=policy_decision.permitted_access_levels,
    )

    evidence = retrieved_state.reranked_results

    # 5. Generate Grounded Response via LLM Provider
    llm_provider: OllamaProvider = request.app.state.llm_provider
    gen_result = await llm_provider.generate(
        prompt=body.query,
        evidence=evidence,
    )

    # 6. Map Citations & Verify Groundedness
    citation_svc = CitationService()
    citations, sources = citation_svc.map_citations(gen_result.answer, evidence)

    verifier = GroundednessVerificationService()
    groundedness = verifier.verify(gen_result.answer, evidence)

    latency_ms = int((time.time() - start_time) * 1000)

    response_data = QueryResponse(
        answer=gen_result.answer,
        citations=citations,
        confidence=retrieved_state.confidence_level,
        confidence_score=retrieved_state.confidence_score,
        evidence=[
            EvidenceSnippet(
                chunk_id=item.chunk_id,
                content=item.content,
                page_number=item.page_number,
                score=item.score,
            )
            for item in evidence
        ],
        sources=sources,
        verification_status=groundedness.verification_status,
        trace_id=trace_id,
        retrieval_strategy=retrieved_state.strategy_used,
        latency_ms=latency_ms,
    )

    # 7. Store in Cache
    if redis_client:
        cache_svc = CacheService(redis_client, postgres)
        await cache_svc.set_response(tenant_ctx, body.query, response_data.model_dump())

    logger.info("chat.request_complete", trace_id=trace_id, latency_ms=latency_ms)
    return response_data


@router.post(
    "/search",
    summary="Raw multi-channel retrieval search (No LLM generation)",
)
async def raw_search(
    request: Request,
    body: QueryRequest,
    tenant_ctx: TenantContextDep,
    settings: SettingsDep,
    postgres: PostgresClient = Depends(PostgresDep),
) -> dict:
    """Execute retrieval and reranking, returning raw SearchResult items."""
    start_time = time.time()
    embedder: EmbeddingService = request.app.state.embedder
    dense_svc = DenseSearchService(request.app.state.weaviate)
    bm25_svc = BM25SearchService(request.app.state.weaviate)
    graph_svc = GraphSearchService(request.app.state.neo4j, postgres)
    reranker = VoyageRerankService(settings.voyage_api_key)
    normalizer = QueryNormalizationService()
    confidence_svc = ConfidenceScoringService()

    retrieval_agent = RetrievalAgent(
        embedder=embedder,
        dense_svc=dense_svc,
        bm25_svc=bm25_svc,
        graph_svc=graph_svc,
        reranker=reranker,
        normalizer=normalizer,
        confidence_svc=confidence_svc,
    )

    initial_qstate = QueryState(
        query=body.query,
        tenant_context=tenant_ctx,
        top_k=body.top_k,
    )

    retrieved_state = await retrieval_agent.retrieve(initial_qstate)
    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "query": body.query,
        "results": [r.model_dump() for r in retrieved_state.reranked_results],
        "confidence": retrieved_state.confidence_level,
        "strategy": retrieved_state.strategy_used,
        "latency_ms": latency_ms,
    }
