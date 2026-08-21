"""
Query and Chat API Routes.
POST /api/v1/chat   — Grounded RAG conversational endpoint (Retrieval + Generation + Citations)
POST /api/v1/search — Raw search endpoint (Retrieval only)
"""
import time
import uuid
from typing import Annotated
from app.infrastructure.postgres.client import PostgresClient
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
from app.agents.query_workflow.nodes import QueryNodes
from app.agents.query_workflow.graph import build_query_graph
from app.agents.query_workflow.state import QueryWorkflowState

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

    # 1. Initialize dependencies
    opa_client: OPAClient = getattr(request.app.state, "opa", OPAClient())
    redis_client: RedisClient | None = getattr(request.app.state, "redis", None)
    cache_svc = CacheService(redis_client, postgres) if redis_client else None
    
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
        cache_svc=cache_svc,
    )
    
    llm_provider: OllamaProvider = request.app.state.llm_provider
    citation_svc = CitationService()
    verifier = GroundednessVerificationService()

    # 2. Build LangGraph Nodes & Graph
    nodes = QueryNodes(
        retrieval_agent=retrieval_agent,
        llm_provider=llm_provider,
        citation_svc=citation_svc,
        verifier_svc=verifier,
        opa_client=opa_client,
        cache_svc=cache_svc,
    )
    graph = build_query_graph(nodes)

    # 3. Initialize State
    initial_state = QueryWorkflowState(
        query=body.query,
        tenant_context=tenant_ctx,
        top_k=body.top_k,
        intent="",
        permitted_access_levels=[],
        cache_hit=False,
        retrieval_strategy="",
        dense_results=[],
        bm25_results=[],
        graph_results=[],
        fused_results=[],
        reranked_results=[],
        confidence_level="LOW",
        confidence_score=0.0,
        retries=0,
        answer="",
        citations=[],
        sources=[],
        verification_status="UNSUPPORTED",
        trace_id=trace_id,
        latency_ms=0,
        error_message=None
    )

    # 4. Execute LangGraph Workflow
    final_state = await graph.ainvoke(initial_state)

    if final_state.get("error_message"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=final_state["error_message"],
        )

    latency_ms = int((time.time() - start_time) * 1000)
    
    # 5. Format Response
    response_data = QueryResponse(
        answer=final_state["answer"],
        citations=final_state["citations"],
        confidence=final_state["confidence_level"],
        confidence_score=final_state["confidence_score"],
        evidence=[
            EvidenceSnippet(
                chunk_id=item.chunk_id,
                content=item.content,
                page_number=item.page_number,
                score=item.score,
            )
            for item in final_state["reranked_results"]
        ],
        sources=final_state["sources"],
        verification_status=final_state["verification_status"],
        trace_id=trace_id,
        retrieval_strategy=final_state["retrieval_strategy"],
        latency_ms=latency_ms,
    )

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
    redis_client: RedisClient | None = getattr(request.app.state, "redis", None)
    cache_svc = CacheService(redis_client, postgres) if redis_client else None

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
        cache_svc=cache_svc,
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
