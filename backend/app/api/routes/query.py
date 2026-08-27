"""
Query and Chat API Routes.
POST /api/v1/chat   — Grounded RAG conversational endpoint
(Retrieval + Generation + Citations)
POST /api/v1/search — Raw search endpoint (Retrieval only)
"""

import time
import uuid

from fastapi import APIRouter, HTTPException, Request, status

from app.agents.query_workflow.graph import build_query_graph
from app.agents.query_workflow.nodes import QueryNodes
from app.agents.query_workflow.state import QueryWorkflowState
from app.agents.retrieval.agent import RetrievalAgent
from app.agents.retrieval.state import QueryState
from app.api.dependencies import (
    PostgresDep,
    SettingsDep,
    TenantContextDep,
)
from app.config.logging import get_logger
from app.infrastructure.redis.client import RedisClient
from app.models.query import (
    EvidenceSnippet,
    QueryRequest,
    QueryResponse,
)
from app.services.cache.service import CacheService
from app.services.confidence.service import ConfidenceScoringService
from app.services.embeddings.service import EmbeddingService
from app.services.generation.citations import CitationService
from app.services.generation.hallucination import (
    GroundednessVerificationService,
)
from app.services.generation.llm import OllamaProvider
from app.services.query.normalization import QueryNormalizationService
from app.services.retrieval.bm25_search import BM25SearchService
from app.services.retrieval.dense_search import DenseSearchService
from app.services.retrieval.graph_search import GraphSearchService
from app.services.retrieval.hierarchical import (
    HierarchicalRetrievalService,
)
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
    2. Evaluate OPA policy decision
    3. Check security-aware cache
    4. Execute RetrievalAgent
       (Dense + BM25 + Graph + RRF + Rerank + Hierarchical Retrieval)
    5. Score confidence and generate answer via LLM
    6. Map citations and verify groundedness
    7. Cache response and return QueryResponse
    """

    start_time = time.time()
    trace_id = f"tr_{uuid.uuid4().hex[:12]}"

    logger.info(
        "chat.request_received",
        trace_id=trace_id,
        query=body.query[:50],
        tenant=tenant_ctx.tenant_id,
    )

    # --------------------------------------------------
    # 1. Initialize dependencies
    # --------------------------------------------------

    redis_client: RedisClient | None = getattr(
        request.app.state,
        "redis",
        None,
    )

    cache_svc = (
        CacheService(redis_client, postgres)
        if redis_client
        else None
    )

    embedder: EmbeddingService = request.app.state.embedder

    dense_svc = DenseSearchService(
        request.app.state.weaviate
    )

    bm25_svc = BM25SearchService(
        request.app.state.weaviate
    )

    graph_svc = GraphSearchService(
        request.app.state.neo4j,
        postgres,
    )

    # NEW: Hierarchical retrieval service
    hierarchical_svc = HierarchicalRetrievalService(
        postgres
    )

    reranker = VoyageRerankService(
        settings.voyage_api_key
    )

    normalizer = QueryNormalizationService()

    confidence_svc = ConfidenceScoringService()

    # --------------------------------------------------
    # 2. Create Retrieval Agent
    # --------------------------------------------------

    retrieval_agent = RetrievalAgent(
        embedder=embedder,
        dense_svc=dense_svc,
        bm25_svc=bm25_svc,
        graph_svc=graph_svc,
        reranker=reranker,
        normalizer=normalizer,
        confidence_svc=confidence_svc,
        hierarchical_svc=hierarchical_svc,
        cache_svc=cache_svc,
        weaviate_client=request.app.state.weaviate,
    )

    # --------------------------------------------------
    # 3. Generation dependencies
    # --------------------------------------------------

    llm_provider: OllamaProvider = (
        request.app.state.llm_provider
    )

    citation_svc = CitationService()

    verifier = GroundednessVerificationService()

<<<<<<< HEAD
    # --------------------------------------------------
    # 4. Build LangGraph Nodes and Graph
    # --------------------------------------------------

=======
    # 2. Build LangGraph Nodes
>>>>>>> 2acbe82 (Complete Enterprise-RAG Phases 1-5 hardening)
    nodes = QueryNodes(
        retrieval_agent=retrieval_agent,
        llm_provider=llm_provider,
        citation_svc=citation_svc,
        verifier_svc=verifier,
        opa_client=getattr(
            request.app.state,
            "opa",
            None,
        ),
        cache_svc=cache_svc,
    )
<<<<<<< HEAD

    graph = build_query_graph(nodes)
=======
>>>>>>> 2acbe82 (Complete Enterprise-RAG Phases 1-5 hardening)

    # --------------------------------------------------
    # 5. Initialize workflow state
    # --------------------------------------------------

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
        error_message=None,
    )

<<<<<<< HEAD
    # --------------------------------------------------
    # 6. Execute LangGraph workflow
    # --------------------------------------------------

    final_state = await graph.ainvoke(
        initial_state
    )
=======
    # 4. Execute LangGraph Workflow with persistent Postgres checkpointer
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        async with AsyncPostgresSaver.from_conn_string(settings.database_url) as checkpointer:
            # Note: in a production setup, setup() should run during migrations
            await checkpointer.asetup() 
            graph = build_query_graph(nodes, checkpointer=checkpointer)
            config = {"configurable": {"thread_id": trace_id, "tenant_id": tenant_ctx.tenant_id}}
            final_state = await graph.ainvoke(initial_state, config=config)
    except ImportError:
        logger.warning("chat.missing_postgres_checkpointer", trace_id=trace_id)
        # Fallback to no checkpointer if library is not installed
        graph = build_query_graph(nodes)
        final_state = await graph.ainvoke(initial_state)
>>>>>>> 2acbe82 (Complete Enterprise-RAG Phases 1-5 hardening)

    if final_state.get("error_message"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=final_state["error_message"],
        )

    latency_ms = int(
        (time.time() - start_time) * 1000
    )

    # --------------------------------------------------
    # 7. Format response
    # --------------------------------------------------

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
        verification_status=final_state[
            "verification_status"
        ],
        trace_id=trace_id,
        retrieval_strategy=final_state[
            "retrieval_strategy"
        ],
        latency_ms=latency_ms,
    )

    logger.info(
        "chat.request_complete",
        trace_id=trace_id,
        latency_ms=latency_ms,
    )

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
    postgres: PostgresDep,
) -> dict:
    """
    Execute multi-channel retrieval and reranking.

    Pipeline:

    Dense + BM25 + Graph
            ↓
           RRF
            ↓
         Reranker
            ↓
    Hierarchical Expansion
            ↓
      Confidence Scoring
    """

    start_time = time.time()

    # --------------------------------------------------
    # 1. Initialize dependencies
    # --------------------------------------------------

    redis_client: RedisClient | None = getattr(
        request.app.state,
        "redis",
        None,
    )

    cache_svc = (
        CacheService(redis_client, postgres)
        if redis_client
        else None
    )

    embedder: EmbeddingService = request.app.state.embedder

    dense_svc = DenseSearchService(
        request.app.state.weaviate
    )

    bm25_svc = BM25SearchService(
        request.app.state.weaviate
    )

    graph_svc = GraphSearchService(
        request.app.state.neo4j,
        postgres,
    )

    # NEW: Hierarchical retrieval service
    hierarchical_svc = HierarchicalRetrievalService(
        postgres
    )

    reranker = VoyageRerankService(
        settings.voyage_api_key
    )

    normalizer = QueryNormalizationService()

    confidence_svc = ConfidenceScoringService()

    # --------------------------------------------------
    # 2. Create Retrieval Agent
    # --------------------------------------------------

    retrieval_agent = RetrievalAgent(
        embedder=embedder,
        dense_svc=dense_svc,
        bm25_svc=bm25_svc,
        graph_svc=graph_svc,
        reranker=reranker,
        normalizer=normalizer,
        confidence_svc=confidence_svc,
        hierarchical_svc=hierarchical_svc,
        cache_svc=cache_svc,
        weaviate_client=request.app.state.weaviate,
    )

    # --------------------------------------------------
    # 3. Initialize query state
    # --------------------------------------------------

    initial_qstate = QueryState(
        query=body.query,
        tenant_context=tenant_ctx,
        top_k=body.top_k,
    )

    # --------------------------------------------------
    # 4. Execute retrieval
    # --------------------------------------------------

    retrieved_state = await retrieval_agent.retrieve(
        initial_qstate
    )

    latency_ms = int(
        (time.time() - start_time) * 1000
    )

    # --------------------------------------------------
    # 5. Return raw results
    # --------------------------------------------------

    return {
        "query": body.query,
        "results": [
            result.model_dump()
            for result in retrieved_state.reranked_results
        ],
        "confidence": retrieved_state.confidence_level,
        "strategy": retrieved_state.strategy_used,
        "latency_ms": latency_ms,
    }
