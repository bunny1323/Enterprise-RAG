"""
Query and Chat API Routes.
POST /api/v1/chat   — Grounded RAG conversational endpoint
POST /api/v1/search — Raw search endpoint (Retrieval only)
POST /api/v1/rag/debug — RAG pipeline debug endpoint
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
    CheckpointerDep,
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
from app.services.cache.semantic import SemanticCacheService
from app.services.confidence.service import ConfidenceScoringService
from app.services.embeddings.service import EmbeddingService
from app.services.generation.citations import CitationService
from app.services.generation.hallucination import (
    GroundednessVerificationService,
)
from app.services.generation.llm import LLMProvider, LLMProviderError
from app.services.query.normalization import QueryNormalizationService
from app.services.retrieval.bm25_search import BM25SearchService
from app.services.retrieval.dense_search import DenseSearchService
from app.services.retrieval.graph_search import GraphSearchService
from app.services.retrieval.hierarchical import HierarchicalRetrievalService
from app.services.retrieval.reranking import RRFRerankerService

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


from app.services.retrieval.structure_search import StructureSearchService


def _build_retrieval_agent(request: Request, postgres, cache_svc) -> RetrievalAgent:
    """Build the RetrievalAgent from app.state services."""
    embedder: EmbeddingService = request.app.state.embedder
    return RetrievalAgent(
        embedder=embedder,
        dense_svc=DenseSearchService(request.app.state.weaviate),
        bm25_svc=BM25SearchService(request.app.state.weaviate),
        graph_svc=GraphSearchService(request.app.state.neo4j, postgres),
        reranker=RRFRerankerService(),
        normalizer=QueryNormalizationService(),
        confidence_svc=ConfidenceScoringService(),
        hierarchical_svc=HierarchicalRetrievalService(postgres),
        cache_svc=cache_svc,
        weaviate_client=request.app.state.weaviate,
        structure_svc=StructureSearchService(postgres),
    )


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
    checkpointer: CheckpointerDep,
) -> QueryResponse:
    """
    Full Enterprise RAG query execution pipeline:

    Query → Intent Detection → Retrieval Strategy Selection
    → Dense + BM25 + Graph → RRF Fusion → Optional Reranker
    → Evidence Selection → LLM → Intent-Aware Response
    """

    start_time = time.time()
    trace_id = f"tr_{uuid.uuid4().hex[:12]}"

    logger.info(
        "chat.request_received",
        trace_id=trace_id,
        query=body.query[:50],
        tenant=tenant_ctx.tenant_id,
    )

    # 1. Build dependencies
    redis_client: RedisClient | None = getattr(request.app.state, "redis", None)
    cache_svc = CacheService(redis_client, postgres) if redis_client else None
    semantic_cache_svc = SemanticCacheService(redis_client) if redis_client else None
    embedder: EmbeddingService = request.app.state.embedder

    retrieval_agent = _build_retrieval_agent(request, postgres, cache_svc)

    llm_provider: LLMProvider = request.app.state.llm_provider
    citation_svc = CitationService()
    verifier = GroundednessVerificationService()

    # 2. Build LangGraph Nodes and Graph
    nodes = QueryNodes(
        retrieval_agent=retrieval_agent,
        llm_provider=llm_provider,
        citation_svc=citation_svc,
        verifier_svc=verifier,
        opa_client=getattr(request.app.state, "opa", None),
        cache_svc=cache_svc,
        semantic_cache_svc=semantic_cache_svc,
        embedder=embedder,
    )

    # 3. Initialize workflow state
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

    # 4. Execute LangGraph workflow
    try:
        graph = build_query_graph(nodes, checkpointer=checkpointer) if checkpointer else build_query_graph(nodes)
        if not checkpointer:
            logger.warning("chat.missing_postgres_checkpointer", trace_id=trace_id)

        config = {"configurable": {"thread_id": trace_id, "tenant_id": tenant_ctx.tenant_id}}
        final_state = await graph.ainvoke(initial_state, config=config)
    except LLMProviderError as err:
        logger.warning("chat.llm_unavailable", trace_id=trace_id, error=str(err))
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(err)) from err
    except Exception as err:
        import traceback
        tb = traceback.format_exc()
        logger.error("chat.unhandled_exception", trace_id=trace_id, error=str(err), traceback=tb)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"{str(err)}\n\n{tb}") from err
    if final_state.get("error_message"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=final_state["error_message"])

    latency_ms = int((time.time() - start_time) * 1000)

    # 5. Build structured multimodal response
    images_list = []
    tables_list = []
    pages_list: set[int] = set()

    for item in final_state["reranked_results"]:
        if item.page_number:
            pages_list.add(item.page_number)
        if item.chunk_type in ("IMAGE", "DIAGRAM"):
            images_list.append({
                "chunk_id": item.chunk_id,
                "document_id": str(item.document_id),
                "page_number": item.page_number,
                "url": f"/api/v1/images/{item.document_id}_page_{item.page_number}.png",
            })
        elif item.chunk_type == "TABLE":
            tables_list.append({
                "chunk_id": item.chunk_id,
                "content": item.content,
                "page_number": item.page_number,
            })

    response_data = QueryResponse(
        answer=final_state["answer"],
        intent=final_state.get("intent", "GENERAL_QA"),
        intent_confidence=1.0,
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
        images=images_list,
        tables=tables_list,
        pages=sorted(pages_list),
        retrieval_trace={
            "intent": final_state.get("intent", ""),
            "strategy": final_state.get("retrieval_strategy", ""),
            "dense_count": len(final_state.get("dense_results", [])),
            "bm25_count": len(final_state.get("bm25_results", [])),
            "graph_count": len(final_state.get("graph_results", [])),
            "fused_count": len(final_state.get("fused_results", [])),
            "final_count": len(final_state.get("reranked_results", [])),
            "evidence_coverage": final_state.get("evidence_coverage", "COMPLETE"),
            "answerable": final_state.get("answerable", True),
            **(final_state.get("retrieval_trace") or {}),
            "latency_ms": latency_ms,
        },
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
    postgres: PostgresDep,
) -> dict:
    """
    Execute multi-channel retrieval and reranking without LLM generation.
    Pipeline: Dense + BM25 + Graph → RRF Fusion → Reranker → Hierarchical Expansion → Confidence Scoring
    """
    start_time = time.time()

    redis_client: RedisClient | None = getattr(request.app.state, "redis", None)
    cache_svc = CacheService(redis_client, postgres) if redis_client else None

    retrieval_agent = _build_retrieval_agent(request, postgres, cache_svc)

    initial_qstate = QueryState(
        query=body.query,
        tenant_context=tenant_ctx,
        top_k=body.top_k,
    )

    retrieved_state = await retrieval_agent.retrieve(initial_qstate)

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "query": body.query,
        "intent": retrieved_state.intent,
        "strategy": retrieved_state.strategy_used,
        "results": [result.model_dump() for result in retrieved_state.reranked_results],
        "confidence": retrieved_state.confidence_level,
        "confidence_score": retrieved_state.confidence_score,
        "dense_count": len(retrieved_state.dense_results),
        "bm25_count": len(retrieved_state.bm25_results),
        "graph_count": len(retrieved_state.graph_results),
        "latency_ms": latency_ms,
    }


@router.post(
    "/rag/debug",
    summary="RAG pipeline debug — safe operational information",
)
async def rag_debug(
    request: Request,
    body: QueryRequest,
    tenant_ctx: TenantContextDep,
    postgres: PostgresDep,
) -> dict:
    """
    Expose RAG pipeline trace without secrets or chain-of-thought.
    Returns: intent, strategy, retrieval counts, latency, source IDs.
    Does NOT expose: API keys, passwords, tokens, or internal reasoning.
    """
    start_time = time.time()

    redis_client: RedisClient | None = getattr(request.app.state, "redis", None)
    cache_svc = CacheService(redis_client, postgres) if redis_client else None

    retrieval_agent = _build_retrieval_agent(request, postgres, cache_svc)

    # Just run the normalizer to detect intent before full retrieval
    from app.services.query.normalization import QueryNormalizationService
    normalizer = QueryNormalizationService()
    norm = normalizer.normalize(body.query)

    initial_qstate = QueryState(
        query=body.query,
        tenant_context=tenant_ctx,
        top_k=body.top_k,
    )

    retrieved_state = await retrieval_agent.retrieve(initial_qstate)

    latency_ms = int((time.time() - start_time) * 1000)

    return {
        "query": body.query,
        "detected_intent": norm.intent,
        "extracted_entities": norm.extracted_entities,
        "selected_retrieval_strategy": retrieved_state.strategy_used,
        "embedding_provider": "local",
        "embedding_model": getattr(request.app.state.embedder, "_model_name", "unknown"),
        "embedding_dimension": getattr(request.app.state.embedder, "EMBEDDING_DIM", 384),
        "dense_result_count": len(retrieved_state.dense_results),
        "bm25_result_count": len(retrieved_state.bm25_results),
        "graph_result_count": len(retrieved_state.graph_results),
        "fused_result_count": len(retrieved_state.fused_results),
        "reranker_status": "rrf_fusion",
        "final_evidence_count": len(retrieved_state.reranked_results),
        "source_ids": [r.chunk_id for r in retrieved_state.reranked_results],
        "page_numbers": list({r.page_number for r in retrieved_state.reranked_results if r.page_number}),
        "confidence": retrieved_state.confidence_level,
        "confidence_score": retrieved_state.confidence_score,
        "retrieval_latency_ms": latency_ms,
        "reranking_latency_ms": 0,
        "total_latency_ms": latency_ms,
    }
