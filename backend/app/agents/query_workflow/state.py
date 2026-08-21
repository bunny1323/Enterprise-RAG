"""
LangGraph state definitions for the Query Workflow.
"""
from typing import TypedDict, Any
from app.models.tenant import TenantContext
from app.models.retrieval import SearchResult
from app.models.query import Citation, SourceRef

class QueryWorkflowState(TypedDict):
    """
    State shared across the query workflow graph.
    """
    query: str
    tenant_context: TenantContext
    top_k: int
    
    # Internal state
    intent: str
    permitted_access_levels: list[str]
    cache_hit: bool
    retrieval_strategy: str
    
    # Evidence
    dense_results: list[SearchResult]
    bm25_results: list[SearchResult]
    graph_results: list[SearchResult]
    fused_results: list[SearchResult]
    reranked_results: list[SearchResult]
    
    # Validation & Refinement
    confidence_level: str
    confidence_score: float
    retries: int
    
    # Generation
    answer: str
    citations: list[Citation]
    sources: list[SourceRef]
    verification_status: str
    
    # Metadata
    trace_id: str
    latency_ms: int
    error_message: str | None
