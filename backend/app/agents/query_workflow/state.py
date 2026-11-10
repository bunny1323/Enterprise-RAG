"""
LangGraph state definitions for the Query Workflow.
"""
from typing import Any, NotRequired, TypedDict
from app.models.tenant import TenantContext
from app.models.retrieval import SearchResult
from app.models.query import Citation, SourceRef

class QueryWorkflowState(TypedDict):
    """
    State shared across the query workflow graph.
    """
    query: str
    original_query: NotRequired[str]
    normalized_query: NotRequired[str]
    refined_query: NotRequired[str]
    tenant_context: TenantContext
    top_k: int
    
    # Internal state
    intent: str
    intent_confidence: float
    permitted_access_levels: list[str]
    policy_authorized: NotRequired[bool]
    cache_hit: bool
    retrieval_strategy: str
    
    # Evidence
    dense_results: list[SearchResult]
    bm25_results: list[SearchResult]
    graph_results: list[SearchResult]
    fused_results: list[SearchResult]
    reranked_results: list[SearchResult]
    retrieved_evidence: NotRequired[list[SearchResult]]
    
    # Validation & Refinement
    confidence_level: str
    confidence_score: float
    retries: int
    
    # Generation
    answer: str
    citations: list[Citation]
    sources: list[SourceRef]
    verification_status: str
    refusal_reason: NotRequired[str | None]
    
    # Coverage & Strict Grounding
    evidence_coverage: str
    answerable: bool
    retrieval_trace: dict[str, Any]
    retrieval_diagnostics: NotRequired[dict[str, Any]]

    # Metadata
    trace_id: str
    latency_ms: int
    error_message: str | None
