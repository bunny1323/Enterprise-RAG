"""
Query and Retrieval Agent State.
"""
from typing import Any

from pydantic import BaseModel, Field

from app.models.retrieval import SearchResult
from app.models.tenant import TenantContext


class QueryState(BaseModel):
    query: str
    tenant_context: TenantContext
    top_k: int = 10
    intent: str = "FACTUAL"
    intent_confidence: float = 1.0
    dense_results: list[SearchResult] = Field(default_factory=list)
    bm25_results: list[SearchResult] = Field(default_factory=list)
    graph_results: list[SearchResult] = Field(default_factory=list)
    fused_results: list[SearchResult] = Field(default_factory=list)
    reranked_results: list[SearchResult] = Field(default_factory=list)
    retrieval_diagnostics: dict[str, Any] = Field(default_factory=dict)
    confidence_level: str = "HIGH"
    confidence_score: float = 1.0
    strategy_used: str = "hybrid_rrf"
    retry_count: int = 0
