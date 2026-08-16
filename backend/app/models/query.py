"""
Query and Generation API models.
"""
from typing import Any
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation back-reference to evidence source."""

    document_id: str
    file_name: str | None = None
    page_number: int
    section: str | None = None
    chunk_id: str
    snippet: str


class EvidenceSnippet(BaseModel):
    """Snippet of evidence provided to LLM."""

    chunk_id: str
    content: str
    page_number: int
    score: float


class SourceRef(BaseModel):
    """Source reference summarizing document level context."""

    document_id: str
    file_name: str | None = None
    page_number: int


class QueryRequest(BaseModel):
    """POST /api/v1/chat or /search request body."""

    query: str = Field(..., description="User prompt or question")
    top_k: int = Field(default=10, ge=1, le=50, description="Max evidence chunks to retrieve")
    retrieval_profile: str = Field(default="default", description="Retrieval strategy profile")


class QueryResponse(BaseModel):
    """Structured evidence-backed response from Enterprise-RAG."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: str = Field(default="HIGH", description="HIGH | MEDIUM | LOW")
    confidence_score: float = Field(default=1.0)
    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    verification_status: str = Field(default="SUPPORTED", description="SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED")
    trace_id: str
    retrieval_strategy: str
    latency_ms: int
