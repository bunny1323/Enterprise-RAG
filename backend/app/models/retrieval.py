"""
Retrieval domain models.
"""
from typing import Any
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Atomic search result item from any retrieval channel (Dense, BM25, Graph)."""

    chunk_id: str
    score: float = 0.0
    content: str
    page_number: int = 0
    document_id: str = ""
    chunk_type: str = "TEXT"
    section: str | None = None
    subsection: str | None = None
    context_prefix: str | None = None
    metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Aggregated retrieval output with timing metadata."""

    results: list[SearchResult]
    strategy: str
    latency_ms: int
