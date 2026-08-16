"""
BM25 Keyword Search Service backed by Weaviate BM25 inverted index.
"""
from typing import Any

from app.config.logging import get_logger
from app.infrastructure.weaviate.client import WeaviateClient
from app.models.retrieval import SearchResult

logger = get_logger(__name__)


class BM25SearchService:
    """
    Stateless service for sparse BM25 keyword search in Weaviate.
    """

    def __init__(self, weaviate: WeaviateClient) -> None:
        self._weaviate = weaviate

    async def search(
        self,
        query_text: str,
        top_k: int = 15,
        tenant_id: str = "default",
        knowledge_base_id: str = "default",
        permitted_access_levels: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Execute BM25 keyword search.
        """
        if not query_text.strip():
            return []

        raw_results = self._weaviate.bm25_search(
            query_text=query_text,
            top_k=top_k,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            permitted_access_levels=permitted_access_levels or ["PUBLIC", "INTERNAL"],
        )

        results: list[SearchResult] = []
        for r in raw_results:
            results.append(
                SearchResult(
                    chunk_id=r.get("chunk_id", ""),
                    score=float(r.get("score", 0.0)),
                    content=r.get("content", ""),
                    page_number=int(r.get("page_number", 0)),
                    document_id=r.get("document_id", ""),
                    chunk_type=r.get("chunk_type", "TEXT"),
                    context_prefix=r.get("context_prefix"),
                    metadata=r,
                )
            )

        logger.info("bm25_search.complete", hits=len(results), tenant=tenant_id)
        return results
