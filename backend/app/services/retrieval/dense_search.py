"""
Dense Vector Search Service backed by Weaviate vector index.
"""
from typing import Any

from app.config.logging import get_logger
from app.infrastructure.weaviate.client import WeaviateClient
from app.models.retrieval import SearchResult

logger = get_logger(__name__)


class DenseSearchService:
    """
    Stateless service for dense vector search in Weaviate.
    """

    def __init__(self, weaviate: WeaviateClient) -> None:
        self._weaviate = weaviate

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 15,
        tenant_id: str = "default",
        knowledge_base_id: str = "default",
        permitted_access_levels: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Execute dense vector search.
        """
        if not query_vector:
            return []

        raw_results = self._weaviate.vector_search(
            query_vector=query_vector,
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
                    parent_id=r.get("parent_id") or None,
                    section=r.get("section") or None,
                    section_number=r.get("section_number"),
                    section_title=r.get("section_title") or None,
                    file_name=r.get("file_name") or None,
                    context_prefix=r.get("context_prefix"),
                    metadata=r,
                )
            )

        logger.info("dense_search.complete", hits=len(results), tenant=tenant_id)
        return results
