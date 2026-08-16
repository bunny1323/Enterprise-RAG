"""
Graph Relationship Search Service backed by Neo4j graph database.
"""
from typing import Any

from app.config.logging import get_logger
from app.infrastructure.neo4j.client import Neo4jClient
from app.infrastructure.postgres.client import PostgresClient
from app.models.retrieval import SearchResult

logger = get_logger(__name__)


class GraphSearchService:
    """
    Service for querying entity relationships and document trees in Neo4j.
    """

    def __init__(self, neo4j: Neo4jClient, postgres: PostgresClient) -> None:
        self._neo4j = neo4j
        self._postgres = postgres

    async def search(
        self,
        entity_name: str,
        tenant_id: str = "default",
        knowledge_base_id: str = "default",
        max_hops: int = 2,
    ) -> list[SearchResult]:
        """
        Execute multi-hop graph search for entities or section names.
        """
        if not entity_name.strip():
            return []

        records = await self._neo4j.graph_search(
            entity_name=entity_name,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            max_hops=max_hops,
        )

        chunk_ids = [r["chunk_id"] for r in records if "chunk_id" in r]
        if not chunk_ids:
            return []

        # Fetch full chunk details from PostgreSQL
        rows = await self._postgres.fetch(
            """
            SELECT chunk_id, document_id, content, page_number, chunk_type, context_prefix, section
            FROM chunks
            WHERE chunk_id = ANY($1::text[]) AND tenant_id = $2
            """,
            chunk_ids,
            tenant_id,
        )

        results: list[SearchResult] = []
        for r in rows:
            results.append(
                SearchResult(
                    chunk_id=r["chunk_id"],
                    score=0.8,  # Default graph match confidence score
                    content=r["content"],
                    page_number=r["page_number"] or 0,
                    document_id=str(r["document_id"]),
                    chunk_type=r["chunk_type"] or "TEXT",
                    section=r.get("section"),
                    context_prefix=r.get("context_prefix"),
                    metadata=r,
                )
            )

        logger.info("graph_search.complete", hits=len(results), tenant=tenant_id)
        return results
