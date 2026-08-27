"""
Hierarchical retrieval service.

After retrieving and reranking precise child chunks, this service
resolves their parent chunks so the generation stage receives
broader contextual information.
"""

from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
from app.models.retrieval import SearchResult

logger = get_logger(__name__)


class HierarchicalRetrievalService:
    """Resolve child retrieval hits to their parent context."""

    def __init__(self, postgres: PostgresClient) -> None:
        self._postgres = postgres

    async def expand_to_parents(
        self,
        results: list[SearchResult],
        tenant_id: str = "default",
        knowledge_base_id: str = "default",
    ) -> list[SearchResult]:
        """Replace child results with their parent context where available."""

        if not results:
            return []

        parent_ids = {
            result.parent_id
            for result in results
            if result.parent_id
        }

        if not parent_ids:
            return results

        rows = await self._postgres.fetch(
            """
            SELECT
                chunk_id,
                parent_id,
                document_id,
                content,
                page_number,
                chunk_type,
                section,
                subsection,
                context_prefix,
                metadata
            FROM chunks
            WHERE chunk_id = ANY($1::text[])
              AND tenant_id = $2
              AND knowledge_base_id = $3
            """,
            list(parent_ids),
            tenant_id,
            knowledge_base_id,
        )

        parents = {row["chunk_id"]: row for row in rows}

        expanded: list[SearchResult] = []
        seen_parents: set[str] = set()

        for result in results:
            if not result.parent_id:
                expanded.append(result)
                continue

            parent = parents.get(result.parent_id)

            if parent is None:
                # Safe fallback: keep original child.
                expanded.append(result)
                continue

            parent_id = parent["chunk_id"]

            if parent_id in seen_parents:
                continue

            seen_parents.add(parent_id)

            expanded.append(
                result.model_copy(
                    update={
                        "chunk_id": parent_id,
                        "parent_id": None,
                        "content": parent["content"],
                        "page_number": parent["page_number"] or result.page_number,
                        "chunk_type": parent["chunk_type"] or result.chunk_type,
                        "section": parent.get("section"),
                        "subsection": parent.get("subsection"),
                        "context_prefix": parent.get("context_prefix"),
                        "metadata": {
                            **result.metadata,
                            "retrieved_child_id": result.chunk_id,
                            "hierarchical_expansion": True,
                        },
                    }
                )
            )

        logger.info(
            "hierarchical_retrieval.complete",
            input_count=len(results),
            output_count=len(expanded),
        )

        return expanded
