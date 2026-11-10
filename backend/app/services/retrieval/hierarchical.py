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

    async def get_accessible_chunk_ids(
        self,
        chunk_ids: list[str],
        tenant_id: str = "default",
        knowledge_base_id: str = "default",
        permitted_access_levels: list[str] | None = None,
    ) -> set[str]:
        """Validate vector-store hits against authoritative PostgreSQL classifications."""
        if not chunk_ids or permitted_access_levels == []:
            return set()
        access_levels = (
            permitted_access_levels
            if permitted_access_levels is not None
            else ["PUBLIC", "INTERNAL"]
        )
        rows = await self._postgres.fetch(
            """
            SELECT chunk_id
            FROM chunks
            WHERE chunk_id = ANY($1::text[])
              AND tenant_id = $2
              AND knowledge_base_id = $3
              AND access_classification = ANY($4::text[])
            """,
            list(set(chunk_ids)),
            tenant_id,
            knowledge_base_id,
            access_levels,
        )
        return {str(row["chunk_id"]) for row in rows}

    async def expand_to_parents(
        self,
        results: list[SearchResult],
        tenant_id: str = "default",
        knowledge_base_id: str = "default",
        permitted_access_levels: list[str] | None = None,
    ) -> list[SearchResult]:
        """Replace child results with their parent context where available."""

        if not results or permitted_access_levels == []:
            return []

        parent_ids = {
            result.parent_id
            for result in results
            if result.parent_id
        }

        if not parent_ids:
            return results
        access_levels = (
            permitted_access_levels
            if permitted_access_levels is not None
            else ["PUBLIC", "INTERNAL"]
        )

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
              AND access_classification = ANY($4::text[])
            """,
            list(parent_ids),
            tenant_id,
            knowledge_base_id,
            access_levels,
        )

        parents = {row["chunk_id"]: row for row in rows}

        expanded: list[SearchResult] = []
        seen_parents: set[str] = set()
        seen_chunks: set[str] = set()

        def append_unique(result: SearchResult) -> None:
            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                expanded.append(result)

        for result in results:
            if not result.parent_id:
                append_unique(result)
                continue

            parent = parents.get(result.parent_id)

            if parent is None:
                # Safe fallback: keep original child.
                append_unique(result)
                continue

            parent_id = parent["chunk_id"]

            if parent_id in seen_parents:
                continue

            seen_parents.add(parent_id)

            append_unique(
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
