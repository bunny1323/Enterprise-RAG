"""
Document Structure Search Service.

Queries the canonical `document_structure` table in PostgreSQL for:
- Document-level section counts (COUNT_QUERY)
- Document-level section lists (LIST_QUERY / DOCUMENT_STRUCTURE)
- Page number format explanations (PAGE_NUMBER_FORMAT / EXACT_LOOKUP)
- Exact structural references
"""
from typing import Any
from app.config.logging import get_logger
from app.infrastructure.postgres.client import PostgresClient
from app.models.retrieval import SearchResult

logger = get_logger(__name__)


class StructureSearchService:
    """
    Service for querying canonical document structure directly from PostgreSQL.
    """

    def __init__(self, postgres_client: PostgresClient) -> None:
        self._postgres = postgres_client

    async def count_sections(
        self,
        tenant_id: str = "default",
        knowledge_base_id: str = "default",
        document_id: str | None = None,
    ) -> int:
        """Count unique major sections."""
        count = await self._postgres.count_sections(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        )
        logger.info("structure_search.count_sections", count=count, tenant=tenant_id)
        return count

    async def list_sections(
        self,
        tenant_id: str = "default",
        knowledge_base_id: str = "default",
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """
        Fetch all canonical sections and format as SearchResults for seamless RAG evidence injection.
        """
        rows = await self._postgres.get_sections(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        )
        results: list[SearchResult] = []
        for row in rows:
            sec_num = row.get("number")
            title = row.get("title") or f"SECTION {sec_num}"
            raw_text = row.get("raw_text") or f"SECTION {sec_num} {title}"
            page_num = row.get("page_number") or 1
            doc_id = str(row.get("document_id") or "")

            results.append(
                SearchResult(
                    chunk_id=f"struct::section::{sec_num}::{page_num}",
                    score=1.0,
                    content=f"SECTION {sec_num}: {title}\n{raw_text}".strip(),
                    page_number=page_num,
                    document_id=doc_id,
                    chunk_type="TEXT",
                    section=f"SECTION {sec_num} {title}".strip(),
                    section_number=sec_num,
                    section_title=title,
                    metadata={"source": "document_structure", **(row.get("metadata") or {})},
                )
            )
        logger.info("structure_search.list_sections", sections_found=len(results), tenant=tenant_id)
        return results

    async def lookup_page_format(
        self,
        notation: str | None = None,
        tenant_id: str = "default",
        knowledge_base_id: str = "default",
    ) -> list[SearchResult]:
        """
        Fetch page-format explanation entries from document_structure.
        """
        rows = await self._postgres.get_page_format_entries(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            notation=notation,
        )
        results: list[SearchResult] = []
        for idx, row in enumerate(rows):
            page_num = row.get("page_number") or 1
            raw_text = row.get("raw_text") or ""
            doc_id = str(row.get("document_id") or "")
            meta = row.get("metadata") or {}

            results.append(
                SearchResult(
                    chunk_id=f"struct::page_format::{idx}::{page_num}",
                    score=1.0,
                    content=raw_text,
                    page_number=page_num,
                    document_id=doc_id,
                    chunk_type="TEXT",
                    section="PAGE NUMBER FORMAT",
                    metadata={"source": "document_structure", "structure_type": "page_format", **meta},
                )
            )
        logger.info(
            "structure_search.lookup_page_format",
            notation=notation,
            entries_found=len(results),
            tenant=tenant_id,
        )
        return results
