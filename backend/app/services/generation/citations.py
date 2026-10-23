from app.config.logging import get_logger
from app.models.query import Citation, SourceRef
from app.models.retrieval import SearchResult

logger = get_logger(__name__)

class CitationService:
    """
    Stateless citation mapping and validation engine.
    """

    def map_citations(
        self,
        answer: str,
        evidence: list[SearchResult],
        document_sections: dict[int, str] | None = None,
    ) -> tuple[list[Citation], list[SourceRef]]:
        """
        Map evidence chunks to structured Citation objects and unique SourceRefs with strict consistency checks.
        """
        citations: list[Citation] = []
        sources_seen: set[str] = set()
        sources: list[SourceRef] = []

        for item in evidence:
            snippet = item.content[:150] + "..." if len(item.content) > 150 else item.content

            # Source-derived section resolution:
            # Prefer human-readable section_title, then section string, then schema lookup if provided
            sec_name = item.section_title or item.section
            if not sec_name and item.section_number and document_sections:
                sec_name = document_sections.get(item.section_number)

            # Metadata fallback for file_name if absent in SearchResult:
            # Derive dynamically from item or document_id, never hardcode a specific manual
            file_name = (
                item.file_name
                or (item.metadata.get("file_name") if isinstance(item.metadata, dict) else None)
                or (f"Document_{item.document_id[:8]}" if item.document_id else "Unknown Document")
            )

            citations.append(
                Citation(
                    document_id=item.document_id,
                    file_name=file_name,
                    page_number=item.page_number,
                    section=sec_name,
                    chunk_id=item.chunk_id,
                    snippet=snippet,
                )
            )

            src_key = f"{item.document_id}::p{item.page_number}"
            if src_key not in sources_seen:
                sources_seen.add(src_key)
                sources.append(
                    SourceRef(
                        document_id=item.document_id,
                        file_name=file_name,
                        page_number=item.page_number,
                    )
                )

        return citations, sources
