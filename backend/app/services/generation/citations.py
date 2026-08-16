"""
Citation Mapping Service.
Extracts structured citations and document source references from retrieved evidence chunks.
"""
from app.models.query import Citation, SourceRef
from app.models.retrieval import SearchResult


class CitationService:
    """
    Stateless citation mapping engine.
    """

    def map_citations(
        self,
        answer: str,
        evidence: list[SearchResult],
    ) -> tuple[list[Citation], list[SourceRef]]:
        """
        Map evidence chunks to structured Citation objects and unique SourceRefs.
        """
        citations: list[Citation] = []
        sources_seen: set[str] = set()
        sources: list[SourceRef] = []

        for item in evidence:
            snippet = item.content[:150] + "..." if len(item.content) > 150 else item.content

            citations.append(
                Citation(
                    document_id=item.document_id,
                    page_number=item.page_number,
                    section=item.section,
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
                        page_number=item.page_number,
                    )
                )

        return citations, sources
