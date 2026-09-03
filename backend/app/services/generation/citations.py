from app.config.logging import get_logger
from app.models.query import Citation, SourceRef
from app.models.retrieval import SearchResult

logger = get_logger(__name__)

# Canonical mapping for Hyundai Service Manual Sections 1 to 9
CANONICAL_SECTIONS: dict[int, str] = {
    1: "GENERAL",
    2: "STRUCTURE AND FUNCTION",
    3: "HYDRAULIC SYSTEM",
    4: "ELECTRICAL SYSTEM",
    5: "MECHATRONICS SYSTEM",
    6: "TROUBLESHOOTING",
    7: "MAINTENANCE STANDARD",
    8: "DISASSEMBLY AND ASSEMBLY",
    9: "COMPONENT MOUNTING TORQUE",
}


class CitationService:
    """
    Stateless citation mapping and validation engine.
    """

    def map_citations(
        self,
        answer: str,
        evidence: list[SearchResult],
    ) -> tuple[list[Citation], list[SourceRef]]:
        """
        Map evidence chunks to structured Citation objects and unique SourceRefs with strict consistency checks.
        """
        citations: list[Citation] = []
        sources_seen: set[str] = set()
        sources: list[SourceRef] = []

        for item in evidence:
            snippet = item.content[:150] + "..." if len(item.content) > 150 else item.content

            # Structural consistency validation:
            # section_number must correspond to canonical section_title if known
            sec_num = item.section_number
            sec_title = (item.section_title or "").upper()
            if sec_num in CANONICAL_SECTIONS:
                expected_title = CANONICAL_SECTIONS[sec_num]
                if sec_title and expected_title not in sec_title and sec_title not in expected_title:
                    logger.warning(
                        "citation.structural_inconsistency",
                        section_number=sec_num,
                        found_title=sec_title,
                        expected_title=expected_title,
                        chunk_id=item.chunk_id,
                    )

            # Metadata fallback for file_name if absent in SearchResult
            file_name = item.file_name or item.metadata.get("file_name") or "Hyundai_R215L_Manual.pdf"

            citations.append(
                Citation(
                    document_id=item.document_id,
                    file_name=file_name,
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
                        file_name=file_name,
                        page_number=item.page_number,
                    )
                )

        return citations, sources
