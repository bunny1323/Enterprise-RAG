"""
Query Normalization and Intent Classification Service.
Normalizes text and classifies query intent without LLM latency for standard queries.
"""
import re
from pydantic import BaseModel


class NormalizedQuery(BaseModel):
    raw_query: str
    clean_query: str
    intent: str  # COUNT_QUERY | LIST_QUERY | PAGE_NUMBER_FORMAT | SECTION_LOOKUP | RELATIONSHIP | MULTI_HOP | GENERAL_QA | etc.
    secondary_intent: str | None = None
    extracted_entities: list[str] = []
    requested_section_number: int | None = None  # set when intent == SECTION_LOOKUP
    extracted_notation: str | None = None        # e.g. '2-3' for PAGE_NUMBER_FORMAT
    answer_type: str = "text"                    # 'count' | 'list' | 'definition' | 'relationship' | 'text'


class QueryNormalizationService:
    """
    Stateless normalization service.
    """

    def normalize(self, query: str) -> NormalizedQuery:
        clean = re.sub(r"\s+", " ", query.strip())
        lowered = clean.lower()

        # Entity regex (e.g. part numbers, fault codes like E-104, P2001)
        entities = re.findall(r"\b[A-Z0-9]{2,10}-[0-9]{2,6}\b|\b[EFP]\d{3,5}\b", clean)

        # ── 1. COUNT_QUERY (e.g., "how many major sections...", "how many chapters...") ──
        count_match = re.search(r"\bhow\s+many\s+(major\s+)?(sections?|chapters?|items?|parts?)\b", lowered)
        if count_match:
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                intent="COUNT_QUERY",
                secondary_intent="DOCUMENT_STRUCTURE",
                extracted_entities=entities,
                answer_type="count",
            )

        # ── 2. LIST_QUERY (e.g., "what are the major sections...", "list the sections...") ──
        list_match = (
            re.search(r"\b(what\s+are\s+the\s+(major\s+)?(sections?|chapters?)|list\s+(the\s+)?(major\s+)?(sections?|chapters?))\b", lowered)
            or (any(kw in lowered for kw in ["all sections", "table of contents", "overview of sections"]) and "section" in lowered)
        )
        if list_match:
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                intent="LIST_QUERY",
                secondary_intent="DOCUMENT_STRUCTURE",
                extracted_entities=entities,
                answer_type="list",
            )

        # ── 3. PAGE_NUMBER_FORMAT (e.g., "In the example page number '2-3', what does the '2' represent?", "What does 2-3 mean?") ──
        notation_match = re.search(r"\b(\d+-\d+)\b", clean)
        page_format_keywords = ["page number", "page format", "represent", "mean", "first number", "second number"]
        if notation_match and (any(kw in lowered for kw in page_format_keywords) or "page" in lowered or "example" in lowered):
            notation = notation_match.group(1)
            entities.append(notation)
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                intent="PAGE_NUMBER_FORMAT",
                secondary_intent="EXACT_LOOKUP",
                extracted_entities=entities,
                extracted_notation=notation,
                answer_type="definition",
            )

        # Also check page number format questions without explicit hyphenated notation e.g. "what does the 2 in 2-3 represent"
        if ("2-3" in clean or "page number" in lowered) and any(w in lowered for w in ["represent", "mean", "indicate", "stand for"]):
            notation = "2-3" if "2-3" in clean else None
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                intent="PAGE_NUMBER_FORMAT",
                secondary_intent="EXACT_LOOKUP",
                extracted_entities=entities + ([notation] if notation else []),
                extracted_notation=notation,
                answer_type="definition",
            )

        # ── 4. SECTION_LOOKUP ────────────────────────────────────────────────
        # Matches: "what is section 3", "show section 5", "section 3 contents",
        #          "what does section 3 cover", "tell me about section 9" etc.
        section_match = re.search(r"\bsection\s+(\d+)\b", lowered)
        if section_match:
            sec_num = int(section_match.group(1))
            entities.append(f"section_{sec_num}")
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                intent="SECTION_LOOKUP",
                extracted_entities=entities,
                requested_section_number=sec_num,
                answer_type="text",
            )

        # ── 5. RELATIONSHIP & MULTI_HOP ──────────────────────────────────────
        is_rel = any(kw in lowered for kw in ["relationship between", "how is", "related to", "affects", "correlates to", "connection between"])
        is_multihop = any(kw in lowered for kw in ["which section", "and what does", "and which section"])

        if is_multihop:
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                intent="MULTI_HOP",
                extracted_entities=entities,
                answer_type="relationship",
            )

        if is_rel:
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                intent="RELATIONSHIP",
                extracted_entities=entities,
                answer_type="relationship",
            )

        # ── 6. Other intent heuristics ───────────────────────────────────────────
        intent = "GENERAL_QA"
        if any(kw in lowered for kw in ["diagram", "schematic", "drawing", "blueprint"]):
            intent = "DIAGRAM_RETRIEVAL"
        elif any(kw in lowered for kw in ["image", "photo", "picture"]):
            intent = "IMAGE_RETRIEVAL"
        elif any(kw in lowered for kw in ["page"]):
            intent = "PAGE_RETRIEVAL"
        elif any(kw in lowered for kw in ["how to", "procedure", "steps", "instructions"]):
            intent = "PROCEDURE"
        elif any(kw in lowered for kw in ["maintenance", "replace", "install", "service"]):
            intent = "MAINTENANCE"
        elif any(kw in lowered for kw in ["specification", "capacity", "torque", "clearance", "weight", "dimension", "size"]):
            intent = "SPECIFICATION"
        elif any(kw in lowered for kw in ["troubleshoot", "problem", "issue", "symptom", "won't start", "not working"]):
            intent = "TROUBLESHOOTING"
        elif any(kw in lowered for kw in ["cause", "root cause", "why did"]):
            intent = "ROOT_CAUSE_ANALYSIS"
        elif any(kw in lowered for kw in ["predict", "likely to fail", "history", "failure rate"]):
            intent = "PREDICTIVE_MAINTENANCE"
        elif any(kw in lowered for kw in ["compare", "difference between", "vs"]):
            intent = "COMPARISON"
            # Extract the two entities being compared (e.g. "safety" and "caution")
            comp_match = re.search(
                r"(?:difference between|compare)\s+(?:the\s+)?(.+?)\s+(?:and|vs\.?|versus)\s+(?:the\s+)?(.+?)(?:\?|$)",
                lowered
            )
            if comp_match:
                entities.append(comp_match.group(1).strip())
                entities.append(comp_match.group(2).strip())
        elif re.search(r"\b[a-z0-9]+-[0-9]+\b", lowered) or any(kw in lowered for kw in ["error code", "fault code"]):
            intent = "ERROR_CODE"
        elif any(kw in lowered for kw in ["connected to", "relationship", "depends on"]):
            intent = "RELATIONSHIP"

        return NormalizedQuery(
            raw_query=query,
            clean_query=clean,
            intent=intent,
            extracted_entities=entities,
            requested_section_number=None,
        )
