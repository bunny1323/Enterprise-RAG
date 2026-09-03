"""
Query Normalization and Intent Classification Service.
Normalizes text and classifies query intent without LLM latency for standard queries.
"""
import re
from pydantic import BaseModel


class NormalizedQuery(BaseModel):
    raw_query: str
    clean_query: str
    intent: str  # FACTUAL | TECHNICAL | RELATIONSHIP | VISUAL | PROCEDURAL
    extracted_entities: list[str] = []


class QueryNormalizationService:
    """
    Stateless normalization service.
    """

    def normalize(self, query: str) -> NormalizedQuery:
        clean = re.sub(r"\s+", " ", query.strip())
        lowered = clean.lower()

        # Intent heuristic detection
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
        elif re.search(r"\b[a-z0-9]+-[0-9]+\b", lowered) or any(kw in lowered for kw in ["error code", "fault code"]):
            intent = "ERROR_CODE"
        elif any(kw in lowered for kw in ["connected to", "relationship", "depends on"]):
            intent = "RELATIONSHIP"

        # Entity regex (e.g. part numbers, fault codes like E-104, P2001)
        entities = re.findall(r"\b[A-Z0-9]{2,10}-[0-9]{2,6}\b|\b[EFP]\d{3,5}\b", clean)

        return NormalizedQuery(
            raw_query=query,
            clean_query=clean,
            intent=intent,
            extracted_entities=entities,
        )
