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
        intent = "FACTUAL"
        if any(kw in lowered for kw in ["diagram", "figure", "schematic", "drawing", "image", "photo"]):
            intent = "VISUAL"
        elif any(kw in lowered for kw in ["connected to", "wired to", "relationship", "replaces", "depends on"]):
            intent = "RELATIONSHIP"
        elif any(kw in lowered for kw in ["how to", "procedure", "steps", "instructions", "maintenance", "repair"]):
            intent = "PROCEDURAL"
        elif re.search(r"\b[a-z0-9]+-[0-9]+\b", lowered) or any(kw in lowered for kw in ["fault", "error", "code", "part #"]):
            intent = "TECHNICAL"

        # Entity regex (e.g. part numbers, fault codes like E-104, P2001)
        entities = re.findall(r"\b[A-Z0-9]{2,10}-[0-9]{2,6}\b|\b[EFP]\d{3,5}\b", clean)

        return NormalizedQuery(
            raw_query=query,
            clean_query=clean,
            intent=intent,
            extracted_entities=entities,
        )
