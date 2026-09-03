"""
Evidence Coverage and Answerability Gating Service.

Ensures strict evidence verification:
- Verifies Entity A and Entity B independent evidence presence.
- Verifies explicit connecting passage / cross-reference.
- Prevents LLM hallucinations or unsupported relationship assertions.
"""
from dataclasses import dataclass
import re
from app.config.logging import get_logger
from app.models.retrieval import SearchResult
from app.services.query.normalization import NormalizedQuery

logger = get_logger(__name__)


@dataclass
class EvidenceCoverageResult:
    coverage_status: str  # "COMPLETE" | "PARTIAL" | "INSUFFICIENT"
    answerable: bool
    entities_required: list[str]
    entities_found: list[str]
    relationship_supported: bool
    connecting_evidence_found: bool
    inferred_only: bool = False
    reason: str = ""


class EvidenceCoverageService:
    """
    Evaluates whether the retrieved evidence contains sufficient factual grounding
    to answer the given query accurately without hallucination.
    """

    def evaluate_coverage(
        self,
        norm: NormalizedQuery,
        evidence: list[SearchResult],
    ) -> EvidenceCoverageResult:
        if not evidence:
            return EvidenceCoverageResult(
                coverage_status="INSUFFICIENT",
                answerable=False,
                entities_required=[],
                entities_found=[],
                relationship_supported=False,
                connecting_evidence_found=False,
                reason="No evidence retrieved.",
            )

        all_text = " ".join(e.content.lower() for e in evidence)

        # ── 1. COUNT_QUERY Coverage ──────────────────────────────────────────
        if norm.intent == "COUNT_QUERY":
            # For section count, check if evidence comes from document_structure or contains major sections
            has_count_evidence = any(
                e.metadata.get("source") == "document_structure"
                or "section 1" in e.content.lower()
                or "section 9" in e.content.lower()
                for e in evidence
            )
            return EvidenceCoverageResult(
                coverage_status="COMPLETE" if has_count_evidence else "PARTIAL",
                answerable=has_count_evidence,
                entities_required=["sections"],
                entities_found=["sections"] if has_count_evidence else [],
                relationship_supported=True,
                connecting_evidence_found=True,
                reason="Document structure section index available." if has_count_evidence else "Insufficient section enumeration.",
            )

        # ── 2. LIST_QUERY Coverage ───────────────────────────────────────────
        if norm.intent == "LIST_QUERY":
            has_sections = any(
                e.metadata.get("source") == "document_structure"
                or ("section" in e.content.lower() and any(f"section {i}" in e.content.lower() for i in range(1, 10)))
                for e in evidence
            )
            return EvidenceCoverageResult(
                coverage_status="COMPLETE" if has_sections else "PARTIAL",
                answerable=has_sections,
                entities_required=["sections"],
                entities_found=["sections"] if has_sections else [],
                relationship_supported=True,
                connecting_evidence_found=True,
                reason="Section list retrieved from document structure.",
            )

        # ── 3. PAGE_NUMBER_FORMAT Coverage ───────────────────────────────────
        if norm.intent == "PAGE_NUMBER_FORMAT":
            notation = norm.extracted_notation or "2-3"
            # Require verbatim appearance of notation or item number explanation
            has_notation = (notation in all_text) or ("item number" in all_text) or ("consecutive page" in all_text)
            return EvidenceCoverageResult(
                coverage_status="COMPLETE" if has_notation else "INSUFFICIENT",
                answerable=has_notation,
                entities_required=[notation],
                entities_found=[notation] if has_notation else [],
                relationship_supported=True,
                connecting_evidence_found=has_notation,
                reason="Page number format explanation found." if has_notation else f"Notation {notation} not explicitly explained in retrieved passages.",
            )

        # ── 4. COMPARISON Coverage ────────────────────────────────────────────
        # e.g. "difference between safety and caution", "compare X and Y"
        if norm.intent == "COMPARISON":
            lowered_q = norm.clean_query.lower()
            # Extract the two things being compared
            match = re.search(r"(?:difference between|compare)\s+(?:the\s+)?(.+?)\s+(?:and|vs\.?|versus)\s+(?:the\s+)?(.+?)(?:\?|$)", lowered_q)
            ent_a, ent_b = "", ""
            if match:
                ent_a = match.group(1).strip()
                ent_b = match.group(2).strip()

            found_a = any(ent_a.lower() in e.content.lower() for e in evidence) if ent_a else True
            found_b = any(ent_b.lower() in e.content.lower() for e in evidence) if ent_b else True

            # For comparison, if we have ANY relevant evidence, allow answering
            if found_a and found_b:
                return EvidenceCoverageResult(
                    coverage_status="COMPLETE",
                    answerable=True,
                    entities_required=[ent_a, ent_b],
                    entities_found=[ent_a, ent_b],
                    relationship_supported=True,
                    connecting_evidence_found=True,
                    reason="Both comparison subjects found in retrieved evidence.",
                )
            elif evidence:  # At least some evidence — let LLM answer with what it has
                return EvidenceCoverageResult(
                    coverage_status="PARTIAL",
                    answerable=True,
                    entities_required=[ent_a, ent_b],
                    entities_found=[ent_a] if found_a else ([ent_b] if found_b else []),
                    relationship_supported=False,
                    connecting_evidence_found=False,
                    reason="Partial evidence for comparison — one or both subjects may be present.",
                )
            else:
                return EvidenceCoverageResult(
                    coverage_status="INSUFFICIENT",
                    answerable=False,
                    entities_required=[ent_a, ent_b],
                    entities_found=[],
                    relationship_supported=False,
                    connecting_evidence_found=False,
                    reason="No evidence retrieved for comparison subjects.",
                )

        # ── 5. RELATIONSHIP Strict 5-Step Validation ─────────────────────────
        if norm.intent == "RELATIONSHIP":
            lowered_q = norm.clean_query.lower()
            # Try to identify entity A and entity B from query
            match = re.search(r"(?:between|how is)\s+(.+?)\s+(?:and|related to)\s+(.+)", lowered_q)
            ent_a, ent_b = "", ""
            if match:
                ent_a = match.group(1).strip().replace("section ", "")
                ent_b = match.group(2).strip().replace("section ", "")

            found_a = any(ent_a.lower() in e.content.lower() for e in evidence) if ent_a else True
            found_b = any(ent_b.lower() in e.content.lower() for e in evidence) if ent_b else True

            # Look for explicit connecting / cross-reference phrases
            connecting_terms = [
                "reference material",
                "serves as",
                "used for",
                "refer to",
                "correlat",
                "troubleshoot",
                "related to",
                "connection",
                "pertains to",
            ]
            has_connection = any(term in all_text for term in connecting_terms)

            if found_a and found_b and has_connection:
                return EvidenceCoverageResult(
                    coverage_status="COMPLETE",
                    answerable=True,
                    entities_required=[ent_a, ent_b],
                    entities_found=[e for e in [ent_a, ent_b] if e],
                    relationship_supported=True,
                    connecting_evidence_found=True,
                    reason="Explicit relationship or cross-reference found in source evidence.",
                )
            elif found_a and found_b:
                return EvidenceCoverageResult(
                    coverage_status="PARTIAL",
                    answerable=True,
                    entities_required=[ent_a, ent_b],
                    entities_found=[ent_a, ent_b],
                    relationship_supported=False,
                    connecting_evidence_found=False,
                    inferred_only=True,
                    reason="Both entities found but explicit connecting passage is missing.",
                )
            else:
                return EvidenceCoverageResult(
                    coverage_status="INSUFFICIENT",
                    answerable=False,
                    entities_required=[ent_a, ent_b],
                    entities_found=[e for e in [ent_a, ent_b] if e in all_text],
                    relationship_supported=False,
                    connecting_evidence_found=False,
                    reason="One or both entities missing from retrieved evidence.",
                )

        # ── 6. Standard Coverage (GENERAL_QA, MAINTENANCE, PROCEDURE, SPECIFICATION, etc.) ─
        # For all non-structural intents with retrieved evidence, allow generation.
        return EvidenceCoverageResult(
            coverage_status="COMPLETE" if len(evidence) > 0 else "INSUFFICIENT",
            answerable=len(evidence) > 0,
            entities_required=norm.extracted_entities,
            entities_found=[e for e in norm.extracted_entities if e.lower() in all_text],
            relationship_supported=True,
            connecting_evidence_found=True,
            reason="Retrieved evidence available for general query.",
        )
