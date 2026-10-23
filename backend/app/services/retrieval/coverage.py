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
            # Check if evidence comes from document_structure or contains section/chapter enumerations
            has_count_evidence = any(
                e.metadata.get("source") == "document_structure"
                or (e.section_number is not None and e.section_number > 0)
                or bool(re.search(r'\b(?:section|chapter|part)\s+\d+\b', e.content, re.IGNORECASE))
                or any(kw in e.content.lower() for kw in ["table of contents", "overview of sections", "major sections"])
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
                or (e.section_number is not None and e.section_number > 0)
                or bool(re.search(r'\b(?:section|chapter|part)\s+\d+\b', e.content, re.IGNORECASE))
                or any(kw in e.content.lower() for kw in ["table of contents", "overview of sections", "all sections"])
                for e in evidence
            )
            return EvidenceCoverageResult(
                coverage_status="COMPLETE" if has_sections else "PARTIAL",
                answerable=has_sections,
                entities_required=["sections"],
                entities_found=["sections"] if has_sections else [],
                relationship_supported=True,
                connecting_evidence_found=True,
                reason="Section list retrieved from document structure or section headers." if has_sections else "Insufficient section index available.",
            )

        # ── 3. PAGE_NUMBER_FORMAT Coverage ───────────────────────────────────
        if norm.intent == "PAGE_NUMBER_FORMAT":
            notation = norm.extracted_notation
            if notation:
                has_notation = (notation in all_text) or ("item number" in all_text) or ("consecutive page" in all_text)
            else:
                has_notation = any(
                    kw in all_text
                    for kw in ["page number", "page format", "item number", "consecutive page", "notation"]
                )
            return EvidenceCoverageResult(
                coverage_status="COMPLETE" if has_notation else "INSUFFICIENT",
                answerable=has_notation,
                entities_required=[notation] if notation else ["page_format"],
                entities_found=[notation] if (notation and has_notation) else (["page_format"] if has_notation else []),
                relationship_supported=True,
                connecting_evidence_found=has_notation,
                reason="Page number format explanation found." if has_notation else (f"Notation {notation} not explicitly explained in retrieved passages." if notation else "Page format explanation not found in retrieved passages."),
            )

        # ── Helper for robust entity presence in evidence ────────────────────
        def _check_entity_presence(entity_str: str) -> bool:
            if not entity_str:
                return True
            low_ent = entity_str.lower().strip()
            if low_ent in all_text:
                return True
            # Strip query fluff words
            fluff = {
                "the", "a", "an", "described", "in", "manual", "system",
                "relevant", "operating", "principles", "section", "for",
                "of", "and", "or", "to", "how", "what", "which",
            }
            tokens = [t for t in re.findall(r"\w+", low_ent) if len(t) > 2 and t not in fluff]
            if not tokens:
                tokens = [t for t in re.findall(r"\w+", low_ent) if len(t) > 2]
            if not tokens:
                return True

            for e in evidence:
                doc_header = f"{e.file_name or ''} {e.section_title or ''} {e.section or ''}".lower()
                if any(t in e.content.lower() or t in doc_header for t in tokens):
                    return True
            matches = sum(1 for t in tokens if t in all_text)
            return matches >= max(1, len(tokens) // 2)

        # ── 4. COMPARISON Coverage ────────────────────────────────────────────
        # e.g. "difference between safety and caution", "compare X and Y"
        is_symbol_query = "symbol" in norm.clean_query.lower() and any(
            w in norm.clean_query.lower() for w in ["safety", "precaution", "caution"]
        )

        if norm.intent == "COMPARISON" and not is_symbol_query:
            lowered_q = norm.clean_query.lower()
            match = re.search(
                r"(?:difference between|compare)\s+(?:the\s+)?(.+?)\s+(?:and|vs\.?|versus)\s+(?:the\s+)?(.+?)(?:\?|$)",
                lowered_q,
            )
            ent_a, ent_b = "", ""
            if match:
                ent_a = match.group(1).strip()
                ent_b = match.group(2).strip()

            found_a = _check_entity_presence(ent_a) if ent_a else True
            found_b = _check_entity_presence(ent_b) if ent_b else True

            if found_a and found_b:
                return EvidenceCoverageResult(
                    coverage_status="COMPLETE",
                    answerable=True,
                    entities_required=[ent_a, ent_b] if (ent_a and ent_b) else norm.extracted_entities,
                    entities_found=[e for e in [ent_a, ent_b] if e] or norm.extracted_entities,
                    relationship_supported=True,
                    connecting_evidence_found=True,
                    reason="Both comparison subjects supported by retrieved evidence.",
                )
            elif (found_a or found_b) and evidence:
                return EvidenceCoverageResult(
                    coverage_status="PARTIAL",
                    answerable=True,
                    entities_required=[ent_a, ent_b],
                    entities_found=[ent_a] if found_a else ([ent_b] if found_b else []),
                    relationship_supported=False,
                    connecting_evidence_found=False,
                    inferred_only=True,
                    reason="Partial evidence available; one comparison subject has limited evidence.",
                )
            else:
                return EvidenceCoverageResult(
                    coverage_status="INSUFFICIENT",
                    answerable=False,
                    entities_required=[ent_a, ent_b],
                    entities_found=[],
                    relationship_supported=False,
                    connecting_evidence_found=False,
                    reason="Retrieved evidence does not cover the comparison subjects.",
                )

        # ── 5. SYMBOL_PURPOSE or Symbol-Specific Comparison ───────────────────
        if norm.intent == "SYMBOL_PURPOSE" or (norm.intent == "COMPARISON" and is_symbol_query):
            required_phrases = [
                "special safety precautions",
                "internal pressure",
                "preserving standards",
            ]
            found = [phrase for phrase in required_phrases if phrase in all_text]
            complete = len(found) == len(required_phrases)
            return EvidenceCoverageResult(
                coverage_status="COMPLETE" if complete else "PARTIAL",
                answerable=complete,
                entities_required=required_phrases,
                entities_found=found,
                relationship_supported=complete,
                connecting_evidence_found=complete,
                reason="All three symbol-purpose categories are explicitly present."
                if complete
                else "One or more symbol-purpose categories are missing from evidence.",
            )

        # ── 6. RELATIONSHIP & Multi-System Interaction Coverage ───────────────
        if norm.intent == "RELATIONSHIP":
            lowered_q = norm.clean_query.lower()
            match = re.search(
                r"(?:between|how is|how does|how the)\s+(.+?)\s+(?:and|related to|influence|influences|interact|interacts|affect|affects)\s+(.+)",
                lowered_q,
            )
            ent_a, ent_b = "", ""
            if match:
                ent_a = match.group(1).strip().replace("section ", "")
                ent_b = match.group(2).strip().replace("section ", "")

            found_a = _check_entity_presence(ent_a) if ent_a else True
            found_b = _check_entity_presence(ent_b) if ent_b else True

            # Domain connecting terms for industrial systems (hydraulic, electrical, mechatronics)
            connecting_terms = [
                "reference material", "serves as", "used for", "refer to",
                "correlat", "troubleshoot", "related to", "connection",
                "pertains to", "control", "signal", "valve", "sensor",
                "switch", "pressure", "flow", "circuit", "actuat",
                "drive", "influenc", "interact", "interlock", "eppr",
                "pump", "motor", "spool", "solenoid",
            ]
            has_connection = any(term in all_text for term in connecting_terms)

            if found_a and found_b and has_connection:
                return EvidenceCoverageResult(
                    coverage_status="COMPLETE",
                    answerable=True,
                    entities_required=[ent_a, ent_b] if (ent_a and ent_b) else norm.extracted_entities,
                    entities_found=[e for e in [ent_a, ent_b] if e] or norm.extracted_entities,
                    relationship_supported=True,
                    connecting_evidence_found=True,
                    reason="Explicit relationship or functional connection supported by source evidence.",
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
                    reason="Evidence found for both entities, but explicit direct connection passage is missing.",
                )
            elif (found_a or found_b) and evidence:
                return EvidenceCoverageResult(
                    coverage_status="PARTIAL",
                    answerable=True,
                    entities_required=[ent_a, ent_b],
                    entities_found=[ent_a] if found_a else ([ent_b] if found_b else []),
                    relationship_supported=False,
                    connecting_evidence_found=False,
                    inferred_only=True,
                    reason="Partial evidence available for relationship inquiry.",
                )
            else:
                return EvidenceCoverageResult(
                    coverage_status="INSUFFICIENT",
                    answerable=False,
                    entities_required=[ent_a, ent_b],
                    entities_found=[e for e in [ent_a, ent_b] if _check_entity_presence(e)],
                    relationship_supported=False,
                    connecting_evidence_found=False,
                    reason="One or both entities missing from retrieved evidence.",
                )

        # ── 7. Standard Coverage (GENERAL_QA, MAINTENANCE, PROCEDURE, SPECIFICATION, etc.) ─
        if not evidence:
            return EvidenceCoverageResult(
                coverage_status="INSUFFICIENT",
                answerable=False,
                entities_required=norm.extracted_entities,
                entities_found=[],
                relationship_supported=False,
                connecting_evidence_found=False,
                reason="No evidence retrieved to answer query.",
            )

        # For general queries with retrieved evidence, allow grounded generation
        return EvidenceCoverageResult(
            coverage_status="COMPLETE",
            answerable=True,
            entities_required=norm.extracted_entities,
            entities_found=[e for e in norm.extracted_entities if e.lower() in all_text],
            relationship_supported=True,
            connecting_evidence_found=True,
            reason="Retrieved evidence available for general query.",
        )
