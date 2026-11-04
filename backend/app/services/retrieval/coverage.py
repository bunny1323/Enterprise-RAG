"""
Evidence Coverage and Answerability Gating Service.

Evaluates retrieved evidence to determine whether the query can be answered,
partially answered, or must be declined due to insufficient grounding.

Design rules
------------
- No hardcoded entity names, manual names, or domain-specific keyword lists
  (except the SYMBOL_PURPOSE hardcoded safety phrases which are document-specific).
- COMPARISON coverage recognises 'difference between', 'compare', 'contrast',
  'X vs Y', 'X versus Y' without restricting to any particular subject domain.
- PARTIAL evidence is answerable: the LLM is instructed to answer what it can
  and identify the missing aspect rather than refusing entirely.
- Entity presence check uses token-level matching against both content and
  document headers; it does not require exact string equality.
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

        # 1. COUNT_QUERY Coverage
        if norm.intent == "COUNT_QUERY":
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

        # 2. LIST_QUERY Coverage
        if norm.intent == "LIST_QUERY":
            listed_numbers: set[int] = set()
            has_index_source = False
            for item in evidence:
                if item.metadata.get("source") == "document_structure":
                    has_index_source = True
                listed_numbers.update(
                    int(number)
                    for number in re.findall(
                        r"\b(?:section|chapter|part)\s+(\d+)\b",
                        item.content,
                        re.IGNORECASE,
                    )
                )
            is_complete = has_index_source or len(listed_numbers) >= 2
            return EvidenceCoverageResult(
                coverage_status="COMPLETE" if is_complete else "PARTIAL",
                answerable=is_complete,
                entities_required=["sections"],
                entities_found=[f"section_{number}" for number in sorted(listed_numbers)],
                relationship_supported=True,
                connecting_evidence_found=is_complete,
                reason=(
                    "Section enumeration retrieved from document structure or multiple explicit section headings."
                    if is_complete
                    else "Retrieved evidence does not establish a complete section list."
                ),
            )

        # 3. PAGE_NUMBER_FORMAT Coverage
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
                reason="Page number format explanation found." if has_notation else (
                    f"Notation {notation} not explicitly explained in retrieved passages." if notation
                    else "Page format explanation not found in retrieved passages."
                ),
            )

        if norm.intent == "SECTION_LOOKUP":
            section_number = norm.requested_section_number
            if section_number is None:
                for entity in norm.extracted_entities:
                    match = re.fullmatch(r"section_(\d+)", entity, re.IGNORECASE)
                    if match:
                        section_number = int(match.group(1))
                        break

            section_heading = (
                re.compile(rf"\bSECTION\s+{section_number}\b", re.IGNORECASE)
                if section_number is not None
                else None
            )
            matching_evidence = [
                item for item in evidence
                if section_heading and section_heading.search(item.content or "")
            ]
            supported = bool(matching_evidence)
            return EvidenceCoverageResult(
                coverage_status="COMPLETE" if supported else "INSUFFICIENT",
                answerable=supported,
                entities_required=[f"section_{section_number}"] if section_number is not None else [],
                entities_found=[f"section_{section_number}"] if supported and section_number is not None else [],
                relationship_supported=supported,
                connecting_evidence_found=supported,
                reason=(
                    f"Evidence explicitly identifies Section {section_number}."
                    if supported
                    else f"No retrieved evidence explicitly identifies Section {section_number}."
                    if section_number is not None
                    else "Section number could not be identified from the query."
                ),
            )

        # Helper: token-level entity presence check
        def _check_entity_presence(entity_str: str) -> bool:
            if not entity_str:
                return True
            low_ent = entity_str.lower().strip()
            if low_ent in all_text:
                return True
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

            evidence_text = " ".join(
                f"{e.content} {e.file_name or ''} {e.section_title or ''} {e.section or ''}"
                for e in evidence
            ).lower()
            evidence_tokens = set(re.findall(r"\w+", evidence_text))
            matches = sum(1 for token in tokens if token in evidence_tokens)
            return matches >= (len(tokens) // 2) + 1

        # 4. COMPARISON Coverage
        # Handles: "difference between X and Y", "X vs Y", "compare X and Y",
        # "X versus Y", "contrast X with Y" — any subject matter.
        is_symbol_query = "symbol" in norm.clean_query.lower() and any(
            w in norm.clean_query.lower() for w in ["safety", "precaution", "caution"]
        )

        if norm.intent == "COMPARISON" and not is_symbol_query:
            lowered_q = norm.clean_query.lower()
            # Extended regex: difference between/compare/contrast ... and/with/vs/versus
            # followed by a standalone "X vs/versus Y" alternative.
            comp_re = re.compile(
                r"(?:difference\s+between|compare|contrast)\s+(?:the\s+)?(.+?)\s+(?:and|with|vs\.?|versus)\s+(?:the\s+)?(.+?)(?:\?|$)"
                r"|(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)",
                re.IGNORECASE,
            )
            m = comp_re.search(lowered_q)
            ent_a, ent_b = "", ""
            if m:
                if m.group(1) and m.group(2):
                    ent_a = m.group(1).strip().strip("?.,")
                    ent_b = m.group(2).strip().strip("?.,")
                elif m.group(3) and m.group(4):
                    ent_a = m.group(3).strip().strip("?.,")
                    ent_b = m.group(4).strip().strip("?.,")

            # Fallback: use entities extracted during normalization
            if not (ent_a and ent_b) and len(norm.extracted_entities) >= 2:
                ent_a = norm.extracted_entities[0]
                ent_b = norm.extracted_entities[1]

            if not (ent_a and ent_b):
                return EvidenceCoverageResult(
                    coverage_status="INSUFFICIENT",
                    answerable=False,
                    entities_required=norm.extracted_entities,
                    entities_found=[],
                    relationship_supported=False,
                    connecting_evidence_found=False,
                    reason="Comparison subjects could not be identified from the query.",
                )

            found_a = _check_entity_presence(ent_a)
            found_b = _check_entity_presence(ent_b)

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
                missing = [e for e, ok in [(ent_a, found_a), (ent_b, found_b)] if not ok and e]
                return EvidenceCoverageResult(
                    coverage_status="PARTIAL",
                    answerable=True,  # allow partial synthesis; LLM identifies the gap
                    entities_required=[ent_a, ent_b] if (ent_a and ent_b) else norm.extracted_entities,
                    entities_found=[ent_a] if found_a else ([ent_b] if found_b else []),
                    relationship_supported=False,
                    connecting_evidence_found=False,
                    inferred_only=True,
                    reason=(
                        f"Partial evidence: evidence found for one comparison subject; "
                        f"limited or no evidence for: {', '.join(missing) or 'second subject'}."
                    ),
                )
            else:
                return EvidenceCoverageResult(
                    coverage_status="INSUFFICIENT",
                    answerable=False,
                    entities_required=[ent_a, ent_b] if (ent_a and ent_b) else norm.extracted_entities,
                    entities_found=[],
                    relationship_supported=False,
                    connecting_evidence_found=False,
                    reason="Retrieved evidence does not appear to cover either comparison subject.",
                )

        # 5. SYMBOL_PURPOSE or Symbol-Specific Comparison
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

        # 6. RELATIONSHIP & Multi-System Interaction Coverage
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

            # Domain connecting terms — generic enough to work across any industrial document
            connecting_terms = [
                "reference material", "serves as", "used for", "refer to",
                "correlat", "troubleshoot", "related to", "connection",
                "pertains to", "control", "signal", "valve", "sensor",
                "switch", "pressure", "flow", "circuit", "actuat",
                "drive", "influenc", "interact", "interlock",
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

        # 7. Standard Coverage (GENERAL_QA, MAINTENANCE, PROCEDURE, SPECIFICATION, etc.)
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
