"""
Query Decomposition Service for Multi-hop, Relationship, and Comparison reasoning.
Breaks down complex queries into focused sub-queries for independent retrieval.

Design rules
------------
- No hardcoded manual names, equipment names, or domain-specific keyword lists.
- Decomposition is driven purely by query structure (connective words, intent label).
- Subqueries are validated and never invented beyond what the original query contains.
- Fallback to the original query if decomposition produces no useful subqueries.
- COMPARISON intent is always decomposed when structural markers are present,
  regardless of subject matter (no special-case for symbols or any other topic).
"""
import re

from pydantic import BaseModel

from app.config.logging import get_logger
from app.services.query.normalization import NormalizedQuery

logger = get_logger(__name__)


class SubQuery(BaseModel):
    query: str
    target_entity: str | None = None
    expected_type: str = "TEXT"


# Structural connectives that signal a comparison or contrast
_COMPARISON_CONNECTIVES = re.compile(
    r"""
    (?:
        \bdifference\s+between\b      # difference between X and Y
        | \bcompare\b                  # compare X and Y / compare X with Y
        | \bvs\.?\b                    # X vs Y / X vs. Y
        | \bversus\b                   # X versus Y
        | \bcontrast\b                 # contrast X and Y
        | \bhow\s+does\s+.{1,60}differ # how does X differ from Y
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_RELATIONSHIP_CONNECTIVES = re.compile(
    r"\b(?:work|operate|function|coordinate)\s+together\b"
    r"|\binteract(?:s)?(?:\s+with)?\b"
    r"|\brelationship\s+between\b"
    r"|\brelated\s+to\b"
    r"|\b(?:influence|affect)(?:s)?\b",
    re.IGNORECASE,
)

# Patterns for extracting two subjects from comparison queries.
# Listed from most-specific to least-specific; first match wins.
_COMPARISON_SPLIT_PATTERNS = [
    # "difference between X and Y", "compare X and Y", "contrast X and Y"
    re.compile(
        r"(?:difference\s+between|compare|contrast)\s+(?:the\s+)?(.+?)\s+(?:and|with)\s+(?:the\s+)?(.+?)(?:\?|$)",
        re.IGNORECASE,
    ),
    # "X vs Y", "X vs. Y", "X versus Y"
    re.compile(
        r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)",
        re.IGNORECASE,
    ),
    # "how does X differ from Y"
    re.compile(
        r"how\s+does\s+(.+?)\s+differ(?:\s+from\s+(.+?))?(?:\?|$)",
        re.IGNORECASE,
    ),
]

# Structural connectives for relationship queries
_RELATIONSHIP_PATTERNS = [
    re.compile(r"between\s+(.+?)\s+and\s+(.+)", re.IGNORECASE),
    re.compile(r"how\s+is\s+(.+?)\s+related\s+to\s+(.+)", re.IGNORECASE),
    re.compile(r"how\s+does\s+(.+?)\s+(?:influence|affect|interact\s+with)\s+(.+)", re.IGNORECASE),
]

_RELATIONSHIP_SUBJECT_PATTERNS = [
    re.compile(r"\bbetween\s+(.+?)(?:\?|$)", re.IGNORECASE),
    re.compile(
        r"\bhow\s+(?:(?:do|does)\s+)?(.+?)\s+"
        r"(?:work|operate|function|interact|coordinate)\s+together\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bhow\s+is\s+(.+?)\s+related\s+to\s+(.+?)(?:\?|$)", re.IGNORECASE),
    re.compile(
        r"\bhow\s+does\s+(.+?)\s+(?:influence|affect|interact\s+with)\s+(.+?)(?:\?|$)",
        re.IGNORECASE,
    ),
]


def _extract_comparison_subjects(query: str) -> tuple[str, str]:
    """
    Extract the two subjects from a comparison query using structural patterns.

    Returns (subject_a, subject_b) as stripped strings, or ("", "") if no
    reliable extraction is possible.
    """
    lowered = query.lower()
    for pat in _COMPARISON_SPLIT_PATTERNS:
        m = pat.search(lowered)
        if m:
            a = m.group(1).strip().strip("?.,") if m.lastindex and m.lastindex >= 1 else ""
            b = m.group(2).strip().strip("?.,") if m.lastindex and m.lastindex >= 2 else ""
            # Discard matches where either subject is blank or excessively long (> 80 chars)
            if a and b and len(a) <= 80 and len(b) <= 80:
                return a, b
    return "", ""


class QueryDecompositionService:
    """
    Decomposes queries into parallel sub-queries for multi-hop, comparison,
    or relationship retrieval.

    Supports any number of subjects. Does not hardcode domain-specific terminology.
    Validates output and falls back to the original query if decomposition fails.
    """

    def decompose(self, norm: NormalizedQuery) -> list[SubQuery]:
        """
        Decompose a NormalizedQuery into focused SubQuery objects.

        Returns a non-empty list. If no useful decomposition is possible the
        list contains a single SubQuery wrapping the original clean_query.
        """
        sub_queries: list[SubQuery] = []
        lowered = norm.clean_query.lower()

        # ── COMPARISON: structural connective present, any subject matter ─────
        if norm.intent == "COMPARISON" or _COMPARISON_CONNECTIVES.search(lowered):
            sub_queries = self._decompose_comparison(norm)

        # ── RELATIONSHIP ──────────────────────────────────────────────────────
        elif norm.intent == "RELATIONSHIP" or _RELATIONSHIP_CONNECTIVES.search(lowered):
            sub_queries = self._decompose_relationship(norm)

        # ── SYMBOL_PURPOSE ────────────────────────────────────────────────────
        elif norm.intent == "SYMBOL_PURPOSE":
            # Broad retrieval: purpose/safety/precaution aspects of symbols
            symbol_query = norm.retrieval_query or norm.clean_query
            sub_queries = [
                SubQuery(query=f"{symbol_query} purpose meaning"),
                SubQuery(query=f"{symbol_query} safety precautions warning"),
                SubQuery(query=f"{symbol_query} technical standards"),
            ]

        # ── MULTI_HOP: split on "and what/which/how" connectives ──────────────
        elif norm.intent == "MULTI_HOP":
            sub_queries = self._decompose_multihop(norm)

        # ── GENERAL_QA / other: decompose if structural comparison present ────
        else:
            if _COMPARISON_CONNECTIVES.search(lowered):
                sub_queries = self._decompose_comparison(norm)

        # Validate and deduplicate
        sub_queries = self._validate(norm, sub_queries)

        logger.debug(
            "decomposition.result",
            intent=norm.intent,
            original=norm.clean_query[:60],
            num_subqueries=len(sub_queries),
            subqueries=[sq.query[:40] for sq in sub_queries],
        )
        return sub_queries

    # ── Private helpers ───────────────────────────────────────────────────────

    def _decompose_comparison(self, norm: NormalizedQuery) -> list[SubQuery]:
        """
        Decompose a comparison query into per-subject retrieval subqueries.

        Generates:
          1. A subquery for subject A (what is / how does subject A work?)
          2. A subquery for subject B
          3. A direct comparison subquery pairing A and B

        Falls back to a single-subquery passthrough if subjects cannot be extracted.
        """
        subj_a, subj_b = _extract_comparison_subjects(norm.clean_query)

        if subj_a and subj_b:
            return [
                SubQuery(query=f"{subj_a}", target_entity=subj_a),
                SubQuery(query=f"{subj_b}", target_entity=subj_b),
                SubQuery(
                    query=f"{subj_a} compared to {subj_b}",
                    target_entity="comparison",
                ),
            ]

        # Could not extract two distinct subjects; return original
        return [SubQuery(query=norm.clean_query)]

    def _decompose_relationship(self, norm: NormalizedQuery) -> list[SubQuery]:
        """Extract every explicitly coordinated subject in a relationship query."""
        subjects_text = ""
        for pattern in _RELATIONSHIP_SUBJECT_PATTERNS:
            match = pattern.search(norm.clean_query)
            if match:
                subjects_text = " and ".join(group for group in match.groups() if group)
                break

        if subjects_text:
            subjects = [
                re.sub(r"\s+", " ", re.sub(r"^(?:and|or)\s+", "", subject.strip(), flags=re.IGNORECASE))
                .strip(" \t\r\n?.,;:")
                for subject in re.split(r"\s*,\s*|\s+(?:and|or)\s+|\s+as\s+well\s+as\s+", subjects_text, flags=re.IGNORECASE)
            ]
            subjects = [subject for subject in subjects if subject]
            unique_subjects = list(dict.fromkeys(subject.casefold() for subject in subjects))
            subject_by_key = {subject.casefold(): subject for subject in subjects}
            if len(unique_subjects) >= 2:
                return [
                    SubQuery(
                        query=subject_by_key[key],
                        target_entity=subject_by_key[key],
                    )
                    for key in unique_subjects
                ]

        return [SubQuery(query=norm.clean_query)]

    def _decompose_multihop(self, norm: NormalizedQuery) -> list[SubQuery]:
        """Split multi-hop queries on 'and what/which/how' connectives."""
        parts = re.split(r"\s+and\s+(?=what|which|how)", norm.clean_query, flags=re.IGNORECASE)
        result = []
        for part in parts:
            cleaned = part.strip().rstrip("?.,")
            if cleaned:
                result.append(SubQuery(query=cleaned))
        return result if len(result) > 1 else [SubQuery(query=norm.clean_query)]

    def _validate(self, norm: NormalizedQuery, sub_queries: list[SubQuery]) -> list[SubQuery]:
        """
        Remove degenerate, duplicate, or empty subqueries.
        Guarantee at least one subquery (the original).
        """
        seen: set[str] = set()
        valid: list[SubQuery] = []
        for sq in sub_queries:
            q = sq.query.strip()
            if not q or q.lower() in seen:
                continue
            seen.add(q.lower())
            valid.append(SubQuery(query=q, target_entity=sq.target_entity, expected_type=sq.expected_type))

        if not valid:
            valid = [SubQuery(query=norm.clean_query)]

        return valid
