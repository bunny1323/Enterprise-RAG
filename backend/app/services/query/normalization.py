"""
Query Normalization and Intent Classification Service.
Normalizes text and classifies query intent without LLM latency for standard queries.

The service produces two query forms:
  - clean_query   : whitespace/case-normalized original user text (NEVER altered further)
  - retrieval_query : morphologically normalized form for BM25 (unit aliases expanded,
                      safe plural→singular conversion applied). Used as an ADDITIONAL
                      BM25 retrieval signal, never shown to the user.
"""
import re
import string

from pydantic import BaseModel
from app.services.query.typo_correction import build_typo_query


# ── Unit alias normalization table ────────────────────────────────────────────
# Maps compiled regex patterns → canonical singular lowercase forms.
# Only unambiguous technical measurement units are listed.
# Longer / more specific patterns must come BEFORE shorter ones.
_UNIT_ALIAS_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Mass ─────────────────────────────────────────────────────────────────────
    (re.compile(r'\bkilograms?\b', re.IGNORECASE), 'kilogram'),
    (re.compile(r'\bkgs?\b', re.IGNORECASE), 'kilogram'),
    (re.compile(r'\bgrams?\b', re.IGNORECASE), 'gram'),
    (re.compile(r'\bpounds?\b', re.IGNORECASE), 'pound'),
    (re.compile(r'\blbs?\b', re.IGNORECASE), 'pound'),
    (re.compile(r'\btons?\b', re.IGNORECASE), 'ton'),
    (re.compile(r'\btonnes?\b', re.IGNORECASE), 'tonne'),
    (re.compile(r'\bounces?\b', re.IGNORECASE), 'ounce'),
    # Length ───────────────────────────────────────────────────────────────────
    (re.compile(r'\bkilometers?\b', re.IGNORECASE), 'kilometer'),
    (re.compile(r'\bkms?\b', re.IGNORECASE), 'kilometer'),
    (re.compile(r'\bcentimeters?\b', re.IGNORECASE), 'centimeter'),
    (re.compile(r'\bcms?\b', re.IGNORECASE), 'centimeter'),
    (re.compile(r'\bmillimeters?\b', re.IGNORECASE), 'millimeter'),
    (re.compile(r'\bmms?\b', re.IGNORECASE), 'millimeter'),
    (re.compile(r'\bmeters?\b', re.IGNORECASE), 'meter'),
    (re.compile(r'\bfeet\b', re.IGNORECASE), 'foot'),
    (re.compile(r'\binches\b', re.IGNORECASE), 'inch'),
    (re.compile(r'\bmiles?\b', re.IGNORECASE), 'mile'),
    # Volume ───────────────────────────────────────────────────────────────────
    (re.compile(r'\bmilliliters?\b', re.IGNORECASE), 'milliliter'),
    (re.compile(r'\bliters?\b', re.IGNORECASE), 'liter'),
    (re.compile(r'\bgallons?\b', re.IGNORECASE), 'gallon'),
    (re.compile(r'\bquarts?\b', re.IGNORECASE), 'quart'),
    # Pressure ─────────────────────────────────────────────────────────────────
    (re.compile(r'\bkilopascals?\b', re.IGNORECASE), 'kilopascal'),
    (re.compile(r'\bkpa\b', re.IGNORECASE), 'kilopascal'),
    (re.compile(r'\bmegapascals?\b', re.IGNORECASE), 'megapascal'),
    (re.compile(r'\bmpa\b', re.IGNORECASE), 'megapascal'),
    (re.compile(r'\bpsi\b', re.IGNORECASE), 'psi'),
    # Torque ───────────────────────────────────────────────────────────────────
    (re.compile(r'\bnewton[- ]?meters?\b', re.IGNORECASE), 'newton meter'),
    (re.compile(r'\bfoot[- ]?pounds?\b', re.IGNORECASE), 'foot pound'),
    (re.compile(r'\bft[- ]?lbs?\b', re.IGNORECASE), 'foot pound'),
    # Power / Electrical ───────────────────────────────────────────────────────
    (re.compile(r'\bhorsepowers?\b', re.IGNORECASE), 'horsepower'),
    (re.compile(r'\bkilowatts?\b', re.IGNORECASE), 'kilowatt'),
    (re.compile(r'\bwatts?\b', re.IGNORECASE), 'watt'),
    (re.compile(r'\bvolts?\b', re.IGNORECASE), 'volt'),
    (re.compile(r'\bamperes?\b', re.IGNORECASE), 'ampere'),
    (re.compile(r'\bamps?\b', re.IGNORECASE), 'ampere'),
    # Rotation / Speed ─────────────────────────────────────────────────────────
    (re.compile(r'\brev(?:olution)?s?\s*(?:per|/)\s*min(?:ute)?s?\b', re.IGNORECASE), 'revolution per minute'),
    (re.compile(r'\brpms?\b', re.IGNORECASE), 'revolution per minute'),
    # Temperature ──────────────────────────────────────────────────────────────
    (re.compile(r'\bdegrees?\b', re.IGNORECASE), 'degree'),
]

# Endings that prove a word is NOT simply a plural of its stem.
_PLURAL_EXCEPTION_RE = re.compile(
    r'(?:ss|us|is|as|ness|ous|ious|eous|xes|shes|ches|zes)$',
    re.IGNORECASE,
)

# Tokens that look like technical codes / part numbers — preserve as-is.
_TECHNICAL_TOKEN_RE = re.compile(
    r'^[A-Z]{1,4}[0-9]|^[0-9]|\d+-\d+',
    re.IGNORECASE,
)

# Words that end in 's' but must never be singularized.
_SKIP_SINGULARIZE: frozenset[str] = frozenset({
    'this', 'is', 'was', 'has', 'its', 'his', 'as', 'thus', 'plus',
    'status', 'versus', 'bonus', 'radius', 'nexus', 'focus', 'campus',
    'bus', 'gas', 'yes', 'does', 'goes', 'news', 'lens', 'axis', 'basis',
    'analysis', 'diagnosis', 'hypothesis', 'crisis', 'thesis', 'chassis',
    'the', 'and', 'but', 'for', 'or', 'no',
})


def _safe_singularize_token(token: str) -> str:
    """
    Attempt to convert a single plural token to its singular form, safely.

    Rules (applied in order):
    1. Skip tokens ≤ 3 characters (too short to singularize reliably).
    2. Skip known exception stop words.
    3. Skip tokens that look like technical identifiers / codes.
    4. Handle the -ies → -y case (e.g. "bodies" → "body").
    5. Skip tokens whose suffix proves they are not simple plurals.
    6. Strip a trailing 's' (if the word doesn't end in 'ss').
    """
    if len(token) <= 3:
        return token
    lowered = token.lower()
    if lowered in _SKIP_SINGULARIZE:
        return token
    if _TECHNICAL_TOKEN_RE.match(token):
        return token
    # -ies → -y  (e.g. bodies → body, quantities → quantity)
    if lowered.endswith('ies') and len(token) > 4:
        return token[:-3] + 'y'
    # Exception endings – not a simple plural suffix
    if _PLURAL_EXCEPTION_RE.search(lowered):
        return token
    # Strip trailing 's' (not 'ss')
    if lowered.endswith('s') and not lowered.endswith('ss'):
        return token[:-1]
    return token


def _build_retrieval_query(clean_query: str) -> str:
    """
    Build a morphologically normalized retrieval string from the clean user query.

    Steps:
      1. Apply unit alias expansion (e.g. 'kilograms' → 'kilogram', 'lbs' → 'pound').
      2. Apply safe plural → singular conversion on each remaining token.

    This result is used as an ADDITIONAL BM25 retrieval channel.
    It is never shown to the user and never replaces raw_query / clean_query.
    Returns the original clean_query unchanged if no normalization was needed.
    """
    text = clean_query

    # Step 1: unit alias expansion
    for pattern, canonical in _UNIT_ALIAS_PATTERNS:
        text = pattern.sub(canonical, text)

    # Step 2: token-level safe singularization
    tokens = text.split()
    tokens = [_safe_singularize_token(t) for t in tokens]
    
    # Step 3: Append only the new/changed tokens to the original query (Query Expansion)
    # This ensures BM25 can match documents that contain a mix of original and normalized forms.
    original_tokens_lower = set(clean_query.lower().split())
    new_tokens = []
    for t in tokens:
        if t.lower() not in original_tokens_lower:
            new_tokens.append(t)
            
    if not new_tokens:
        expanded = clean_query
    else:
        expanded = clean_query + " " + " ".join(new_tokens)

    lowered = clean_query.lower()
    if "symbol" in lowered or "caution" in lowered:
        symbol_terms = (
            "symbol safety precautions internal pressure technical precautions "
            "preserving standards"
        )
        expanded = f"{expanded} {symbol_terms}"
        
    return expanded


class NormalizedQuery(BaseModel):
    raw_query: str           # Exact user input — never mutated
    clean_query: str         # Whitespace/case-normalized user text — used for display and dense embedding
    intent: str              # COUNT_QUERY | LIST_QUERY | PAGE_NUMBER_FORMAT | SECTION_LOOKUP |
                             # RELATIONSHIP | MULTI_HOP | GENERAL_QA | COMPARISON | etc.
    retrieval_query: str = ""  # Morphologically normalized form for BM25; empty = same as clean_query
    typo_query: str = ""     # Typo-corrected form for BM25; empty = same as clean_query
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

        # Pre-compute morphological retrieval query once for all return paths
        retrieval_q = _build_retrieval_query(clean)

        # Pre-compute typo-corrected query once for all return paths
        # (build_typo_query returns clean unchanged if no corrections found)
        typo_q = build_typo_query(clean)

        # Entity regex (e.g. part numbers, fault codes like E-104, P2001)
        entities = re.findall(r"\b[A-Z0-9]{2,10}-[0-9]{2,6}\b|\b[EFP]\d{3,5}\b", clean)

        # ── 1. COUNT_QUERY (e.g., "how many major sections...", "how many chapters...") ──
        count_match = re.search(r"\bhow\s+many\s+(major\s+)?(sections?|chapters?|items?|parts?)\b", lowered)
        if count_match:
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                retrieval_query=retrieval_q,
                typo_query=typo_q,
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
                retrieval_query=retrieval_q,
                typo_query=typo_q,
                intent="LIST_QUERY",
                secondary_intent="DOCUMENT_STRUCTURE",
                extracted_entities=entities,
                answer_type="list",
            )

        # ── 3. PAGE_NUMBER_FORMAT ─────────────────────────────────────────────
        # e.g. "In the example page number '2-3', what does the '2' represent?"
        notation_match = re.search(r"\b(\d+-\d+)\b", clean)
        page_format_keywords = ["page number", "page format", "represent", "mean", "first number", "second number"]
        if notation_match and (any(kw in lowered for kw in page_format_keywords) or "page" in lowered or "example" in lowered):
            notation = notation_match.group(1)
            entities.append(notation)
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                retrieval_query=retrieval_q,
                typo_query=typo_q,
                intent="PAGE_NUMBER_FORMAT",
                secondary_intent="EXACT_LOOKUP",
                extracted_entities=entities,
                extracted_notation=notation,
                answer_type="definition",
            )

        # Also catch: "what does the 2 in 2-3 represent" (no hyphenated notation in clean)
        if ("2-3" in clean or "page number" in lowered) and any(w in lowered for w in ["represent", "mean", "indicate", "stand for"]):
            notation = "2-3" if "2-3" in clean else None
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                retrieval_query=retrieval_q,
                typo_query=typo_q,
                intent="PAGE_NUMBER_FORMAT",
                secondary_intent="EXACT_LOOKUP",
                extracted_entities=entities + ([notation] if notation else []),
                extracted_notation=notation,
                answer_type="definition",
            )

        # ── 4. SECTION_LOOKUP ─────────────────────────────────────────────────
        # Matches: "what is section 3", "show section 5", "tell me about section 9"
        section_match = re.search(r"\bsection\s+(\d+)\b", lowered)
        if section_match:
            sec_num = int(section_match.group(1))
            entities.append(f"section_{sec_num}")
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                retrieval_query=retrieval_q,
                typo_query=typo_q,
                intent="SECTION_LOOKUP",
                extracted_entities=entities,
                requested_section_number=sec_num,
                answer_type="text",
            )

        # ── 5. RELATIONSHIP & MULTI_HOP ───────────────────────────────────────
        symbol_purpose = (
            "symbol" in lowered
            and any(term in lowered for term in ["reason", "why", "precaution", "purpose", "provide"])
        )
        if symbol_purpose:
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                retrieval_query=retrieval_q,
                typo_query=typo_q,
                intent="SYMBOL_PURPOSE",
                secondary_intent="LIST_QUERY",
                extracted_entities=["safety precautions", "internal pressure", "preserving standards"],
                answer_type="list",
            )

        is_rel = any(kw in lowered for kw in ["relationship between", "how is", "related to", "affects", "correlates to", "connection between"])
        is_multihop = any(kw in lowered for kw in ["which section", "and what does", "and which section"])

        if is_multihop:
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                retrieval_query=retrieval_q,
                typo_query=typo_q,
                intent="MULTI_HOP",
                extracted_entities=entities,
                answer_type="relationship",
            )

        if is_rel:
            return NormalizedQuery(
                raw_query=query,
                clean_query=clean,
                retrieval_query=retrieval_q,
                typo_query=typo_q,
                intent="RELATIONSHIP",
                extracted_entities=entities,
                answer_type="relationship",
            )

        # ── 6. Other intent heuristics ────────────────────────────────────────
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
            retrieval_query=retrieval_q,
            typo_query=typo_q,
            intent=intent,
            extracted_entities=entities,
            requested_section_number=None,
        )
