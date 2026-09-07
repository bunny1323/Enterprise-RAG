"""
Safe Typo Correction Service for Query-Time Retrieval.

Corrects spelling mistakes in non-technical query tokens before BM25 retrieval,
producing a `typo_query` that is used as a third, additional BM25 channel.

SAFETY GUARANTEES:
  - Technical identifiers (model numbers, error codes, part numbers, page
    numbers, section numbers, hyphenated codes) are NEVER corrected.
  - Numeric-only tokens are NEVER corrected.
  - Corrections are only applied when a unique, high-confidence match exists
    in the trusted vocabulary.
  - The original clean_query is NEVER mutated.

IMPLEMENTATION:
  - Uses Python stdlib difflib.SequenceMatcher for edit-distance ratio scoring.
    (Zero-dependency — no additional packages needed beyond stdlib.)
  - rapidfuzz can be used as an optional faster backend if installed.
  - Trusted vocabulary is built from unit canonical forms + curated domain
    terminology relevant to construction-equipment service manuals.
"""
import re
from difflib import SequenceMatcher
from app.config.logging import get_logger

logger = get_logger(__name__)

# ── Protected-token patterns ─────────────────────────────────────────────────
# A token matching ANY of these patterns must not be fuzzy-corrected.

# Purely numeric (e.g. "2", "215", "3.5")
_NUMERIC_RE = re.compile(r'^\d+(\.\d+)?$')

# Hyphenated technical identifier (e.g. "E-104", "2-3", "HX-220", "R215L-9S")
_HYPHEN_TECH_RE = re.compile(r'^[A-Z0-9]+-[A-Z0-9]+$', re.IGNORECASE)

# Model / part / error codes: letter prefix + digits (e.g. "R215L", "P2001", "E104", "HX220")
_ALPHANUM_CODE_RE = re.compile(r'^[A-Z]{1,4}\d{2,}[A-Z0-9]*$', re.IGNORECASE)

# All-uppercase abbreviation (e.g. "RPM", "PSI", "LBS")
_UPPER_ABBREV_RE = re.compile(r'^[A-Z]{2,5}$')

# Tokens ≤ 3 chars — too short for safe fuzzy correction
_SHORT_TOKEN_MIN_LEN = 4

# Stop words that should never be corrected (they aren't typos of anything useful)
_STOP_WORDS: frozenset[str] = frozenset({
    'the', 'a', 'an', 'is', 'it', 'in', 'of', 'to', 'and', 'or', 'but',
    'for', 'on', 'at', 'by', 'be', 'as', 'up', 'if', 'no', 'so', 'we',
    'do', 'my', 'he', 'she', 'we', 'us', 'me', 'am', 'are', 'was', 'were',
    'has', 'had', 'have', 'does', 'did', 'its', 'not', 'can', 'will', 'from',
    'with', 'this', 'that', 'what', 'which', 'how', 'why', 'who', 'when',
    'where', 'does', 'mean', 'many', 'much', 'some', 'any', 'all', 'more',
    'into', 'about', 'also', 'than', 'then', 'each', 'been', 'both', 'such',
    'there', 'their', 'they', 'would', 'could', 'should', 'your', 'after',
    'between',
})

# ── Trusted vocabulary ────────────────────────────────────────────────────────
# Words that are valid correction targets. Only words appearing here can be
# suggested as corrections.  This prevents hallucinating nonsense corrections.
#
# The vocabulary is intentionally curated for construction / heavy-equipment
# service manuals. Extend this list as new documents are indexed.

_TRUSTED_VOCABULARY: frozenset[str] = frozenset({
    # ── Measurement units (canonical singular forms) ──────────────────────────
    'kilogram', 'gram', 'pound', 'ton', 'tonne', 'ounce',
    'kilometer', 'centimeter', 'millimeter', 'meter', 'foot', 'inch', 'mile',
    'liter', 'milliliter', 'gallon', 'quart',
    'kilopascal', 'megapascal', 'bar', 'psi',
    'newton', 'joule', 'watt', 'kilowatt', 'horsepower',
    'volt', 'ampere', 'ohm', 'hertz',
    'celsius', 'fahrenheit', 'kelvin', 'degree',
    # ── Conversion / factor terms ─────────────────────────────────────────────
    'conversion', 'factor', 'convert', 'equivalent', 'equals', 'approximate',
    'value', 'ratio', 'multiply', 'divide', 'formula', 'unit', 'units',
    'table', 'chart', 'reference',
    # ── General technical / manual terminology ────────────────────────────────
    'specification', 'specifications', 'capacity', 'torque', 'clearance',
    'dimension', 'dimensions', 'tolerance', 'weight', 'length', 'width',
    'height', 'volume', 'pressure', 'temperature', 'flow', 'speed', 'force',
    'power', 'current', 'voltage', 'resistance', 'frequency', 'cycle',
    # ── Safety ────────────────────────────────────────────────────────────────
    'safety', 'caution', 'warning', 'danger', 'hazard', 'symbol', 'symbols',
    'precaution', 'precautions', 'instruction', 'instructions',
    # ── Machine / component terms ─────────────────────────────────────────────
    'hydraulic', 'cylinder', 'pump', 'valve', 'motor', 'engine', 'filter',
    'coolant', 'lubricant', 'grease', 'oil', 'fuel', 'battery', 'cable',
    'hose', 'pipe', 'fitting', 'bolt', 'nut', 'screw', 'washer', 'seal',
    'gasket', 'bearing', 'bushing', 'pin', 'bracket', 'plate', 'cover',
    'housing', 'gear', 'shaft', 'belt', 'chain', 'sprocket', 'pulley',
    'actuator', 'sensor', 'switch', 'relay', 'solenoid', 'fuse', 'breaker',
    'alternator', 'starter', 'radiator', 'compressor', 'injector', 'nozzle',
    'piston', 'ring', 'liner', 'crankshaft', 'camshaft', 'connecting',
    # ── Operations / maintenance ──────────────────────────────────────────────
    'maintenance', 'inspection', 'replacement', 'installation', 'removal',
    'disassembly', 'assembly', 'adjustment', 'calibration', 'lubrication',
    'cleaning', 'testing', 'overhaul', 'repair', 'diagnosis', 'troubleshoot',
    'troubleshooting', 'procedure', 'procedures', 'steps', 'operation',
    'service', 'manual', 'document', 'section', 'chapter', 'page', 'figure',
    'diagram', 'schematic', 'drawing', 'component', 'components', 'parts',
    'system', 'systems', 'circuit', 'assembly', 'function', 'description',
    # ── Error / fault terms ───────────────────────────────────────────────────
    'error', 'fault', 'code', 'alarm', 'indicator', 'warning', 'failure',
    'problem', 'symptom', 'cause', 'solution', 'remedy',
    # ── Structural doc terms ──────────────────────────────────────────────────
    'general', 'structure', 'standard', 'specification', 'appendix', 'index',
    'foreword', 'introduction', 'overview', 'summary', 'contents',
    # ── Common question words ─────────────────────────────────────────────────
    'what', 'which', 'how', 'when', 'where', 'why', 'explain', 'describe',
    'list', 'show', 'tell', 'define', 'meaning', 'purpose', 'reason',
    'difference', 'compare', 'between', 'represent', 'indicate',
    'conversion', 'factor', 'maximum', 'minimum', 'required', 'recommended',
    'correct', 'proper', 'appropriate', 'standard', 'typical', 'normal',
})

# Fuzzy correction config
_SIMILARITY_THRESHOLD = 0.78  # conservative threshold for short technical typos
_MAX_CANDIDATES_TO_CHECK = 256  # trusted vocabulary is intentionally small


def _is_protected_token(token: str) -> bool:
    """Return True if this token must NOT be fuzzy-corrected."""
    if len(token) < _SHORT_TOKEN_MIN_LEN:
        return True
    if token.lower() in _STOP_WORDS:
        return True
    if _NUMERIC_RE.match(token):
        return True
    if _HYPHEN_TECH_RE.match(token):
        return True
    if _ALPHANUM_CODE_RE.match(token):
        return True
    if _UPPER_ABBREV_RE.match(token):
        return True
    return False


def _similarity(a: str, b: str) -> float:
    """Return normalized edit-distance similarity in [0, 1]."""
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _find_correction(token: str) -> str | None:
    """
    Attempt to find a high-confidence typo correction for a single token.

    Returns the corrected word if a unique, unambiguous, high-similarity
    match is found in the trusted vocabulary.  Returns None otherwise.
    """
    lowered = token.lower()

    # Already in vocabulary — no correction needed
    if lowered in _TRUSTED_VOCABULARY:
        return None

    best_word: str | None = None
    best_score = 0.0
    second_score = 0.0

    # Only compare against vocabulary words of similar length (±3 chars)
    # to speed up and avoid nonsense matches across very different lengths.
    tok_len = len(lowered)
    candidates = [
        w for w in _TRUSTED_VOCABULARY
        if abs(len(w) - tok_len) <= 3
    ]

    # Cap candidates to avoid pathological runtime
    if len(candidates) > _MAX_CANDIDATES_TO_CHECK:
        # Sort deterministically so a frozenset iteration order cannot hide the
        # correct vocabulary entry behind the candidate cap.
        candidates = sorted(candidates, key=lambda w: (abs(len(w) - tok_len), w))
        candidates = candidates[:_MAX_CANDIDATES_TO_CHECK]

    for vocab_word in candidates:
        score = _similarity(lowered, vocab_word)
        if score > best_score:
            second_score = best_score
            best_score = score
            best_word = vocab_word
        elif score > second_score:
            second_score = score

    if best_word is None or best_score < _SIMILARITY_THRESHOLD:
        return None  # No confident match

    # Require unambiguous: best must be clearly better than second-best
    margin = best_score - second_score
    if margin < 0.07 and not (best_score >= 0.9 and margin >= 0.03):
        return None  # Too ambiguous

    logger.debug(
        "typo_correction.corrected",
        original=token,
        correction=best_word,
        score=round(best_score, 3),
        margin=round(margin, 3),
    )
    return best_word


def build_typo_query(clean_query: str) -> str:
    """
    Produce a typo-corrected retrieval query from the clean user query.

    Algorithm:
      1. Tokenize on whitespace (preserve punctuation attached to tokens).
      2. For each token:
         a. If it is a protected identifier → keep verbatim.
         b. If it is in the trusted vocabulary → keep verbatim.
         c. Otherwise, attempt fuzzy correction against the vocabulary.
      3. Reassemble tokens.
      4. Return the corrected string, or the original clean_query unchanged
         if no corrections were made (avoids unnecessary BM25 calls).

    The returned string is ALWAYS a valid retrieval query derived from
    clean_query and is NEVER used as the displayed user query.
    """
    tokens = clean_query.split()
    corrected_tokens = []
    any_correction = False

    for token in tokens:
        # Strip trailing punctuation for matching, restore it after
        stripped = token.rstrip('?,.')
        suffix = token[len(stripped):]

        if _is_protected_token(stripped):
            corrected_tokens.append(token)
            continue

        correction = _find_correction(stripped)
        if correction is not None and correction != stripped.lower():
            # Preserve original capitalisation style
            if stripped[0].isupper():
                correction = correction.capitalize()
            corrected_tokens.append(correction + suffix)
            any_correction = True
        else:
            corrected_tokens.append(token)

    if not any_correction:
        return clean_query  # signal: no change needed

    return " ".join(corrected_tokens)
