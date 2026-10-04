"""
Candidate Selector for Vision Analysis.

Filters figures using fast, deterministic OR-based heuristics so that
only diagrams, schematics, vector drawings, or meaningful images undergo
expensive vision LLM inference.
"""
from pathlib import Path
from typing import Any
from PIL import Image

from app.config.logging import get_logger

logger = get_logger(__name__)

# Keywords that indicate technical schematics or engineering diagrams
DIAGRAM_KEYWORDS = {
    "diagram",
    "schematic",
    "circuit",
    "flowchart",
    "blueprint",
    "exploded",
    "mounting torque",
    "hydraulic",
    "wiring",
    "cross-section",
    "cross section",
    "layout",
    "assembly",
}

PAGE_DIAGRAM_TITLES = {
    "DIAGRAM",
    "SCHEMATIC",
    "CIRCUIT",
    "MOUNTING TORQUE",
    "EXPLODED VIEW",
    "COMPONENT MOUNTING",
    "HYDRAULIC SYSTEM",
    "MECHATRONICS",
    "SYSTEM DRAWING",
}


class CandidateSelector:
    """
    Evaluates whether an extracted figure is a candidate for vision analysis.
    Uses OR-based evaluation across multiple strong domain signals.
    """

    @classmethod
    def is_candidate(
        cls,
        figure: dict[str, Any],
        page_data: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """
        Determine if figure qualifies for vision analysis.

        Returns:
            dict with keys: classification, needs_vlm, reason, score
        """
        image_path_str = figure.get("image_path", "")
        if not image_path_str:
            return {"classification": "UNKNOWN", "needs_vlm": False, "reason": "no_image_path", "score": 0}

        image_path = Path(image_path_str)
        if not image_path.exists():
            return {"classification": "UNKNOWN", "needs_vlm": False, "reason": "file_not_found", "score": 0}

        try:
            file_size = image_path.stat().st_size
            if file_size < 100:
                return {"classification": "UNKNOWN", "needs_vlm": False, "reason": "empty_or_corrupt_file", "score": 0}
        except OSError:
            return {"classification": "UNKNOWN", "needs_vlm": False, "reason": "file_stat_failed", "score": 0}

        # Check resolution via PIL header (reject tiny bullets/icons immediately)
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                if w < 60 or h < 60:
                    return {"classification": "TEXT_ONLY", "needs_vlm": False, "reason": "image_too_small", "score": 0}
        except Exception:
            pass

        score = 0
        reasons = []

        # -----------------------------------------------------------------
        # Contextual filter: When page_data is available, distinguish
        # genuine diagram pages from dense text/table pages with border rules
        # -----------------------------------------------------------------
        is_pymupdf_render = "pymupdf" in image_path.name.lower() or figure.get("caption") == "Rendered vector content"
        
        if page_data and isinstance(page_data, dict):
            text_blocks = page_data.get("text_blocks", [])
            total_text_chars = sum(len(str(b.get("text", ""))) for b in text_blocks if isinstance(b, dict))
            all_text_upper = " ".join(str(b.get("text", "")) for b in text_blocks if isinstance(b, dict)).upper()

            # Signal A: Diagram-dominant page (sparse text on page with figures)
            if total_text_chars <= 400 and len(page_data.get("figures", [])) >= 1:
                score += 2
                reasons.append("diagram_dominant_page")

            # Signal B: Page contains explicit diagram title/heading
            if any(title in all_text_upper for title in PAGE_DIAGRAM_TITLES):
                score += 2
                reasons.append("page_diagram_title")

            # If it's a full-page PyMuPDF render and the page has dense text (> 500 chars)
            # with no diagram heading or arrow/label flags, it is a text/table page with rule lines
            if is_pymupdf_render and total_text_chars > 500:
                if not (figure.get("has_arrows") or figure.get("has_labels") or figure.get("is_diagram")):
                    return {"classification": "TEXT_ONLY", "needs_vlm": False, "reason": "dense_text_page_not_diagram", "score": 0}

        # -----------------------------------------------------------------
        # Deterministic Scoring System
        # -----------------------------------------------------------------
        
        # 1. High drawing complexity / Vector drawings
        fig_type = str(
            figure.get("figure_type")
            or figure.get("type")
            or figure.get("item_type")
            or ""
        ).lower()
        if any(kw in fig_type for kw in {"vector", "drawing", "diagram", "schematic", "cad"}):
            score += 2
            reasons.append("vector_or_diagram_type")

        # 2. Unusual / mixed content (Meaningful author caption)
        caption = str(figure.get("caption", "")).lower()
        if caption and caption != "rendered vector content":
            if any(kw in caption for kw in DIAGRAM_KEYWORDS):
                score += 1
                reasons.append("diagram_keyword_in_caption")

        if is_pymupdf_render:
            score += 2
            reasons.append("rendered_vector_drawings")

        if figure.get("is_vector") is True or figure.get("is_diagram") is True:
            score += 2
            reasons.append("vector_flag")

        # 3. Many labels with uncertain relationships (Arrows and Labels)
        if figure.get("has_arrows") is True:
            score += 2
            reasons.append("has_arrows")
        elif "arrow" in caption or "arrow" in str(figure.get("description", "")).lower():
            score += 1
            reasons.append("arrow_in_metadata")

        if figure.get("has_labels") is True:
            score += 2
            reasons.append("has_labels")
        elif any(kw in caption for kw in {"label", "callout", "balloon", "item no", "legend"}):
            score += 1
            reasons.append("label_in_metadata")

        # 4. Insufficient structural evidence (Large image area without structure)
        bbox = figure.get("bbox")
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                bw = abs(float(bbox[2]) - float(bbox[0]))
                bh = abs(float(bbox[3]) - float(bbox[1]))
                if (bw >= 250 and bh >= 250) or (bw * bh >= 80000):
                    score += 1
                    reasons.append("large_bbox")
            except (ValueError, TypeError):
                pass

        try:
            with Image.open(image_path) as img:
                w, h = img.size
                if (w >= 300 and h >= 300) or (w * h >= 120000):
                    score += 1
                    reasons.append("large_image_dimensions")
        except Exception as exc:
            logger.debug("candidate_selector.image_open_failed", path=str(image_path), error=str(exc))

        # 5. Low OCR confidence (Simulated for now, replace with actual OCR confidence check)
        ocr_confidence = figure.get("ocr_confidence", 1.0)
        if ocr_confidence < 0.6:
            score += 2
            reasons.append("low_ocr_confidence")

        classification = "SIMPLE_DIAGRAM"
        needs_vlm = False

        if score >= 5:
            classification = "COMPLEX_DIAGRAM"
            needs_vlm = True
        elif score >= 3:
            classification = "AMBIGUOUS"
            needs_vlm = True
        elif score > 0:
            classification = "SIMPLE_DIAGRAM"
            needs_vlm = False
        else:
            classification = "TEXT_ONLY"
            needs_vlm = False

        return {
            "classification": classification,
            "needs_vlm": needs_vlm,
            "reason": " + ".join(reasons) if reasons else "no_diagram_signals",
            "score": score
        }
