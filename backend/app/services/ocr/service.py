"""
OCR Service — Tesseract primary, EasyOCR fallback for low-confidence results.
Handles scanned pages and image regions that contain embedded text.
"""
from app.config.logging import get_logger

logger = get_logger(__name__)

# Confidence threshold below which we switch to EasyOCR
_MIN_CONFIDENCE = 50


class OCRService:
    """
    Stateless OCR service.

    Primary: pytesseract (Tesseract v5).
    Fallback: EasyOCR when Tesseract confidence is low or result is empty.
    """

    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image file.

        First attempts Tesseract; if confidence is below threshold or result
        is empty, falls back to EasyOCR for better accuracy on difficult images.

        Args:
            image_path: Absolute path to image file (PNG, JPEG, TIFF, etc.).

        Returns:
            Extracted text string. Empty string if no text found.
        """
        # ── Tesseract ──────────────────────────────────────────────────────────
        try:
            text, confidence = self._run_tesseract(image_path)
            if text.strip() and confidence >= _MIN_CONFIDENCE:
                logger.debug(
                    "ocr.tesseract_success",
                    path=image_path,
                    confidence=confidence,
                    chars=len(text),
                )
                return text
        except Exception as err:
            logger.warning("ocr.tesseract_error", error=str(err), path=image_path)

        # ── EasyOCR fallback ───────────────────────────────────────────────────
        try:
            text = self._run_easyocr(image_path)
            logger.debug("ocr.easyocr_fallback_used", path=image_path, chars=len(text))
            return text
        except Exception as err:
            logger.error("ocr.easyocr_error", error=str(err), path=image_path)
            return ""

    # ── Private helpers ────────────────────────────────────────────────────────

    def _run_tesseract(self, image_path: str) -> tuple[str, float]:
        """
        Run Tesseract OCR and return (text, mean_confidence).

        Returns:
            Tuple of (extracted_text, mean_confidence_0_100).
        """
        import pytesseract  # type: ignore[import-untyped]
        from PIL import Image

        image = Image.open(image_path)
        # Convert to RGB if needed (Tesseract handles RGB best)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Get per-word data including confidence scores
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 6",
        )
        confidences = [
            int(c)
            for c in data.get("conf", [])
            if str(c).lstrip("-").isdigit() and int(c) >= 0
        ]
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        text = pytesseract.image_to_string(image, config="--oem 3 --psm 6")
        return text, mean_confidence

    def _run_easyocr(self, image_path: str) -> str:
        """
        Run EasyOCR and return concatenated text from all detected regions.

        EasyOCR is lazily imported to avoid slow GPU initialization until needed.
        """
        import easyocr  # type: ignore[import-untyped]

        reader = easyocr.Reader(
            ["en"],
            gpu=False,  # Set True if GPU is available for speed
            verbose=False,
        )
        results = reader.readtext(image_path)
        # results: list of ([bbox], text, confidence)
        texts = [text for (_, text, _confidence) in results]
        return " ".join(texts)
