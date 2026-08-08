"""
Vision Service — local Ollama llava for diagram/schematic analysis.
Zero API cost; runs entirely on local GPU/CPU via Ollama.
"""
import base64
import json
import re
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config.logging import get_logger

logger = get_logger(__name__)

_ANALYSIS_PROMPT = (
    "Analyze this technical diagram or engineering schematic carefully. "
    "Return ONLY a valid JSON object with exactly these keys:\n"
    "{\n"
    '  "functional_summary": "string describing what this diagram shows",\n'
    '  "components": ["list", "of", "identified", "components"],\n'
    '  "relationships": [\n'
    '    {"from": "component_a", "to": "component_b", "type": "RELATIONSHIP_TYPE"}\n'
    "  ],\n"
    '  "spatial_layout": "string describing the spatial organization"\n'
    "}\n"
    "Do not include any text outside the JSON object."
)


class VisionService:
    """
    Stateless diagram/schematic vision analysis via local Ollama llava model.

    Uses Ollama's /api/generate endpoint with base64-encoded image.
    No external API calls — entirely free and offline.
    """

    def __init__(self, ollama_base_url: str, model: str = "llava:13b") -> None:
        self._base_url = ollama_base_url.rstrip("/")
        self._model = model
        # Reuse connection pool across requests
        self._http = httpx.Client(timeout=120.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    def analyze_diagram(self, image_path: str) -> dict[str, Any]:
        """
        Analyze a technical diagram image using the local llava model.

        Args:
            image_path: Absolute path to the image file.

        Returns:
            Dictionary with keys:
                - functional_summary (str)
                - components (list[str])
                - relationships (list[dict])
                - spatial_layout (str)

        Raises:
            FileNotFoundError: If the image file does not exist.
            httpx.HTTPError: On Ollama API communication failure (after retries).
            ValueError: If the model returns malformed JSON.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Encode image as base64
        image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")

        payload = {
            "model": self._model,
            "prompt": _ANALYSIS_PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temp for structured JSON output
                "num_predict": 512,
            },
        }

        logger.debug("vision.analyze_start", model=self._model, image=image_path)

        response = self._http.post(
            f"{self._base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()

        raw_text: str = response.json().get("response", "")
        result = self._parse_json_response(raw_text)

        logger.info(
            "vision.analyze_complete",
            image=image_path,
            components=len(result.get("components", [])),
        )
        return result

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        """
        Extract and parse a JSON object from the model's raw text output.

        Handles cases where the model wraps JSON in markdown code fences.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        # Find the first { ... } block
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("vision.json_not_found", raw=raw[:200])
            return self._empty_result()

        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError as err:
            logger.warning("vision.json_parse_error", error=str(err), raw=raw[:200])
            return self._empty_result()

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "functional_summary": "Vision analysis unavailable",
            "components": [],
            "relationships": [],
            "spatial_layout": "Unknown",
        }

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()
