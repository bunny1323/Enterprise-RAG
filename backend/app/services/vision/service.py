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
    "Identify only the most important information visible in the diagram. "
    "Return ONLY a valid JSON object with exactly these keys:\n"
    "{\n"
    '  "functional_summary": "concise summary, maximum 3 sentences",\n'
    '  "components": ["maximum 20 important component names"],\n'
    '  "relationships": [\n'
    '    {"from": "component_a", "to": "component_b", "type": "RELATIONSHIP_TYPE"}\n'
    "  ],\n"
    '  "spatial_layout": "concise description of the spatial organization"\n'
    "}\n"
    "Return at most 15 important relationships. "
    "Do not repeat relationships. "
    "Do not include markdown or any text outside the JSON object."
)

class VisionService:
    """
    Stateless diagram/schematic vision analysis via local Ollama llava model.

    Uses Ollama's /api/generate endpoint with base64-encoded image.
    No external API calls — entirely free and offline.
    """

    def __init__(self, ollama_base_url: str, model: str = "qwen2.5vl:3b") -> None:
        self._base_url = ollama_base_url.rstrip("/")
        self._model = model
        # Reuse connection pool across requests
        self._http = httpx.Client(timeout=600.0)

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
            "format": "json",
            "options": {
                "temperature": 0.1,  # Low temp for structured JSON output
                "num_predict": 1024,
            },
        }

        logger.debug("vision.analyze_start", model=self._model, image=image_path)

        response = self._http.post(
            f"{self._base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()

        raw_text: str = response.json().get("response", "")
        result = self._parse_json_response(raw_text, image_path)

        logger.info(
            "vision.analyze_complete",
            image=image_path,
            model=self._model,
            components=len(result.get("components", [])),
        )
        return result

    def _parse_json_response(self, raw: str, image_path: str | None = None) -> dict[str, Any]:
        """
        Extract and parse a JSON object from the model's raw text output.

        Handles cases where the model wraps JSON in markdown code fences.
        Accepts either the required structured schema or a generic Ollama answer
        and normalizes it into a diagram-analysis payload.
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", raw or "").strip().rstrip("`").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1

        if start == -1 or end == 0:
            msg = "Ollama returned no valid JSON object for diagram analysis."
            logger.error("vision.json_not_found", image=image_path, raw=raw[:200] if raw else "")
            raise ValueError(msg)

        try:
            parsed = json.loads(cleaned[start:end])
            if not isinstance(parsed, dict):
                raise ValueError("Ollama response was not a JSON object.")
            return self._normalize_result(parsed)
        except (json.JSONDecodeError, ValueError) as err:
            logger.error(
                "vision.json_parse_error",
                image=image_path,
                error=str(err),
                raw=raw[:200] if raw else "",
            )
            raise ValueError(f"Failed to parse Ollama vision response: {err}") from err

    @staticmethod
    def _normalize_result(parsed: dict[str, Any]) -> dict[str, Any]:
        required_keys = {"functional_summary", "components", "relationships", "spatial_layout"}
        if required_keys.issubset(parsed.keys()):
            result = dict(parsed)
            result.setdefault("status", "ok")
            return result

        summary = (
            parsed.get("functional_summary")
            or parsed.get("summary")
            or parsed.get("description")
            or parsed.get("text")
            or parsed.get("answer")
            or "Vision analysis unavailable"
        )
        components = parsed.get("components") or parsed.get("objects") or parsed.get("visible_components") or []
        if isinstance(components, str):
            components = [components]
        if not isinstance(components, list):
            components = []

        relationships = parsed.get("relationships") or []
        if not isinstance(relationships, list):
            relationships = []

        spatial_layout = (
            parsed.get("spatial_layout")
            or parsed.get("layout")
            or parsed.get("diagram_layout")
            or "Unknown"
        )

        return {
            "functional_summary": str(summary),
            "components": [str(item) for item in components if item is not None],
            "relationships": relationships,
            "spatial_layout": str(spatial_layout),
            "status": "ok",
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "functional_summary": "Vision analysis unavailable",
            "components": [],
            "relationships": [],
            "spatial_layout": "Unknown",
            "status": "failed",
        }

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()
