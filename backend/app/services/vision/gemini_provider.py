"""
Gemini Cloud Vision Provider.

Uses the official google-genai Python SDK to analyze technical engineering
diagrams and schematics via Gemini's vision capabilities.

Design principles:
- Structured output via Pydantic schema (no brittle free-form JSON parsing)
- Never modifies the original image on disk
- Raises standard Python exceptions so the caller (s04_vision.py) can apply
  timeout / retry / failure-isolation logic uniformly
- Never logs the API key
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from app.config.logging import get_logger

logger = get_logger(__name__)

# ── Versioned prompt ──────────────────────────────────────────────────────────
VISION_PROMPT_VERSION = "v2"

_ANALYSIS_PROMPT = """\
You are analyzing a technical engineering/industrial maintenance manual image.

Analyze ONLY information visibly supported by the image.
Do NOT invent components, relationships, measurements, labels, part numbers, or procedures.

Identify:
1. functional_summary — one concise paragraph describing what the diagram shows
2. components — list of visible named components, assemblies, or parts
3. relationships — directional or structural connections between components
4. spatial_layout — brief description of how components are arranged spatially
5. important_labels — visible text labels, part numbers, callouts
6. important_numbers — torque values, dimensions, specifications, counts
7. technical_observations — key technical details (arrows, flow direction, mounting info)
8. uncertainty — things you cannot determine confidently from the image

Pay special attention to:
- mechanical assemblies, hydraulic components, electrical/wiring diagrams
- exploded views, arrows, torque values, dimensions, part numbers
- connectors, hoses, cylinders, valves, bolts, brackets, mounting relationships
- maintenance procedures visible in the diagram

If something cannot be confidently determined from the image, leave it empty or note it in uncertainty.
Do NOT hallucinate missing information.
"""


class GeminiVisionProvider:
    """
    Cloud-based vision provider backed by Google Gemini.

    analyze_diagram() is synchronous (blocking) so it can be called from
    asyncio.run_in_executor inside s04_vision.py, matching the existing
    OllamaVisionProvider interface.
    """

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for GeminiVisionProvider. "
                "Set it in your .env file."
            )
        self.model_name = model
        self.timeout = timeout
        self._api_key = api_key  # kept private; never logged

        # Import lazily so missing package doesn't crash the import of this module
        try:
            from google import genai  # type: ignore[import]
            self._client = genai.Client(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. "
                "Run: pip install google-genai"
            ) from exc

        logger.info(
            "vision.provider_initialized",
            provider=self.provider_name,
            model=self.model_name,
        )

    def analyze_diagram(self, image_path: str, timeout: float | None = None) -> dict[str, Any]:
        """
        Analyze a technical diagram image using Gemini.

        Args:
            image_path: Path to the (possibly pre-processed/resized) image file.
            timeout:    Per-call timeout in seconds. Falls back to self.timeout.

        Returns:
            dict conforming to the Enterprise-RAG vision analysis contract:
                functional_summary, components, relationships, spatial_layout,
                important_labels, important_numbers, technical_observations,
                uncertainty, model_used, prompt_version

        Raises:
            FileNotFoundError: Image not found on disk.
            google.api_core.exceptions.GoogleAPIError subclasses: propagated so
                the caller can distinguish rate-limit (429) from other errors.
            ValueError: Structured output validation failure.
        """
        from google import genai  # type: ignore[import]
        from google.genai import types  # type: ignore[import]

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_bytes = path.read_bytes()
        suffix = path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        mime_type = mime_map.get(suffix, "image/png")

        logger.info(
            "vision.figure_started",
            provider=self.provider_name,
            model=self.model_name,
            image=image_path,
            size_kb=round(len(image_bytes) / 1024, 1),
        )

        req_timeout = timeout or self.timeout

        # Build request with inline image bytes
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=[image_part, _ANALYSIS_PROMPT],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "functional_summary": {"type": "string"},
                        "components": {"type": "array", "items": {"type": "string"}},
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                    "type": {"type": "string"},
                                },
                                "required": ["from", "to", "type"],
                            },
                        },
                        "spatial_layout": {"type": "string"},
                        "important_labels": {"type": "array", "items": {"type": "string"}},
                        "important_numbers": {"type": "array", "items": {"type": "string"}},
                        "technical_observations": {"type": "array", "items": {"type": "string"}},
                        "uncertainty": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "functional_summary",
                        "components",
                        "relationships",
                        "spatial_layout",
                    ],
                },
                http_options=types.HttpOptions(timeout=int(req_timeout * 1000)),  # ms
            ),
        )

        # Parse and normalize
        parsed = self._parse_response(response, image_path)
        parsed["model_used"] = self.model_name
        parsed["prompt_version"] = VISION_PROMPT_VERSION

        logger.info(
            "vision.figure_success",
            provider=self.provider_name,
            model=self.model_name,
            image=image_path,
            components=len(parsed.get("components", [])),
            relationships=len(parsed.get("relationships", [])),
        )
        return parsed

    @staticmethod
    def _parse_response(response: Any, image_path: str) -> dict[str, Any]:
        """Extract structured data from a Gemini response object."""
        import json

        # Gemini structured output: response.text is valid JSON when
        # response_mime_type="application/json" is set.
        raw_text = ""
        try:
            raw_text = response.text or ""
            data = json.loads(raw_text)
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            logger.warning(
                "vision.gemini.parse_warning",
                image=image_path,
                error=str(exc),
                raw=raw_text[:300],
            )
            data = {}

        return {
            "functional_summary": str(data.get("functional_summary", "Vision analysis unavailable")),
            "components": _safe_list(data.get("components", [])),
            "relationships": _safe_relationship_list(data.get("relationships", [])),
            "spatial_layout": str(data.get("spatial_layout", "Unknown")),
            "important_labels": _safe_list(data.get("important_labels", [])),
            "important_numbers": _safe_list(data.get("important_numbers", [])),
            "technical_observations": _safe_list(data.get("technical_observations", [])),
            "uncertainty": _safe_list(data.get("uncertainty", [])),
            "status": "ok",
        }

    def close(self) -> None:
        """No persistent connections to close for Gemini HTTP client."""
        pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _safe_relationship_list(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, dict) and "from" in item and "to" in item:
            out.append({
                "from": str(item.get("from", "")),
                "to": str(item.get("to", "")),
                "type": str(item.get("type", "RELATED_TO")),
            })
    return out
