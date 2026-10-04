"""
Vision Provider Abstraction.
Supports:
  - DisabledVisionProvider  — safe no-op for cloud/Render without API key
  - GeminiVisionProvider    — primary cloud vision via Google Gemini
Set VISION_PROVIDER=gemini in .env to use Gemini.
"""
from abc import ABC, abstractmethod
from typing import Any

from app.config.logging import get_logger

logger = get_logger(__name__)


class BaseVisionProvider(ABC):
    """Abstract interface for technical diagram/schematic vision analysis."""

    provider_name: str = "base"
    model_name: str = "n/a"

    @abstractmethod
    def analyze_diagram(self, image_path: str, timeout: float | None = None) -> dict[str, Any]:
        """
        Analyze a diagram image and return structured analysis dict.
        Must return a dict with keys:
            - functional_summary (str)
            - components (list[str])
            - relationships (list[dict])
            - spatial_layout (str)
        """
        pass

    def close(self) -> None:
        """Cleanup resources if needed."""
        pass


class DisabledVisionProvider(BaseVisionProvider):
    """
    No-op vision provider used when vision analysis is disabled.
    Returns empty analysis structure without failing.
    Images are still extracted and served via /api/v1/images/...
    """

    provider_name = "disabled"
    model_name = "disabled"

    def analyze_diagram(self, image_path: str, timeout: float | None = None) -> dict[str, Any]:
        logger.info(
            "vision.disabled.skip_analysis",
            provider=self.provider_name,
            model=self.model_name,
            image=image_path,
        )
        return {
            "functional_summary": "Vision analysis disabled",
            "components": [],
            "relationships": [],
            "spatial_layout": "Unknown",
            "status": "SKIPPED",
        }

    def close(self) -> None:
        pass


class GeminiVisionProvider(BaseVisionProvider):
    """
    Google Gemini cloud vision provider (primary production provider).

    Delegates to app.services.vision.gemini_provider.GeminiVisionProvider
    which handles the google-genai SDK details.
    """

    provider_name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", timeout: float = 120.0) -> None:
        from app.services.vision.gemini_provider import GeminiVisionProvider as _GeminiImpl
        self._impl = _GeminiImpl(api_key=api_key, model=model, timeout=timeout)
        self.model_name = model
        self.timeout = timeout

    def analyze_diagram(self, image_path: str, timeout: float | None = None) -> dict[str, Any]:
        return self._impl.analyze_diagram(image_path, timeout=timeout)

    def close(self) -> None:
        self._impl.close()





def build_vision_provider(settings: Any) -> BaseVisionProvider:
    """
    Factory: instantiate the appropriate VisionProvider from configuration.

    VISION_PROVIDER=gemini  → GeminiVisionProvider (primary, cloud-based)
    VISION_PROVIDER=disabled → DisabledVisionProvider (no-op)
    """
    provider_type = getattr(settings, "vision_provider", "disabled").lower().strip()

    if provider_type == "gemini":
        api_key = getattr(settings, "gemini_api_key", "") or ""
        if not api_key:
            logger.error(
                "vision.gemini.missing_api_key",
                hint="Set GEMINI_API_KEY in your .env file",
            )
            raise ValueError(
                "GEMINI_API_KEY is required when VISION_PROVIDER=gemini. "
                "Add it to your .env file."
            )
        model = getattr(settings, "gemini_vision_model", "gemini-2.5-flash")
        timeout = getattr(settings, "vision_request_timeout", 120.0)
        logger.info("vision.provider_initialized", provider="gemini", model=model)
        return GeminiVisionProvider(api_key=api_key, model=model, timeout=timeout)



    logger.info("vision.provider_initialized", provider="disabled")
    return DisabledVisionProvider()
