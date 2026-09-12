"""
Vision Provider Abstraction.
Supports DisabledVisionProvider (safe for cloud/Render) and OllamaVisionProvider (for local ingestion).
"""
from abc import ABC, abstractmethod
from typing import Any

from app.config.logging import get_logger
from app.services.vision.service import VisionService

logger = get_logger(__name__)


class BaseVisionProvider(ABC):
    """Abstract interface for technical diagram/schematic vision analysis."""

    @abstractmethod
    def analyze_diagram(self, image_path: str) -> dict[str, Any]:
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
    No-op vision provider used when vision analysis is disabled (e.g. Render 512MB RAM).
    Returns empty analysis structure without failing. Images are still extracted and served.
    """

    def analyze_diagram(self, image_path: str) -> dict[str, Any]:
        logger.debug("vision.disabled.skip_analysis", image=image_path)
        return {
            "functional_summary": "Vision analysis disabled",
            "components": [],
            "relationships": [],
            "spatial_layout": "Unknown",
        }

    def close(self) -> None:
        pass


class OllamaVisionProvider(BaseVisionProvider):
    """
    Ollama-backed vision provider for local/dedicated ingestion environments.
    Wraps existing VisionService.
    """

    def __init__(self, ollama_base_url: str, model: str = "qwen2.5vl:3b") -> None:
        self._service = VisionService(ollama_base_url=ollama_base_url, model=model)

    def analyze_diagram(self, image_path: str) -> dict[str, Any]:
        return self._service.analyze_diagram(image_path)

    def close(self) -> None:
        self._service.close()


def build_vision_provider(settings: Any) -> BaseVisionProvider:
    """Factory to instantiate the appropriate VisionProvider based on configuration."""
    provider_type = getattr(settings, "vision_provider", "disabled").lower().strip()

    if provider_type == "ollama":
        model = getattr(settings, "ollama_vision_model", "qwen2.5vl:3b")
        logger.info("vision.provider_initialized", provider="ollama", model=model)
        return OllamaVisionProvider(
            ollama_base_url=settings.ollama_base_url,
            model=model,
        )

    logger.info("vision.provider_initialized", provider="disabled")
    return DisabledVisionProvider()
