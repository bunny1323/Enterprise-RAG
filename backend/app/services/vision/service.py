"""
Deprecated: The local Ollama vision service was replaced by the Gemini Cloud Vision provider.
VisionService is retained as a stub for backward compatibility.
"""
from typing import Any


class VisionService:
    """Deprecated stub: Use build_vision_provider() from app.services.vision.vision_provider instead."""

    def __init__(self, ollama_base_url: str = "", model: str = "", **kwargs: Any) -> None:
        self.ollama_base_url = ollama_base_url
        self.model = model

    def analyze_diagram(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "VisionService (Ollama) has been deprecated and replaced by Gemini Cloud Vision. "
            "Use build_vision_provider() instead."
        )
