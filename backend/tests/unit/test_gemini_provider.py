"""
Unit tests for GeminiVisionProvider and the Gemini integration path.

All tests inject a fake google.genai module into sys.modules so that
NO real GEMINI_API_KEY and NO google-genai installation is required.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from PIL import Image


# ── Fake google.genai module injection ───────────────────────────────────────

def _make_fake_genai_modules() -> tuple[ModuleType, ModuleType, MagicMock]:
    """
    Build fake `google`, `google.genai`, and `google.genai.types` modules
    that can be injected into sys.modules before importing gemini_provider.

    Returns (fake_google, fake_genai, mock_client_instance).
    """
    mock_client_instance = MagicMock()

    # google.genai.types
    fake_types = types.ModuleType("google.genai.types")
    fake_types.Part = MagicMock()
    fake_types.Part.from_bytes = MagicMock(return_value=MagicMock())
    fake_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
    fake_types.HttpOptions = MagicMock(return_value=MagicMock())

    # google.genai
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = MagicMock(return_value=mock_client_instance)
    fake_genai.types = fake_types

    # google (namespace package)
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai

    return fake_google, fake_genai, mock_client_instance


@pytest.fixture(autouse=False)
def inject_fake_genai(monkeypatch):
    """
    Fixture: injects fake google.genai into sys.modules for the duration of
    the test, then restores the original state.
    """
    fake_google, fake_genai, mock_client_instance = _make_fake_genai_modules()

    orig_google = sys.modules.get("google")
    orig_genai = sys.modules.get("google.genai")
    orig_types = sys.modules.get("google.genai.types")

    orig_google_genai = None
    if orig_google is not None:
        orig_google_genai = getattr(orig_google, "genai", None)
        orig_google.genai = fake_genai

    sys.modules["google"] = fake_google if orig_google is None else orig_google
    sys.modules["google.genai"] = fake_genai
    sys.modules["google.genai.types"] = fake_genai.types

    yield fake_google, fake_genai, mock_client_instance

    # Restore
    if orig_google is None:
        sys.modules.pop("google", None)
    else:
        sys.modules["google"] = orig_google
        if orig_google_genai is None:
            if hasattr(orig_google, "genai"):
                delattr(orig_google, "genai")
        else:
            orig_google.genai = orig_google_genai

    if orig_genai is None:
        sys.modules.pop("google.genai", None)
    else:
        sys.modules["google.genai"] = orig_genai
    if orig_types is None:
        sys.modules.pop("google.genai.types", None)
    else:
        sys.modules["google.genai.types"] = orig_types


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture
def diagram_image(tmp_path) -> Path:
    """400×400 PNG diagram fixture."""
    p = tmp_path / "hydraulic_diagram.png"
    Image.new("RGB", (400, 400), color="blue").save(p)
    return p


def _make_mock_response(data: dict) -> MagicMock:
    """Build a fake Gemini response whose .text is the JSON string of data."""
    resp = MagicMock()
    resp.text = json.dumps(data)
    return resp


_GOOD_PAYLOAD = {
    "functional_summary": "Hydraulic control circuit showing pump, valve, and cylinder.",
    "components": ["Pump", "Control Valve", "Hydraulic Cylinder", "Filter"],
    "relationships": [
        {"from": "Pump", "to": "Control Valve", "type": "FEEDS"},
        {"from": "Control Valve", "to": "Hydraulic Cylinder", "type": "CONTROLS"},
    ],
    "spatial_layout": "Left-to-right flow: pump on left, cylinder on right.",
    "important_labels": ["P1", "V1", "C1"],
    "important_numbers": ["210 bar", "45 Nm"],
    "technical_observations": ["Arrow indicates flow direction from pump to valve."],
    "uncertainty": [],
}


def _make_provider(fake_genai_modules):
    """Instantiate GeminiVisionProvider with the injected fake SDK."""
    from app.services.vision.gemini_provider import GeminiVisionProvider
    return GeminiVisionProvider(api_key="test-key", model="gemini-2.5-flash")


# ── 1. Provider creation ──────────────────────────────────────────────────────

class TestGeminiProviderCreation:
    def test_created_with_valid_key(self, inject_fake_genai):
        fake_google, fake_genai, mock_client = inject_fake_genai
        provider = _make_provider(inject_fake_genai)
        assert provider.model_name == "gemini-2.5-flash"
        assert provider.provider_name == "gemini"
        fake_genai.Client.assert_called_with(api_key="test-key")

    def test_raises_on_missing_api_key(self, inject_fake_genai):
        import app.services.vision.gemini_provider as gmod
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            gmod.GeminiVisionProvider(api_key="", model="gemini-2.5-flash")

    def test_factory_builds_gemini_provider(self, inject_fake_genai):
        """build_vision_provider returns GeminiVisionProvider when VISION_PROVIDER=gemini."""
        from app.services.vision.vision_provider import build_vision_provider, GeminiVisionProvider

        settings = MagicMock()
        settings.vision_provider = "gemini"
        settings.gemini_api_key = "test-key"
        settings.gemini_vision_model = "gemini-2.5-flash"
        settings.vision_request_timeout = 120.0

        provider = build_vision_provider(settings)
        assert isinstance(provider, GeminiVisionProvider)

    def test_factory_raises_missing_key(self):
        """build_vision_provider raises ValueError if API key is empty."""
        from app.services.vision.vision_provider import build_vision_provider

        settings = MagicMock()
        settings.vision_provider = "gemini"
        settings.gemini_api_key = ""

        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            build_vision_provider(settings)

    def test_factory_builds_disabled_provider(self):
        from app.services.vision.vision_provider import build_vision_provider, DisabledVisionProvider

        settings = MagicMock()
        settings.vision_provider = "disabled"

        provider = build_vision_provider(settings)
        assert isinstance(provider, DisabledVisionProvider)




# ── 2. Structured response parsing ───────────────────────────────────────────

class TestGeminiResponseParsing:
    def test_successful_response_parsed(self, inject_fake_genai, diagram_image):
        _, fake_genai, mock_client = inject_fake_genai
        provider = _make_provider(inject_fake_genai)
        mock_resp = _make_mock_response(_GOOD_PAYLOAD)
        mock_client.models.generate_content.return_value = mock_resp

        result = provider.analyze_diagram(str(diagram_image))

        assert result["functional_summary"] == _GOOD_PAYLOAD["functional_summary"]
        assert result["components"] == _GOOD_PAYLOAD["components"]
        assert len(result["relationships"]) == 2
        assert result["relationships"][0]["from"] == "Pump"
        assert result["model_used"] == "gemini-2.5-flash"
        assert result["prompt_version"] == "v2"
        assert result["status"] == "ok"

    def test_malformed_response_returns_empty_fields(self, inject_fake_genai, diagram_image):
        """When Gemini returns non-JSON text, provider returns safe empty structure."""
        _, _, mock_client = inject_fake_genai
        provider = _make_provider(inject_fake_genai)
        bad_resp = MagicMock()
        bad_resp.text = "Sorry, I cannot process this image."
        mock_client.models.generate_content.return_value = bad_resp

        result = provider.analyze_diagram(str(diagram_image))

        assert "functional_summary" in result
        assert isinstance(result["components"], list)
        assert isinstance(result["relationships"], list)

    def test_empty_response_returns_safe_structure(self, inject_fake_genai, diagram_image):
        _, _, mock_client = inject_fake_genai
        provider = _make_provider(inject_fake_genai)
        empty_resp = MagicMock()
        empty_resp.text = ""
        mock_client.models.generate_content.return_value = empty_resp

        result = provider.analyze_diagram(str(diagram_image))

        assert result["functional_summary"] == "Vision analysis unavailable"
        assert result["components"] == []

    def test_partial_response_normalized(self, inject_fake_genai, diagram_image):
        """Response missing optional fields still has required fields."""
        _, _, mock_client = inject_fake_genai
        provider = _make_provider(inject_fake_genai)
        partial = {"functional_summary": "A valve assembly.", "components": ["Valve"]}
        mock_client.models.generate_content.return_value = _make_mock_response(partial)

        result = provider.analyze_diagram(str(diagram_image))

        assert result["functional_summary"] == "A valve assembly."
        assert result["components"] == ["Valve"]
        assert result["relationships"] == []
        assert result["important_labels"] == []


# ── 3. Error handling ─────────────────────────────────────────────────────────

class TestGeminiErrorHandling:
    def test_file_not_found_raises(self, inject_fake_genai, tmp_path):
        provider = _make_provider(inject_fake_genai)
        with pytest.raises(FileNotFoundError):
            provider.analyze_diagram(str(tmp_path / "nonexistent.png"))

    def test_api_error_propagated(self, inject_fake_genai, diagram_image):
        """API errors are propagated so s04_vision.py can apply retry/timeout logic."""
        _, _, mock_client = inject_fake_genai
        provider = _make_provider(inject_fake_genai)
        mock_client.models.generate_content.side_effect = RuntimeError("API unavailable")

        with pytest.raises(RuntimeError, match="API unavailable"):
            provider.analyze_diagram(str(diagram_image))

    def test_timeout_propagated(self, inject_fake_genai, diagram_image):
        """Timeout exceptions propagate; s04_vision sets vision_status=TIMEOUT."""
        _, _, mock_client = inject_fake_genai
        provider = _make_provider(inject_fake_genai)
        mock_client.models.generate_content.side_effect = TimeoutError("Request timed out")

        with pytest.raises(TimeoutError):
            provider.analyze_diagram(str(diagram_image))


# ── 4. Ollama not required when Gemini is active ──────────────────────────────

class TestOllamaNotRequired:
    def test_gemini_provider_has_no_ollama_dependency(self, inject_fake_genai):
        """GeminiVisionProvider must not import or instantiate VisionService."""
        with patch("app.services.vision.service.VisionService") as mock_svc:
            _make_provider(inject_fake_genai)
            mock_svc.assert_not_called()

    def test_disabled_provider_when_ollama_down(self, diagram_image):
        """DisabledVisionProvider works even if Ollama is completely unreachable."""
        from app.services.vision.vision_provider import DisabledVisionProvider
        provider = DisabledVisionProvider()
        result = provider.analyze_diagram(str(diagram_image))
        assert result["status"] == "SKIPPED"
        assert result["components"] == []


# ── 5. Integration with s04_vision step ──────────────────────────────────────

@pytest.mark.asyncio
async def test_s04_uses_gemini_provider(tmp_path, monkeypatch):
    """s04_vision step calls analyze_diagram on whatever provider is injected."""
    from app.agents.supervisor.state import IngestionState
    from app.config.settings import get_settings
    from app.pipelines.ingestion.steps.s04_vision import step as vision_step

    cache_dir = tmp_path / "isolated_cache"
    monkeypatch.setattr(get_settings(), "vision_cache_dir", str(cache_dir))

    img = tmp_path / f"fig_{uuid4().hex}.png"
    im = Image.new("RGB", (400, 400), color="red")
    im.putpixel((0, 0), (int(uuid4().int % 255), 10, 20))
    im.save(img)

    fig = {
        "image_path": str(img),
        "has_arrows": True,
        "figure_type": "vector_drawing",  # scores ≥ 3 → candidate
    }

    mock_provider = MagicMock()
    mock_provider.provider_name = "gemini"
    mock_provider.model_name = "gemini-2.5-flash"
    mock_provider.fallback_model = None
    mock_provider.analyze_diagram.return_value = {
        "functional_summary": "Pump schematic",
        "components": ["Pump", "Valve"],
        "relationships": [{"from": "Pump", "to": "Valve", "type": "FEEDS"}],
        "spatial_layout": "Left to right",
        "status": "ok",
        "model_used": "gemini-2.5-flash",
    }

    state = IngestionState(
        document_id=uuid4(),
        filename="manual.pdf",
        tenant_id="tenant-1",
        storage_path=str(tmp_path / "manual.pdf"),
        parsed_doc={"pages": [{"page_num": 1, "figures": [fig]}]},
    )

    result = await vision_step(state, {"vision": mock_provider})
    out_fig = result.parsed_doc["pages"][0]["figures"][0]

    assert mock_provider.analyze_diagram.call_count == 1
    assert out_fig["vision_status"] in ("PRIMARY_SUCCESS", "CACHED")
    assert out_fig["vision_analysis"]["functional_summary"] == "Pump schematic"
