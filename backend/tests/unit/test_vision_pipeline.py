"""
Unit tests for the scalable vision pipeline.
Tests candidate selection, content-based caching, image preprocessing,
concurrency limits, timeout isolation, retry behavior, and contract preservation.
"""
import asyncio
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from PIL import Image

from app.agents.supervisor.state import IngestionState
from app.config.settings import get_settings
from app.pipelines.ingestion.steps.s04_vision import step as s04_vision_step
from app.services.vision.candidate_selector import CandidateSelector
from app.services.vision.image_preprocessor import preprocess_for_vision
from app.services.vision.vision_cache import VisionCache
from app.services.vision.vision_provider import BaseVisionProvider


@pytest.fixture
def sample_image(tmp_path):
    """Create a standard test image."""
    img_path = tmp_path / "diagram.png"
    img = Image.new("RGB", (400, 400), color="blue")
    img.save(img_path)
    return img_path


@pytest.fixture
def large_image(tmp_path):
    """Create an oversized test image exceeding 1920px."""
    img_path = tmp_path / "large_schematic.png"
    img = Image.new("RGB", (3000, 2000), color="green")
    img.save(img_path)
    return img_path


# =========================================================================
# 1. Candidate Selection (OR-based signals)
# =========================================================================

def test_candidate_selector_vector_signal(sample_image):
    fig = {"image_path": str(sample_image), "figure_type": "vector_drawing"}
    cand_result = CandidateSelector.is_candidate(fig)
    assert cand_result["needs_vlm"] is True
    assert "vector" in cand_result["reason"] or "diagram" in cand_result["reason"]


def test_candidate_selector_caption_keyword(sample_image):
    fig = {"image_path": str(sample_image), "caption": "Hydraulic circuit diagram", "has_arrows": True}
    cand_result = CandidateSelector.is_candidate(fig)
    assert cand_result["needs_vlm"] is True
    assert "caption" in cand_result["reason"]


def test_candidate_selector_pymupdf_filename(tmp_path):
    p = tmp_path / "page2_pymupdf.png"
    Image.new("RGB", (400, 400), color="red").save(p)
    fig = {"image_path": str(p), "has_arrows": True}
    cand_result = CandidateSelector.is_candidate(fig)
    assert cand_result["needs_vlm"] is True
    assert "pymupdf" in cand_result["reason"] or "vector" in cand_result["reason"]


def test_candidate_selector_has_arrows(sample_image):
    fig = {"image_path": str(sample_image), "has_arrows": True}
    cand_result = CandidateSelector.is_candidate(fig)
    assert cand_result["needs_vlm"] is True
    assert "arrow" in cand_result["reason"]


def test_candidate_selector_has_labels(sample_image):
    fig = {"image_path": str(sample_image), "has_labels": True}
    cand_result = CandidateSelector.is_candidate(fig)
    assert cand_result["needs_vlm"] is True
    assert "label" in cand_result["reason"]


def test_candidate_selector_large_image(sample_image):
    fig = {"image_path": str(sample_image), "has_arrows": True}
    cand_result = CandidateSelector.is_candidate(fig)
    assert cand_result["needs_vlm"] is True
    assert "large_image" in cand_result["reason"]


def test_candidate_selector_diagram_dominant_page(tmp_path):
    small_img = tmp_path / "small.png"
    Image.new("RGB", (150, 150), color="white").save(small_img)
    fig = {"image_path": str(small_img), "has_arrows": True}
    page_data = {
        "text_blocks": [{"text": "Short caption only"}],
        "figures": [fig],
    }
    cand_result = CandidateSelector.is_candidate(fig, page_data=page_data)
    assert cand_result["needs_vlm"] is True
    assert "diagram_dominant_page" in cand_result["reason"]


def test_candidate_selector_skips_tiny_image_without_signals(tmp_path):
    tiny_img = tmp_path / "icon.png"
    Image.new("RGB", (32, 32), color="gray").save(tiny_img)
    fig = {"image_path": str(tiny_img), "caption": "bullet point"}
    cand_result = CandidateSelector.is_candidate(fig)
    assert cand_result["needs_vlm"] is False
    assert cand_result["reason"] == "image_too_small"


def test_candidate_selector_missing_file():
    fig = {"image_path": "/nonexistent/path/image.png"}
    cand_result = CandidateSelector.is_candidate(fig)
    assert cand_result["needs_vlm"] is False
    assert cand_result["reason"] == "file_not_found"


# =========================================================================
# 2. Content-Based Persistent Cache
# =========================================================================

def test_cache_content_hashing(tmp_path):
    img1 = tmp_path / "copy1.png"
    img2 = tmp_path / "copy2.png"
    data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRtest_data"
    img1.write_bytes(data)
    img2.write_bytes(data)

    hash1 = VisionCache.compute_hash(img1)
    hash2 = VisionCache.compute_hash(img2)
    assert hash1 == hash2


def test_cache_store_and_retrieve(tmp_path):
    cache = VisionCache(cache_dir=tmp_path / "v_cache")
    test_hash = "abcdef1234567890"
    analysis = {
        "functional_summary": "Hydraulic system diagram",
        "components": ["Pump", "Valve"],
        "relationships": [{"from": "Pump", "to": "Valve", "type": "CONNECTS"}],
        "spatial_layout": "Top to bottom",
        "status": "ok",
    }
    cache.store(test_hash, analysis)
    assert cache.has(test_hash)

    cached = cache.get(test_hash)
    assert cached is not None
    assert cached["functional_summary"] == "Hydraulic system diagram"
    assert len(cached["components"]) == 2


def test_cache_does_not_store_failures(tmp_path):
    cache = VisionCache(cache_dir=tmp_path / "v_cache")
    test_hash = "failed_hash_1"
    cache.store(test_hash, {"status": "FAILED", "functional_summary": "Failed"})
    assert not cache.has(test_hash)

    cache.store("timeout_hash_1", {"status": "TIMEOUT", "functional_summary": "Timed out"})
    assert not cache.has("timeout_hash_1")


# =========================================================================
# 3. Image Preprocessor (Aspect Ratio & Immutability)
# =========================================================================

def test_image_preprocessor_noop_for_normal_images(sample_image):
    orig_path_str = str(sample_image)
    orig_mtime = sample_image.stat().st_mtime
    with preprocess_for_vision(orig_path_str, max_dimension=1920) as preprocessed:
        assert preprocessed == orig_path_str

    assert sample_image.stat().st_mtime == orig_mtime


def test_image_preprocessor_resizes_oversized_images(large_image):
    orig_path_str = str(large_image)
    orig_size = large_image.stat().st_size

    with preprocess_for_vision(orig_path_str, max_dimension=1000) as preprocessed:
        assert preprocessed != orig_path_str
        assert Path(preprocessed).exists()
        with Image.open(preprocessed) as img:
            w, h = img.size
            assert max(w, h) == 1000
            assert w == 1000  # Aspect ratio 3000x2000 -> 1000x666
            assert h == 666

    # Verify temp file is cleaned up
    assert not Path(preprocessed).exists()
    # Verify original file untouched
    assert large_image.stat().st_size == orig_size


# =========================================================================
# 4. Pipeline Step Integration: Concurrency, Caching & Timeouts
# =========================================================================

class MockVisionProvider(BaseVisionProvider):
    provider_name = "mock"
    model_name = "mock-gemini"

    def __init__(self, delay: float = 0.01, fail_image: str | None = None, timeout_image: str | None = None):
        self.delay = delay
        self.fail_image = fail_image
        self.timeout_image = timeout_image
        self.call_count = 0
        self.active_calls = 0
        self.max_observed_concurrency = 0

    def analyze_diagram(self, image_path: str):
        self.call_count += 1
        self.active_calls += 1
        self.max_observed_concurrency = max(self.max_observed_concurrency, self.active_calls)

        import time
        time.sleep(self.delay)
        self.active_calls -= 1

        if self.timeout_image and self.timeout_image in image_path:
            raise TimeoutError("Vision inference timed out")

        if self.fail_image and self.fail_image in image_path:
            raise ValueError("Malformed model response")

        return {
            "functional_summary": f"Analyzed diagram at {Path(image_path).name}",
            "components": ["Part A", "Part B"],
            "relationships": [{"from": "Part A", "to": "Part B", "type": "ATTACHED"}],
            "spatial_layout": "Left to right",
            "status": "ok",
        }


@pytest.mark.asyncio
async def test_s04_vision_step_concurrency_and_caching(tmp_path, monkeypatch):
    cache_dir = tmp_path / "v_cache"
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_max_concurrency", 2)
    monkeypatch.setattr(settings, "vision_cache_dir", str(cache_dir))
    monkeypatch.setattr(settings, "vision_request_timeout", 5.0)

    # Create 4 distinct test images
    figures = []
    for i in range(4):
        p = tmp_path / f"figure_{i}.png"
        Image.new("RGB", (300, 300), color=(i * 40, i * 40, i * 40)).save(p)
        figures.append({
            "image_path": str(p),
            "figure_type": "diagram",
            "caption": f"Figure {i} schematic",
        })

    mock_provider = MockVisionProvider(delay=0.05)
    services = {"vision": mock_provider}

    state = IngestionState(
        document_id=uuid4(),
        filename="doc.pdf",
        tenant_id="tenant-1",
        storage_path=str(tmp_path / "doc.pdf"),
        parsed_doc={"pages": [{"page_num": 1, "figures": figures}]},
    )

    # First run: uncached, bounded by concurrency=2
    updated_state = await s04_vision_step(state, services)
    out_figures = updated_state.parsed_doc["pages"][0]["figures"]

    assert mock_provider.call_count == 4
    assert mock_provider.max_observed_concurrency <= 2
    for fig in out_figures:
        assert fig["vision_status"] == "PRIMARY_SUCCESS"
        assert len(fig["vision_analysis"]["components"]) == 2
        # CRITICAL: original image_path must remain untouched
        assert "figure_" in fig["image_path"]

    # Second run: should all be cache hits, provider NOT called again!
    mock_provider_2 = MockVisionProvider(delay=0.01)
    services_2 = {"vision": mock_provider_2}

    cached_state = await s04_vision_step(updated_state, services_2)
    cached_figures = cached_state.parsed_doc["pages"][0]["figures"]

    assert mock_provider_2.call_count == 0  # Zero calls to vision model
    for fig in cached_figures:
        assert fig["vision_status"] == "CACHED"
        assert len(fig["vision_analysis"]["components"]) == 2


@pytest.mark.asyncio
async def test_s04_vision_step_timeout_not_retried(tmp_path, monkeypatch):
    """Verify Rule 2: Timeouts are isolated, marked 'timeout', and NEVER retried."""
    cache_dir = tmp_path / "v_cache"
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_max_concurrency", 2)
    monkeypatch.setattr(settings, "vision_cache_dir", str(cache_dir))
    monkeypatch.setattr(settings, "vision_request_timeout", 5.0)
    monkeypatch.setattr(settings, "vision_max_retries", 1)

    p_normal = tmp_path / "normal_fig.png"
    Image.new("RGB", (350, 350), color="blue").save(p_normal)

    p_timeout = tmp_path / "slow_fig.png"
    Image.new("RGB", (350, 350), color="red").save(p_timeout)

    figures = [
        {"image_path": str(p_normal), "figure_type": "diagram"},
        {"image_path": str(p_timeout), "figure_type": "diagram"},
    ]

    mock_provider = MockVisionProvider(delay=0.01, timeout_image="slow_fig")
    services = {"vision": mock_provider}

    state = IngestionState(
        document_id=uuid4(),
        filename="doc.pdf",
        tenant_id="tenant-1",
        storage_path=str(tmp_path / "doc.pdf"),
        parsed_doc={"pages": [{"page_num": 1, "figures": figures}]},
    )

    result_state = await s04_vision_step(state, services)
    res_figures = result_state.parsed_doc["pages"][0]["figures"]

    normal_res = next(f for f in res_figures if "normal_fig" in f["image_path"])
    timeout_res = next(f for f in res_figures if "slow_fig" in f["image_path"])

    assert normal_res["vision_status"] == "PRIMARY_SUCCESS"
    assert timeout_res["vision_status"] == "TIMEOUT"
    assert timeout_res["vision_analysis"]["status"] == "TIMEOUT"
    assert timeout_res["vision_analysis"]["components"] == []

    # Call count should be exactly 2 (1 for normal + 1 for timeout, NOT retried!)
    assert mock_provider.call_count == 2


@pytest.mark.asyncio
async def test_s04_vision_step_transient_retry(tmp_path, monkeypatch):
    """Verify that transient network errors are retried up to VISION_MAX_RETRIES."""
    import httpx
    cache_dir = tmp_path / "v_cache"
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_max_concurrency", 2)
    monkeypatch.setattr(settings, "vision_cache_dir", str(cache_dir))
    monkeypatch.setattr(settings, "vision_request_timeout", 5.0)
    monkeypatch.setattr(settings, "vision_max_retries", 1)

    p = tmp_path / "transient_fig.png"
    Image.new("RGB", (300, 300), color="yellow").save(p)
    figures = [{"image_path": str(p), "figure_type": "diagram"}]

    class FlakyVisionProvider(BaseVisionProvider):
        provider_name = "flaky"
        model_name = "mock"
        attempts = 0

        def analyze_diagram(self, image_path: str):
            self.attempts += 1
            if self.attempts == 1:
                # Transient network error on attempt 1
                raise httpx.ConnectError("Connection refused by Vision API")
            return {
                "functional_summary": "Recovered on attempt 2",
                "components": ["Part X"],
                "relationships": [],
                "spatial_layout": "Center",
                "status": "ok",
            }

    provider = FlakyVisionProvider()
    services = {"vision": provider}

    state = IngestionState(
        document_id=uuid4(),
        filename="doc.pdf",
        tenant_id="tenant-1",
        storage_path=str(tmp_path / "doc.pdf"),
        parsed_doc={"pages": [{"page_num": 1, "figures": figures}]},
    )

    result_state = await s04_vision_step(state, services)
    fig_res = result_state.parsed_doc["pages"][0]["figures"][0]

    assert provider.attempts == 2
    assert fig_res["vision_status"] == "PRIMARY_SUCCESS"
    assert fig_res["vision_analysis"]["functional_summary"] == "Recovered on attempt 2"


@pytest.mark.asyncio
async def test_s04_vision_step_skips_non_candidate(tmp_path, monkeypatch):
    """Verify that figures without diagram signals are skipped."""
    cache_dir = tmp_path / "v_cache"
    settings = get_settings()
    monkeypatch.setattr(settings, "vision_cache_dir", str(cache_dir))

    tiny = tmp_path / "tiny_bullet.png"
    Image.new("RGB", (20, 20), color="black").save(tiny)

    figures = [{"image_path": str(tiny), "caption": "decorative bullet"}]
    mock_provider = MockVisionProvider()
    services = {"vision": mock_provider}

    state = IngestionState(
        document_id=uuid4(),
        filename="doc.pdf",
        tenant_id="tenant-1",
        storage_path=str(tmp_path / "doc.pdf"),
        parsed_doc={"pages": [{"page_num": 1, "figures": figures}]},
    )

    result_state = await s04_vision_step(state, services)
    fig_res = result_state.parsed_doc["pages"][0]["figures"][0]

    assert mock_provider.call_count == 0
    assert fig_res["vision_status"] == "SKIPPED"
    assert "Skipped" in fig_res["vision_analysis"]["functional_summary"]


def test_downstream_chunking_contract(sample_image):
    """Verify that chunking creates a DIAGRAM chunk referencing the unchanged image_path."""
    from app.models.chunk import ChunkType
    from app.services.chunking.service import ChunkingService

    chunker = ChunkingService()
    doc_id = uuid4()
    figure = {
        "image_path": str(sample_image),
        "caption": "Hydraulic Schematic",
        "vision_analysis": {
            "functional_summary": "Hydraulic control circuit",
            "components": ["Pump", "Control Valve", "Cylinder"],
            "relationships": [{"from": "Pump", "to": "Control Valve", "type": "FEEDS"}],
            "spatial_layout": "Left to right flow",
            "status": "success",
        },
        "bbox": [10.0, 20.0, 400.0, 300.0],
    }

    chunk = chunker._make_figure_chunk(
        figure=figure,
        page_num=2,
        doc_str="doc1",
        document_id=doc_id,
        industry="construction",
        tenant_id="tenant-1",
        assistant_id="asst-1",
        knowledge_base_id="kb-1",
        filename="manual.pdf",
    )

    assert chunk is not None
    assert chunk.chunk_type == ChunkType.DIAGRAM
    assert chunk.image_path == str(sample_image)
    assert "Hydraulic control circuit" in chunk.content
    assert "Pump" in chunk.content

