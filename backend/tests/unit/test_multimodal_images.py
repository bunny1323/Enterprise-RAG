"""
Unit tests for multimodal query images and vision provider features.
"""
from pathlib import Path
import tempfile
import pytest

from app.api.routes.query import _image_path_to_url
from app.models.chunk import Chunk
from app.models.query import ImageEvidence, QueryResponse
from app.models.retrieval import SearchResult
from app.services.vision.vision_provider import (
    BaseVisionProvider,
    DisabledVisionProvider,
    OllamaVisionProvider,
    build_vision_provider,
)


def test_chunk_model_has_image_path():
    chunk = Chunk(
        chunk_id="chk-1",
        document_id="doc-1",
        tenant_id="tenant-1",
        content="Test content",
        chunk_type="IMAGE",
        image_path="/data/processed/figures/doc1/page1_123.png",
    )
    assert chunk.image_path == "/data/processed/figures/doc1/page1_123.png"


def test_search_result_has_image_path():
    res = SearchResult(
        chunk_id="chk-1",
        content="Figure caption",
        page_number=3,
        chunk_type="IMAGE",
        image_path="/data/processed/figures/doc1/page3_456.png",
    )
    assert res.image_path == "/data/processed/figures/doc1/page3_456.png"


def test_image_evidence_model():
    img = ImageEvidence(
        url="/api/v1/images/figures/manual/page2_101.png",
        page=2,
        source="manual.pdf",
        caption="Hydraulic circuit diagram",
        chunk_id="chk-img-1",
    )
    assert img.url == "/api/v1/images/figures/manual/page2_101.png"
    assert img.page == 2

    # Verify QueryResponse accepts ImageEvidence objects
    resp = QueryResponse(
        answer="The diagram shows the hydraulic pump setup.",
        images=[img],
    )
    assert len(resp.images) == 1
    assert resp.images[0].url == img.url


def test_image_path_to_url_valid(tmp_path):
    # Setup temporary processed storage structure
    processed_dir = tmp_path / "processed"
    figures_dir = processed_dir / "figures" / "sample_doc"
    figures_dir.mkdir(parents=True)
    test_img = figures_dir / "page1_img.png"
    test_img.write_bytes(b"\x89PNG\r\n\x1a\nfake_image_data")

    url = _image_path_to_url(str(test_img), str(processed_dir))
    assert url == "/api/v1/images/figures/sample_doc/page1_img.png"


def test_image_path_to_url_security_traversal(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Secret file outside processed_dir
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret_img = outside_dir / "secret.png"
    secret_img.write_bytes(b"secret")

    # Directory traversal attempt
    url = _image_path_to_url(str(secret_img), str(processed_dir))
    assert url is None


def test_image_path_to_url_nonexistent(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    nonexistent = processed_dir / "figures" / "ghost.png"
    assert _image_path_to_url(str(nonexistent), str(processed_dir)) is None


def test_image_path_to_url_disallowed_extension(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    txt_file = processed_dir / "test.txt"
    txt_file.write_text("not an image")
    assert _image_path_to_url(str(txt_file), str(processed_dir)) is None


def test_disabled_vision_provider():
    provider = DisabledVisionProvider()
    res = provider.analyze_diagram("/any/path.png")
    assert isinstance(res, dict)
    assert res["functional_summary"] == "Vision analysis disabled"
    assert res["components"] == []
    provider.close()


def test_build_vision_provider_factory():
    class MockSettings:
        vision_provider = "disabled"
        ollama_base_url = "http://localhost:11434"
        ollama_vision_model = "qwen2.5vl:3b"

    settings = MockSettings()
    provider = build_vision_provider(settings)
    assert isinstance(provider, DisabledVisionProvider)

    settings.vision_provider = "ollama"
    provider_ollama = build_vision_provider(settings)
    assert isinstance(provider_ollama, OllamaVisionProvider)
