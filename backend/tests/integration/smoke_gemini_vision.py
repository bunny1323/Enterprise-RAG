"""
Gemini Vision Smoke Test — requires a real GEMINI_API_KEY.

Usage:
    GEMINI_API_KEY=<your-key> python tests/integration/smoke_gemini_vision.py [image_path]

If no image_path is given, uses the first .png found in data/processed/figures/.
This test is NOT run automatically by pytest.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)


def _find_test_image() -> Path:
    """Find a real diagram image from the processed figures directory."""
    candidates = sorted(Path("data/processed/figures").rglob("*.png"))
    if not candidates:
        raise FileNotFoundError(
            "No processed figures found. "
            "Ingest a document first, or pass an image path as CLI argument."
        )
    return candidates[0]


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable is not set.")
        print("  Run:  GEMINI_API_KEY=<your-key> python tests/integration/smoke_gemini_vision.py")
        sys.exit(1)

    model = os.environ.get("GEMINI_VISION_MODEL", "gemini-3.8-flash")

    # Resolve image path
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_path = _find_test_image()

    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Enterprise-RAG — Gemini Vision Smoke Test")
    print(f"{'='*60}")
    print(f"  Image  : {image_path}")
    print(f"  Model  : {model}")
    print(f"  Size   : {image_path.stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}\n")

    from app.services.vision.gemini_provider import GeminiVisionProvider

    provider = GeminiVisionProvider(api_key=api_key, model=model, timeout=120.0)

    t0 = time.perf_counter()
    try:
        result = provider.analyze_diagram(str(image_path))
    except Exception as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}")
        sys.exit(1)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    print(f"  ✅ Success")
    print(f"  Provider      : {provider.provider_name}")
    print(f"  Model used    : {result.get('model_used', model)}")
    print(f"  Prompt ver.   : {result.get('prompt_version', 'n/a')}")
    print(f"  Latency       : {latency_ms} ms")
    print(f"  Components    : {len(result.get('components', []))}")
    print(f"  Relationships : {len(result.get('relationships', []))}")
    print(f"  Labels        : {len(result.get('important_labels', []))}")
    print(f"  Numbers       : {len(result.get('important_numbers', []))}")
    print(f"\n  Summary: {result.get('functional_summary', '')[:200]}")
    print(f"\n  Components:")
    for c in result.get("components", [])[:10]:
        print(f"    - {c}")
    print(f"\n  Relationships:")
    for r in result.get("relationships", [])[:5]:
        print(f"    {r.get('from')} --[{r.get('type')}]--> {r.get('to')}")

    # Validate required keys
    required = {"functional_summary", "components", "relationships", "spatial_layout"}
    missing = required - set(result.keys())
    if missing:
        print(f"\n[WARNING] Missing required keys: {missing}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Smoke test PASSED in {latency_ms} ms")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
