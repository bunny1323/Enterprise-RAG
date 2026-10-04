"""
Benchmark script for Multimodal Vision Pipeline.

Benchmarks 4-5 figures across three stages:
1. Concurrency = 1 (Baseline Time)
2. Concurrency = 2 (Optimized Time)
3. Second Run (Cache-Hit Time)
"""
import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from app.agents.supervisor.state import IngestionState
from app.config.settings import get_settings
from app.pipelines.ingestion.steps.s04_vision import step as s04_vision_step
from app.services.vision.vision_provider import build_vision_provider


def get_sample_figures(num_figures: int = 4) -> list[dict]:
    """Find 4-5 real figures from data/processed/figures/."""
    figures_base = Path("data/processed/figures")
    candidates = []

    if figures_base.exists():
        for doc_dir in figures_base.iterdir():
            if doc_dir.is_dir():
                for img_path in sorted(doc_dir.glob("*.png")):
                    candidates.append({
                        "image_path": str(img_path.resolve()),
                        "caption": "Rendered vector content",
                        "figure_type": "diagram",
                    })
                    if len(candidates) >= num_figures:
                        return candidates

    if len(candidates) < num_figures:
        # Fallback: create mock diagrams if directory not found
        from PIL import Image
        temp_dir = Path(tempfile.gettempdir()) / "benchmark_figures"
        temp_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_figures):
            p = temp_dir / f"benchmark_fig_{i}.png"
            Image.new("RGB", (400, 400), color=(50 * i, 80, 120)).save(p)
            candidates.append({
                "image_path": str(p),
                "caption": "Technical diagram schematic",
                "figure_type": "diagram",
            })

    return candidates[:num_figures]


async def run_benchmark():
    settings = get_settings()
    figures = get_sample_figures(num_figures=4)
    print("=" * 70)
    print(f"BENCHMARKING VISION PIPELINE WITH {len(figures)} FIGURES")
    print("=" * 70)
    for i, fig in enumerate(figures):
        print(f"  [{i+1}] {fig['image_path']}")

    # Setup vision provider
    vision_provider = build_vision_provider(settings)
    print(f"\nVision Provider: {vision_provider.provider_name} (Model: {vision_provider.model_name})")
    services = {"vision": vision_provider}

    benchmark_cache_dir = Path("data/processed/vision_cache_benchmark")
    if benchmark_cache_dir.exists():
        shutil.rmtree(benchmark_cache_dir)
    benchmark_cache_dir.mkdir(parents=True, exist_ok=True)

    settings.vision_cache_dir = str(benchmark_cache_dir)

    try:
        # ─────────────────────────────────────────────────────────────
        # RUN 1: Baseline (Concurrency = 1, Uncached)
        # ─────────────────────────────────────────────────────────────
        print("\n" + "-" * 70)
        print("RUN 1: Baseline (Concurrency = 1, Cold Cache)")
        print("-" * 70)
        settings.vision_max_concurrency = 1

        run1_figures = [dict(f) for f in figures]
        state1 = IngestionState(
            document_id=uuid4(),
            filename="benchmark.pdf",
            tenant_id="benchmark",
            storage_path="benchmark.pdf",
            parsed_doc={"pages": [{"page_num": 1, "figures": run1_figures}]},
        )

        t0 = time.perf_counter()
        res1 = await s04_vision_step(state1, services)
        run1_time = time.perf_counter() - t0
        print(f"\n>>> Run 1 Completed in {run1_time:.2f} seconds")

        # ─────────────────────────────────────────────────────────────
        # RUN 2: Optimized (Concurrency = 2, Cold Cache)
        # ─────────────────────────────────────────────────────────────
        print("\n" + "-" * 70)
        print("RUN 2: Optimized (Concurrency = 2, Cold Cache)")
        print("-" * 70)
        # Clear cache to measure uncached concurrency speedup
        shutil.rmtree(benchmark_cache_dir)
        benchmark_cache_dir.mkdir(parents=True, exist_ok=True)

        settings.vision_max_concurrency = 2

        run2_figures = [dict(f) for f in figures]
        state2 = IngestionState(
            document_id=uuid4(),
            filename="benchmark.pdf",
            tenant_id="benchmark",
            storage_path="benchmark.pdf",
            parsed_doc={"pages": [{"page_num": 1, "figures": run2_figures}]},
        )

        t0 = time.perf_counter()
        res2 = await s04_vision_step(state2, services)
        run2_time = time.perf_counter() - t0
        print(f"\n>>> Run 2 Completed in {run2_time:.2f} seconds")

        # ─────────────────────────────────────────────────────────────
        # RUN 3: Cache-Hit (Concurrency = 2, Warm Cache)
        # ─────────────────────────────────────────────────────────────
        print("\n" + "-" * 70)
        print("RUN 3: Second Run (Cache-Hit Time)")
        print("-" * 70)
        # Keep populated cache

        run3_figures = [dict(f) for f in figures]
        state3 = IngestionState(
            document_id=uuid4(),
            filename="benchmark.pdf",
            tenant_id="benchmark",
            storage_path="benchmark.pdf",
            parsed_doc={"pages": [{"page_num": 1, "figures": run3_figures}]},
        )

        t0 = time.perf_counter()
        res3 = await s04_vision_step(state3, services)
        run3_time = time.perf_counter() - t0
        print(f"\n>>> Run 3 Completed in {run3_time:.4f} seconds")

        # ─────────────────────────────────────────────────────────────
        # SUMMARY REPORT
        # ─────────────────────────────────────────────────────────────
        speedup_concurrency = (run1_time / run2_time) if run2_time > 0 else 1.0
        speedup_cache = (run1_time / run3_time) if run3_time > 0 else 1.0

        print("\n" + "=" * 70)
        print("VISION PIPELINE BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Run / Configuration':<35} | {'Wall Time':<15} | {'Speedup':<15}")
        print("-" * 70)
        print(f"{'Concurrency = 1 (Baseline)':<35} | {f'{run1_time:.2f}s':<15} | {'1.00x':<15}")
        print(f"{'Concurrency = 2 (Optimized)':<35} | {f'{run2_time:.2f}s':<15} | {f'{speedup_concurrency:.2f}x':<15}")
        print(f"{'Second Run (Cache Hit)':<35} | {f'{run3_time:.4f}s':<15} | {f'{speedup_cache:.1f}x':<15}")
        print("=" * 70)

    finally:
        if benchmark_cache_dir.exists():
            shutil.rmtree(benchmark_cache_dir)
        vision_provider.close()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
