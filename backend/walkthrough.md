# Scalable Multimodal Vision Pipeline — Walkthrough

## Summary of Changes

We implemented the scalable vision pipeline strategy for Enterprise-RAG according to the exact architectural specifications and adjustments:

1. **Preserved Existing Architecture & Contracts**:
   - Original flow intact: `PDF` -> `Parser` -> `figures[]` -> `s04_vision.py` -> `figure["vision_analysis"]` -> `Chunking` -> `DIAGRAM chunks` -> `Weaviate` -> `Retrieval` -> `ImageEvidence` -> `RAG answer`.
   - `figure["image_path"]` remains unmodified. Preprocessed resized images are strictly temporary and deleted after Qwen inference.

2. **OR-Based Candidate Selection** ([`candidate_selector.py`](file:///c:/Users/vigne/Enterprise-RAG/backend/app/services/vision/candidate_selector.py)):
   - Cheap, deterministic pre-filtering without running heavy ML models.
   - Evaluates any of 5 strong signals (OR logic):
     - Vector / drawing / diagram signal (`figure_type`, caption keywords, `pymupdf` vector renders, `is_vector` flag)
     - Has arrows
     - Has labels / callouts / balloons
     - Large meaningful image resolution (>= 300x300 px or large bbox)
     - Diagram-dominant page (sparse text <= 300 chars)

3. **Content-Based Persistent Cache** ([`vision_cache.py`](file:///c:/Users/vigne/Enterprise-RAG/backend/app/services/vision/vision_cache.py)):
   - `image bytes` -> `SHA256` -> `data/processed/vision_cache/<sha256>.json`.
   - Filename and path agnostic.
   - Only caches successful analyses; failures and timeouts are never cached.

4. **Temporary Image Preprocessing** ([`image_preprocessor.py`](file:///c:/Users/vigne/Enterprise-RAG/backend/app/services/vision/image_preprocessor.py)):
   - Bounds oversized images to `VISION_MAX_IMAGE_DIMENSION` (default `1920` px) while preserving aspect ratio.
   - Uses temporary file context manager; cleans up on completion.

5. **Bounded Concurrency & Strict Timeout Policy** ([`s04_vision.py`](file:///c:/Users/vigne/Enterprise-RAG/backend/app/pipelines/ingestion/steps/s04_vision.py)):
   - Concurrency initialized to `VISION_MAX_CONCURRENCY=2`.
   - Per-figure timeout: `VISION_REQUEST_TIMEOUT=420.0` s.
   - **Timeout isolation**: Timeouts are marked `timeout` and **never retried**.
   - Transient failures (connection resets, 502/503/504) are retried once (`VISION_MAX_RETRIES=1`).
   - Per-figure status tracking across 5 distinct states: `success`, `cached`, `skipped`, `timeout`, `failed`.
   - Removed tenacity `@retry` from [`service.py`](file:///c:/Users/vigne/Enterprise-RAG/backend/app/services/vision/service.py) so timeouts and retries are cleanly managed by `s04_vision.py`.

6. **Settings & Configuration** ([`settings.py`](file:///c:/Users/vigne/Enterprise-RAG/backend/app/config/settings.py)):
   - Added `vision_max_concurrency` (default 2)
   - Added `vision_request_timeout` (default 420.0s)
   - Added `vision_max_retries` (default 1)
   - Added `vision_max_image_dimension` (default 1920)
   - Added `vision_cache_dir` (default `./data/processed/vision_cache`)
   - Added `ingestion_timeout_vision` (default 1800s)

7. **Comprehensive Test Suite** ([`test_vision_pipeline.py`](file:///c:/Users/vigne/Enterprise-RAG/backend/tests/unit/test_vision_pipeline.py)):
   - Unit tests covering all candidate selection signals (OR logic), cache storage/retrieval, image preprocessor bounds, semaphore concurrency limit, timeout isolation without retry, transient retry recovery, and downstream DIAGRAM chunk creation.

---

## Verification Commands for User Terminal

Run in `backend/`:

```powershell
# 1. Run the new vision pipeline test suite
.\.venv\Scripts\python.exe -m pytest tests/unit/test_vision_pipeline.py -v

# 2. Run multimodal image contract tests
.\.venv\Scripts\python.exe -m pytest tests/unit/test_multimodal_images.py -v

# 3. Ingest Hyundai manual & run benchmark
python test_ingest.py
```
