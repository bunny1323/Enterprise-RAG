# Multimodal Query Images — Task List

- [x] Fix 1: Parser — save figures to PROCESSED_STORAGE_PATH (not tempdir)
- [x] Fix 2: Chunk model — add `image_path` field
- [x] Fix 3: Chunking service — populate `chunk.image_path`
- [x] Fix 4: Weaviate schema, upsert, and search return props
- [x] Fix 5: SearchResult model — add `image_path` field
- [x] Fix 6: Dense search map `image_path`
- [x] Fix 7: BM25 search map `image_path`
- [x] Fix 8: Query models — add `ImageEvidence` typed model
- [x] Fix 9: Query route — build real image URLs with security validation, deduplication, and `ImageEvidence`
- [x] Fix 10: Vision Provider Abstraction (`DisabledVisionProvider`, `OllamaVisionProvider`, factory)
- [x] Fix 11: Settings — add `vision_provider: str = "disabled"`
- [x] Fix 12: Main & Pipeline wiring — wire vision provider into `main.py` and `s04_vision.py`
- [x] Fix 13: Health route — make Ollama/vision health optional so overall health stays 200
- [x] Verification — Unit tests created in `tests/unit/test_multimodal_images.py` and static code inspection complete
