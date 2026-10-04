"""
Step 04 — Scalable Diagram Vision Analysis.

Analyzes technical diagrams and schematics with Qwen2.5-VL with:
1. OR-based candidate selection (only diagrams, vector drawings, meaningful images)
2. Content-based persistent caching (SHA256 of raw image bytes)
3. Bounded concurrency (VISION_MAX_CONCURRENCY, default 2)
4. Strict per-figure timeouts without retrying timeouts
5. Transient error retry (VISION_MAX_RETRIES, default 1)
6. Status classification: success | cached | skipped | timeout | failed
7. Preservation of original image_path (temporary preprocessing only)
"""
import asyncio
from collections import defaultdict
from typing import Any

import httpx

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.config.settings import get_settings
from app.services.vision.candidate_selector import CandidateSelector
from app.services.vision.image_preprocessor import preprocess_for_vision
from app.services.vision.vision_cache import VisionCache
from app.services.vision.vision_provider import BaseVisionProvider

logger = get_logger(__name__)


def _is_transient_error(exc: Exception) -> bool:
    """Identify transient network/transport errors that warrant a retry."""
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            ConnectionResetError,
            ConnectionRefusedError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {502, 503, 504}
    return False


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Analyze all eligible figures/diagrams found during parsing.

    Args:
        state: Current ingestion state (parsed_doc must be set).
        services: Must contain 'vision' key -> BaseVisionProvider.

    Returns:
        Updated state with vision_analysis and vision_status embedded in parsed_doc figures.
    """
    logger.info("step.vision.start", document_id=str(state.document_id))

    if state.parsed_doc is None:
        logger.warning("step.vision.skipped", reason="parsed_doc is None")
        return state

    settings = get_settings()
    max_concurrency: int = getattr(settings, "vision_max_concurrency", 2)
    request_timeout: float = getattr(settings, "vision_request_timeout", 420.0)
    max_retries: int = getattr(settings, "vision_max_retries", 1)
    max_dim: int = getattr(settings, "vision_max_image_dimension", 1920)
    cache_dir: str = getattr(settings, "vision_cache_dir", "./data/processed/vision_cache")

    vision: BaseVisionProvider = services["vision"]
    provider_name = getattr(vision, "provider_name", vision.__class__.__name__.lower())
    model_name = getattr(vision, "model_name", "n/a")
    fallback_model = getattr(vision, "fallback_model", None)
    
    valid_models = [model_name]
    if fallback_model:
        valid_models.append(fallback_model)
        
    loop = asyncio.get_running_loop()

    cache = VisionCache(cache_dir=cache_dir)
    pages = state.parsed_doc.get("pages", [])

    total_figures = 0
    analyzed_figures = 0
    cached_figures = 0
    skipped_figures = 0
    timeout_figures = 0
    failed_figures = 0

    # Figures requiring model execution: list of (figure_dict, sha256_hash)
    candidates_to_process: list[tuple[dict[str, Any], str]] = []

    # 1. Pass: Filter figures, check disabled provider, evaluate candidates & pre-check cache
    for page_data in pages:
        for figure in page_data.get("figures", []):
            image_path: str = figure.get("image_path", "")
            total_figures += 1

            if not image_path:
                figure["vision_status"] = "SKIPPED"
                skipped_figures += 1
                logger.info(
                    "step.vision.figure_skip",
                    provider=provider_name,
                    model=model_name,
                    page=page_data.get("page_num"),
                    reason="no image_path",
                )
                continue

            if provider_name == "disabled":
                skipped_figures += 1
                figure["vision_status"] = "SKIPPED"
                figure["vision_analysis"] = {
                    "functional_summary": "Vision analysis disabled",
                    "components": [],
                    "relationships": [],
                    "spatial_layout": "Unknown",
                    "status": "SKIPPED",
                }
                logger.info(
                    "step.vision.figure_skipped",
                    provider=provider_name,
                    model=model_name,
                    image=image_path,
                    reason="VISION_PROVIDER=disabled",
                )
                continue

            # Candidate Selection (Deterministic Scoring)
            cand_result = CandidateSelector.is_candidate(figure, page_data)
            is_cand = cand_result["needs_vlm"]
            reason = cand_result["reason"]
            classification = cand_result["classification"]
            
            # Store classification in figure metadata regardless of VLM usage
            figure["classification"] = classification
            figure["vision_score"] = cand_result["score"]
            
            if not is_cand:
                skipped_figures += 1
                figure["vision_status"] = "SKIPPED"
                figure["vision_analysis"] = {
                    "functional_summary": f"Skipped: {reason} ({classification})",
                    "components": [],
                    "relationships": [],
                    "spatial_layout": "Unknown",
                    "status": "SKIPPED",
                }
                logger.info(
                    "step.vision.figure_skipped",
                    image=image_path,
                    reason=reason,
                    classification=classification
                )
                continue

            # Compute content-based hash
            try:
                content_hash = VisionCache.compute_hash(
                    image_path, 
                    model_version=model_name, 
                    prompt_version="v2", 
                    resize_dim=max_dim
                )
            except Exception as err:
                failed_figures += 1
                figure["vision_status"] = "FAILED"
                figure["vision_analysis"] = {
                    "functional_summary": "Failed to read image for hashing",
                    "components": [],
                    "relationships": [],
                    "spatial_layout": "Unknown",
                    "status": "FAILED",
                    "error": str(err),
                }
                logger.error("step.vision.hash_failed", image=image_path, error=str(err))
                continue

            # Content cache check
            cached_result = cache.get(content_hash, valid_models=valid_models if provider_name != "disabled" else None)
            if cached_result is not None:
                cached_figures += 1
                figure["vision_status"] = "CACHED"
                analysis_data = dict(cached_result)
                analysis_data["status"] = "CACHED"
                figure["vision_analysis"] = analysis_data
                logger.info(
                    "step.vision.cache_hit",
                    image=image_path,
                    hash=content_hash,
                    components=len(analysis_data.get("components", [])),
                )
                continue

            # Figure is a candidate and not yet cached
            candidates_to_process.append((figure, content_hash))

    # 2. Concurrency-limited processing for uncached candidates
    if candidates_to_process:
        semaphore = asyncio.Semaphore(max_concurrency)
        hash_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        async def process_candidate(fig: dict[str, Any], c_hash: str) -> None:
            nonlocal analyzed_figures, cached_figures, timeout_figures, failed_figures

            orig_image_path = fig["image_path"]

            # Deduplication lock per image content hash
            async with hash_locks[c_hash]:
                # Check cache again inside lock
                cached = cache.get(c_hash, valid_models=valid_models if provider_name != "disabled" else None)
                if cached is not None:
                    cached_figures += 1
                    fig["vision_status"] = "CACHED"
                    analysis_data = dict(cached)
                    analysis_data["status"] = "CACHED"
                    fig["vision_analysis"] = analysis_data
                    return

                # Acquire concurrency semaphore
                async with semaphore:
                    # Final check before calling Qwen
                    cached = cache.get(c_hash, valid_models=valid_models if provider_name != "disabled" else None)
                    if cached is not None:
                        cached_figures += 1
                        fig["vision_status"] = "CACHED"
                        analysis_data = dict(cached)
                        analysis_data["status"] = "CACHED"
                        fig["vision_analysis"] = analysis_data
                        return

                    # Execute vision inference with retry for transient errors, but NEVER retry timeout
                    for attempt in range(max_retries + 1):
                        try:
                            with preprocess_for_vision(orig_image_path, max_dimension=max_dim) as preprocessed_path:
                                logger.info(
                                    "step.vision.analyze_start",
                                    image=orig_image_path,
                                    attempt=attempt + 1,
                                    concurrency_limit=max_concurrency,
                                )
                                analysis = await asyncio.wait_for(
                                    loop.run_in_executor(
                                        None,
                                        vision.analyze_diagram,
                                        preprocessed_path,
                                    ),
                                    # Timeout is handled internally by httpx requests per model (primary + fallback)
                                    # We give a generous upper bound for safety.
                                    timeout=request_timeout + getattr(settings, "vision_fallback_timeout", 420.0) + 60.0,
                                )

                            # Success or Fallback Success
                            model_used = analysis.get("model_used", model_name)
                            if analysis.get("fallback_triggered"):
                                fig["vision_status"] = "FALLBACK_SUCCESS"
                                analysis.setdefault("status", "FALLBACK_SUCCESS")
                                logger.info(
                                    "step.vision.fallback_success",
                                    image=orig_image_path,
                                    model=model_used
                                )
                            else:
                                fig["vision_status"] = "PRIMARY_SUCCESS"
                                analysis.setdefault("status", "PRIMARY_SUCCESS")
                                
                            analysis["model_used"] = model_used
                            
                            fig["vision_analysis"] = analysis
                            cache.store(c_hash, analysis)
                            analyzed_figures += 1

                            logger.info(
                                "step.vision.figure_analyzed",
                                provider=provider_name,
                                model=model_name,
                                image=orig_image_path,
                                components=len(analysis.get("components", [])),
                            )
                            return

                        except (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException) as exc:
                            # CRITICAL: Do NOT retry timeouts
                            timeout_figures += 1
                            fig["vision_status"] = "TIMEOUT"
                            fig["vision_analysis"] = {
                                "functional_summary": "Vision analysis timed out",
                                "components": [],
                                "relationships": [],
                                "spatial_layout": "Unknown",
                                "status": "TIMEOUT",
                            }
                            logger.warning(
                                "step.vision.figure_timeout",
                                image=orig_image_path,
                                timeout=request_timeout,
                                error=str(exc),
                            )
                            return

                        except Exception as exc:
                            is_transient = _is_transient_error(exc)
                            if is_transient and attempt < max_retries:
                                logger.warning(
                                    "step.vision.transient_retry",
                                    image=orig_image_path,
                                    attempt=attempt + 1,
                                    max_retries=max_retries,
                                    error=str(exc),
                                )
                                await asyncio.sleep(1.0)
                                continue

                            # Non-transient or retries exhausted
                            failed_figures += 1
                            fig["vision_status"] = "FAILED"
                            fig["vision_analysis"] = {
                                "functional_summary": "Vision analysis failed",
                                "components": [],
                                "relationships": [],
                                "spatial_layout": "Unknown",
                                "status": "FAILED",
                                "error": str(exc),
                            }
                            logger.error(
                                "step.vision.figure_failed",
                                image=orig_image_path,
                                attempt=attempt + 1,
                                error=str(exc),
                            )
                            return

        await asyncio.gather(*(process_candidate(f, h) for f, h in candidates_to_process))

    logger.info(
        "step.vision.complete",
        document_id=str(state.document_id),
        total=total_figures,
        analyzed=analyzed_figures,
        cached=cached_figures,
        skipped=skipped_figures,
        timeout=timeout_figures,
        failed=failed_figures,
    )

    return state.model_copy(update={"parsed_doc": state.parsed_doc})
