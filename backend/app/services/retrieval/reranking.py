"""
Cross-Encoder Reranker Service.

Uses ``cross-encoder/ms-marco-MiniLM-L-6-v2`` for genuine query-document
semantic reranking when the model is available.

Falls back gracefully to score-sorted passthrough when:
- The model is not installed / unavailable.
- RERANKER_ENABLED=false is set.
- The reranker raises any exception at inference time.

Design rules
------------
- Model is loaded once as a module-level singleton (thread-safe).
- Caller receives plain SearchResult objects with updated .score values.
- Metadata (chunk_id, document_id, page_number, citations) is always preserved.
- The log clearly distinguishes semantic reranking from passthrough.
"""
import os
import threading
from typing import Any

from app.config.logging import get_logger
from app.models.retrieval import SearchResult

logger = get_logger(__name__)

# ── Cross-encoder singleton ───────────────────────────────────────────────────
_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker_lock = threading.Lock()
_reranker_instance: Any | None = None
_reranker_unavailable: bool = False   # set True once we confirm model missing


def _get_reranker() -> Any | None:
    """
    Load ``cross-encoder/ms-marco-MiniLM-L-6-v2`` on first call.

    Returns None and sets _reranker_unavailable=True if the model or the
    ``sentence-transformers`` package is not installed.
    """
    global _reranker_instance, _reranker_unavailable  # noqa: PLW0603
    if _reranker_unavailable:
        return None
    if _reranker_instance is not None:
        return _reranker_instance

    with _reranker_lock:
        if _reranker_instance is not None:
            return _reranker_instance
        if _reranker_unavailable:
            return None

        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            logger.info(
                "reranker.loading",
                model=_RERANKER_MODEL_NAME,
            )
            _reranker_instance = CrossEncoder(_RERANKER_MODEL_NAME, max_length=512)
            logger.info(
                "reranker.ready",
                model=_RERANKER_MODEL_NAME,
            )
        except ImportError:
            logger.warning(
                "reranker.unavailable",
                reason="sentence-transformers not installed",
                hint="Run: pip install sentence-transformers",
                fallback="score-sorted passthrough",
            )
            _reranker_unavailable = True
        except Exception as exc:
            logger.warning(
                "reranker.load_failed",
                model=_RERANKER_MODEL_NAME,
                error=str(exc),
                fallback="score-sorted passthrough",
            )
            _reranker_unavailable = True

    return _reranker_instance


def _is_reranker_enabled() -> bool:
    """Check RERANKER_ENABLED env-var (default: True)."""
    val = os.environ.get("RERANKER_ENABLED", "true").lower().strip()
    return val not in {"false", "0", "no", "off"}


class RRFRerankerService:
    """
    Query-document semantic reranker backed by cross-encoder/ms-marco-MiniLM-L-6-v2.

    When the cross-encoder is available and RERANKER_ENABLED=true, candidates
    are scored by genuine query-document relevance and re-ranked accordingly.

    When the cross-encoder is unavailable or disabled, candidates are returned
    in their current score order (passthrough) and the log clearly states this.
    """

    def __init__(self) -> None:
        pass

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_n: int = 10,
    ) -> list[SearchResult]:
        """
        Rerank candidates using the cross-encoder.

        Args:
            query:      The original user query.
            candidates: RRF-fused candidates from the retrieval agent.
            top_n:      Maximum number of results to return.

        Returns:
            Up to top_n candidates ordered by cross-encoder relevance score.
            If the cross-encoder is unavailable, returns the top_n candidates
            ordered by their existing fusion score.
        """
        if not candidates:
            return []

        top_n = max(1, top_n)
        sorted_by_fusion = sorted(candidates, key=lambda x: x.score, reverse=True)

        if not _is_reranker_enabled():
            logger.info(
                "rerank.disabled",
                mode="passthrough",
                input_count=len(candidates),
                output_count=min(top_n, len(candidates)),
            )
            return sorted_by_fusion[:top_n]

        reranker = _get_reranker()

        if reranker is None:
            logger.info(
                "rerank.passthrough",
                reason="cross-encoder unavailable",
                input_count=len(candidates),
                output_count=min(top_n, len(candidates)),
            )
            return sorted_by_fusion[:top_n]

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, c.content or "") for c in sorted_by_fusion]

        try:
            scores = reranker.predict(pairs)
        except Exception as exc:
            logger.warning(
                "rerank.inference_failed",
                model=_RERANKER_MODEL_NAME,
                error=str(exc),
                fallback="score-sorted passthrough",
            )
            return sorted_by_fusion[:top_n]

        # Attach cross-encoder scores and re-sort
        reranked = [
            result.model_copy(update={"score": float(score)})
            for result, score in zip(sorted_by_fusion, scores)
        ]
        reranked.sort(key=lambda x: x.score, reverse=True)
        final = reranked[:top_n]

        logger.info(
            "rerank.complete",
            mode="cross-encoder",
            model=_RERANKER_MODEL_NAME,
            input_count=len(candidates),
            output_count=len(final),
            top_score=round(float(final[0].score), 4) if final else None,
        )
        return final
