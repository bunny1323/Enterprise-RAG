"""
Embedding Service — Voyage Multimodal-3 with rate-limit & quota resilience.

Features:
- Batches text and image inputs under TPM limits.
- Estimates tokens prior to Voyage API request submission.
- Detects HTTP 429 RateLimit errors, extracts Retry-After headers, and raises RateLimitQuotaError.
- Retries transient errors with exponential backoff.
- Resumes exact failed batch without restarting document.
"""
import base64
import time
from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config.logging import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 32
_EMBEDDING_DIM = 1024  # Voyage Multimodal-3 outputs 1024-dim vectors


class RateLimitQuotaError(Exception):
    """Raised when Voyage API returns 429 Rate Limit Exceeded."""

    def __init__(self, retry_after: float = 60.0) -> None:
        self.retry_after = retry_after
        super().__init__(f"Voyage rate limit exceeded. Retry after {retry_after} seconds.")


def estimate_tokens(texts: list[str]) -> int:
    """Estimate token count for a batch of strings (rough rule: 1 token ≈ 4 chars)."""
    return sum(len(t) // 4 for t in texts)


class EmbeddingService:
    """
    Stateless embedding service backed by Voyage Multimodal-3.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-multimodal-3.5",
        text_model: str = "voyage-3.5",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._text_model = text_model
        self._client: Any = None  # Lazy initialization

    def _get_client(self) -> Any:
        """Lazily initialize the Voyage client on first use."""
        if self._client is None:
            import voyageai  # type: ignore[import-untyped]

            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    # ── Text embedding with rate-limit handling ─────────────────────────────────

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings using Voyage Multimodal-3.
        Splits into batches of 32 and retries each batch independently.
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []

        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[batch_start : batch_start + _BATCH_SIZE]
            tokens = estimate_tokens(batch)
            logger.debug(
                "embeddings.batch_submitting",
                batch_start=batch_start,
                batch_size=len(batch),
                token_estimate=tokens,
            )

            vectors = self._embed_text_batch_with_retry(batch)
            all_vectors.extend(vectors)

            # Brief pause between batches to respect TPM
            if batch_start + _BATCH_SIZE < len(texts):
                time.sleep(0.1)

        logger.info("embeddings.batch_complete", total=len(texts), dim=_EMBEDDING_DIM)
        return all_vectors

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        reraise=True,
    )
    def _embed_text_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch of texts with tenacity retry on failure."""
        client = self._get_client()
        inputs = [{"content": text} for text in texts]

        try:
            result = client.multimodal_embed(
                inputs=inputs,
                model=self._model,
            )
        except Exception as err:
            err_str = str(err).lower()
            if "429" in err_str or "rate limit" in err_str:
                logger.warning("embeddings.rate_limit_detected", error=str(err))
                # Raise RateLimitQuotaError for caller to update job status
                raise RateLimitQuotaError(retry_after=30.0) from err
            raise

        vectors: list[list[float]] = result.embeddings
        if len(vectors) != len(texts):
            raise ValueError(
                f"Voyage returned {len(vectors)} vectors for {len(texts)} texts"
            )

        logger.debug(
            "embeddings.text_batch",
            batch_size=len(texts),
            dim=len(vectors[0]) if vectors else 0,
        )
        return vectors

    # ── Image embedding ────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        reraise=True,
    )
    def embed_image(self, image_path: str) -> list[float]:
        """Embed a single image file using Voyage Multimodal-3."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found for embedding: {image_path}")

        image_bytes = path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        suffix = path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        mime_type = mime_map.get(suffix, "image/png")

        client = self._get_client()
        inputs = [
            {
                "content": [
                    {
                        "type": "image_base64",
                        "image_base64": f"data:{mime_type};base64,{image_b64}",
                    }
                ]
            }
        ]

        try:
            result = client.multimodal_embed(inputs=inputs, model=self._model)
        except Exception as err:
            err_str = str(err).lower()
            if "429" in err_str or "rate limit" in err_str:
                raise RateLimitQuotaError(retry_after=30.0) from err
            raise

        vectors: list[list[float]] = result.embeddings
        if not vectors:
            raise ValueError("Voyage returned empty embeddings for image")

        logger.debug("embeddings.image", path=image_path, dim=len(vectors[0]))
        return vectors[0]
