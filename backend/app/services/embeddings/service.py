"""
Embedding Service — Voyage Multimodal-3 for all modalities.

This is the ONLY external paid API call in Phase 1.
All text, tables, images, and diagram descriptions are embedded here.
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


class EmbeddingService:
    """
    Stateless embedding service backed by Voyage Multimodal-3.

    - Batches text inputs in groups of 32 to respect API rate limits.
    - Retries on rate-limit (HTTP 429) and transient errors with exponential backoff.
    - Supports both text content and image files (base64 encoded).
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

    # ── Text embedding ─────────────────────────────────────────────────────────

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings using Voyage Multimodal-3.

        Splits into batches of 32 and retries each batch independently.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of 1024-dim float vectors, parallel to input texts.
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []

        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[batch_start : batch_start + _BATCH_SIZE]
            vectors = self._embed_text_batch_with_retry(batch)
            all_vectors.extend(vectors)

            # Brief pause between batches to stay within rate limits
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

        # Voyage multimodal_embed expects list of content dicts
        inputs = [{"content": text} for text in texts]

        result = client.multimodal_embed(
            inputs=inputs,
            model=self._model,
        )

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
        """
        Embed a single image file using Voyage Multimodal-3.

        The image is read as base64 and submitted as an image modality input.

        Args:
            image_path: Absolute path to image file (PNG, JPEG, etc.).

        Returns:
            1024-dim float vector.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found for embedding: {image_path}")

        # Read and base64-encode the image
        image_bytes = path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Detect MIME type from extension
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

        result = client.multimodal_embed(inputs=inputs, model=self._model)
        vectors: list[list[float]] = result.embeddings

        if not vectors:
            raise ValueError("Voyage returned empty embeddings for image")

        logger.debug("embeddings.image", path=image_path, dim=len(vectors[0]))
        return vectors[0]
