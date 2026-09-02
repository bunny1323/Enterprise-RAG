"""
Local Embedding Provider — BAAI/bge-small-en-v1.5

Loads the model once via SentenceTransformers and reuses the singleton
for the lifetime of the process.  No API key required.

Output dimension : 384
Device           : CPU (configurable via LOCAL_EMBEDDING_DEVICE)
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from app.config.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # noqa: F401

logger = get_logger(__name__)

_lock = threading.Lock()
_model_instance: "SentenceTransformer | None" = None


def _get_model(model_name: str, device: str) -> "SentenceTransformer":
    """Return the singleton SentenceTransformer model, loading it on first call."""
    global _model_instance  # noqa: PLW0603
    if _model_instance is None:
        with _lock:
            if _model_instance is None:
                try:
                    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
                except ImportError as err:
                    raise ImportError(
                        "sentence-transformers is not installed. "
                        "Run: pip install sentence-transformers"
                    ) from err

                logger.info(
                    "local_embedder.loading_model",
                    model=model_name,
                    device=device,
                )
                _model_instance = SentenceTransformer(model_name, device=device)
                logger.info(
                    "local_embedder.model_ready",
                    model=model_name,
                    dim=_model_instance.get_sentence_embedding_dimension(),
                )
    return _model_instance  # type: ignore[return-value]


class LocalEmbeddingProvider:
    """
    CPU-based embedding provider backed by BAAI/bge-small-en-v1.5.

    Implements the same interface as EmbeddingService (embed_batch / embed_image)
    so it can be used as a drop-in for development and testing.

    embed_image() falls back to embedding the supplied alt-text because
    BGE is a text-only model.
    """

    #: Model identifier, also used as the ``embedding_model`` field in IngestionState.
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    MODEL_VERSION = "1.5"
    EMBEDDING_DIM = 384

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "cpu",
        batch_size: int = 8,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size

    # ── Public API (matches EmbeddingService interface) ────────────────────────

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings.  Returns a list of 384-dim float vectors.
        Runs synchronously on the calling thread (use run_in_executor for async code).
        """
        if not texts:
            return []

        model = _get_model(self._model_name, self._device)
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine similarity without extra step
            convert_to_numpy=True,
        )

        vectors = [emb.tolist() for emb in embeddings]

        logger.debug(
            "local_embedder.batch_complete",
            count=len(texts),
            dim=len(vectors[0]) if vectors else 0,
        )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for retrieval."""
        if not text.strip():
            return []
        
        return self.embed_batch([text])[0]

    def get_dimension(self) -> int:
        """Return the output dimension of the loaded embedding model."""
        model = _get_model(self._model_name, self._device)
        return model.get_sentence_embedding_dimension()

    def health_check(self) -> dict:
        """Verify the model can actually produce vectors."""
        try:
            vec = self.embed_query("health check test")
            dim = len(vec) if vec else 0
            return {
                "status": "ok",
                "model": self._model_name,
                "dimension": dim,
                "device": self._device,
            }
        except Exception as e:
            return {
                "status": "error",
                "model": self._model_name,
                "error": str(e)
            }

    def embed_image(self, image_path: str) -> list[float]:
        """
        BGE is a text-only model; this method is not supported for true image
        embeddings. Raises NotImplementedError so s07_embed falls back to
        embedding the chunk's alt-text content instead.
        """
        raise NotImplementedError(
            "LocalEmbeddingProvider (BGE-small) does not support image embeddings. "
            "The pipeline will automatically fall back to embedding the image alt-text."
        )
