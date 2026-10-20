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


# BGE-small-en-v1.5 documented query instruction for asymmetric retrieval.
# IMPORTANT: This prefix must be applied consistently to BOTH the query AND
# all indexed document embeddings.  If you enable this for queries, you MUST
# also re-embed and re-index the entire document corpus with the same prefix.
# See: https://huggingface.co/BAAI/bge-small-en-v1.5
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Token budget for BGE-small-en-v1.5 (max 512 subword tokens).
# ~350 words is a conservative word-count proxy for the 512-token limit.
_SAFE_WORD_BUDGET = 350


def _check_input_length(texts: list[str], context: str = "embed_batch") -> None:
    """
    Warn when any input text exceeds the safe word budget for BGE-small-en-v1.5.

    BGE-small silently truncates inputs exceeding 512 subword tokens.
    This guard fires early at ~350 words (conservative proxy) so callers
    are aware before important content is silently dropped.
    """
    for i, text in enumerate(texts):
        word_count = len(text.split())
        if word_count > _SAFE_WORD_BUDGET:
            logger.warning(
                "local_embedder.input_too_long",
                context=context,
                index=i,
                word_count=word_count,
                safe_budget=_SAFE_WORD_BUDGET,
            )


class LocalEmbeddingProvider:
    """
    CPU-based embedding provider backed by BAAI/bge-small-en-v1.5.

    Implements the same interface as EmbeddingService (embed_batch / embed_image)
    so it can be used as a drop-in for development and testing.

    embed_image() falls back to embedding the supplied alt-text because
    BGE is a text-only model.

    BGE query prefix (use_query_prefix):
    BGE-small-en-v1.5 supports asymmetric retrieval where queries use the
    documented instruction prefix.  This flag is False by default to preserve
    compatibility with existing indexed vectors that were produced without the
    prefix.  To enable: set use_query_prefix=True AND re-embed the entire
    document corpus before querying.  Mixing prefix/no-prefix vectors produces
    incorrect retrieval results.
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
        use_query_prefix: bool = False,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._use_query_prefix = use_query_prefix
        if use_query_prefix:
            logger.info(
                "local_embedder.query_prefix_enabled",
                model=model_name,
                warning=(
                    "All indexed document vectors must also have been produced "
                    "with the same prefix or retrieval results will be incorrect."
                ),
            )

    # ── Public API (matches EmbeddingService interface) ────────────────────────

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings.  Returns a list of 384-dim float vectors.
        Runs synchronously on the calling thread (use run_in_executor for async code).

        Used for DOCUMENT embedding during ingestion.  Does NOT apply the query
        prefix even when use_query_prefix is True.
        """
        if not texts:
            return []

        _check_input_length(texts, context="embed_batch")

        model = _get_model(self._model_name, self._device)
        try:
            embeddings = model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,   # cosine similarity without extra step
                convert_to_numpy=True,
            )
        except Exception as exc:
            logger.error(
                "local_embedder.batch_failed",
                count=len(texts),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise

        vectors = [emb.tolist() for emb in embeddings]

        logger.debug(
            "local_embedder.batch_complete",
            count=len(texts),
            dim=len(vectors[0]) if vectors else 0,
        )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string for retrieval.

        When use_query_prefix=True, prepends the BGE-documented query instruction
        for asymmetric retrieval mode.  Only enable after re-indexing all documents
        with the same prefix.  See class docstring for migration requirements.
        """
        if not text.strip():
            return []

        _check_input_length([text], context="embed_query")

        if self._use_query_prefix:
            prefixed = _BGE_QUERY_INSTRUCTION + text
            logger.debug("local_embedder.query_prefix_applied", prefixed_length=len(prefixed))
            return self.embed_batch([prefixed])[0]

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
                "use_query_prefix": self._use_query_prefix,
            }
        except Exception as e:
            return {
                "status": "error",
                "model": self._model_name,
                "error": str(e),
                "use_query_prefix": self._use_query_prefix,
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
