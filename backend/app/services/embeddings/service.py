"""
Embedding Service — Local BGE provider adapter.

This module acts as a clean adapter around LocalEmbeddingProvider.

The public interface is:
    embed_batch(texts)   → list[list[float]]
    embed_query(text)    → list[float]
    embed_image(path)    → NotImplementedError (BGE is text-only; falls back to alt-text)
"""

from app.services.embeddings.local_provider import LocalEmbeddingProvider

# ── Re-export for backward compatibility ──────────────────────────────────────
# Code that imports EmbeddingService can still work without changes.
EmbeddingService = LocalEmbeddingProvider


class EmbeddingQuotaError(Exception):
    """Raised when local embedding inference fails (resource error, OOM, etc.)."""

    def __init__(self, message: str = "Local embedding inference failed.") -> None:
        super().__init__(message)


# Preserve the old exception alias for callers while keeping failures local.
RateLimitQuotaError = EmbeddingQuotaError
