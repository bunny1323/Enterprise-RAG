"""
Embedding Service — Local BGE provider adapter.

This module now acts as a clean adapter around LocalEmbeddingProvider.
All Voyage AI code has been removed.

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


# Remove the old Voyage-specific name so imports fail loudly if anything
# still tries to reference it by the old symbol.
RateLimitQuotaError = EmbeddingQuotaError
