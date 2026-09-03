"""
Embedding Provider Factory.

Returns the configured local embedding provider (BAAI/bge-small-en-v1.5).
Voyage AI has been removed.
"""
from __future__ import annotations

from app.config.logging import get_logger

logger = get_logger(__name__)


def build_embedding_provider(settings: object) -> object:
    """
    Instantiate and return the configured embedding provider.

    Parameters
    ----------
    settings:
        The application Settings object (app.config.settings.Settings).

    Returns
    -------
    LocalEmbeddingProvider
    """
    from app.services.embeddings.local_provider import LocalEmbeddingProvider

    model_name: str = getattr(settings, "local_embedding_model", LocalEmbeddingProvider.MODEL_NAME)
    device: str = getattr(settings, "local_embedding_device", "cpu")
    batch_size: int = getattr(settings, "local_embedding_batch_size", 8)

    logger.info(
        "embedding_factory.local_provider",
        model=model_name,
        device=device,
        batch_size=batch_size,
    )
    return LocalEmbeddingProvider(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
