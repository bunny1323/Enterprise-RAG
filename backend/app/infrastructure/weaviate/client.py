"""
Weaviate Cloud infrastructure client.
Manages connection, schema (collection) initialization, and vector upsert.
Uses weaviate-client v4 API.
"""
from typing import Any

import weaviate
import weaviate.classes as wvc
from weaviate.client import WeaviateClient as _WeaviateSDKClient

from app.config.logging import get_logger
from app.models.chunk import Chunk

logger = get_logger(__name__)

_COLLECTION_NAME = "DocumentChunk"


class WeaviateClient:
    """Wrapper around the Weaviate v4 Python client for WCS clusters."""

    def __init__(self, url: str, api_key: str) -> None:
        self._url = url
        self._api_key = api_key
        self._client: _WeaviateSDKClient | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open a synchronous connection to Weaviate Cloud."""
        self._client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self._url,
            auth_credentials=wvc.init.Auth.api_key(self._api_key),
            additional_config=wvc.init.AdditionalConfig(
                timeout=wvc.init.Timeout(init=10, query=30, insert=60),
            ),
        )
        logger.info("weaviate.connected", url=self._url)

    def is_connected(self) -> bool:
        """Return True if the client has an open connection."""
        if self._client is None:
            return False
        try:
            return self._client.is_connected()
        except Exception:
            return False

    def close(self) -> None:
        """Close the Weaviate connection."""
        if self._client:
            self._client.close()
            logger.info("weaviate.closed")

    # ── Schema ─────────────────────────────────────────────────────────────────

    def init_schema(self) -> None:
        """Create the DocumentChunk collection if it does not already exist."""
        client = self._require_client()
        if client.collections.exists(_COLLECTION_NAME):
            logger.info("weaviate.collection_exists", name=_COLLECTION_NAME)
            return

        client.collections.create(
            name=_COLLECTION_NAME,
            description="Hierarchical document chunks with Voyage multimodal embeddings",
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # BYO vectors
            properties=[
                wvc.config.Property(
                    name="chunk_id",
                    data_type=wvc.config.DataType.TEXT,
                    description="Unique chunk identifier",
                ),
                wvc.config.Property(
                    name="parent_id",
                    data_type=wvc.config.DataType.TEXT,
                    description="Parent chunk ID for hierarchical retrieval",
                    skip_vectorization=True,
                ),
                wvc.config.Property(
                    name="document_id",
                    data_type=wvc.config.DataType.TEXT,
                    description="Parent document UUID",
                ),
                wvc.config.Property(
                    name="content",
                    data_type=wvc.config.DataType.TEXT,
                    description="Raw chunk text",
                ),
                wvc.config.Property(
                    name="page_number",
                    data_type=wvc.config.DataType.INT,
                    description="Source page number",
                ),
                wvc.config.Property(
                    name="chunk_type",
                    data_type=wvc.config.DataType.TEXT,
                    description="TEXT | TABLE | IMAGE | DIAGRAM",
                ),
                wvc.config.Property(
                    name="industry_domain",
                    data_type=wvc.config.DataType.TEXT,
                    description="Industry domain label",
                ),
                wvc.config.Property(
                    name="access_classification",
                    data_type=wvc.config.DataType.TEXT,
                    description="PUBLIC | INTERNAL | RESTRICTED",
                ),
                wvc.config.Property(
                    name="bounding_box",
                    data_type=wvc.config.DataType.TEXT,
                    description="JSON-serialized [x0,y0,x1,y1] bbox",
                    skip_vectorization=True,
                ),
                wvc.config.Property(
                    name="hierarchy_path",
                    data_type=wvc.config.DataType.TEXT,
                    description="Dot-separated hierarchy path",
                    skip_vectorization=True,
                ),
            ],
        )
        logger.info("weaviate.collection_created", name=_COLLECTION_NAME)

    # ── Write operations ───────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """
        Batch-upsert chunks with pre-computed Voyage vectors.

        Args:
            chunks: Chunk objects to store.
            vectors: Parallel list of 1024-dim embedding vectors.

        Raises:
            ValueError: If chunks and vectors lengths do not match.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Chunk/vector count mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
            )

        import json

        client = self._require_client()
        collection = client.collections.get(_COLLECTION_NAME)

        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                properties: dict[str, Any] = {
                    "chunk_id": chunk.chunk_id,
                    "parent_id": chunk.parent_id or "",
                    "document_id": str(chunk.document_id),
                    "content": chunk.content,
                    "page_number": chunk.page_number,
                    "chunk_type": chunk.chunk_type.value,
                    "industry_domain": chunk.industry_domain,
                    "access_classification": chunk.access_classification,
                    "bounding_box": json.dumps(chunk.bounding_box) if chunk.bounding_box else "[]",
                    "hierarchy_path": chunk.hierarchy_path,
                }
                batch.add_object(properties=properties, vector=vector)

        logger.info("weaviate.chunks_upserted", count=len(chunks))

    def delete_by_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        client = self._require_client()
        collection = client.collections.get(_COLLECTION_NAME)
        collection.data.delete_many(
            where=wvc.query.Filter.by_property("document_id").equal(document_id)
        )
        logger.info("weaviate.chunks_deleted", document_id=document_id)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _require_client(self) -> _WeaviateSDKClient:
        if self._client is None:
            raise RuntimeError("WeaviateClient.connect() must be called first")
        return self._client
