"""
Weaviate Cloud infrastructure client.
Manages connection, schema initialization, vector upsert, and tenant-isolated searches.
Uses weaviate-client v4 API.
"""
import json
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
                    name="tenant_id",
                    data_type=wvc.config.DataType.TEXT,
                    description="Tenant ID for data isolation",
                ),
                wvc.config.Property(
                    name="assistant_id",
                    data_type=wvc.config.DataType.TEXT,
                    description="Assistant ID",
                ),
                wvc.config.Property(
                    name="knowledge_base_id",
                    data_type=wvc.config.DataType.TEXT,
                    description="Knowledge base ID",
                ),
                wvc.config.Property(
                    name="content",
                    data_type=wvc.config.DataType.TEXT,
                    description="Raw chunk text",
                ),
                wvc.config.Property(
                    name="context_prefix",
                    data_type=wvc.config.DataType.TEXT,
                    description="Section & Document contextual prefix",
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
        """Batch-upsert chunks with pre-computed Voyage vectors."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Chunk/vector count mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
            )

        client = self._require_client()
        collection = client.collections.get(_COLLECTION_NAME)

        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                properties: dict[str, Any] = {
                    "chunk_id": chunk.chunk_id,
                    "parent_id": chunk.parent_id or "",
                    "document_id": str(chunk.document_id),
                    "tenant_id": chunk.tenant_id,
                    "assistant_id": chunk.assistant_id,
                    "knowledge_base_id": chunk.knowledge_base_id,
                    "content": chunk.content,
                    "context_prefix": chunk.context_prefix or "",
                    "page_number": chunk.page_number,
                    "chunk_type": chunk.chunk_type.value,
                    "industry_domain": chunk.industry_domain,
                    "access_classification": chunk.access_classification,
                    "bounding_box": json.dumps(chunk.bounding_box) if chunk.bounding_box else "[]",
                    "hierarchy_path": chunk.hierarchy_path,
                }
                batch.add_object(properties=properties, vector=vector)

        logger.info("weaviate.chunks_upserted", count=len(chunks))

    def delete_by_document(self, document_id: str, tenant_id: str = "default") -> None:
        """Delete all chunks belonging to a document under tenant isolation."""
        client = self._require_client()
        collection = client.collections.get(_COLLECTION_NAME)
        collection.data.delete_many(
            where=(
                wvc.query.Filter.by_property("document_id").equal(document_id)
                & wvc.query.Filter.by_property("tenant_id").equal(tenant_id)
            )
        )
        logger.info("weaviate.chunks_deleted", document_id=document_id, tenant_id=tenant_id)

    # ── Retrieval Operations ───────────────────────────────────────────────────

    def vector_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        tenant_id: str = "default",
        knowledge_base_id: str = "default",
        permitted_access_levels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Dense vector search filtered by tenant, knowledge_base, and access levels."""
        client = self._require_client()
        collection = client.collections.get(_COLLECTION_NAME)

        levels = permitted_access_levels or ["PUBLIC", "INTERNAL"]
        filters = (
            wvc.query.Filter.by_property("tenant_id").equal(tenant_id)
            & wvc.query.Filter.by_property("knowledge_base_id").equal(knowledge_base_id)
            & wvc.query.Filter.by_property("access_classification").contains_any(levels)
        )

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=filters,
            return_metadata=wvc.query.MetadataQuery(distance=True, score=True),
        )

        results: list[dict[str, Any]] = []
        for obj in response.objects:
            props = dict(obj.properties)
            dist = obj.metadata.distance if obj.metadata else 1.0
            props["score"] = 1.0 - (dist if dist is not None else 1.0)
            results.append(props)

        logger.info("weaviate.vector_search_complete", hits=len(results), tenant=tenant_id)
        return results

    def bm25_search(
        self,
        query_text: str,
        top_k: int = 10,
        tenant_id: str = "default",
        knowledge_base_id: str = "default",
        permitted_access_levels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 sparse keyword search filtered by tenant and access levels."""
        client = self._require_client()
        collection = client.collections.get(_COLLECTION_NAME)

        levels = permitted_access_levels or ["PUBLIC", "INTERNAL"]
        filters = (
            wvc.query.Filter.by_property("tenant_id").equal(tenant_id)
            & wvc.query.Filter.by_property("knowledge_base_id").equal(knowledge_base_id)
            & wvc.query.Filter.by_property("access_classification").contains_any(levels)
        )

        response = collection.query.bm25(
            query=query_text,
            limit=top_k,
            filters=filters,
            return_metadata=wvc.query.MetadataQuery(score=True),
        )

        results: list[dict[str, Any]] = []
        for obj in response.objects:
            props = dict(obj.properties)
            props["score"] = obj.metadata.score if obj.metadata else 0.0
            results.append(props)

        logger.info("weaviate.bm25_search_complete", hits=len(results), tenant=tenant_id)
        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _require_client(self) -> _WeaviateSDKClient:
        if self._client is None:
            raise RuntimeError("WeaviateClient.connect() must be called first")
        return self._client
