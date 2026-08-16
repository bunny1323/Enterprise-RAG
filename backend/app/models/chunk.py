"""
Chunk domain models.
Chunks are the atomic units stored in Weaviate (with vectors) and Supabase (metadata).
"""
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    """Semantic type of chunk content for retrieval routing."""

    TEXT = "TEXT"
    TABLE = "TABLE"
    IMAGE = "IMAGE"
    DIAGRAM = "DIAGRAM"


class Chunk(BaseModel):
    """Atomic chunk stored in vector store and relational DB."""

    model_config = {"from_attributes": True}

    chunk_id: str = Field(..., description="Unique chunk identifier (doc_id::page::seq)")
    parent_id: str | None = Field(
        default=None, description="Parent chunk ID for hierarchical retrieval"
    )
    document_id: UUID = Field(..., description="Parent document UUID")
    content: str = Field(..., description="Raw text content for embedding and retrieval")
    page_number: int = Field(..., ge=0, description="1-based page number (0 for doc-level)")
    bounding_box: list[float] | None = Field(
        default=None,
        description="[x0, y0, x1, y1] normalized bounding box on page",
    )
    chunk_type: ChunkType = Field(default=ChunkType.TEXT, description="Semantic content type")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    assistant_id: str = Field(default="default", description="Assistant identifier")
    knowledge_base_id: str = Field(default="default", description="Knowledge base identifier")
    content_hash: str | None = Field(default=None, description="Deterministic hash of chunk content")
    section: str | None = Field(default=None, description="Section heading")
    subsection: str | None = Field(default=None, description="Subsection heading")
    context_prefix: str | None = Field(default=None, description="Document + Section contextual prefix")
    embedding_representation: str = Field(default="text", description="Representation type (text | image | text_summary_of_image)")
    access_classification: str = Field(
        default="INTERNAL",
        description="Security classification (PUBLIC | INTERNAL | RESTRICTED)",
    )
    industry_domain: str = Field(
        default="manufacturing", description="Industry domain for metadata routing"
    )
    hierarchy_path: str = Field(
        default="",
        description="Dot-separated path e.g. 'doc.section.subsection.chunk'",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Arbitrary enrichment metadata from parser and domain config",
    )

    # ── Derived helpers ────────────────────────────────────────────────────────
    @property
    def token_estimate(self) -> int:
        """Rough word-count based token estimate (1 token ≈ 0.75 words)."""
        return int(len(self.content.split()) / 0.75)

    def is_parent(self) -> bool:
        """Return True if this chunk has no parent (it is a parent chunk)."""
        return self.parent_id is None
