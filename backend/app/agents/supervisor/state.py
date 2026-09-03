"""
Ingestion Pipeline State.
Immutable Pydantic model passed between pipeline steps.
All steps return a new state via model_copy(update={...}).
"""
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.chunk import Chunk
from app.models.document import DocumentStatus


class IngestionState(BaseModel):
    """
    Shared mutable state flowing through the ingestion pipeline.

    Steps receive this state and return a new copy with updated fields.
    No step mutates the state directly (Pydantic model_copy pattern).
    """

    model_config = {"arbitrary_types_allowed": True}

    # ── Identity & Multi-tenancy ───────────────────────────────────────────────
    document_id: UUID = Field(..., description="UUID of the document being ingested")
    job_id: UUID | None = Field(default=None, description="UUID of the ingestion job")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    assistant_id: str = Field(default="default", description="Assistant identifier")
    knowledge_base_id: str = Field(default="default", description="Knowledge base identifier")
    filename: str = Field(..., description="Original filename as uploaded")
    industry: str = Field(default="manufacturing", description="Industry domain")

    # ── Storage ────────────────────────────────────────────────────────────────
    file_bytes: bytes | None = Field(
        default=None,
        description="Raw file bytes (cleared after save to conserve memory)",
    )
    storage_path: str = Field(
        default="", description="Absolute path to raw file on local filesystem"
    )

    # ── Processing state ───────────────────────────────────────────────────────
    sha256: str | None = Field(
        default=None, description="SHA-256 hex digest (set by step 02)"
    )
    content_hash: str | None = Field(
        default=None, description="Normalized content hash (set by step 02/03)"
    )
    parsed_doc: dict | None = Field(
        default=None, description="Structured parse output from step 03"
    )
    page_count: int | None = Field(
        default=None, description="Number of pages (set by step 03)"
    )
    chunks: list[Chunk] | None = Field(
        default=None, description="Hierarchical chunks (set by step 05)"
    )
    vectors: list[list[float]] | None = Field(
        default=None, description="384-dim BGE (BAAI/bge-small-en-v1.5) vectors (set by step 07)"
    )

    # ── Resumability & Checkpointing ────────────────────────────────────────────
    last_successful_stage: str | None = Field(default=None, description="Last completed pipeline step")
    stage_checkpoints: dict[str, bool] = Field(default_factory=dict, description="Completed stage map")
    retry_count: int = Field(default=0, ge=0, description="Current stage retry count")

    # ── Extended Deduplication & Versioning ──────────────────────────────────
    dup_classification: str | None = Field(default=None, description="Deduplication result classification")
    canonical_document_id: UUID | None = Field(default=None, description="Canonical document UUID if duplicate")
    reusable_chunk_ids: list[str] = Field(default_factory=list, description="IDs of unchanged chunks to reuse")
    new_chunk_ids: list[str] = Field(default_factory=list, description="IDs of new/modified chunks")
    document_version: int = Field(default=1, ge=1, description="Version number of document")
    supersedes_document_id: UUID | None = Field(default=None, description="Prior document UUID superseded by this")

    # ── Pipeline Versioning ───────────────────────────────────────────────────
    parser_version: str = Field(default="docling-2.x")
    chunking_version: str = Field(default="1.0")
    # Set by the pipeline at startup from the active embedding provider
    embedding_model: str = Field(default="")
    embedding_model_version: str = Field(default="")

    # ── Status & Control ───────────────────────────────────────────────────────
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING, description="Current pipeline status"
    )
    error_message: str | None = Field(
        default=None, description="Error message if status is FAILED"
    )
    progress_percent: int = Field(
        default=0, ge=0, le=100, description="Progress 0-100"
    )
    cancelled: bool = Field(default=False, description="Cancellation flag")

