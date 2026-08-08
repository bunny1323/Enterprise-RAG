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

    # ── Identity ───────────────────────────────────────────────────────────────
    document_id: UUID = Field(..., description="UUID of the document being ingested")
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
        default=None, description="1024-dim Voyage vectors (set by step 07)"
    )

    # ── Status ─────────────────────────────────────────────────────────────────
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING, description="Current pipeline status"
    )
    error_message: str | None = Field(
        default=None, description="Error message if status is FAILED"
    )
    progress_percent: int = Field(
        default=0, ge=0, le=100, description="Progress 0-100"
    )
