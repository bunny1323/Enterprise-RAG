"""
Document domain models.
DocumentStatus tracks pipeline progress; Document is the canonical record.
"""
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Lifecycle states for a document moving through the ingestion pipeline."""

    PENDING = "PENDING"
    VALIDATING = "VALIDATING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Document(BaseModel):
    """Canonical document record stored in Supabase."""

    model_config = {"from_attributes": True}

    id: UUID = Field(default_factory=uuid4, description="Primary key — UUID v4")
    sha256: str = Field(..., description="SHA-256 hex digest of raw file bytes (deduplication key)")
    file_name: str = Field(..., description="Original uploaded filename")
    storage_path: str = Field(..., description="Absolute local path to raw file")
    industry: str = Field(
        default="manufacturing", description="Industry domain for metadata enrichment"
    )
    page_count: int | None = Field(default=None, description="Number of pages (set after parsing)")
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING, description="Current pipeline status"
    )
    progress_percent: int = Field(
        default=0, ge=0, le=100, description="Ingestion progress 0-100"
    )
    metadata: dict = Field(
        default_factory=dict, description="Arbitrary metadata from parsing and enrichment"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Record creation timestamp (UTC)"
    )
    completed_at: datetime | None = Field(
        default=None, description="Timestamp when ingestion completed or failed"
    )
    error_message: str | None = Field(
        default=None, description="Error details if status is FAILED"
    )


class DocumentCreateRequest(BaseModel):
    """Request body for creating a new document ingestion job."""

    file_name: str
    industry: str = "manufacturing"


class DocumentStatusResponse(BaseModel):
    """Lightweight response for polling document status."""

    document_id: UUID
    status: DocumentStatus
    progress_percent: int
    created_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
