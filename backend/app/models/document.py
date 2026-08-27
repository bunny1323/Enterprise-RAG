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
    RECEIVED = "RECEIVED"
    VALIDATING = "VALIDATING"
    CHECKING_DUPLICATE = "CHECKING_DUPLICATE"
    PARSING = "PARSING"
    OCR = "OCR"
    VISION = "VISION"
    CHUNKING = "CHUNKING"
    METADATA = "METADATA"
    EMBEDDING = "EMBEDDING"
    WAITING_FOR_EMBEDDING_QUOTA = "WAITING_FOR_EMBEDDING_QUOTA"
    INDEXING = "INDEXING"
    PARTIAL = "PARTIAL"
    DUPLICATE = "DUPLICATE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    QUARANTINED = "QUARANTINED"


class Document(BaseModel):
    """Canonical document record stored in Supabase."""

    model_config = {"from_attributes": True}

    id: UUID = Field(default_factory=uuid4, description="Primary key — UUID v4")
    sha256: str = Field(..., description="SHA-256 hex digest of raw file bytes")
    file_name: str = Field(..., description="Original uploaded filename")
    storage_path: str = Field(..., description="Absolute local path to raw file")
    industry: str = Field(
        default="manufacturing", description="Industry domain for metadata enrichment"
    )
    tenant_id: str = Field(default="default", description="Tenant identifier")
    assistant_id: str = Field(default="default", description="Assistant identifier")
    knowledge_base_id: str = Field(default="default", description="Knowledge base identifier")
    content_hash: str | None = Field(default=None, description="Normalized content hash")
    version: int = Field(default=1, ge=1, description="Document version number")
    canonical_document_id: UUID | None = Field(default=None, description="Canonical document ID if duplicate")
    supersedes: UUID | None = Field(default=None, description="Prior document ID superseded by this")
    parser_version: str | None = Field(default=None, description="Parser version used")
    embedding_model: str | None = Field(default=None, description="Embedding model used")
    embedding_model_version: str | None = Field(default=None, description="Embedding model version")
    page_count: int | None = Field(default=None, description="Number of pages")
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING, description="Current pipeline status"
    )
    progress_percent: int = Field(
        default=0, ge=0, le=100, description="Ingestion progress 0-100"
    )
    metadata: dict = Field(
        default_factory=dict, description="Arbitrary metadata"
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
