"""
Ingestion Job models for asynchronous worker execution and tracking.
"""
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Lifecycle states for an ingestion job."""

    RECEIVED = "RECEIVED"
    VALIDATING = "VALIDATING"
    CHECKING_DUPLICATE = "CHECKING_DUPLICATE"
    PARSING = "PARSING"
    OCR = "OCR"
    VISION = "VISION"
    CHUNKING = "CHUNKING"
    METADATA = "METADATA"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    PARTIAL = "PARTIAL"
    DUPLICATE = "DUPLICATE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    QUARANTINED = "QUARANTINED"


class IngestionJob(BaseModel):
    """Canonical ingestion job record."""

    model_config = {"from_attributes": True}

    job_id: UUID = Field(default_factory=uuid4, description="Primary key — UUID v4")
    document_id: UUID = Field(..., description="Target document UUID")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    assistant_id: str = Field(default="default", description="Assistant identifier")
    knowledge_base_id: str = Field(default="default", description="Knowledge base identifier")
    status: JobStatus = Field(default=JobStatus.RECEIVED, description="Current job status")
    current_stage: str | None = Field(default=None, description="Current executing pipeline stage")
    progress_percent: int = Field(default=0, ge=0, le=100, description="Job progress percent")
    retry_count: int = Field(default=0, ge=0, description="Current stage retry count")
    last_successful_stage: str | None = Field(default=None, description="Last completed stage")
    stage_checkpoints: dict[str, bool] = Field(default_factory=dict, description="Checkpoint flags per stage")
    error_message: str | None = Field(default=None, description="Error message if failed")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    timeout_at: datetime | None = Field(default=None)
    cancelled_at: datetime | None = Field(default=None)
    metadata: dict = Field(default_factory=dict)
