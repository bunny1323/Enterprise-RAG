"""
Tenant, Security Context, and Knowledge Base domain models.
"""
from datetime import datetime
from pydantic import BaseModel, Field


class TenantContext(BaseModel):
    """Authenticated user/tenant security context passed through request pipeline."""

    tenant_id: str = Field(default="default", description="Tenant ID for data isolation")
    assistant_id: str = Field(default="default", description="Assistant ID")
    knowledge_base_id: str = Field(default="default", description="Knowledge base ID")
    access_level: str = Field(default="INTERNAL", description="User security classification: PUBLIC | INTERNAL | RESTRICTED")
    user_id: str | None = Field(default=None, description="Authenticated user ID")
    claims: dict = Field(default_factory=dict, description="JWT / Security claims for OPA evaluation")


class KnowledgeBaseVersion(BaseModel):
    """Version tracking for knowledge base cache invalidation."""

    knowledge_base_id: str
    tenant_id: str
    version: int = 1
    document_count: int = 0
    chunk_count: int = 0
    updated_at: datetime = Field(default_factory=datetime.utcnow)
