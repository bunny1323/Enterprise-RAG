"""
FastAPI dependency injection functions.
All singletons are cached on app.state to share across requests.
"""
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, Request

from app.config.settings import Settings, get_settings
from app.infrastructure.neo4j.client import Neo4jClient
from app.infrastructure.postgres.client import PostgresClient
from app.infrastructure.storage.client import StorageClient
from app.infrastructure.weaviate.client import WeaviateClient
from app.services.chunking.service import ChunkingService
from app.services.document_parser.service import DocumentParserService
from app.services.embeddings.service import EmbeddingService
from app.services.embeddings.factory import build_embedding_provider
from app.services.metadata.service import MetadataService
from app.services.ocr.service import OCRService
from app.services.storage.service import StorageService
from app.services.vision.service import VisionService


from app.infrastructure.opa.client import OPAClient
from app.infrastructure.redis.client import RedisClient
from app.models.tenant import TenantContext
from app.services.cache.service import CacheService


# ── Settings ───────────────────────────────────────────────────────────────────

def get_settings_dep() -> Settings:
    """Return the cached Settings singleton."""
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


# ── Infrastructure singletons (from app.state) ─────────────────────────────────

def get_postgres(request: Request) -> PostgresClient:
    """Return the shared PostgresClient from application state."""
    return request.app.state.postgres


def get_weaviate(request: Request) -> WeaviateClient:
    """Return the shared WeaviateClient from application state."""
    return request.app.state.weaviate


def get_neo4j(request: Request) -> Neo4jClient:
    """Return the shared Neo4jClient from application state."""
    return request.app.state.neo4j


def get_redis(request: Request) -> RedisClient | None:
    """Return the shared RedisClient from application state."""
    return getattr(request.app.state, "redis", None)


def get_opa(request: Request) -> OPAClient:
    """Return the shared OPAClient from application state."""
    return getattr(request.app.state, "opa", OPAClient())


def get_supervisor(request: Request):  # type: ignore[return]
    """Return the shared IngestionSupervisor from application state."""
    return request.app.state.supervisor


# ── Security & Multi-tenancy ───────────────────────────────────────────────────

def get_tenant_context(request: Request) -> TenantContext:
    """
    Extract authenticated identity and tenant context from request headers.
    Headers:
        X-Tenant-ID
        X-Assistant-ID
        X-Knowledge-Base-ID
        X-Access-Level
    """
    tenant_id = request.headers.get("X-Tenant-ID", "default")
    assistant_id = request.headers.get("X-Assistant-ID", "default")
    kb_id = request.headers.get("X-Knowledge-Base-ID", "default")
    access_level = request.headers.get("X-Access-Level", "INTERNAL")
    user_id = request.headers.get("X-User-ID", None)

    return TenantContext(
        tenant_id=tenant_id,
        assistant_id=assistant_id,
        knowledge_base_id=kb_id,
        access_level=access_level,
        user_id=user_id,
    )


TenantContextDep = Annotated[TenantContext, Depends(get_tenant_context)]


# ── Per-request service factories (stateless, cheap to create) ─────────────────

def get_storage_service(
    settings: SettingsDep,
) -> StorageService:
    """Create a StorageService backed by configured storage paths."""
    client = StorageClient(
        raw_path=settings.raw_storage_path,
        processed_path=settings.processed_storage_path,
        chunks_path=settings.chunks_storage_path,
    )
    return StorageService(client)


def get_parser_service() -> DocumentParserService:
    """Return a stateless DocumentParserService."""
    return DocumentParserService()


def get_ocr_service() -> OCRService:
    """Return a stateless OCRService."""
    return OCRService()


def get_vision_service(settings: SettingsDep) -> VisionService:
    """Return a stateless VisionService configured for local Ollama."""
    return VisionService(
        ollama_base_url=settings.ollama_base_url,
        model=settings.ollama_vision_model,
    )


def get_chunking_service() -> ChunkingService:
    """Return a stateless ChunkingService."""
    return ChunkingService()


def get_metadata_service() -> MetadataService:
    """Return a stateless MetadataService pointed at the config directory."""
    return MetadataService(config_dir="./config/industries")


def get_embedding_service(settings: SettingsDep) -> EmbeddingService:
    """Return the configured local embedding provider (BAAI/bge-small-en-v1.5)."""
    return build_embedding_provider(settings)  # type: ignore[return-value]


def get_checkpointer(request: Request) -> Any:
    """Return the persistent checkpointer for LangGraph."""
    return request.app.state.checkpointer


# ── Annotated shorthand types for route injection ──────────────────────────────

PostgresDep = Annotated[PostgresClient, Depends(get_postgres)]
WeaviateDep = Annotated[WeaviateClient, Depends(get_weaviate)]
Neo4jDep = Annotated[Neo4jClient, Depends(get_neo4j)]
CheckpointerDep = Annotated[Any, Depends(get_checkpointer)]
SupervisorDep = Annotated[object, Depends(get_supervisor)]

