"""
FastAPI dependency injection functions.
All singletons are cached on app.state to share across requests.
"""
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from app.config.settings import Settings, get_settings
from app.infrastructure.neo4j.client import Neo4jClient
from app.infrastructure.postgres.client import PostgresClient
from app.infrastructure.storage.client import StorageClient
from app.infrastructure.weaviate.client import WeaviateClient
from app.services.chunking.service import ChunkingService
from app.services.document_parser.service import DocumentParserService
from app.services.embeddings.service import EmbeddingService
from app.services.metadata.service import MetadataService
from app.services.ocr.service import OCRService
from app.services.storage.service import StorageService
from app.services.vision.service import VisionService


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


def get_supervisor(request: Request):  # type: ignore[return]
    """Return the shared IngestionSupervisor from application state."""
    return request.app.state.supervisor


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
    """Return a stateless EmbeddingService with Voyage API key."""
    return EmbeddingService(
        api_key=settings.voyage_api_key,
        model=settings.voyage_model,
    )


# ── Annotated shorthand types for route injection ──────────────────────────────

PostgresDep = Annotated[PostgresClient, Depends(get_postgres)]
WeaviateDep = Annotated[WeaviateClient, Depends(get_weaviate)]
Neo4jDep = Annotated[Neo4jClient, Depends(get_neo4j)]
SupervisorDep = Annotated[object, Depends(get_supervisor)]
