"""
FastAPI application factory with lifespan management.
Startup: initialize all infrastructure clients, start background queue worker.
Shutdown: gracefully close all connections.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health, jobs, query
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.infrastructure.neo4j.client import Neo4jClient
from app.infrastructure.opa.client import OPAClient
from app.infrastructure.postgres.client import PostgresClient
from app.infrastructure.redis.client import RedisClient
from app.infrastructure.storage.client import StorageClient
from app.infrastructure.weaviate.client import WeaviateClient
from app.agents.supervisor.agent import IngestionSupervisor
from app.pipelines.ingestion.pipeline import IngestionPipeline
from app.services.chunking.service import ChunkingService
from app.services.document_parser.service import DocumentParserService
from app.services.embeddings.service import EmbeddingService
from app.services.generation.llm import OllamaProvider
from app.services.metadata.service import MetadataService
from app.services.ocr.service import OCRService
from app.services.storage.service import StorageService
from app.services.vision.service import VisionService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    Everything before `yield` runs at startup; everything after at shutdown.
    """
    settings = get_settings()
    configure_logging(debug=settings.debug)
    logger = get_logger(__name__)

    logger.info("startup.begin", app=settings.app_name)

    # ── 1. PostgreSQL pool ─────────────────────────────────────────────────────
    postgres = PostgresClient(database_url=settings.database_url)
    await postgres.init_pool()
    await postgres.init_schema()
    app.state.postgres = postgres

    # ── 2. Weaviate Cloud ──────────────────────────────────────────────────────
    weaviate_client = WeaviateClient(
        url=settings.weaviate_url,
        api_key=settings.weaviate_api_key,
    )
    weaviate_client.connect()
    weaviate_client.init_schema()
    app.state.weaviate = weaviate_client

    # ── 3. Neo4j ───────────────────────────────────────────────────────────────
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await neo4j_client.connect()
    await neo4j_client.init_schema()
    app.state.neo4j = neo4j_client

    # ── 4. Redis Client ────────────────────────────────────────────────────────
    redis_client = RedisClient(redis_url=settings.redis_url)
    await redis_client.connect()
    app.state.redis = redis_client

    # ── 5. OPA Client ──────────────────────────────────────────────────────────
    opa_client = OPAClient()
    app.state.opa = opa_client

    # ── 6. Storage client ──────────────────────────────────────────────────────
    storage_client = StorageClient(
        raw_path=settings.raw_storage_path,
        processed_path=settings.processed_storage_path,
        chunks_path=settings.chunks_storage_path,
    )
    app.state.storage = storage_client

    # ── 7. Services & Providers ────────────────────────────────────────────────
    storage_service = StorageService(storage_client)
    parser_service = DocumentParserService()
    ocr_service = OCRService()
    vision_service = VisionService(
        ollama_base_url=settings.ollama_base_url,
        model=settings.ollama_vision_model,
    )
    chunker_service = ChunkingService()
    metadata_service = MetadataService(config_dir="./config/industries")
    embedding_service = EmbeddingService(
        api_key=settings.voyage_api_key,
        model=settings.voyage_model,
    )
    llm_provider = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_vision_model,
    )
    app.state.embedder = embedding_service
    app.state.llm_provider = llm_provider

    # ── 8. Pipeline and supervisor ─────────────────────────────────────────────
    services_registry: dict = {
        "postgres": postgres,
        "weaviate": weaviate_client,
        "neo4j": neo4j_client,
        "storage": storage_service,
        "parser": parser_service,
        "ocr": ocr_service,
        "vision": vision_service,
        "chunker": chunker_service,
        "metadata": metadata_service,
        "embedder": embedding_service,
    }

    pipeline = IngestionPipeline(services=services_registry)
    supervisor = IngestionSupervisor(
        pipeline=pipeline,
        storage=storage_service,
        postgres=postgres,
    )
    app.state.supervisor = supervisor
    app.state.settings = settings

    # ── 9. Start background queue worker ──────────────────────────────────────
    queue_task = asyncio.create_task(supervisor.process_queue())
    app.state.queue_task = queue_task
    logger.info("startup.queue_worker_started")

    logger.info("startup.complete", app=settings.app_name, port=settings.port)

    yield  # ── Application is running ────────────────────────────────────────

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("shutdown.begin")

    # Cancel the background queue worker
    queue_task.cancel()
    try:
        await queue_task
    except asyncio.CancelledError:
        pass

    # Close all infrastructure connections
    await postgres.close()
    weaviate_client.close()
    await neo4j_client.close()
    await redis_client.close()
    await opa_client.close()
    await llm_provider.close()
    vision_service.close()

    logger.info("shutdown.complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Enterprise Multi-Agent RAG Platform — Phase 1 & 2 Complete Architecture",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(jobs.router)
    app.include_router(query.router)

    return app


# Module-level app instance for uvicorn
app = create_app()
