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
from app.services.embeddings.factory import build_embedding_provider
from app.services.generation.llm import build_llm_provider
from app.services.metadata.service import MetadataService
from app.services.ocr.service import OCRService
from app.services.storage.service import StorageService
from app.services.vision.service import VisionService
from app.config.opentelemetry import setup_telemetry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    Everything before `yield` runs at startup; everything after at shutdown.
    """
    settings = get_settings()
    configure_logging(debug=settings.debug)
    logger = get_logger(__name__)

    # ── 0. OpenTelemetry setup ─────────────────────────────────────────────────
    setup_telemetry(
        service_name=settings.app_name,
        otlp_endpoint=settings.otel_endpoint or None,
        enable_console=settings.otel_console or settings.debug,
    )

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
    try:
        weaviate_client.connect()
        weaviate_client.init_schema()
    except Exception as err:
        # The API can still provide liveness and report a truthful readiness
        # failure. Ingestion will record failed/partial vector indexing instead
        # of claiming success.
        logger.error("startup.weaviate_unavailable", error=str(err))
    app.state.weaviate = weaviate_client

    # ── 3. Neo4j ───────────────────────────────────────────────────────────────
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    try:
        await neo4j_client.connect()
        await neo4j_client.init_schema()
    except Exception as err:
        # Keep the process available for diagnostics; graph indexing/search will
        # expose the dependency failure through their existing error paths.
        logger.error("startup.neo4j_unavailable", error=str(err))
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

    # Pick embedding provider based on EMBEDDING_PROVIDER env var
    embedding_service = build_embedding_provider(settings)

    llm_provider = build_llm_provider(
        provider=settings.llm_provider,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        ollama_timeout=settings.ollama_timeout,
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

    # ── 10. Start checkpointer ────────────────────────────────────────────────
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        checkpointer_cm = AsyncPostgresSaver.from_conn_string(settings.database_url)
    except Exception as e:
        logger.warning("startup.checkpointer_failed", error=str(e))
        checkpointer_cm = None

    if checkpointer_cm:
        async with checkpointer_cm as checkpointer:
            await checkpointer.setup()
            app.state.checkpointer = checkpointer
            logger.info("startup.checkpointer_ready")
            
            logger.info("startup.complete", app=settings.app_name, port=settings.port)
            yield  # ── Application is running ────────────────────────────────────────
    else:
        app.state.checkpointer = None
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

    # ── Rate Limiting & Audit Logging ──────────────────────────────────────────
    from app.api.middleware import RateLimitMiddleware, AuditLogMiddleware
    app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)
    app.add_middleware(AuditLogMiddleware)

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(jobs.router)
    app.include_router(query.router)

    # ── Static Images ──────────────────────────────────────────────────────────
    from fastapi.staticfiles import StaticFiles
    import os
    os.makedirs(settings.processed_storage_path, exist_ok=True)
    app.mount(
        "/api/v1/images",
        StaticFiles(directory=settings.processed_storage_path),
        name="images"
    )

    return app


# Module-level app instance for uvicorn
app = create_app()
