"""
Health check route.
GET /health — verifies connectivity to all dependent services.
"""
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health/live", summary="Process liveness check")
async def liveness_check() -> dict[str, str]:
    """Return success whenever the FastAPI process is serving requests."""
    return {"status": "alive"}


@router.get(
    "/health",
    summary="System health check",
    response_description="Service status for all dependencies",
)
async def health_check(request: Request) -> dict:
    """
    Check connectivity to all external services.

    Returns:
        {
            "status": "healthy" | "degraded",
            "services": {
                "supabase": "ok" | "error: <message>",
                "weaviate": "ok" | "error: <message>",
                "neo4j":    "ok" | "error: <message>",
                "ollama":   "ok" | "error: <message>"
            }
        }
    """
    services: dict[str, str] = {}
    overall_healthy = True

    # ── Supabase (PostgreSQL) ──────────────────────────────────────────────────
    try:
        postgres = request.app.state.postgres
        await postgres.fetchval("SELECT 1")
        services["supabase"] = "ok"
    except Exception as err:
        services["supabase"] = f"error: {str(err)}"
        overall_healthy = False
        logger.warning("health.supabase_failed", error=str(err))

    # ── Weaviate Cloud ─────────────────────────────────────────────────────────
    try:
        weaviate_client = request.app.state.weaviate
        if weaviate_client.is_connected():
            services["weaviate"] = "ok"
        else:
            services["weaviate"] = "error: not connected"
            overall_healthy = False
    except Exception as err:
        services["weaviate"] = f"error: {str(err)}"
        overall_healthy = False
        logger.warning("health.weaviate_failed", error=str(err))

    # ── Neo4j ──────────────────────────────────────────────────────────────────
    try:
        neo4j_client = request.app.state.neo4j
        reachable = await neo4j_client.verify_connectivity()
        if reachable:
            services["neo4j"] = "ok"
        else:
            services["neo4j"] = "error: not reachable"
            overall_healthy = False
    except Exception as err:
        services["neo4j"] = f"error: {str(err)}"
        overall_healthy = False
        logger.warning("health.neo4j_failed", error=str(err))

    # ── Ollama ─────────────────────────────────────────────────────────────────
    try:
        import httpx

        settings = request.app.state.settings
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
        services["ollama"] = "ok"
    except Exception as err:
        services["ollama"] = f"error: {str(err)}"
        # Ollama is non-critical — degraded but not unhealthy for Phase 1
        logger.warning("health.ollama_unavailable", error=str(err))

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "services": services,
    }


@router.get("/health/ready", summary="Dependency readiness check")
async def readiness_check(request: Request) -> JSONResponse:
    """Return 503 when a required runtime dependency cannot be used."""
    payload = await health_check(request)
    code = status.HTTP_200_OK if payload["status"] == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=payload)


@router.get("/health/embedding", summary="Embedding health check")
async def test_embedding(request: Request) -> dict:
    """Test generating a small vector from the configured embedding provider."""
    try:
        from app.services.embeddings.service import EmbeddingService
        embedder = request.app.state.embedder
        return embedder.health_check()
    except Exception as err:
        logger.error("health.embedding_test_failed", error=str(err))
        return {"status": "error", "error": str(err)}


@router.get("/health/llm", summary="LLM Generation Test (GET)")
@router.post("/health/llm", summary="LLM Generation Test (POST)")
@router.get("/api/v1/health/llm", summary="LLM Generation Test (GET v1)")
@router.post("/api/v1/health/llm", summary="LLM Generation Test (POST v1)")
async def test_llm(request: Request) -> dict:
    """Test generating a simple response from the configured LLM."""
    try:
        from app.services.generation.llm import LLMProvider
        llm_provider: LLMProvider = request.app.state.llm_provider
        
        # If the provider has a health_check() method, invoke it for detailed diagnostics
        if hasattr(llm_provider, "health_check"):
            result = await llm_provider.health_check()
            return result

        response = await llm_provider.generate(
            prompt="__HEALTH_CHECK__: Reply with 'LLM is working'",
            evidence=[],
            system_instruction="You are a health check. Reply with exactly: LLM is working",
        )
        return {
            "status": "healthy",
            "provider": request.app.state.settings.llm_provider,
            "model": request.app.state.settings.llm_model,
            "response": response.answer,
        }
    except Exception as err:
        logger.error("health.llm_test_failed", error=str(err))
        return {"status": "error", "error": str(err)}


