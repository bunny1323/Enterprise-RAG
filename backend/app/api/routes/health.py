"""
Health check route.
GET /health — verifies connectivity to all dependent services.
"""
from fastapi import APIRouter, Request

from app.config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


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
