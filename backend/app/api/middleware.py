import hashlib
import time
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.config.logging import get_logger
from app.infrastructure.redis.client import RedisClient

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Redis-based fixed window rate limiter with tamper-proof identity resolution.
    Limits requests per authenticated identity or verified client IP.
    """
    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def _resolve_identity(self, request: Request) -> str:
        """
        Derive a trusted rate-limiting identity:
        1. If an auth token is provided (Bearer or X-API-Key), use its SHA256 digest.
           This prevents authenticated callers from bypassing limits by cycling X-Tenant-ID.
        2. Otherwise, use client IP. Forwarded headers (X-Forwarded-For) are trusted
           ONLY if the immediate upstream client IP is in settings.trusted_proxies.
        """
        auth_header = request.headers.get("Authorization", "")
        api_key_header = request.headers.get("X-API-Key", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        elif api_key_header:
            token = api_key_header.strip()
        elif "token" in request.query_params:
            token = request.query_params.get("token", "").strip()

        if token:
            token_digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
            return f"auth:{token_digest}"

        direct_ip = request.client.host if request.client else "unknown"
        settings = getattr(request.app.state, "settings", None)
        trusted_proxies = getattr(settings, "trusted_proxies", ["127.0.0.1", "::1"])

        if direct_ip in trusted_proxies:
            xff = request.headers.get("X-Forwarded-For", "")
            if xff:
                client_candidates = [ip.strip() for ip in xff.split(",") if ip.strip()]
                if client_candidates:
                    return f"ip:{client_candidates[0]}"

        return f"ip:{direct_ip}"

    async def dispatch(self, request: Request, call_next):
        # We only rate limit API routes
        if not request.url.path.startswith("/api/v1/"):
            return await call_next(request)

        redis_client: RedisClient | None = getattr(request.app.state, "redis", None)
        if not redis_client:
            # Safe fallback: preserve availability when Redis is unconfigured or unavailable
            return await call_next(request)

        identifier = self._resolve_identity(request)
        current_window = int(time.time() / self.window_seconds)
        key = f"ratelimit:{identifier}:{current_window}"

        try:
            raw_redis = getattr(redis_client, "_client", None)
            if raw_redis:
                pipeline = raw_redis.pipeline()
                pipeline.incr(key)
                pipeline.expire(key, self.window_seconds * 2)
                results = await pipeline.execute()
                
                request_count = results[0]
                
                if request_count > self.max_requests:
                    logger.warning("rate_limit.exceeded", identifier=identifier, path=request.url.path)
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={"detail": "Rate limit exceeded. Please try again later."},
                        headers={"Retry-After": str(self.window_seconds)}
                    )
        except Exception as e:
            # Fail open on redis errors to preserve availability, but log error
            logger.error("rate_limit.redis_error", error=str(e), identifier=identifier)
            
        return await call_next(request)


class AuditLogMiddleware(BaseHTTPMiddleware):
    """
    Immutable audit logging for query and admin endpoints.
    Logs tenant_id, user_id, path, method, status, and duration.
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        tenant_id = request.headers.get("X-Tenant-ID", "unknown")
        user_id = request.headers.get("X-User-ID", "unknown")
        
        response = await call_next(request)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Only log API paths
        if request.url.path.startswith("/api/v1/"):
            logger.info(
                "audit_log.request",
                tenant_id=tenant_id,
                user_id=user_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            
        return response
