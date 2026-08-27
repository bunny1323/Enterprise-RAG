import time
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.config.logging import get_logger
from app.infrastructure.redis.client import RedisClient

logger = get_logger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple Redis-based fixed window rate limiter.
    Limits requests per tenant or IP address.
    """
    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def dispatch(self, request: Request, call_next):
        redis_client: RedisClient | None = getattr(request.app.state, "redis", None)
        if not redis_client:
            # Skip rate limiting if redis is not available
            return await call_next(request)

        # Rate limit by Tenant ID, fallback to client IP
        tenant_id = request.headers.get("X-Tenant-ID")
        client_ip = request.client.host if request.client else "unknown"
        identifier = tenant_id if tenant_id and tenant_id != "default" else client_ip
        
        # We only rate limit API routes
        if not request.url.path.startswith("/api/v1/"):
            return await call_next(request)

        current_window = int(time.time() / self.window_seconds)
        key = f"ratelimit:{identifier}:{current_window}"

        try:
            # Note: A proper atomic INCR + EXPIRE is ideal, using standard redis operations
            # Assuming our RedisClient has basic get/set or we can use the raw aioredis client
            raw_redis = getattr(redis_client, "_redis", None)
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
            # Fail open on redis errors to preserve availability
            logger.error("rate_limit.redis_error", error=str(e))
            
        return await call_next(request)
