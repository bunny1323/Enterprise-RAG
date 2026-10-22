"""
FastAPI dependency injection functions.
All singletons are cached on app.state to share across requests.
"""
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status

from app.config.logging import get_logger
from app.config.settings import Settings, get_settings

logger = get_logger(__name__)
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
from app.services.vision.vision_provider import BaseVisionProvider, build_vision_provider


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

import hashlib
import json
import secrets


def get_tenant_context(
    request: Request,
    settings: SettingsDep,
) -> TenantContext:
    """
    Extract and validate authenticated identity and tenant context.

    Security & Access Control:
    1. Authentication:
       - Production Mode (`settings.is_production`): Authentication is mandatory.
       - Token matching uses constant-time comparison (`secrets.compare_digest`).
       - Role separation: `admin_api_token` grants `is_admin=True`. Ordinary `api_auth_token`
         or tokens mapped in `api_tokens_json` grant standard user rights (`is_admin=False`).
    2. Tenant Isolation & Identity-to-Tenant Mapping:
       - Client-supplied `X-Tenant-ID` is NEVER blindly trusted.
       - Authenticated identities have an authorized tenant and allowed tenant list.
       - Tenant overrides via `X-Tenant-ID` are permitted ONLY if the identity is
         an admin or explicitly authorized for the requested tenant. Unauthorized overrides
         are rejected with HTTP 403 Forbidden.
    3. Privilege Escalation Defense:
       - `X-Access-Level: RESTRICTED` is accepted ONLY from verified administrators.
         All unverified callers are clamped to INTERNAL or PUBLIC.
    4. Development Mode:
       - Explicit local development mode (when auth disabled and non-production)
         defaults to `settings.default_tenant_id`.
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

    is_authenticated = False
    is_admin = False
    bound_tenant_id = settings.default_tenant_id
    allowed_tenants: list[str] = []
    user_id = request.headers.get("X-User-ID", None)

    # 1. Parse server-side token mapping if configured
    token_mapping: dict[str, dict[str, Any]] = {}
    if settings.api_tokens_json:
        try:
            token_mapping = json.loads(settings.api_tokens_json)
        except Exception:
            logger.error("security.api_tokens_mapping_invalid_json")

    # 2. Authenticate against configured secrets using constant-time comparison
    if token:
        # Check Admin Token
        if settings.admin_api_token and secrets.compare_digest(token, settings.admin_api_token):
            is_authenticated = True
            is_admin = True
            allowed_tenants = ["*"]
        # Check Server-Side Token Mapping
        elif token in token_mapping:
            info = token_mapping[token]
            is_authenticated = True
            is_admin = bool(info.get("is_admin", False) or "admin" in info.get("roles", []))
            bound_tenant_id = str(info.get("tenant_id", settings.default_tenant_id))
            allowed_tenants = list(info.get("allowed_tenants", [bound_tenant_id]))
            user_id = info.get("user_id", user_id)
        # Check Standard API Auth Token
        elif settings.api_auth_token and secrets.compare_digest(token, settings.api_auth_token):
            is_authenticated = True
            is_admin = False
            bound_tenant_id = settings.default_tenant_id
            allowed_tenants = [bound_tenant_id]

    # Explicit open development mode (only when NOT in production and auth is explicitly disabled)
    elif not settings.api_key_auth_enabled and not settings.is_production:
        is_authenticated = True
        is_admin = False
        bound_tenant_id = settings.default_tenant_id
        allowed_tenants = ["*"]  # Permissive only in local dev

    # Reject unauthenticated requests when authentication is required
    if (settings.api_key_auth_enabled or settings.is_production) and not is_authenticated:
        logger.warning(
            "security.authentication_failed",
            client_ip=getattr(request.client, "host", "unknown"),
            path=request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials required or invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Derive and validate tenant context
    requested_tenant = request.headers.get("X-Tenant-ID")
    if requested_tenant:
        requested_tenant = requested_tenant.strip()

    if requested_tenant and requested_tenant != bound_tenant_id:
        # Caller is attempting a tenant override
        if is_admin or "*" in allowed_tenants or requested_tenant in allowed_tenants:
            tenant_id = requested_tenant
        else:
            logger.warning(
                "security.unauthorized_tenant_override",
                requested=requested_tenant,
                bound=bound_tenant_id,
                allowed=allowed_tenants,
                client_ip=getattr(request.client, "host", "unknown"),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: identity is not authorized to access the requested tenant",
            )
    else:
        tenant_id = bound_tenant_id or "default"

    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required tenant context",
        )

    # 4. Context identifiers (untrusted headers)
    assistant_id = (request.headers.get("X-Assistant-ID", "default") or "default").strip()
    kb_id = (request.headers.get("X-Knowledge-Base-ID", "default") or "default").strip()

    # 5. Sanitize and clamp access level
    raw_access_level = (request.headers.get("X-Access-Level", "INTERNAL") or "INTERNAL").strip().upper()
    if raw_access_level == "RESTRICTED":
        if not is_admin:
            logger.warning(
                "security.privilege_escalation_prevented",
                tenant_id=tenant_id,
                requested_level="RESTRICTED",
                granted_level="INTERNAL",
                client_ip=getattr(request.client, "host", "unknown"),
            )
            access_level = "INTERNAL"
        else:
            access_level = "RESTRICTED"
    elif raw_access_level in ("PUBLIC", "INTERNAL"):
        access_level = raw_access_level
    else:
        access_level = "PUBLIC"

    claims: dict[str, Any] = {
        "is_authenticated": is_authenticated,
        "is_admin": is_admin,
        "token_present": bool(token),
        "allowed_tenants": allowed_tenants,
        "bound_tenant_id": bound_tenant_id,
    }

    return TenantContext(
        tenant_id=tenant_id,
        assistant_id=assistant_id,
        knowledge_base_id=kb_id,
        access_level=access_level,
        user_id=user_id,
        claims=claims,
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


def get_vision_service(settings: SettingsDep) -> BaseVisionProvider:
    """Return the configured vision provider (Gemini Cloud Vision or disabled)."""
    return build_vision_provider(settings)


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

