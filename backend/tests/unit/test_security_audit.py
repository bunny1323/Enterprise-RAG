"""
Comprehensive Security and Multi-Tenant Isolation Tests (Tasks 1 through 7).

Covers all 18 test cases specified in the security remediation task:
1. Missing credentials rejected when auth enabled.
2. Invalid credentials rejected.
3. Production configuration rejects disabled authentication.
4. Authenticated ordinary user does not automatically become an administrator.
5. Client cannot obtain elevated access by changing X-Access-Level.
6. Missing or invalid tenant identity is rejected where required.
7. Tenant A cannot list Tenant B's documents.
8. Tenant A cannot retrieve Tenant B's document.
9. Tenant A cannot delete Tenant B's document.
10. The default tenant cannot bypass document ownership checks.
11. Tenant A cannot download Tenant B's images.
12. Spoofing X-Tenant-ID does not change the authenticated tenant.
13. Tenant A cannot retrieve Tenant B's cached answers.
14. Background jobs cannot be accessed or cancelled by unauthorized tenants.
15. Cycling arbitrary tenant headers does not bypass rate limits.
16. Spoofed forwarded IP headers cannot create arbitrary identities without trusted proxy.
17. Frontend streaming requests include expected authorization header.
18. Frontend handles 401 and 403 responses appropriately.
"""
import hashlib
import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.dependencies import get_tenant_context
from app.api.middleware import RateLimitMiddleware
from app.api.routes.documents import delete_document, get_document_status, list_documents
from app.api.routes.images import get_document_image
from app.api.routes.jobs import cancel_job, get_job_status
from app.config.settings import Settings
from app.models.tenant import TenantContext
from app.services.cache.semantic import _semantic_cache_namespace
from app.services.generation.prompts import PromptRegistry
from app.services.pii.interface import SimpleRegexPIIDetector


# ─────────────────────────────────────────────────────────────────────────────
# 1-6: Authentication & Authorization Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_01_missing_credentials_rejected_when_auth_required():
    """Missing credentials must be rejected with HTTP 401 when auth is enabled."""
    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.client.host = "10.0.0.1"
    mock_request.url.path = "/api/v1/documents"

    settings = MagicMock(spec=Settings)
    settings.api_key_auth_enabled = True
    settings.api_auth_token = "valid_secret_token"
    settings.admin_api_token = "admin_secret_token"
    settings.api_tokens_json = ""
    settings.is_production = False
    settings.default_tenant_id = "default"

    with pytest.raises(HTTPException) as exc:
        get_tenant_context(mock_request, settings)

    assert exc.value.status_code == 401
    assert "credentials required" in exc.value.detail.lower()


def test_02_invalid_credentials_rejected():
    """Invalid token credentials must be rejected with HTTP 401."""
    mock_request = MagicMock()
    mock_request.headers = {"Authorization": "Bearer totally_wrong_secret"}
    mock_request.client.host = "10.0.0.1"
    mock_request.url.path = "/api/v1/documents"

    settings = MagicMock(spec=Settings)
    settings.api_key_auth_enabled = True
    settings.api_auth_token = "valid_secret_token"
    settings.admin_api_token = "admin_secret_token"
    settings.api_tokens_json = ""
    settings.is_production = False
    settings.default_tenant_id = "default"

    with pytest.raises(HTTPException) as exc:
        get_tenant_context(mock_request, settings)

    assert exc.value.status_code == 401


def test_03_production_config_rejects_disabled_authentication():
    """Startup config validation must fail if APP_ENV is production and auth is disabled."""
    settings = Settings(
        app_env="production",
        api_key_auth_enabled=False,
        database_url="postgresql://user:pass@localhost:5432/db",
        weaviate_url="https://weaviate.example.com",
        weaviate_api_key="real-weaviate-key",
        neo4j_password="real-neo4j-password",
    )
    with pytest.raises(ValueError) as exc:
        settings.validate_startup_config()

    assert "API_KEY_AUTH_ENABLED must be set to 'true'" in str(exc.value)


def test_04_authenticated_ordinary_user_not_admin():
    """An ordinary user token must grant is_admin=False and be locked to their tenant."""
    mock_request = MagicMock()
    mock_request.headers = {"Authorization": "Bearer standard_user_token"}
    mock_request.client.host = "10.0.0.1"
    mock_request.url.path = "/api/v1/query"

    settings = MagicMock(spec=Settings)
    settings.api_key_auth_enabled = True
    settings.api_auth_token = "standard_user_token"
    settings.admin_api_token = "separate_super_admin_token"
    settings.api_tokens_json = ""
    settings.is_production = False
    settings.default_tenant_id = "tenant_standard"

    ctx = get_tenant_context(mock_request, settings)
    assert ctx.claims["is_authenticated"] is True
    assert ctx.claims["is_admin"] is False
    assert ctx.tenant_id == "tenant_standard"


def test_05_client_cannot_escalate_access_level():
    """Non-admin client sending X-Access-Level: RESTRICTED is clamped to INTERNAL."""
    mock_request = MagicMock()
    mock_request.headers = {
        "Authorization": "Bearer standard_user_token",
        "X-Access-Level": "RESTRICTED",
    }
    mock_request.client.host = "10.0.0.1"
    mock_request.url.path = "/api/v1/query"

    settings = MagicMock(spec=Settings)
    settings.api_key_auth_enabled = True
    settings.api_auth_token = "standard_user_token"
    settings.admin_api_token = "separate_super_admin_token"
    settings.api_tokens_json = ""
    settings.is_production = False
    settings.default_tenant_id = "tenant_standard"

    ctx = get_tenant_context(mock_request, settings)
    assert ctx.access_level == "INTERNAL"
    assert ctx.claims["is_admin"] is False


def test_06_missing_or_invalid_tenant_identity_rejected():
    """Empty tenant identifier where tenant context is required raises HTTP 400."""
    mock_request = MagicMock()
    mock_request.headers = {"Authorization": "Bearer user_token"}
    mock_request.client.host = "10.0.0.1"
    mock_request.url.path = "/api/v1/documents"

    settings = MagicMock(spec=Settings)
    settings.api_key_auth_enabled = True
    settings.api_auth_token = "user_token"
    settings.admin_api_token = "admin_token"
    settings.api_tokens_json = ""
    settings.is_production = False
    settings.default_tenant_id = ""  # Misconfigured empty tenant

    with pytest.raises(HTTPException) as exc:
        get_tenant_context(mock_request, settings)

    assert exc.value.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 7-14: Tenant Isolation Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_07_tenant_a_cannot_list_tenant_b_documents():
    """list_documents queries postgres with caller's tenant_id and KB, isolating other tenants."""
    postgres = MagicMock()
    postgres.fetch = AsyncMock(return_value=[])

    tenant_a_ctx = TenantContext(
        tenant_id="tenant_a",
        knowledge_base_id="kb_1",
        access_level="INTERNAL",
    )

    await list_documents(postgres=postgres, tenant_ctx=tenant_a_ctx, status_filter=None)

    # Verify query strictly filtered by tenant_a
    called_sql, *called_args = postgres.fetch.call_args[0]
    assert "tenant_id = $1" in called_sql
    assert called_args[0] == "tenant_a"
    assert called_args[1] == "kb_1"


@pytest.mark.asyncio
async def test_08_tenant_a_cannot_retrieve_tenant_b_document():
    """get_document_status returns HTTP 404 when querying another tenant's document."""
    postgres = MagicMock()
    # Postgres query filtered by tenant_id returns None for cross-tenant request
    postgres.fetchrow = AsyncMock(return_value=None)

    tenant_a_ctx = TenantContext(
        tenant_id="tenant_a",
        knowledge_base_id="kb_1",
    )
    doc_id = str(uuid.uuid4())

    with pytest.raises(HTTPException) as exc:
        await get_document_status(document_id=doc_id, postgres=postgres, tenant_ctx=tenant_a_ctx)

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_09_tenant_a_cannot_delete_tenant_b_document():
    """delete_document rejects deletion with HTTP 403 when document belongs to Tenant B."""
    postgres = MagicMock()
    doc_uuid = uuid.uuid4()
    # Document row belongs to tenant_b
    postgres.fetchrow = AsyncMock(return_value={
        "id": doc_uuid,
        "tenant_id": "tenant_b",
        "knowledge_base_id": "kb_1",
        "storage_path": "/data/raw/tenant_b_doc.pdf",
        "file_name": "tenant_b_doc.pdf",
    })

    tenant_a_ctx = TenantContext(
        tenant_id="tenant_a",
        claims={"is_admin": False},
    )
    mock_request = MagicMock()

    with pytest.raises(HTTPException) as exc:
        await delete_document(
            document_id=str(doc_uuid),
            postgres=postgres,
            tenant_ctx=tenant_a_ctx,
            request=mock_request,
        )

    assert exc.value.status_code == 403
    assert "Forbidden" in exc.value.detail


@pytest.mark.asyncio
async def test_10_default_tenant_cannot_bypass_document_ownership():
    """Default tenant cannot delete a document belonging to tenant_b."""
    postgres = MagicMock()
    doc_uuid = uuid.uuid4()
    postgres.fetchrow = AsyncMock(return_value={
        "id": doc_uuid,
        "tenant_id": "tenant_b",
        "knowledge_base_id": "kb_default",
        "storage_path": "/data/raw/b_doc.pdf",
        "file_name": "b_doc.pdf",
    })

    # Default tenant with no admin claims
    default_ctx = TenantContext(
        tenant_id="default",
        claims={"is_admin": False},
    )
    mock_request = MagicMock()

    with pytest.raises(HTTPException) as exc:
        await delete_document(
            document_id=str(doc_uuid),
            postgres=postgres,
            tenant_ctx=default_ctx,
            request=mock_request,
        )

    assert exc.value.status_code == 403
    assert "Forbidden" in exc.value.detail


@pytest.mark.asyncio
async def test_11_tenant_a_cannot_download_tenant_b_images(tmp_path):
    """Tenant A requesting Tenant B's image receives HTTP 404 without leaking file existence."""
    # Create fake image file on disk in processed storage
    doc_stem = "tenant_b_doc_123"
    figures_dir = tmp_path / "figures" / doc_stem
    figures_dir.mkdir(parents=True, exist_ok=True)
    img_file = figures_dir / "page1_img.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\nfakeimagecontent")

    settings = MagicMock(spec=Settings)
    settings.processed_storage_path = str(tmp_path)
    settings.is_production = True
    settings.api_key_auth_enabled = True

    postgres = MagicMock()
    # DB indicates document belongs to tenant_b
    postgres.fetchrow = AsyncMock(return_value={
        "id": uuid.uuid4(),
        "tenant_id": "tenant_b",
    })

    tenant_a_ctx = TenantContext(
        tenant_id="tenant_a",
        claims={"is_admin": False},
    )

    with pytest.raises(HTTPException) as exc:
        await get_document_image(
            image_path=f"figures/{doc_stem}/page1_img.png",
            tenant_ctx=tenant_a_ctx,
            postgres=postgres,
            settings=settings,
        )

    # Must be 404 to avoid leaking existence
    assert exc.value.status_code == 404


def test_12_spoofing_tenant_header_rejected():
    """Caller mapped to tenant_a attempting to send X-Tenant-ID: tenant_b gets HTTP 403."""
    mock_request = MagicMock()
    mock_request.headers = {
        "Authorization": "Bearer token_for_tenant_a",
        "X-Tenant-ID": "tenant_b",  # Attempted spoof
    }
    mock_request.client.host = "10.0.0.1"
    mock_request.url.path = "/api/v1/query"

    settings = MagicMock(spec=Settings)
    settings.api_key_auth_enabled = True
    settings.api_auth_token = ""
    settings.admin_api_token = "super_admin_only"
    settings.api_tokens_json = json.dumps({
        "token_for_tenant_a": {
            "tenant_id": "tenant_a",
            "allowed_tenants": ["tenant_a"],
            "is_admin": False,
        }
    })
    settings.is_production = False
    settings.default_tenant_id = "default"

    with pytest.raises(HTTPException) as exc:
        get_tenant_context(mock_request, settings)

    assert exc.value.status_code == 403
    assert "Forbidden" in exc.value.detail


def test_13_semantic_cache_isolates_tenants():
    """Cache namespaces for Tenant A and Tenant B with identical queries are completely isolated."""
    ctx_a = TenantContext(tenant_id="tenant_a", assistant_id="default", knowledge_base_id="kb1")
    ctx_b = TenantContext(tenant_id="tenant_b", assistant_id="default", knowledge_base_id="kb1")

    ns_a = _semantic_cache_namespace(ctx_a, kb_version=1)
    ns_b = _semantic_cache_namespace(ctx_b, kb_version=1)

    assert ns_a != ns_b
    assert len(ns_a) == 24


@pytest.mark.asyncio
async def test_14_background_jobs_isolated_by_tenant():
    """cancel_job fails with HTTP 404 when job belongs to another tenant."""
    postgres = MagicMock()
    postgres.fetchrow = AsyncMock(return_value=None)  # No job found under caller's tenant

    tenant_a_ctx = TenantContext(tenant_id="tenant_a", knowledge_base_id="kb1")
    job_uuid = str(uuid.uuid4())

    with pytest.raises(HTTPException) as exc:
        await cancel_job(job_id=job_uuid, postgres=postgres, tenant_ctx=tenant_a_ctx)

    assert exc.value.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# 15-16: Rate Limiting Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_15_cycling_tenant_headers_does_not_bypass_rate_limit():
    """Cycling X-Tenant-ID while presenting the same auth token yields the same rate-limit key."""
    middleware = RateLimitMiddleware(app=MagicMock())

    req1 = MagicMock()
    req1.headers = {
        "Authorization": "Bearer attacker_token",
        "X-Tenant-ID": "tenant_1",
    }
    req1.client.host = "1.2.3.4"
    req1.app.state.settings = None

    req2 = MagicMock()
    req2.headers = {
        "Authorization": "Bearer attacker_token",
        "X-Tenant-ID": "tenant_2",
    }
    req2.client.host = "1.2.3.4"
    req2.app.state.settings = None

    id1 = middleware._resolve_identity(req1)
    id2 = middleware._resolve_identity(req2)

    # Identifiers must be identical because token hash is used instead of client tenant header
    assert id1 == id2
    assert id1.startswith("auth:")


def test_16_spoofed_forwarded_ip_ignored_from_untrusted_source():
    """X-Forwarded-For is ignored when request arrives from an untrusted direct IP."""
    middleware = RateLimitMiddleware(app=MagicMock())

    req = MagicMock()
    req.headers = {
        "X-Forwarded-For": "8.8.8.8, 9.9.9.9",
    }
    # Direct client is an untrusted host (not in trusted_proxies)
    req.client.host = "203.0.113.50"
    mock_settings = MagicMock()
    mock_settings.trusted_proxies = ["127.0.0.1", "::1"]
    req.app.state.settings = mock_settings

    ident = middleware._resolve_identity(req)
    # Must use the direct untrusted IP, not the spoofed X-Forwarded-For!
    assert ident == "ip:203.0.113.50"


# ─────────────────────────────────────────────────────────────────────────────
# 17-18: Frontend Streaming Contract & PII / Injection Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_17_18_frontend_streaming_auth_and_error_contract():
    """Verify frontend streamChatCompletion code incorporates Bearer auth and 401/403 handlers."""
    frontend_chat_file = Path(__file__).resolve().parents[3] / "frontend" / "src" / "services" / "chat.ts"
    assert frontend_chat_file.exists()

    code = frontend_chat_file.read_text(encoding="utf-8")

    # 17. Auth header attached
    assert 'sessionStorage.getItem("erag_token")' in code
    assert 'headers["Authorization"] = `Bearer ${token}`' in code

    # 18. 401 and 403 handling
    assert "response.status === 401" in code
    assert "response.status === 403" in code


def test_pii_and_prompt_injection_delimiters():
    """Verify prompt template contains sandboxing markers and PII detector redacts sensitive data."""
    # Injection defense
    registry = PromptRegistry()
    template = registry.get_template()
    assert "<UNTRUSTED_EVIDENCE>" in template.user_prompt_template
    assert "</UNTRUSTED_EVIDENCE>" in template.user_prompt_template
    assert "passive data" in template.system_prompt.lower()

    # PII Redaction
    detector = SimpleRegexPIIDetector()
    res = detector.detect_and_redact("My email is alice@corp.com and ssn is 987-65-4321")
    assert res.contains_pii is True
    assert "[REDACTED_EMAIL]" in res.redacted_text
    assert "[REDACTED_SSN]" in res.redacted_text
