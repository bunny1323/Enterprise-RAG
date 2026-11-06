import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

from app.api.dependencies import get_tenant_context
from app.api.dependencies import require_authenticated_access
from app.api.routes.health import health_check, readiness_check
from app.api.routes.images import get_document_image
from app.config.settings import Settings
from app.infrastructure.opa.client import OPAClient
from app.models.tenant import TenantContext


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        MagicMock(status_code=503),
        MagicMock(status_code=200, json=MagicMock(return_value={"result": None})),
    ],
)
async def test_invalid_or_unavailable_opa_response_denies_retrieval(response):
    client = OPAClient("https://policy.example.com")
    client._http.post = AsyncMock(return_value=response)

    decision = await client.evaluate_retrieval_policy(TenantContext())

    assert decision.allowed is False
    assert decision.permitted_access_levels == []
    await client.close()


@pytest.mark.asyncio
async def test_opa_transport_error_denies_retrieval():
    client = OPAClient("https://policy.example.com")
    client._http.post = AsyncMock(side_effect=httpx.ConnectError("offline"))

    decision = await client.evaluate_retrieval_policy(TenantContext())

    assert decision.allowed is False
    assert decision.denied_reason == "policy_engine_unavailable"
    await client.close()


@pytest.mark.asyncio
async def test_disabled_opa_allows_only_authenticated_public_reads_in_authorized_scopes():
    client = OPAClient(enabled=False)
    context = TenantContext(
        tenant_id="tenant-a",
        knowledge_base_id="manuals",
        claims={
            "is_authenticated": True,
            "bound_tenant_id": "tenant-a",
            "allowed_tenants": ["tenant-a"],
            "allowed_knowledge_bases": ["manuals"],
        },
    )

    decision = await client.evaluate_retrieval_policy(context)

    assert decision.allowed is True
    assert decision.permitted_access_levels == ["PUBLIC"]
    assert decision.permitted_tenants == ["tenant-a"]
    client._http.post = AsyncMock()
    await client.evaluate_retrieval_policy(context)
    client._http.post.assert_not_called()
    await client.close()


@pytest.mark.asyncio
async def test_disabled_opa_allows_unauthenticated_only_in_explicit_default_public_scope():
    settings = Settings(
        _env_file=None,
        app_env="production",
        api_key_auth_enabled=False,
        opa_enabled=False,
        default_tenant_id="tenant-primary",
    )
    request = SimpleNamespace(
        headers={},
        query_params={},
        client=SimpleNamespace(host="127.0.0.1"),
        url=SimpleNamespace(path="/api/v1/chat"),
    )
    context = get_tenant_context(request, settings)
    client = OPAClient(enabled=False)

    decision = await client.evaluate_retrieval_policy(context)
    protected_action = await client.evaluate_retrieval_policy(context, action="write")
    other_scope = context.model_copy(update={"knowledge_base_id": "other"})
    other_kb_decision = await client.evaluate_retrieval_policy(other_scope)

    assert decision.allowed is True
    assert decision.permitted_access_levels == ["PUBLIC"]
    assert decision.permitted_tenants == ["tenant-primary"]
    assert protected_action.allowed is False
    assert other_kb_decision.allowed is False
    await client.close()


def test_protected_data_operations_are_denied_in_production_auth_disabled_mode():
    context = TenantContext(
        claims={"api_auth_disabled_public_only": True, "is_authenticated": False}
    )

    with pytest.raises(HTTPException, match="unavailable while API token authentication is disabled"):
        require_authenticated_access(context, "Document upload")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("context", "action"),
    [
        (TenantContext(claims={"is_authenticated": False}), "read"),
        (
            TenantContext(
                tenant_id="tenant-b",
                claims={
                    "is_authenticated": True,
                    "bound_tenant_id": "tenant-a",
                    "allowed_tenants": ["tenant-a"],
                    "allowed_knowledge_bases": ["default"],
                },
            ),
            "read",
        ),
        (
            TenantContext(
                claims={
                    "is_authenticated": True,
                    "bound_tenant_id": "default",
                    "allowed_tenants": ["default"],
                    "allowed_knowledge_bases": ["default"],
                },
            ),
            "write",
        ),
        (
            TenantContext(
                knowledge_base_id="other",
                claims={
                    "is_authenticated": True,
                    "bound_tenant_id": "default",
                    "allowed_tenants": ["default"],
                    "allowed_knowledge_bases": ["default"],
                },
            ),
            "read",
        ),
    ],
)
async def test_disabled_opa_denies_unauthenticated_or_protected_requests(context, action):
    client = OPAClient(enabled=False)

    decision = await client.evaluate_retrieval_policy(context, action=action)

    assert decision.allowed is False
    assert decision.permitted_access_levels == []
    await client.close()


def test_production_opa_disabled_allows_missing_opa_url_and_api_tokens():
    settings = Settings(
        _env_file=None,
        app_env="production",
        api_key_auth_enabled=False,
        api_auth_token="",
        admin_api_token="",
        api_tokens_json="",
        opa_enabled=False,
        opa_url="",
    )

    assert settings.opa_enabled is False


@pytest.mark.asyncio
async def test_mapped_token_knowledge_base_scope_reaches_disabled_policy():
    settings = Settings(
        app_env="development",
        api_key_auth_enabled=True,
        api_tokens_json=json.dumps(
            {
                "mapped-token": {
                    "tenant_id": "tenant-a",
                    "allowed_knowledge_bases": ["manuals"],
                }
            }
        ),
    )
    request = SimpleNamespace(
        headers={
            "Authorization": "Bearer mapped-token",
            "X-Knowledge-Base-ID": "manuals",
        },
        query_params={},
        client=SimpleNamespace(host="127.0.0.1"),
        url=SimpleNamespace(path="/api/v1/chat"),
    )
    context = get_tenant_context(request, settings)
    client = OPAClient(enabled=False)

    decision = await client.evaluate_retrieval_policy(context)

    assert decision.allowed is True
    assert decision.permitted_access_levels == ["PUBLIC"]
    await client.close()


@pytest.mark.asyncio
async def test_health_reports_limited_public_only_mode_and_keeps_readiness_explicit():
    postgres = SimpleNamespace(fetchval=AsyncMock(return_value=1))
    weaviate = SimpleNamespace(is_connected=MagicMock(return_value=True))
    neo4j = SimpleNamespace(health_check=AsyncMock(return_value={"status": "ok"}))
    opa = OPAClient(enabled=False)
    state = SimpleNamespace(
        postgres=postgres,
        weaviate=weaviate,
        neo4j=neo4j,
        opa=opa,
        settings=SimpleNamespace(
            vision_provider="disabled",
            redis_url="",
            llm_provider="groq",
            llm_model="test-model",
            api_key_auth_enabled=False,
        ),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    payload = await health_check(request)
    ready = await readiness_check(request)

    assert payload["status"] == "limited"
    assert payload["security_mode"] == "API_AUTH_AND_OPA_DISABLED_PUBLIC_ONLY"
    assert payload["services"]["opa"].startswith("disabled")
    assert payload["services"]["api_authentication"].startswith("disabled")
    assert ready.status_code == 200
    assert b"OPA_DISABLED_PUBLIC_ONLY" in ready.body
    await opa.close()


@pytest.mark.asyncio
async def test_images_are_denied_when_opa_is_disabled():
    with pytest.raises(HTTPException) as exc_info:
        await get_document_image(
            image_path="figures/manual/page1.png",
            tenant_ctx=TenantContext(
                claims={"is_authenticated": True, "bound_tenant_id": "default"}
            ),
            postgres=None,
            settings=SimpleNamespace(opa_enabled=False),
        )

    assert exc_info.value.status_code == 403
