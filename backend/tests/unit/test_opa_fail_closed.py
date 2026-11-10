import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_settings_dep, get_tenant_context, require_authenticated_access
from app.api.routes import query as query_routes
from app.api.routes.documents import delete_document, upload_document
from app.api.routes.health import health_check, readiness_check
from app.api.routes.images import get_document_image
from app.api.routes.jobs import cancel_job
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


def test_opa_enabled_environment_flag_is_read_as_disabled(monkeypatch):
    monkeypatch.setenv("OPA_ENABLED", "false")

    settings = Settings(
        _env_file=None,
        app_env="production",
        api_key_auth_enabled=False,
        database_url="postgresql://localhost/test",
        weaviate_url="https://weaviate.example.com",
        weaviate_api_key="test-key",
        neo4j_password="test-password",
    )

    assert settings.opa_enabled is False


@pytest.mark.asyncio
async def test_disabled_opa_allows_only_authenticated_public_reads_in_authorized_scopes():
    client = OPAClient(enabled=False)
    client._http.post = AsyncMock()
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
    client._http.post.assert_not_awaited()

    assert decision.allowed is True
    assert decision.permitted_access_levels == ["PUBLIC"]
    assert decision.permitted_tenants == ["tenant-a"]
    await client.evaluate_retrieval_policy(context)
    client._http.post.assert_not_awaited()
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
    client._http.post = AsyncMock()
    client._http.get = AsyncMock()

    decision = await client.evaluate_retrieval_policy(context)
    protected_action = await client.evaluate_retrieval_policy(context, action="write")
    other_scope = context.model_copy(update={"knowledge_base_id": "other"})
    other_kb_decision = await client.evaluate_retrieval_policy(other_scope)
    health_check = await client.health_check()

    assert decision.allowed is True
    assert decision.permitted_access_levels == ["PUBLIC"]
    assert decision.permitted_tenants == ["tenant-primary"]
    assert protected_action.allowed is False
    assert other_kb_decision.allowed is False
    assert health_check is False
    client._http.post.assert_not_awaited()
    client._http.get.assert_not_awaited()
    await client.close()


@pytest.mark.parametrize(
    "operation",
    [
        "Document upload",
        "Document deletion",
        "Job cancellation",
        "Feedback submission",
        "Administrative operation",
    ],
)
def test_protected_data_operations_are_denied_in_production_auth_disabled_mode(operation):
    context = TenantContext(
        claims={"api_auth_disabled_public_only": True, "is_authenticated": False}
    )

    with pytest.raises(
        HTTPException,
        match="unavailable while API token authentication is disabled",
    ):
        require_authenticated_access(context, operation)


@pytest.mark.asyncio
async def test_anonymous_feedback_write_is_denied():
    context = TenantContext(
        claims={"api_auth_disabled_public_only": True, "is_authenticated": False}
    )

    with pytest.raises(HTTPException, match="Feedback submission is unavailable"):
        await query_routes.submit_feedback(
            query_routes.FeedbackRequest(query_id="query-1", rating=1),
            context,
        )


@pytest.mark.asyncio
async def test_anonymous_document_and_job_mutations_are_denied_before_backend_access():
    context = TenantContext(
        claims={"api_auth_disabled_public_only": True, "is_authenticated": False}
    )

    with pytest.raises(HTTPException, match="Document upload is unavailable"):
        await upload_document(None, tenant_ctx=context, supervisor=None)
    with pytest.raises(HTTPException, match="Document deletion is unavailable"):
        await delete_document("document-id", None, context, None)
    with pytest.raises(HTTPException, match="Job cancellation is unavailable"):
        await cancel_job("job-id", None, context)


@pytest.mark.asyncio
@pytest.mark.parametrize("access_level", ["INTERNAL", "RESTRICTED"])
async def test_disabled_opa_denies_protected_access_levels(access_level):
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
    context = get_tenant_context(request, settings).model_copy(
        update={"access_level": access_level}
    )
    client = OPAClient(enabled=False)
    client._http.post = AsyncMock()

    decision = await client.evaluate_retrieval_policy(context)

    assert decision.allowed is False
    assert decision.permitted_access_levels == []
    client._http.post.assert_not_awaited()
    await client.close()


def test_anonymous_headers_cannot_override_tenant_or_knowledge_base():
    settings = Settings(
        _env_file=None,
        app_env="production",
        api_key_auth_enabled=False,
        opa_enabled=False,
        default_tenant_id="tenant-primary",
    )
    request = SimpleNamespace(
        headers={
            "X-Tenant-ID": "tenant-other",
            "X-Knowledge-Base-ID": "other",
            "X-Access-Level": "RESTRICTED",
        },
        query_params={},
        client=SimpleNamespace(host="127.0.0.1"),
        url=SimpleNamespace(path="/api/v1/chat"),
    )

    with pytest.raises(HTTPException) as exc_info:
        get_tenant_context(request, settings)

    assert exc_info.value.status_code == 403

    request.headers = {"X-Knowledge-Base-ID": "other"}
    with pytest.raises(HTTPException) as exc_info:
        get_tenant_context(request, settings)
    assert exc_info.value.status_code == 403

    request.headers = {"X-Access-Level": "RESTRICTED"}
    context = get_tenant_context(request, settings)
    assert context.tenant_id == "tenant-primary"
    assert context.knowledge_base_id == "default"
    assert context.access_level == "PUBLIC"


@pytest.mark.asyncio
async def test_opa_enabled_calls_configured_policy_endpoint():
    client = OPAClient("https://policy.example.com", enabled=True)
    client._http.post = AsyncMock(
        return_value=MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "result": {
                        "allowed": True,
                        "permitted_access_levels": ["PUBLIC"],
                    }
                }
            ),
        )
    )
    context = TenantContext(
        tenant_id="tenant-a",
        claims={"is_authenticated": True},
    )

    decision = await client.evaluate_retrieval_policy(context)

    assert decision.allowed is True
    client._http.post.assert_awaited_once()
    assert client._http.post.await_args.args[0] == (
        "https://policy.example.com/v1/data/rag/authz/allow"
    )
    await client.close()


@pytest.mark.asyncio
async def test_authorized_chat_policy_is_not_rechecked_by_langgraph(monkeypatch):
    settings = Settings(
        _env_file=None,
        app_env="production",
        api_key_auth_enabled=False,
        opa_enabled=False,
        default_tenant_id="tenant-primary",
        database_url="postgresql://localhost/test",
        weaviate_url="https://weaviate.example.com",
        weaviate_api_key="test-key",
        neo4j_password="test-password",
        pii_detection_enabled=False,
    )
    opa = OPAClient(enabled=False)
    opa._http.post = AsyncMock()
    evaluate_policy = AsyncMock(wraps=opa.evaluate_retrieval_policy)
    opa.evaluate_retrieval_policy = evaluate_policy
    app = FastAPI()
    app.include_router(query_routes.router)
    app.dependency_overrides[get_settings_dep] = lambda: settings
    app.state.postgres = MagicMock()
    app.state.checkpointer = None
    app.state.redis = None
    app.state.embedder = MagicMock()
    app.state.llm_provider = MagicMock()
    app.state.weaviate = MagicMock()
    app.state.neo4j = MagicMock()
    app.state.opa = opa

    async def invoke_authorized_workflow(nodes, state):
        assert state["tenant_context"].tenant_id == "tenant-primary"
        assert state["tenant_context"].knowledge_base_id == "default"
        assert state["tenant_context"].access_level == "PUBLIC"
        assert state["permitted_access_levels"] == ["PUBLIC"]
        assert state["policy_authorized"] is True

        authorized_state = await nodes.evaluate_policy(state)
        assert authorized_state.get("error_message") is None
        return {
            **authorized_state,
            "answer": "Answer based on public evidence.",
            "intent": "GENERAL_QA",
            "intent_confidence": 1.0,
            "reranked_results": [],
            "citations": [],
            "sources": [],
            "confidence_level": "HIGH",
            "confidence_score": 1.0,
            "verification_status": "SUPPORTED",
            "retrieval_strategy": "dense",
        }

    class FakeGraph:
        def __init__(self, nodes):
            self.nodes = nodes

        async def ainvoke(self, state, config):
            return await invoke_authorized_workflow(self.nodes, state)

    def fake_build_query_graph(nodes, checkpointer=None):
        return FakeGraph(nodes)

    monkeypatch.setattr(query_routes, "_build_retrieval_agent", lambda *args: MagicMock())
    monkeypatch.setattr(query_routes, "build_query_graph", fake_build_query_graph)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post("/api/v1/chat", json={"query": "What is in the manual?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Answer based on public evidence."
    evaluate_policy.assert_awaited_once()
    opa._http.post.assert_not_awaited()
    await opa.close()


@pytest.mark.asyncio
async def test_disabled_opa_chat_rejects_anonymous_tenant_and_kb_overrides(monkeypatch):
    settings = Settings(
        _env_file=None,
        app_env="production",
        api_key_auth_enabled=False,
        opa_enabled=False,
        default_tenant_id="tenant-primary",
    )
    app = FastAPI()
    app.include_router(query_routes.router)
    app.dependency_overrides[get_settings_dep] = lambda: settings
    app.state.postgres = MagicMock()
    app.state.checkpointer = None
    app.state.opa = OPAClient(enabled=False)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        tenant_response = await client.post(
            "/api/v1/chat",
            headers={"X-Tenant-ID": "tenant-other"},
            json={"query": "test"},
        )
        knowledge_base_response = await client.post(
            "/api/v1/chat",
            headers={"X-Knowledge-Base-ID": "other"},
            json={"query": "test"},
        )

    assert tenant_response.status_code == 403
    assert knowledge_base_response.status_code == 403
    await app.state.opa.close()


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
