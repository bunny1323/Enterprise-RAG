"""
Open Policy Agent (OPA) client for authorization policy evaluation.
Note: OPA is NOT authentication. It evaluates already-authenticated claims
and returns permitted access filters for retrieval (e.g. permitted_access_levels).
Default-deny approach when security metadata is missing.
"""

import httpx
from pydantic import BaseModel, Field

from app.config.logging import get_logger
from app.models.tenant import TenantContext

logger = get_logger(__name__)


class PolicyDecision(BaseModel):
    """Result of OPA policy evaluation."""

    allowed: bool
    permitted_access_levels: list[str] = Field(default_factory=list)
    permitted_tenants: list[str] = Field(default_factory=list)
    denied_reason: str | None = None


class OPAClient:
    """
    HTTP client for evaluating policies via OPA's REST API.
    """

    def __init__(
        self,
        opa_url: str = "http://localhost:8181",
        auth_token: str = "",
        enabled: bool = True,
    ) -> None:
        self._opa_url = opa_url.rstrip("/")
        self.enabled = enabled
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        self._http = httpx.AsyncClient(timeout=5.0, headers=headers)
        if not enabled:
            logger.warning(
                "opa.disabled_public_only",
                message="OPA is disabled; only authenticated PUBLIC retrieval in authorized tenant/knowledge-base scopes is allowed.",
            )

    async def evaluate_retrieval_policy(
        self,
        ctx: TenantContext,
        action: str = "read",
    ) -> PolicyDecision:
        """
        Evaluate OPA policy for retrieval access.

        A missing, unreachable, malformed, or non-successful policy response denies
        retrieval. Claims never substitute for a failed policy decision.
        """
        if not self.enabled:
            claims = ctx.claims
            allowed_tenants = claims.get("allowed_tenants")
            tenant_is_allowed = (
                claims.get("bound_tenant_id") == ctx.tenant_id
                or (
                    isinstance(allowed_tenants, list)
                    and ("*" in allowed_tenants or ctx.tenant_id in allowed_tenants)
                )
            )
            allowed_knowledge_bases = claims.get("allowed_knowledge_bases")
            knowledge_base_is_allowed = (
                isinstance(allowed_knowledge_bases, list)
                and ctx.knowledge_base_id in allowed_knowledge_bases
            )
            if (
                action != "read"
                or claims.get("is_authenticated") is not True
                or not ctx.tenant_id
                or not tenant_is_allowed
                or not knowledge_base_is_allowed
            ):
                return PolicyDecision(
                    allowed=False,
                    denied_reason="opa_disabled_protected_operation",
                )
            return PolicyDecision(
                allowed=True,
                permitted_access_levels=["PUBLIC"],
                permitted_tenants=[ctx.tenant_id],
            )

        payload = {
            "input": {
                "tenant_id": ctx.tenant_id,
                "assistant_id": ctx.assistant_id,
                "knowledge_base_id": ctx.knowledge_base_id,
                "access_level": ctx.access_level,
                "user_id": ctx.user_id,
                "claims": ctx.claims,
                "action": action,
            }
        }

        try:
            resp = await self._http.post(
                f"{self._opa_url}/v1/data/rag/authz/allow",
                json=payload,
            )
            if resp.status_code != 200:
                logger.error(
                    "opa.policy_request_rejected",
                    status_code=resp.status_code,
                    tenant=ctx.tenant_id,
                )
                return PolicyDecision(
                    allowed=False,
                    permitted_access_levels=[],
                    denied_reason="policy_engine_unavailable",
                )

            res_json = resp.json()
            result = res_json.get("result") if isinstance(res_json, dict) else None
            if isinstance(result, bool):
                allowed = result
                is_admin = ctx.claims.get("is_admin") is True
                permitted = (
                    ["PUBLIC", "INTERNAL", "RESTRICTED"]
                    if is_admin and ctx.access_level == "RESTRICTED"
                    else ["PUBLIC", "INTERNAL"]
                    if ctx.access_level == "INTERNAL"
                    else ["PUBLIC"]
                )
                denied_reason = None
            elif isinstance(result, dict):
                allowed = result.get("allowed") is True
                raw_permitted = result.get("permitted_access_levels", [])
                permitted = (
                    [level for level in raw_permitted if level in {"PUBLIC", "INTERNAL", "RESTRICTED"}]
                    if isinstance(raw_permitted, list)
                    else []
                )
                denied_reason = result.get("denied_reason")
            else:
                return PolicyDecision(
                    allowed=False,
                    permitted_access_levels=[],
                    denied_reason="invalid_policy_response",
                )

            if not allowed or not permitted:
                return PolicyDecision(
                    allowed=False,
                    permitted_access_levels=[],
                    denied_reason=denied_reason or "policy_denied",
                )

            logger.info(
                "opa.evaluated",
                allowed=True,
                tenant=ctx.tenant_id,
                user=ctx.user_id,
            )
            return PolicyDecision(
                allowed=True,
                permitted_access_levels=permitted,
                permitted_tenants=[ctx.tenant_id],
            )
        except (httpx.HTTPError, ValueError, TypeError) as err:
            logger.error("opa.policy_evaluation_failed", error=str(err))
            return PolicyDecision(
                allowed=False,
                permitted_access_levels=[],
                denied_reason="policy_engine_unavailable",
            )

    async def close(self) -> None:
        """Close HTTP client."""
        await self._http.aclose()

    async def health_check(self) -> bool:
        """Check that the configured policy service is reachable."""
        try:
            response = await self._http.get(f"{self._opa_url}/health")
        except httpx.HTTPError as err:
            logger.error("opa.health_check_failed", error=str(err))
            return False
        if response.status_code != 200:
            logger.error(
                "opa.health_check_failed",
                status_code=response.status_code,
            )
            return False
        return True
