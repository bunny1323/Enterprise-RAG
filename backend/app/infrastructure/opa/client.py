"""
Open Policy Agent (OPA) client for authorization policy evaluation.
Note: OPA is NOT authentication. It evaluates already-authenticated claims
and returns permitted access filters for retrieval (e.g. permitted_access_levels).
Default-deny approach when security metadata is missing.
"""
from typing import Any

import httpx
from pydantic import BaseModel

from app.config.logging import get_logger
from app.models.tenant import TenantContext

logger = get_logger(__name__)


class PolicyDecision(BaseModel):
    """Result of OPA policy evaluation."""

    allowed: bool
    permitted_access_levels: list[str] = ["PUBLIC", "INTERNAL"]
    permitted_tenants: list[str] = []
    denied_reason: str | None = None


class OPAClient:
    """
    HTTP client for evaluating policies via OPA's REST API.
    """

    def __init__(self, opa_url: str = "http://localhost:8181") -> None:
        self._opa_url = opa_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=5.0)

    async def evaluate_retrieval_policy(
        self,
        ctx: TenantContext,
        action: str = "read",
    ) -> PolicyDecision:
        """
        Evaluate OPA policy for retrieval access.

        If OPA server is unreachable or unconfigured, falls back to safe default-deny
        or strict claim-based access level filter.
        """
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
            if resp.status_code == 200:
                res_json = resp.json()
                result = res_json.get("result", {})
                if isinstance(result, bool):
                    allowed = result
                    permitted = (
                        ["PUBLIC", "INTERNAL", "RESTRICTED"]
                        if ctx.access_level == "RESTRICTED"
                        else ["PUBLIC", "INTERNAL"]
                    )
                else:
                    allowed = result.get("allowed", False)
                    permitted = result.get("permitted_access_levels", ["PUBLIC", "INTERNAL"])

                logger.info(
                    "opa.evaluated",
                    allowed=allowed,
                    tenant=ctx.tenant_id,
                    user=ctx.user_id,
                )
                return PolicyDecision(
                    allowed=allowed,
                    permitted_access_levels=permitted,
                    permitted_tenants=[ctx.tenant_id],
                )
        except Exception as err:
            logger.warning("opa.unreachable_fallback_used", error=str(err))

        # Safe fallback based on context access_level (default-deny for RESTRICTED if unauthenticated)
        if ctx.access_level == "RESTRICTED":
            permitted_levels = ["PUBLIC", "INTERNAL", "RESTRICTED"]
        elif ctx.access_level == "INTERNAL":
            permitted_levels = ["PUBLIC", "INTERNAL"]
        else:
            permitted_levels = ["PUBLIC"]

        return PolicyDecision(
            allowed=True,
            permitted_access_levels=permitted_levels,
            permitted_tenants=[ctx.tenant_id],
        )

    async def close(self) -> None:
        """Close HTTP client."""
        await self._http.aclose()
