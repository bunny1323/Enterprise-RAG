"""
LangGraph Nodes for Query Workflow.
"""
import time
from app.agents.query_workflow.state import QueryWorkflowState
from app.agents.retrieval.agent import RetrievalAgent
from app.agents.retrieval.state import QueryState
from app.services.generation.citations import CitationService
from app.services.generation.hallucination import GroundednessVerificationService
from app.services.generation.llm import OllamaProvider
from app.services.cache.service import CacheService
from app.infrastructure.opa.client import OPAClient
from app.config.logging import get_logger

logger = get_logger(__name__)


class QueryNodes:
    """
    Stateless nodes for the LangGraph Query Workflow.
    """

    def __init__(
        self,
        retrieval_agent: RetrievalAgent,
        llm_provider: OllamaProvider,
        citation_svc: CitationService,
        verifier_svc: GroundednessVerificationService,
        opa_client: OPAClient,
        cache_svc: CacheService | None,
    ):
        self._retrieval_agent = retrieval_agent
        self._llm = llm_provider
        self._citation_svc = citation_svc
        self._verifier = verifier_svc
        self._opa = opa_client
        self._cache = cache_svc

    async def evaluate_policy(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Check OPA policies."""
        ctx = state["tenant_context"]
        decision = await self._opa.evaluate_retrieval_policy(ctx)
        
        if not decision.allowed:
            state["error_message"] = f"Access denied: {decision.denied_reason}"
        
        state["permitted_access_levels"] = decision.permitted_access_levels
        return state

    async def check_cache(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Check security-aware cache for existing response."""
        if not self._cache or state.get("error_message"):
            state["cache_hit"] = False
            return state

        ctx = state["tenant_context"]
        cached = await self._cache.get_response(ctx, state["query"])
        
        if cached:
            state["cache_hit"] = True
            state["answer"] = cached.get("answer", "")
            state["citations"] = cached.get("citations", [])
            state["confidence_level"] = cached.get("confidence", "HIGH")
            state["confidence_score"] = cached.get("confidence_score", 1.0)
            state["verification_status"] = cached.get("verification_status", "SUPPORTED")
            state["retrieval_strategy"] = cached.get("retrieval_strategy", "cache")
        else:
            state["cache_hit"] = False

        return state

    async def retrieve_evidence(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Execute parallel retrieval using RetrievalAgent."""
        if state.get("error_message") or state.get("cache_hit"):
            return state

        qstate = QueryState(
            query=state["query"],
            tenant_context=state["tenant_context"],
            top_k=state["top_k"]
        )

        result = await self._retrieval_agent.retrieve(
            qstate, 
            permitted_access_levels=state.get("permitted_access_levels")
        )

        state["intent"] = result.intent
        state["dense_results"] = result.dense_results
        state["bm25_results"] = result.bm25_results
        state["graph_results"] = result.graph_results
        state["fused_results"] = result.fused_results
        state["reranked_results"] = result.reranked_results
        state["confidence_level"] = result.confidence_level
        state["confidence_score"] = result.confidence_score
        state["retrieval_strategy"] = result.strategy_used

        # Context diversity deduplication based on content hash
        seen_hashes = set()
        deduped = []
        for item in state["reranked_results"]:
            h = item.metadata.get("content_hash", "")
            if h and h in seen_hashes:
                continue
            seen_hashes.add(h)
            deduped.append(item)
        
        state["reranked_results"] = deduped

        return state

    async def refine_query(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Refine query if confidence is low."""
        # Simple refinement: just append keyword modifier and increment retries
        state["retries"] = state.get("retries", 0) + 1
        state["query"] = state["query"] + " explain details"
        return state

    async def generate_response(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Generate response via LLM."""
        evidence = state.get("reranked_results", [])
        
        if not evidence:
            state["answer"] = "I don't have enough verified information in the knowledge base to answer this question."
            state["citations"] = []
            state["sources"] = []
            state["verification_status"] = "UNSUPPORTED"
            return state

        gen_result = await self._llm.generate(state["query"], evidence)
        state["answer"] = gen_result.answer
        return state

    async def verify_citations(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Map citations and verify groundedness."""
        evidence = state.get("reranked_results", [])
        if not evidence or not state.get("answer"):
            return state

        citations, sources = self._citation_svc.map_citations(state["answer"], evidence)
        groundedness = self._verifier.verify(state["answer"], evidence)

        state["citations"] = citations
        state["sources"] = sources
        state["verification_status"] = groundedness.verification_status

        # Update cache if it's a good response
        if self._cache and groundedness.verification_status == "SUPPORTED":
            resp_data = {
                "answer": state["answer"],
                "citations": [c.model_dump() for c in citations],
                "confidence": state["confidence_level"],
                "confidence_score": state["confidence_score"],
                "verification_status": state["verification_status"],
                "retrieval_strategy": state["retrieval_strategy"]
            }
            # We don't await this so it can run in background, but since it's a fast async call we can
            await self._cache.set_response(state["tenant_context"], state["query"], resp_data)

        return state
