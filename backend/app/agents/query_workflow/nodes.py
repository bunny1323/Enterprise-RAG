"""
LangGraph Nodes for Query Workflow.
"""
import time
import asyncio
from app.agents.query_workflow.state import QueryWorkflowState
from app.agents.retrieval.agent import RetrievalAgent
from app.agents.retrieval.state import QueryState
from app.services.generation.citations import CitationService
from app.services.generation.hallucination import GroundednessVerificationService
from app.services.generation.llm import LLMProvider
from app.services.cache.service import CacheService
from app.services.cache.semantic import SemanticCacheService
from app.services.embeddings.service import EmbeddingService
from app.infrastructure.opa.client import OPAClient
from app.config.logging import get_logger
from app.config.opentelemetry import get_tracer

logger = get_logger(__name__)
tracer = get_tracer()


class QueryNodes:
    """
    Stateless nodes for the LangGraph Query Workflow.
    """

    def __init__(
        self,
        retrieval_agent: RetrievalAgent,
        llm_provider: LLMProvider,
        citation_svc: CitationService,
        verifier_svc: GroundednessVerificationService,
        opa_client: OPAClient,
        cache_svc: CacheService | None,
        semantic_cache_svc: SemanticCacheService | None = None,
        embedder: EmbeddingService | None = None,
    ):
        self._retrieval_agent = retrieval_agent
        self._llm = llm_provider
        self._citation_svc = citation_svc
        self._verifier = verifier_svc
        self._opa = opa_client
        self._cache = cache_svc
        self._semantic_cache = semantic_cache_svc
        self._embedder = embedder

    async def evaluate_policy(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Check OPA policies."""
        with tracer.start_as_current_span("QueryNodes.evaluate_policy") as span:
            ctx = state["tenant_context"]
            span.set_attribute("tenant_id", ctx.tenant_id)
            decision = await self._opa.evaluate_retrieval_policy(ctx)
            
            if not decision.allowed:
                state["error_message"] = f"Access denied: {decision.denied_reason}"
                span.set_attribute("error", True)
                span.set_attribute("error_message", state["error_message"])
            
            state["permitted_access_levels"] = decision.permitted_access_levels
            return state

    async def check_cache(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Check security-aware cache for existing response."""
        with tracer.start_as_current_span("QueryNodes.check_cache") as span:
            if not self._cache or state.get("error_message"):
                state["cache_hit"] = False
                return state

            ctx = state["tenant_context"]
            span.set_attribute("tenant_id", ctx.tenant_id)
            cached = await self._cache.get_response(ctx, state["query"])
            
            if cached:
                span.set_attribute("cache_hit", True)
                state["cache_hit"] = True
                state["answer"] = cached.get("answer", "")
                state["citations"] = cached.get("citations", [])
                state["confidence_level"] = cached.get("confidence", "HIGH")
                state["confidence_score"] = cached.get("confidence_score", 1.0)
                state["verification_status"] = cached.get("verification_status", "SUPPORTED")
                state["retrieval_strategy"] = cached.get("retrieval_strategy", "exact_cache")
                return state

            # 2. Check semantic cache
            if self._semantic_cache and self._embedder:
                # Need kb_version for the semantic cache lookup
                kb_ver = await self._cache.get_kb_version(ctx.tenant_id, ctx.knowledge_base_id)
                # embed_query is synchronous (local BGE), run in executor
                loop = asyncio.get_event_loop()
                query_vec = await loop.run_in_executor(None, self._embedder.embed_query, state["query"])
                if query_vec:
                    sem_cached = await self._semantic_cache.get(
                        ctx=ctx,
                        query_embedding=query_vec,
                        kb_version=kb_ver,
                    )
                    if sem_cached:
                        span.set_attribute("cache_hit", True)
                        span.set_attribute("semantic_hit", True)
                        state["cache_hit"] = True
                        state["answer"] = sem_cached.get("answer", "")
                        state["citations"] = sem_cached.get("citations", [])
                        state["confidence_level"] = sem_cached.get("confidence", "HIGH")
                        state["confidence_score"] = sem_cached.get("confidence_score", 1.0)
                        state["verification_status"] = sem_cached.get("verification_status", "SUPPORTED")
                        state["retrieval_strategy"] = sem_cached.get("retrieval_strategy", "semantic_cache")
                        return state

            span.set_attribute("cache_hit", False)
            state["cache_hit"] = False
            return state

    async def retrieve_evidence(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Execute parallel retrieval using RetrievalAgent."""
        with tracer.start_as_current_span("QueryNodes.retrieve_evidence") as span:
            if state.get("error_message") or state.get("cache_hit"):
                span.set_attribute("skipped", True)
                return state

            span.set_attribute("tenant_id", state["tenant_context"].tenant_id)

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

            span.set_attribute("confidence_level", result.confidence_level)
            span.set_attribute("confidence_score", result.confidence_score)
            span.set_attribute("retrieval_strategy", result.strategy_used)

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
            span.set_attribute("deduped_results_count", len(deduped))

            return state

    async def refine_query(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Refine query if confidence is low."""
        with tracer.start_as_current_span("QueryNodes.refine_query") as span:
            # Simple refinement: just append keyword modifier and increment retries
            state["retries"] = state.get("retries", 0) + 1
            state["query"] = state["query"] + " explain details"
            span.set_attribute("retries", state["retries"])
            span.set_attribute("new_query", state["query"])
            return state

    async def generate_response(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Generate response via LLM with complexity-based model routing."""
        with tracer.start_as_current_span("QueryNodes.generate_response") as span:
            evidence = state.get("reranked_results", [])
            
            if not evidence:
                span.set_attribute("skipped", True)
                state["answer"] = "I don't have enough verified information in the knowledge base to answer this question."
                state["citations"] = []
                state["sources"] = []
                state["verification_status"] = "UNSUPPORTED"
                return state

            gen_result = await self._llm.generate(
                prompt=state["query"],
                evidence=evidence,
            )
            state["answer"] = gen_result.answer

            # Add data lineage mapping (source chunk UUIDs) to trace for auditability
            lineage_ids = [str(item.chunk_id) for item in evidence]
            span.set_attribute("lineage.source_chunks", lineage_ids)

            return state

    async def verify_citations(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Map citations and verify groundedness."""
        with tracer.start_as_current_span("QueryNodes.verify_citations") as span:
            evidence = state.get("reranked_results", [])
            if not evidence or not state.get("answer"):
                span.set_attribute("skipped", True)
                return state

            citations, sources = self._citation_svc.map_citations(state["answer"], evidence)
            groundedness = self._verifier.verify(state["answer"], evidence)

            state["citations"] = citations
            state["sources"] = sources
            state["verification_status"] = groundedness.verification_status
            span.set_attribute("verification_status", groundedness.verification_status)

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
