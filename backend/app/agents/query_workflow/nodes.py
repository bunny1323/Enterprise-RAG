"""
LangGraph Nodes for Query Workflow.
"""
import asyncio

from app.agents.query_workflow.state import QueryWorkflowState
from app.agents.retrieval.agent import RetrievalAgent
from app.agents.retrieval.state import QueryState
from app.config.logging import get_logger
from app.config.opentelemetry import get_tracer
from app.infrastructure.opa.client import OPAClient
from app.models.retrieval import SearchResult
from app.services.cache.semantic import SemanticCacheService
from app.services.cache.service import CacheService
from app.services.embeddings.service import EmbeddingService
from app.services.generation.citations import CitationService
from app.services.generation.hallucination import GroundednessVerificationService
from app.services.generation.llm import LLMProvider
from app.services.query.normalization import QueryNormalizationService
from app.services.retrieval.coverage import EvidenceCoverageService

logger = get_logger(__name__)
tracer = get_tracer()


def _deduplicate_evidence_by_source(results: list[SearchResult]) -> list[SearchResult]:
    seen_hashes: set[tuple[str, str]] = set()
    deduped: list[SearchResult] = []
    for item in results:
        content_hash = item.metadata.get("content_hash", "")
        if not content_hash:
            deduped.append(item)
            continue
        source = (
            item.document_id
            or item.file_name
            or item.metadata.get("file_name")
            or "unknown"
        )
        identity = (source, content_hash)
        if identity in seen_hashes:
            continue
        seen_hashes.add(identity)
        deduped.append(item)
    return deduped


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
        coverage_svc: EvidenceCoverageService | None = None,
        normalizer: QueryNormalizationService | None = None,
    ):
        self._retrieval_agent = retrieval_agent
        self._llm = llm_provider
        self._citation_svc = citation_svc
        self._verifier = verifier_svc
        self._opa = opa_client
        self._cache = cache_svc
        self._semantic_cache = semantic_cache_svc
        self._embedder = embedder
        self._coverage_svc = coverage_svc or EvidenceCoverageService()
        self._normalizer = normalizer or QueryNormalizationService()

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
            if not decision.allowed:
                state["refusal_reason"] = decision.denied_reason or "policy_denied"
            return state

    async def check_cache(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Check security-aware cache for existing response."""
        with tracer.start_as_current_span("QueryNodes.check_cache") as span:
            if not self._cache or state.get("error_message"):
                state["cache_hit"] = False
                return state

            ctx = state["tenant_context"]
            span.set_attribute("tenant_id", ctx.tenant_id)
            permitted_levels = state.get("permitted_access_levels", [])
            try:
                cached = await self._cache.get_response(
                    ctx,
                    state["query"],
                    permitted_access_levels=permitted_levels,
                )
            except Exception as err:
                logger.exception("query_cache.lookup_failed", error=str(err))
                state["cache_hit"] = False
                return state
            
            if cached:
                span.set_attribute("cache_hit", True)
                state["cache_hit"] = True
                state["answer"] = cached.get("answer", "")
                state["citations"] = cached.get("citations", [])
                state["sources"] = cached.get("sources", [])
                state["confidence_level"] = cached.get("confidence", "HIGH")
                state["confidence_score"] = cached.get("confidence_score", 1.0)
                state["verification_status"] = cached.get("verification_status", "SUPPORTED")
                state["retrieval_strategy"] = cached.get("retrieval_strategy", "exact_cache")
                return state

            # 2. Check semantic cache
            if self._semantic_cache and self._embedder:
                # Need kb_version for the semantic cache lookup
                try:
                    kb_ver = await self._cache.get_kb_version(
                        ctx.tenant_id, ctx.knowledge_base_id
                    )
                    loop = asyncio.get_running_loop()
                    query_vec = await loop.run_in_executor(
                        None, self._embedder.embed_query, state["query"]
                    )
                    if query_vec:
                        sem_cached = await self._semantic_cache.get(
                            ctx=ctx,
                            query_embedding=query_vec,
                            kb_version=kb_ver,
                            permitted_access_levels=permitted_levels,
                        )
                        if sem_cached:
                            span.set_attribute("cache_hit", True)
                            span.set_attribute("semantic_hit", True)
                            state["cache_hit"] = True
                            state["answer"] = sem_cached.get("answer", "")
                            state["citations"] = sem_cached.get("citations", [])
                            state["sources"] = sem_cached.get("sources", [])
                            state["confidence_level"] = sem_cached.get("confidence", "HIGH")
                            state["confidence_score"] = sem_cached.get("confidence_score", 1.0)
                            state["verification_status"] = sem_cached.get("verification_status", "SUPPORTED")
                            state["retrieval_strategy"] = sem_cached.get("retrieval_strategy", "semantic_cache")
                            return state
                except Exception as err:
                    logger.exception("query_cache.semantic_lookup_failed", error=str(err))

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

            retrieval_query = (
                state.get("refined_query")
                or state.get("original_query")
                or state["query"]
            )
            qstate = QueryState(
                query=retrieval_query,
                tenant_context=state["tenant_context"],
                top_k=state["top_k"]
            )

            result = await self._retrieval_agent.retrieve(
                qstate, 
                permitted_access_levels=state.get("permitted_access_levels")
            )

            state["intent"] = result.intent
            state["intent_confidence"] = getattr(result, "intent_confidence", 0.95)
            state["dense_results"] = result.dense_results
            state["bm25_results"] = result.bm25_results
            state["graph_results"] = result.graph_results
            state["fused_results"] = result.fused_results
            state["reranked_results"] = result.reranked_results
            state["confidence_level"] = result.confidence_level
            state["confidence_score"] = result.confidence_score
            state["retrieval_strategy"] = result.strategy_used
            state["normalized_query"] = getattr(
                result, "normalized_query", ""
            ) or self._normalizer.normalize(retrieval_query).clean_query

            span.set_attribute("confidence_level", result.confidence_level)
            span.set_attribute("confidence_score", result.confidence_score)
            span.set_attribute("retrieval_strategy", result.strategy_used)

            # Deduplicate within each source while preserving cross-manual lineage.
            state["reranked_results"] = _deduplicate_evidence_by_source(
                state["reranked_results"]
            )
            state["retrieved_evidence"] = list(state["reranked_results"])
            span.set_attribute("deduped_results_count", len(state["reranked_results"]))

            return state

    async def check_answerability(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Evaluate evidence coverage and gate hallucination-prone generation."""
        with tracer.start_as_current_span("QueryNodes.check_answerability") as span:
            if state.get("cache_hit") or state.get("error_message"):
                return state

            if state.get("intent") == "GREETING":
                state["evidence_coverage"] = "COMPLETE"
                state["answerable"] = True
                state["retrieval_trace"] = {
                    "intent": "GREETING",
                    "retrieval_strategy": "conversational",
                    "evidence_count": 0,
                    "evidence_coverage": "COMPLETE",
                    "answerable": True,
                    "reason": "Direct conversational greeting.",
                }
                return state

            norm = self._normalizer.normalize(
                state.get("original_query", state["query"])
            )
            evidence = state.get("reranked_results", [])
            coverage_result = self._coverage_svc.evaluate_coverage(norm, evidence)

            from app.services.generation.llm import get_context_summary
            ctx_summary = get_context_summary(evidence)

            state["evidence_coverage"] = coverage_result.coverage_status
            state["answerable"] = coverage_result.answerable
            state["retrieval_trace"] = {
                "intent": norm.intent,
                "entities_required": coverage_result.entities_required,
                "entities_found": coverage_result.entities_found,
                "retrieval_strategy": state.get("retrieval_strategy", "hybrid"),
                "evidence_count": len(evidence),
                "evidence_coverage": coverage_result.coverage_status,
                "answerable": coverage_result.answerable,
                "relationship_supported": coverage_result.relationship_supported,
                "inferred_only": coverage_result.inferred_only,
                "reason": coverage_result.reason,
                **ctx_summary,
            }

            span.set_attribute("evidence_coverage", coverage_result.coverage_status)
            span.set_attribute("answerable", coverage_result.answerable)

            if not coverage_result.answerable:
                state["answer"] = "The provided evidence does not contain enough information to answer this question."
                state["verification_status"] = "UNSUPPORTED"
                state["refusal_reason"] = coverage_result.reason or "insufficient_evidence"
                state["citations"] = []
                state["sources"] = []

            return state

    async def refine_query(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Refine query if confidence is low."""
        with tracer.start_as_current_span("QueryNodes.refine_query") as span:
            state["retries"] = state.get("retries", 0) + 1
            original_query = state.get("original_query", state["query"])
            normalized_query = state.get("normalized_query") or original_query
            state["refined_query"] = f"{normalized_query} explain details"
            span.set_attribute("retries", state["retries"])
            span.set_attribute("new_query", state["refined_query"])
            return state

    async def generate_response(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Generate response via LLM with complexity-based model routing."""
        with tracer.start_as_current_span("QueryNodes.generate_response") as span:
            if state.get("intent") == "GREETING":
                span.set_attribute("conversational_greeting", True)
                state["answer"] = (
                    "Hello! I am your Enterprise Document Intelligence Assistant. "
                    "How can I assist you today? You can ask me questions about your uploaded documents, "
                    "technical procedures, specifications, troubleshooting, or diagrams."
                )
                state["citations"] = []
                state["sources"] = []
                state["confidence_level"] = "HIGH"
                state["confidence_score"] = 1.0
                state["verification_status"] = "SUPPORTED"
                return state

            # If answerability gate already set answer due to insufficient evidence, skip LLM
            if state.get("answerable") is False and state.get("answer"):
                span.set_attribute("gated_insufficient_evidence", True)
                return state

            evidence = state.get("reranked_results", [])
            
            if not evidence:
                span.set_attribute("skipped", True)
                state["answer"] = "I don't have enough verified information in the knowledge base to answer this question."
                state["refusal_reason"] = "no_retrieved_evidence"
                state["citations"] = []
                state["sources"] = []
                state["verification_status"] = "UNSUPPORTED"
                return state

            from app.services.generation.llm import get_intent_system_instruction
            sys_inst = get_intent_system_instruction(state.get("intent"), query=state.get("query"))

            gen_result = await self._llm.generate(
                prompt=state["query"],
                evidence=evidence,
                system_instruction=sys_inst,
            )
            state["answer"] = gen_result.answer

            # Add data lineage mapping (source chunk UUIDs) to trace for auditability
            lineage_ids = [str(item.chunk_id) for item in evidence]
            span.set_attribute("lineage.source_chunks", lineage_ids)

            return state

    async def verify_citations(self, state: QueryWorkflowState) -> QueryWorkflowState:
        """Map citations and verify groundedness."""
        with tracer.start_as_current_span("QueryNodes.verify_citations") as span:
            if state.get("intent") == "GREETING":
                state["citations"] = []
                state["sources"] = []
                state["verification_status"] = "SUPPORTED"
                return state
            if state.get("answerable") is False:
                state["citations"] = []
                state["sources"] = []
                state["verification_status"] = "UNSUPPORTED"
                state["refusal_reason"] = state.get("refusal_reason") or "insufficient_evidence"
                span.set_attribute("gated_insufficient_evidence", True)
                return state
            evidence = state.get("reranked_results", [])
            if not evidence or not state.get("answer"):
                span.set_attribute("skipped", True)
                return state

            groundedness = self._verifier.verify(state["answer"], evidence)
            citations, sources = self._citation_svc.map_citations(
                state["answer"],
                evidence,
                supporting_chunk_ids=set(groundedness.supported_chunk_ids),
            )

            state["citations"] = citations
            state["sources"] = sources
            state["verification_status"] = groundedness.verification_status
            if groundedness.verification_status == "UNSUPPORTED":
                state["answer"] = "The provided evidence does not contain enough information to answer this question."
                state["refusal_reason"] = "unsupported_by_evidence"
                state["citations"] = []
                state["sources"] = []
                state["confidence_level"] = "LOW"
                state["confidence_score"] = 0.0
            elif groundedness.verification_status == "PARTIALLY_SUPPORTED":
                state["refusal_reason"] = "partially_supported"
                state["answer"] = " ".join(groundedness.supported_sentences)
                if not state["answer"]:
                    state["answer"] = "The provided evidence does not contain enough information to answer this question."
                    state["verification_status"] = "UNSUPPORTED"
                    state["citations"] = []
                    state["sources"] = []
                    state["confidence_level"] = "LOW"
                    state["confidence_score"] = 0.0
                else:
                    state["answer"] += " Other requested details were omitted because the retrieved evidence did not support them."
                state["confidence_level"] = "LOW"
                state["confidence_score"] = min(state.get("confidence_score", 0.0), groundedness.score)
            span.set_attribute("verification_status", groundedness.verification_status)

            # Update cache if it's a good response
            if self._cache and groundedness.verification_status == "SUPPORTED":
                resp_data = {
                    "answer": state["answer"],
                    "citations": [c.model_dump() for c in citations],
                    "sources": [source.model_dump() for source in state.get("sources", [])],
                    "confidence": state["confidence_level"],
                    "confidence_score": state["confidence_score"],
                    "verification_status": state["verification_status"],
                    "retrieval_strategy": state["retrieval_strategy"]
                }
                # We don't await this so it can run in background, but since it's a fast async call we can
                try:
                    await self._cache.set_response(
                        state["tenant_context"],
                        state.get("original_query", state["query"]),
                        resp_data,
                        permitted_access_levels=state.get("permitted_access_levels", []),
                    )
                except Exception as err:
                    logger.exception("query_cache.store_failed", error=str(err))

            return state
