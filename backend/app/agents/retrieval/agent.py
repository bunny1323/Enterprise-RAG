"""
Retrieval Agent.

Orchestrates multi-channel retrieval (Dense + BM25 + Graph),
fusion via RRF, reranking, and hierarchical retrieval expansion.

Calls stateless services — does not reimplement business logic directly.
"""

import asyncio
from typing import Any
from app.agents.retrieval.state import QueryState
from app.config.logging import get_logger
from app.services.cache.service import CacheService
from app.services.confidence.service import ConfidenceScoringService
from app.services.embeddings.service import EmbeddingService
from app.services.query.normalization import QueryNormalizationService
from app.services.retrieval.bm25_search import BM25SearchService
from app.services.retrieval.dense_search import DenseSearchService
from app.services.retrieval.graph_search import GraphSearchService
from app.services.retrieval.hierarchical import HierarchicalRetrievalService
from app.services.retrieval.reranking import RRFRerankerService
from app.services.retrieval.structure_search import StructureSearchService
from app.services.query.decomposition import QueryDecompositionService
from app.utils.fusion import reciprocal_rank_fusion
from app.models.retrieval import SearchResult

from app.config.opentelemetry import get_tracer

logger = get_logger(__name__)
tracer = get_tracer()

class RetrievalAgent:
    """
    Agent responsible for retrieval strategy decisions and pipeline execution.
    """

    def __init__(
        self,
        embedder: EmbeddingService,
        dense_svc: DenseSearchService,
        bm25_svc: BM25SearchService,
        graph_svc: GraphSearchService,
        reranker: RRFRerankerService,
        normalizer: QueryNormalizationService,
        confidence_svc: ConfidenceScoringService,
        hierarchical_svc: HierarchicalRetrievalService,
        cache_svc: CacheService | None = None,
        weaviate_client: Any | None = None,
        structure_svc: StructureSearchService | None = None,
        decomposer_svc: QueryDecompositionService | None = None,
    ) -> None:
        self._embedder = embedder
        self._dense_svc = dense_svc
        self._bm25_svc = bm25_svc
        self._graph_svc = graph_svc
        self._reranker = reranker
        self._normalizer = normalizer
        self._confidence_svc = confidence_svc
        self._hierarchical_svc = hierarchical_svc
        self._cache_svc = cache_svc
        self._weaviate = weaviate_client
        self._structure_svc = structure_svc
        self._decomposer_svc = decomposer_svc or QueryDecompositionService()

    async def retrieve(
        self,
        query_state: QueryState,
        permitted_access_levels: list[str] | None = None,
    ) -> QueryState:
        """
        Execute multi-channel retrieval, fusion, reranking,
        and hierarchical parent expansion.
        """
        ctx = query_state.tenant_context
        norm = self._normalizer.normalize(query_state.query)
        intent = norm.intent

        with tracer.start_as_current_span("RetrievalAgent.retrieve") as span:
            span.set_attribute("tenant_id", ctx.tenant_id)
            span.set_attribute("query", norm.clean_query)
            span.set_attribute("intent", intent)

            logger.info(
                "retrieval_agent.start",
                query=norm.clean_query[:50],
                intent=intent,
                tenant=ctx.tenant_id,
            )

            # ── 0. STRUCTURAL RETRIEVAL FAST-PATH ─────────────────────────────
            # For COUNT_QUERY and LIST_QUERY (e.g., "How many major sections...",
            # "What are the major sections?"), query the canonical document_structure index.
            if self._structure_svc:
                if intent == "COUNT_QUERY":
                    count = await self._structure_svc.count_sections(
                        tenant_id=ctx.tenant_id,
                        knowledge_base_id=ctx.knowledge_base_id,
                    )
                    sections = await self._structure_svc.list_sections(
                        tenant_id=ctx.tenant_id,
                        knowledge_base_id=ctx.knowledge_base_id,
                    )
                    evidence = [
                        SearchResult(
                            chunk_id="struct::count::sections",
                            score=1.0,
                            content=(
                                f"There are {count} major sections described in the service manual:\n"
                                + "\n".join(f"- {s.section}" for s in sections)
                            ),
                            page_number=1,
                            section="DOCUMENT STRUCTURE - MAJOR SECTIONS",
                            metadata={"source": "document_structure", "count": count},
                        )
                    ] + sections
                    return query_state.model_copy(
                        update={
                            "intent": intent,
                            "fused_results": evidence,
                            "reranked_results": evidence,
                            "confidence_level": "HIGH",
                            "confidence_score": 1.0,
                            "strategy_used": "document_structure_count",
                        }
                    )

                elif intent == "LIST_QUERY":
                    sections = await self._structure_svc.list_sections(
                        tenant_id=ctx.tenant_id,
                        knowledge_base_id=ctx.knowledge_base_id,
                    )
                    if sections:
                        return query_state.model_copy(
                            update={
                                "intent": intent,
                                "fused_results": sections,
                                "reranked_results": sections,
                                "confidence_level": "HIGH",
                                "confidence_score": 1.0,
                                "strategy_used": "document_structure_list",
                            }
                        )

                elif intent == "PAGE_NUMBER_FORMAT":
                    notation = norm.extracted_notation or "2-3"
                    pf_results = await self._structure_svc.lookup_page_format(
                        notation=notation,
                        tenant_id=ctx.tenant_id,
                        knowledge_base_id=ctx.knowledge_base_id,
                    )
                    if pf_results:
                        return query_state.model_copy(
                            update={
                                "intent": intent,
                                "fused_results": pf_results,
                                "reranked_results": pf_results,
                                "confidence_level": "HIGH",
                                "confidence_score": 1.0,
                                "strategy_used": "document_structure_page_format",
                            }
                        )

            # 1. Generate query embedding (with caching)
            query_vec = None
            if self._cache_svc:
                query_vec = await self._cache_svc.get_query_embedding(
                    ctx,
                    norm.clean_query,
                )

            if not query_vec:
                # LocalEmbeddingProvider is synchronous, so run embedding in executor.
                loop = asyncio.get_event_loop()

                query_vectors = await loop.run_in_executor(
                    None,
                    self._embedder.embed_batch,
                    [norm.clean_query],
                )

                query_vec = query_vectors[0] if query_vectors else []

                if query_vec and self._cache_svc:
                    await self._cache_svc.set_query_embedding(
                        ctx,
                        norm.clean_query,
                        query_vec,
                    )

            # 2. Parallel retrieval channels
            tasks = {
                "dense": self._dense_svc.search(
                    query_vector=query_vec,
                    top_k=query_state.top_k * 2,
                    tenant_id=ctx.tenant_id,
                    knowledge_base_id=ctx.knowledge_base_id,
                    permitted_access_levels=permitted_access_levels,
                ),
                "bm25": self._bm25_svc.search(
                    query_text=norm.clean_query,
                    top_k=query_state.top_k * 2,
                    tenant_id=ctx.tenant_id,
                    knowledge_base_id=ctx.knowledge_base_id,
                    permitted_access_levels=permitted_access_levels,
                ),
            }

            # ── Multi-hop / Relationship sub-query decomposition ───────────────
            extra_bm25: list[SearchResult] = []
            if intent in ("MULTI_HOP", "RELATIONSHIP"):
                sub_queries = self._decomposer_svc.decompose(norm)
                for sq in sub_queries:
                    if sq.query != norm.clean_query:
                        try:
                            sub_res = await self._bm25_svc.search(
                                query_text=sq.query,
                                top_k=5,
                                tenant_id=ctx.tenant_id,
                                knowledge_base_id=ctx.knowledge_base_id,
                                permitted_access_levels=permitted_access_levels,
                            )
                            extra_bm25.extend(sub_res)
                        except Exception as e:
                            logger.warning("retrieval_agent.subquery_bm25_failed", sub_query=sq.query, error=str(e))

            if intent in ("RELATIONSHIP", "ROOT_CAUSE_ANALYSIS", "COMPONENT_IDENTIFICATION", "PREDICTIVE_MAINTENANCE"):
                entity_query = (
                    norm.extracted_entities[0]
                    if norm.extracted_entities
                    else norm.clean_query[:30]
                )

                tasks["graph"] = self._graph_svc.search(
                    entity_name=entity_query,
                    tenant_id=ctx.tenant_id,
                    knowledge_base_id=ctx.knowledge_base_id,
                )

            # Execute retrieval channels concurrently.
            results = await asyncio.gather(
                *tasks.values(),
                return_exceptions=True,
            )

            dense_results = []
            bm25_results = []
            graph_results = []
            degraded_channels = []

            for i, key in enumerate(tasks.keys()):
                res = results[i]

                if isinstance(res, Exception):
                    logger.error(
                        f"retrieval_agent.{key}_failed",
                        error=str(res),
                    )
                    degraded_channels.append(key)
                    continue

                if key == "dense":
                    dense_results = res
                elif key == "bm25":
                    bm25_results = res
                elif key == "graph":
                    graph_results = res

            channel_lists = []
            if dense_results:
                channel_lists.append(dense_results)
            if bm25_results:
                channel_lists.append(bm25_results)
            if extra_bm25:
                channel_lists.append(extra_bm25)
            if graph_results:
                channel_lists.append(graph_results)
                
            # Apply intent-based strategy modifications
            if intent in ("IMAGE_RETRIEVAL", "DIAGRAM_RETRIEVAL", "PAGE_RETRIEVAL", "SPECIFICATION"):
                for lst in channel_lists:
                    for res in lst:
                        if intent == "IMAGE_RETRIEVAL" and res.chunk_type == "IMAGE":
                            res.score += 2.0
                        elif intent == "DIAGRAM_RETRIEVAL" and res.chunk_type in ("DIAGRAM", "IMAGE"):
                            res.score += 2.0
                        elif intent == "SPECIFICATION" and res.chunk_type == "TABLE":
                            res.score += 1.0
                        elif intent == "PAGE_RETRIEVAL" and res.chunk_type == "PAGE":
                            res.score += 2.0
                    lst.sort(key=lambda x: x.score, reverse=True)
            elif intent == "PROCEDURE":
                for lst in channel_lists:
                    for res in lst:
                        if res.chunk_type == "TEXT" and any(w in res.content.lower() for w in ["step", "1.", "first"]):
                            res.score += 0.5
                    lst.sort(key=lambda x: x.score, reverse=True)
            elif intent == "ERROR_CODE":
                if bm25_results:
                    for res in bm25_results:
                        if any(e.lower() in res.content.lower() for e in norm.extracted_entities):
                            res.score += 3.0
                    bm25_results.sort(key=lambda x: x.score, reverse=True)
            elif intent == "SECTION_LOOKUP" and norm.requested_section_number is not None:
                # Pre-RRF: boost all results whose section_number already matches
                req_sec = norm.requested_section_number
                for lst in channel_lists:
                    for res in lst:
                        if res.section_number == req_sec:
                            res.score += 5.0
                    lst.sort(key=lambda x: x.score, reverse=True)
            elif intent in ("COMPARISON", "GENERAL_QA", "MAINTENANCE", "PROCEDURE",
                            "SPECIFICATION", "TROUBLESHOOTING", "ROOT_CAUSE_ANALYSIS",
                            "MULTI_HOP", "ERROR_CODE"):
                # Boost chunks whose content contains the most query keywords
                # This prevents large sections (like Section 9 MAINTENANCE) from
                # dominating purely by volume of indexed content.
                query_keywords = [
                    w for w in norm.clean_query.lower().split()
                    if len(w) > 3 and w not in {"what", "does", "this", "that", "from", "with",
                                                 "have", "when", "where", "which", "there", "their",
                                                 "between", "difference", "explain", "tell", "give"}
                ]
                if query_keywords:
                    for lst in channel_lists:
                        for res in lst:
                            hit_count = sum(1 for kw in query_keywords if kw in res.content.lower())
                            if hit_count > 0:
                                res.score += hit_count * 0.4  # boost proportional to keyword hits
                        lst.sort(key=lambda x: x.score, reverse=True)

            if not channel_lists:
                # Complete failure of all retrieval channels
                logger.error("retrieval_agent.all_channels_failed", tenant=ctx.tenant_id)
                fused = []
                reranked = []
                hierarchical_results = []
                strategy_str = "failed"
                conf_level = "LOW"
                conf_score = 0.0
            else:
                fused = reciprocal_rank_fusion(
                    channel_lists,
                    k=60,
                    top_n=30,
                )

                # ── SECTION_LOOKUP post-fusion boost ──────────────────────────
                # Multiply score for chunks that exactly match the requested
                # section_number. This ensures "3. CONVERSION TABLE" chunks
                # (section_number=None or section_number != 3) don't surface as
                # Section 3 when the user asks "what is section 3?".
                if intent == "SECTION_LOOKUP" and norm.requested_section_number is not None:
                    req_sec = norm.requested_section_number
                    for res in fused:
                        if res.section_number == req_sec:
                            res.score *= 2.0  # boost exact section match
                    fused.sort(key=lambda x: x.score, reverse=True)
                    logger.info(
                        "retrieval_agent.section_lookup_boost",
                        requested_section=req_sec,
                        top_section=fused[0].section_number if fused else None,
                    )

                # 4. Rerank fused candidates
                try:
                    reranked = await self._reranker.rerank(
                        query=norm.clean_query,
                        candidates=fused,
                        top_n=query_state.top_k,
                    )
                except Exception as e:
                    logger.error("retrieval_agent.rerank_failed", error=str(e))
                    degraded_channels.append("rerank")
                    reranked = fused[:query_state.top_k]

                # 5. Expand retrieved child chunks to their parent chunks
                hierarchical_results = await self._hierarchical_svc.expand_to_parents(
                    reranked,
                    tenant_id=ctx.tenant_id,
                    knowledge_base_id=ctx.knowledge_base_id,
                )

                # 6. Score confidence using the final hierarchical results
                conf = self._confidence_svc.score(
                    norm.clean_query,
                    hierarchical_results,
                )
                
                conf_level = conf.level
                conf_score = conf.score

                strategy_str = (
                    f"dense+bm25"
                    f"{'+graph' if graph_results else ''}"
                    "_rrf"
                )
                if "rerank" not in degraded_channels:
                    strategy_str += "_rerank"
                strategy_str += "_hierarchical"
                if degraded_channels:
                    strategy_str += f" (degraded: {','.join(degraded_channels)})"

            logger.info(
                "retrieval_agent.complete",
                fused_count=len(fused),
                reranked_count=len(reranked),
                hierarchical_count=len(hierarchical_results) if channel_lists else 0,
                confidence=conf_level,
                score=conf_score,
                strategy=strategy_str,
            )

            return query_state.model_copy(
                update={
                    "intent": intent,
                    "dense_results": dense_results,
                    "bm25_results": bm25_results,
                    "graph_results": graph_results,
                    "fused_results": fused,
                    "reranked_results": hierarchical_results if channel_lists else [],
                    "confidence_level": conf_level,
                    "confidence_score": conf_score,
                    "strategy_used": strategy_str,
                }
            )
