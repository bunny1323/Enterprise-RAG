"""
Retrieval Agent.

Orchestrates multi-channel retrieval (Dense + BM25 + Graph),
fusion via RRF, reranking, and hierarchical retrieval expansion.

Calls stateless services — does not reimplement business logic directly.
"""

import asyncio
import os
import re
from typing import Any

from app.agents.retrieval.state import QueryState
from app.config.logging import get_logger
from app.config.opentelemetry import get_tracer
from app.models.retrieval import SearchResult
from app.services.cache.service import CacheService
from app.services.confidence.service import ConfidenceScoringService
from app.services.embeddings.service import EmbeddingService
from app.services.query.decomposition import (
    _COMPARISON_CONNECTIVES,
    _RELATIONSHIP_CONNECTIVES,
    QueryDecompositionService,
)
from app.services.query.normalization import QueryNormalizationService
from app.services.retrieval.bm25_search import BM25SearchService
from app.services.retrieval.dense_search import DenseSearchService
from app.services.retrieval.graph_search import GraphSearchService
from app.services.retrieval.hierarchical import HierarchicalRetrievalService
from app.services.retrieval.reranking import RRFRerankerService
from app.services.retrieval.structure_search import StructureSearchService
from app.utils.fusion import reciprocal_rank_fusion

logger = get_logger(__name__)
tracer = get_tracer()


def _source_key(result: SearchResult) -> str:
    """Return a stable source identifier for a SearchResult."""
    return (
        result.file_name
        or (result.metadata.get("file_name") if isinstance(result.metadata, dict) else None)
        or result.document_id
        or "unknown"
    )


def _trace_results(results: list[SearchResult]) -> list[dict[str, Any]]:
    """Return source-safe ranking metadata without exposing chunk contents."""
    return [
        {
            "rank": rank,
            "chunk_id": result.chunk_id,
            "document_id": result.document_id,
            "file_name": _source_key(result),
            "page_number": result.page_number,
            "section": result.section_title or result.section,
            "score": result.score,
            "matched_subqueries": result.metadata.get("_retrieval_subqueries", []),
            "requirement_scores": result.metadata.get("_rerank_requirement_scores", {}),
        }
        for rank, result in enumerate(results, start=1)
    ]


def _deduplicate_parent_candidates(results: list[SearchResult]) -> list[SearchResult]:
    """Keep the highest-ranked result for each eventual parent evidence chunk."""
    seen: set[str] = set()
    unique = []
    for result in results:
        evidence_id = result.parent_id or result.chunk_id
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        unique.append(result)
    return unique


def _section_lookup_evidence(
    results: list[SearchResult],
    section_number: int | None,
) -> list[SearchResult]:
    """Keep only chunks explicitly belonging to or introducing the requested section."""
    if section_number is None:
        return results

    heading_pattern = re.compile(rf"\bSECTION\s+{section_number}\b", re.IGNORECASE)
    return [
        result for result in results
        if heading_pattern.search(result.content or "")
    ]


def _diversity_aware_select(
    reranked: list[SearchResult],
    top_k: int,
    min_secondary_score_ratio: float = 0.30,
    required_queries: list[str] | None = None,
) -> list[SearchResult]:
    """Select top evidence while retaining the best candidate per query requirement."""
    if not reranked or top_k <= 0:
        return []

    try:
        ratio = float(os.environ.get("RAG_MIN_SECONDARY_SCORE_RATIO", str(min_secondary_score_ratio)))
    except (ValueError, TypeError):
        ratio = min_secondary_score_ratio
    ratio = max(0.0, min(1.0, ratio))

    selected = list(reranked[:top_k])
    selected_ids = {item.chunk_id for item in selected}
    best_by_requirement: dict[str, SearchResult] = {}
    for requirement in required_queries or []:
        matching = [
            item for item in reranked
            if requirement in item.metadata.get("_retrieval_subqueries", [])
        ]
        best = max(
            matching,
            key=lambda item: item.metadata.get("_rerank_requirement_scores", {}).get(
                requirement,
                item.score,
            ),
            default=None,
        )
        if best is not None:
            best_by_requirement[requirement] = best

    for requirement, candidate in best_by_requirement.items():
        if candidate.chunk_id in selected_ids:
            continue
        for index in range(len(selected) - 1, -1, -1):
            current = selected[index]
            if any(
                other_requirement != requirement and best.chunk_id == current.chunk_id
                for other_requirement, best in best_by_requirement.items()
            ):
                continue
            selected_ids.discard(current.chunk_id)
            selected[index] = candidate
            selected_ids.add(candidate.chunk_id)
            break

    selected_sources: dict[str, list[int]] = {}
    for index, item in enumerate(selected):
        selected_sources.setdefault(_source_key(item), []).append(index)

    top_score = reranked[0].score
    min_acceptable = top_score * ratio
    for candidate in reranked[top_k:]:
        source = _source_key(candidate)
        if candidate.chunk_id in selected_ids or source in selected_sources:
            continue
        if candidate.score < min_acceptable:
            continue
        for index in range(len(selected) - 1, -1, -1):
            current_source = _source_key(selected[index])
            if len(selected_sources.get(current_source, [])) < 2:
                continue
            evicted = selected[index]
            if any(best.chunk_id == evicted.chunk_id for best in best_by_requirement.values()):
                continue
            selected[index] = candidate
            selected_ids.discard(evicted.chunk_id)
            selected_ids.add(candidate.chunk_id)
            selected_sources[current_source].remove(index)
            selected_sources.setdefault(source, []).append(index)
            break

    return sorted(selected, key=lambda item: item.score, reverse=True)


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

            # ── 0. GREETING FAST-PATH ─────────────────────────────────────────
            if intent == "GREETING":
                return query_state.model_copy(
                    update={
                        "intent": intent,
                        "intent_confidence": norm.intent_confidence,
                        "dense_results": [],
                        "bm25_results": [],
                        "graph_results": [],
                        "fused_results": [],
                        "reranked_results": [],
                        "confidence_level": "HIGH",
                        "confidence_score": 1.0,
                        "strategy_used": "conversational",
                    }
                )

            # ── 1. STRUCTURAL RETRIEVAL FAST-PATH ─────────────────────────────
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
                            "intent_confidence": norm.intent_confidence,
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
                                "intent_confidence": norm.intent_confidence,
                                "fused_results": sections,
                                "reranked_results": sections,
                                "confidence_level": "HIGH",
                                "confidence_score": 1.0,
                                "strategy_used": "document_structure_list",
                            }
                        )

                elif intent == "PAGE_NUMBER_FORMAT":
                    notation = norm.extracted_notation
                    pf_results = await self._structure_svc.lookup_page_format(
                        notation=notation,
                        tenant_id=ctx.tenant_id,
                        knowledge_base_id=ctx.knowledge_base_id,
                    )
                    if pf_results:
                        return query_state.model_copy(
                            update={
                                "intent": intent,
                                "intent_confidence": norm.intent_confidence,
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
                loop = asyncio.get_running_loop()

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
            extra_ranked_lists: list[list[SearchResult]] = []
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

            # ── Morphological BM25 channel ────────────────────────────────────
            # When the retrieval_query differs from the clean_query (e.g. the user
            # typed 'kilograms' but the indexed text uses 'kilogram'), fire a second
            # BM25 search with the morphologically normalized form so that BM25's
            # exact inverted-index matching can find the relevant chunks.
            _retrieval_q = norm.retrieval_query or norm.clean_query
            _has_morph_variant = _retrieval_q.lower() != norm.clean_query.lower()
            if _has_morph_variant:
                tasks["bm25_morph"] = self._bm25_svc.search(
                    query_text=_retrieval_q,
                    top_k=query_state.top_k * 2,
                    tenant_id=ctx.tenant_id,
                    knowledge_base_id=ctx.knowledge_base_id,
                    permitted_access_levels=permitted_access_levels,
                )
                logger.info(
                    "retrieval_agent.morphological_bm25",
                    original=norm.clean_query[:60],
                    normalized=_retrieval_q[:60],
                )

            # ── Typo-corrected BM25 channel ───────────────────────────────────
            # When the user's query contains spelling errors, the typo_query
            # holds the corrected form. This is a SEPARATE channel from both
            # the original BM25 and the morphological BM25 — it fires only when
            # the typo-corrected form is distinct from the other two queries.
            _typo_q = norm.typo_query or norm.clean_query
            _has_typo_variant = (
                _typo_q.lower() != norm.clean_query.lower()
                and _typo_q.lower() != _retrieval_q.lower()
            )
            if _has_typo_variant:
                tasks["bm25_typo"] = self._bm25_svc.search(
                    query_text=_typo_q,
                    top_k=query_state.top_k * 2,
                    tenant_id=ctx.tenant_id,
                    knowledge_base_id=ctx.knowledge_base_id,
                    permitted_access_levels=permitted_access_levels,
                )
                logger.info(
                    "retrieval_agent.typo_bm25",
                    original=norm.clean_query[:60],
                    corrected=_typo_q[:60],
                )

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

            main_task_names = list(tasks)
            main_task = asyncio.gather(*tasks.values(), return_exceptions=True)

            # ── Multi-hop / Comparison / Relationship sub-query decomposition ──
            # Triggered by explicit intent labels AND by structural comparison
            # connectives present in any query (vs., versus, difference between, etc.).
            # Each subquery fires HYBRID retrieval (dense + BM25) so semantic and
            # lexical signals both contribute to the candidate pool.
            _should_decompose = (
                intent in ("SYMBOL_PURPOSE", "MULTI_HOP", "RELATIONSHIP", "COMPARISON")
                or _COMPARISON_CONNECTIVES.search(norm.clean_query)
                or _RELATIONSHIP_CONNECTIVES.search(norm.clean_query)
            )
            subquery_diagnostics: list[dict[str, Any]] = []
            sub_queries = []
            if _should_decompose:
                sub_queries = self._decomposer_svc.decompose(norm)
                loop = asyncio.get_running_loop()

                async def retrieve_subquery(
                    sq: Any,
                ) -> tuple[list[list[SearchResult]], dict[str, Any]]:
                    try:
                        sq_vec = await loop.run_in_executor(
                            None,
                            self._embedder.embed_batch,
                            [sq.query],
                        )
                        sq_vector = sq_vec[0] if sq_vec else []

                        sub_tasks = []
                        sub_channel_names = []
                        if sq_vector:
                            sub_channel_names.append("dense")
                            sub_tasks.append(
                                self._dense_svc.search(
                                    query_vector=sq_vector,
                                    top_k=query_state.top_k * 2,
                                    tenant_id=ctx.tenant_id,
                                    knowledge_base_id=ctx.knowledge_base_id,
                                    permitted_access_levels=permitted_access_levels,
                                )
                            )
                        sub_channel_names.append("bm25")
                        sub_tasks.append(
                            self._bm25_svc.search(
                                query_text=sq.query,
                                top_k=query_state.top_k * 2,
                                tenant_id=ctx.tenant_id,
                                knowledge_base_id=ctx.knowledge_base_id,
                                permitted_access_levels=permitted_access_levels,
                            )
                        )

                        sub_results = await asyncio.gather(*sub_tasks, return_exceptions=True)
                        ranked_lists = []
                        channel_trace = {}
                        for channel_name, sr in zip(sub_channel_names, sub_results):
                            if isinstance(sr, Exception):
                                logger.warning(
                                    "retrieval_agent.subquery_channel_failed",
                                    sub_query=sq.query[:60],
                                    error=str(sr),
                                )
                            else:
                                if sr:
                                    sr = [
                                        result.model_copy(
                                            update={
                                                "metadata": {
                                                    **result.metadata,
                                                    "_retrieval_subqueries": [sq.query],
                                                }
                                            }
                                        )
                                        for result in sr
                                    ]
                                    ranked_lists.append(sr)
                                    channel_trace[channel_name] = _trace_results(sr)

                        logger.debug(
                            "retrieval_agent.subquery_hybrid",
                            sub_query=sq.query[:60],
                            dense_ok=bool(sq_vector),
                            hits=len([r for r in sub_results if not isinstance(r, Exception)]),
                        )
                        return ranked_lists, {
                            "query": sq.query,
                            "channels": channel_trace,
                        }
                    except Exception as e:
                        logger.warning(
                            "retrieval_agent.subquery_failed",
                            sub_query=sq.query[:60],
                            error=str(e),
                        )
                        return [], {"query": sq.query, "error": str(e)}

                subquery_jobs = [
                    retrieve_subquery(sq)
                    for sq in sub_queries
                    if sq.query.strip().lower() != norm.clean_query.strip().lower()
                ]
                subquery_results = await asyncio.gather(*subquery_jobs, return_exceptions=True)
                for subquery_result in subquery_results:
                    if isinstance(subquery_result, Exception):
                        logger.warning(
                            "retrieval_agent.subquery_failed",
                            error=str(subquery_result),
                        )
                    else:
                        ranked_lists, subquery_trace = subquery_result
                        extra_ranked_lists.extend(ranked_lists)
                        subquery_diagnostics.append(subquery_trace)

            # Main and decomposed retrieval are now complete; each channel remains
            # an independent ranked list for RRF.
            results = await main_task

            dense_results = []
            bm25_results = []
            graph_results = []
            morph_results = []
            typo_results = []
            reranked_wide = []
            degraded_channels = []

            for i, key in enumerate(main_task_names):
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
                elif key == "bm25_morph":
                    # Preserve this query's independent ranking for RRF.
                    morph_results = res
                    extra_ranked_lists.append(res)
                    logger.debug(
                        "retrieval_agent.morphological_bm25_hits",
                        hits=len(res),
                    )
                elif key == "bm25_typo":
                    # Preserve this query's independent ranking for RRF.
                    typo_results = res
                    extra_ranked_lists.append(res)
                    logger.debug(
                        "retrieval_agent.typo_bm25_hits",
                        hits=len(res),
                    )
                elif key == "graph":
                    graph_results = res

            channel_lists = []
            if dense_results:
                channel_lists.append(dense_results)
            if bm25_results:
                channel_lists.append(bm25_results)
            channel_lists.extend(result_list for result_list in extra_ranked_lists if result_list)
            if graph_results:
                channel_lists.append(graph_results)

            logger.info(
                "retrieval_agent.candidates",
                dense=len(dense_results),
                bm25=len(bm25_results),
                subquery=sum(len(result_list) for result_list in extra_ranked_lists),
                graph=len(graph_results),
            )
                
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
                combined_text = f"{norm.clean_query} {norm.retrieval_query} {norm.typo_query}"
                # Strip punctuation for reliable substring matching during keyword boost
                import string
                raw_keywords = {w.strip(string.punctuation) for w in combined_text.lower().split()}
                query_keywords = [
                    w for w in raw_keywords
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
                    top_n=sum(len(channel) for channel in channel_lists),
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

                # 4. Compare the complete global pool before selecting final evidence.
                reranker_top_n = len(fused)
                try:
                    reranked_wide = await self._reranker.rerank(
                        query=norm.clean_query,
                        candidates=fused,
                        top_n=reranker_top_n,
                        relevance_queries=[sq.query for sq in sub_queries],
                    )
                except Exception as e:
                    logger.error("retrieval_agent.rerank_failed", error=str(e))
                    degraded_channels.append("rerank")
                    reranked_wide = fused[:reranker_top_n]

                # 4b. Collapse parent/child representations before final selection.
                selection_pool = _deduplicate_parent_candidates(reranked_wide)
                reranked = _diversity_aware_select(
                    selection_pool,
                    top_k=query_state.top_k,
                    required_queries=[sq.query for sq in sub_queries],
                )

                logger.info(
                    "retrieval_agent.rerank_complete",
                    wide_pool=len(reranked_wide),
                    unique_evidence_pool=len(selection_pool),
                    selected=len(reranked),
                    sources=len({_source_key(r) for r in reranked}),
                )

                # 5. Expand retrieved child chunks to their parent chunks
                hierarchical_results = await self._hierarchical_svc.expand_to_parents(
                    reranked,
                    tenant_id=ctx.tenant_id,
                    knowledge_base_id=ctx.knowledge_base_id,
                )
                if intent == "SECTION_LOOKUP":
                    hierarchical_results = _section_lookup_evidence(
                        hierarchical_results,
                        norm.requested_section_number,
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
                    "intent_confidence": norm.intent_confidence,
                    "dense_results": dense_results,
                    "bm25_results": bm25_results,
                    "graph_results": graph_results,
                    "fused_results": fused,
                    "reranked_results": hierarchical_results if channel_lists else [],
                    "retrieval_diagnostics": {
                        "query": norm.raw_query,
                        "normalized_query": norm.clean_query,
                        "retrieval_query": norm.retrieval_query,
                        "typo_query": norm.typo_query,
                        "intent": intent,
                        "subqueries": subquery_diagnostics,
                        "channels": {
                            "dense": _trace_results(dense_results),
                            "bm25": _trace_results(bm25_results),
                            "bm25_morph": _trace_results(morph_results),
                            "bm25_typo": _trace_results(typo_results),
                            "graph": _trace_results(graph_results),
                        },
                        "fused_candidates": _trace_results(fused),
                        "reranked_candidates": _trace_results(reranked_wide),
                        "selected_candidates": _trace_results(reranked),
                        "selected_evidence": _trace_results(hierarchical_results),
                        "selected_sources": sorted({_source_key(item) for item in hierarchical_results}),
                        "degraded_channels": degraded_channels,
                    },
                    "confidence_level": conf_level,
                    "confidence_score": conf_score,
                    "strategy_used": strategy_str,
                }
            )
