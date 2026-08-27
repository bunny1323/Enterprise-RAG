"""
Retrieval Agent.

Orchestrates multi-channel retrieval (Dense + BM25 + Graph),
fusion via RRF, reranking, and hierarchical retrieval expansion.

Calls stateless services — does not reimplement business logic directly.
"""

import asyncio
<<<<<<< HEAD

=======
from typing import Any
>>>>>>> 2acbe82 (Complete Enterprise-RAG Phases 1-5 hardening)
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
from app.services.retrieval.reranking import VoyageRerankService
from app.utils.fusion import reciprocal_rank_fusion

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
        reranker: VoyageRerankService,
        normalizer: QueryNormalizationService,
        confidence_svc: ConfidenceScoringService,
        hierarchical_svc: HierarchicalRetrievalService,
        cache_svc: CacheService | None = None,
        weaviate_client: Any | None = None,
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

<<<<<<< HEAD
        # 1. Generate query embedding (with caching)
        query_vec = None

        if self._cache_svc:
            query_vec = await self._cache_svc.get_query_embedding(
                ctx,
                norm.clean_query,
            )

        if not query_vec:
            # Voyage client is synchronous, so run embedding in executor.
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

        if intent in ("RELATIONSHIP", "TECHNICAL"):
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
=======
            logger.info(
                "retrieval_agent.start",
                query=norm.clean_query[:50],
                intent=intent,
                tenant=ctx.tenant_id,
            )

            # 1. Generate query embedding (with caching)
            query_vec = None
            if self._cache_svc:
                query_vec = await self._cache_svc.get_query_embedding(ctx, norm.clean_query)
>>>>>>> 2acbe82 (Complete Enterprise-RAG Phases 1-5 hardening)

            if not query_vec:
                # Need to run in executor since embedder is sync under the hood in Voyage client
                loop = asyncio.get_event_loop()
                query_vectors = await loop.run_in_executor(None, self._embedder.embed_batch, [norm.clean_query])
                query_vec = query_vectors[0] if query_vectors else []
                if query_vec and self._cache_svc:
                    await self._cache_svc.set_query_embedding(ctx, norm.clean_query, query_vec)

<<<<<<< HEAD
        for i, key in enumerate(tasks.keys()):
            res = results[i]

            if isinstance(res, Exception):
                logger.error(
                    f"retrieval_agent.{key}_failed",
                    error=str(res),
                )
                continue

            if key == "dense":
                dense_results = res
            elif key == "bm25":
                bm25_results = res
            elif key == "graph":
                graph_results = res

        # 3. Fuse results using Reciprocal Rank Fusion (RRF)
        channel_lists = [
            dense_results,
            bm25_results,
        ]

        if graph_results:
            channel_lists.append(graph_results)

        fused = reciprocal_rank_fusion(
            channel_lists,
            k=60,
            top_n=30,
        )

        # 4. Rerank fused candidates
        reranked = await self._reranker.rerank(
            query=norm.clean_query,
            candidates=fused,
            top_n=query_state.top_k,
        )

        # 5. Expand retrieved child chunks to their parent chunks
        #
        # Reranking identifies the most relevant child chunks.
        # Hierarchical retrieval then replaces those children with
        # their larger parent context before confidence scoring
        # and generation.
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

        strategy_str = (
            f"dense+bm25"
            f"{'+graph' if graph_results else ''}"
            "_rrf_rerank_hierarchical"
        )

        logger.info(
            "retrieval_agent.complete",
            fused_count=len(fused),
            reranked_count=len(reranked),
            hierarchical_count=len(hierarchical_results),
            confidence=conf.level,
            score=conf.score,
        )

        return query_state.model_copy(
            update={
                "intent": intent,
                "dense_results": dense_results,
                "bm25_results": bm25_results,
                "graph_results": graph_results,
                "fused_results": fused,
                "reranked_results": hierarchical_results,
                "confidence_level": conf.level,
                "confidence_score": conf.score,
                "strategy_used": strategy_str,
            }
        )
=======
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
                )
            }

            if intent in ("RELATIONSHIP", "TECHNICAL"):
                entity_query = norm.extracted_entities[0] if norm.extracted_entities else norm.clean_query[:30]
                tasks["graph"] = self._graph_svc.search(
                    entity_name=entity_query,
                    tenant_id=ctx.tenant_id,
                    knowledge_base_id=ctx.knowledge_base_id,
                )

            # Execute concurrently and handle exceptions safely
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            dense_results = []
            bm25_results = []
            graph_results = []
            degraded_channels = []

            for i, key in enumerate(tasks.keys()):
                res = results[i]
                if isinstance(res, Exception):
                    logger.error(f"retrieval_agent.{key}_failed", error=str(res))
                    degraded_channels.append(key)
                    continue

                if key == "dense":
                    dense_results = res
                elif key == "bm25":
                    bm25_results = res
                elif key == "graph":
                    graph_results = res

            # 3. Fuse results via RRF
            channel_lists = []
            if dense_results:
                channel_lists.append(dense_results)
            if bm25_results:
                channel_lists.append(bm25_results)
            if graph_results:
                channel_lists.append(graph_results)
                
            if not channel_lists:
                # Complete failure of all retrieval channels
                logger.error("retrieval_agent.all_channels_failed", tenant=ctx.tenant_id)
                fused = []
                reranked = []
                strategy_str = "failed"
                conf_level = "LOW"
                conf_score = 0.0
            else:
                fused = reciprocal_rank_fusion(channel_lists, k=60, top_n=30)

                # 4. Rerank via Voyage Rerank
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

                # 4.5 Parent Context Restoration
                if self._weaviate:
                    parent_ids_to_fetch = set()
                    for item in reranked:
                        if item.metadata and item.metadata.get("parent_id"):
                            parent_ids_to_fetch.add(item.metadata["parent_id"])

                    if parent_ids_to_fetch:
                        # Run synchronous weaviate fetch in executor
                        loop = asyncio.get_event_loop()
                        parents_data = await loop.run_in_executor(
                            None, 
                            self._weaviate.get_chunks_by_ids, 
                            list(parent_ids_to_fetch), 
                            ctx.tenant_id
                        )
                        parent_map = {p.get("chunk_id"): p for p in parents_data if p.get("chunk_id")}
                        
                        for item in reranked:
                            p_id = item.metadata.get("parent_id") if item.metadata else None
                            if p_id and p_id in parent_map:
                                parent_chunk = parent_map[p_id]
                                # Expand context: use parent's content instead of just child's
                                item.content = parent_chunk.get("content", item.content)
                                item.metadata["restored_from_parent"] = True

                # 5. Score confidence
                conf = self._confidence_svc.score(norm.clean_query, reranked)
                conf_level = conf.level
                conf_score = conf.score

                strategy_str = f"dense+bm25{'+graph' if graph_results else ''}_rrf"
                if "rerank" not in degraded_channels:
                    strategy_str += "_rerank"
                if degraded_channels:
                    strategy_str += f" (degraded: {','.join(degraded_channels)})"

            logger.info(
                "retrieval_agent.complete",
                fused_count=len(fused) if channel_lists else 0,
                reranked_count=len(reranked) if channel_lists else 0,
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
                    "reranked_results": reranked,
                    "confidence_level": conf_level,
                    "confidence_score": conf_score,
                    "strategy_used": strategy_str,
                }
            )
>>>>>>> 2acbe82 (Complete Enterprise-RAG Phases 1-5 hardening)
