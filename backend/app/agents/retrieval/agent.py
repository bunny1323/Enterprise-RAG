"""
Retrieval Agent.

Orchestrates multi-channel retrieval (Dense + BM25 + Graph),
fusion via RRF, reranking, and hierarchical retrieval expansion.

Calls stateless services — does not reimplement business logic directly.
"""

import asyncio

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

logger = get_logger(__name__)


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

        logger.info(
            "retrieval_agent.start",
            query=norm.clean_query[:50],
            intent=intent,
            tenant=ctx.tenant_id,
        )

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

        dense_results = []
        bm25_results = []
        graph_results = []

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