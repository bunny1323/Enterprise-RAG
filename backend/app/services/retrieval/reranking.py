"""
Cross-Encoder Reranking Service.
Uses Voyage Rerank API to re-score and re-order fused retrieval candidates.
"""
import asyncio
import time
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config.logging import get_logger
from app.models.retrieval import SearchResult

logger = get_logger(__name__)


class VoyageRerankService:
    """
    Stateless cross-encoder reranking service.
    """

    def __init__(self, api_key: str, model: str = "rerank-2") -> None:
        self._api_key = api_key
        self._model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import voyageai  # type: ignore[import-untyped]

            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_n: int = 10,
    ) -> list[SearchResult]:
        """
        Rerank candidate SearchResults relative to query using Voyage Rerank API.
        Never reranks more than 100 candidates in one call.
        """
        if not candidates or not query:
            return candidates[:top_n]

        # Limit candidate count
        candidates_to_rerank = candidates[:50]
        documents = [c.content for c in candidates_to_rerank]

        try:
            loop = asyncio.get_event_loop()
            rerank_results = await loop.run_in_executor(
                None,
                lambda: self._rerank_with_retry(query, documents, top_n),
            )

            reranked_results: list[SearchResult] = []
            for item in rerank_results:
                idx = item.index
                re_score = float(item.relevance_score)
                cand = candidates_to_rerank[idx]
                reranked_results.append(cand.model_copy(update={"score": round(re_score, 4)}))

            logger.info("rerank.complete", input_count=len(candidates), output_count=len(reranked_results))
            return reranked_results

        except Exception as err:
            logger.warning("rerank.failed_fallback_to_original", error=str(err))
            return candidates[:top_n]

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _rerank_with_retry(self, query: str, documents: list[str], top_k: int) -> list[Any]:
        client = self._get_client()
        result = client.rerank(
            query=query,
            documents=documents,
            model=self._model,
            top_k=top_k,
        )
        return result.results
