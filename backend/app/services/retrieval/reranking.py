"""
RRF (Reciprocal Rank Fusion) / Basic Re-Scoring Service.
Fuses and re-scores candidates from multiple retrieval strategies.
Currently implemented as a pass-through/truncation since fusion
is handled by the RetrievalAgent.
No external API calls are made.
"""
from typing import Any

from app.config.logging import get_logger
from app.models.retrieval import SearchResult

logger = get_logger(__name__)


class RRFRerankerService:
    """
    Stateless RRF / Basic reranking service.
    """

    def __init__(self) -> None:
        pass

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_n: int = 10,
    ) -> list[SearchResult]:
        """
        Since fusion already happened in the RetrievalAgent, this method
        simply normalizes scores and takes the top N.
        If a true cross-encoder local reranker is added later, it goes here.
        """
        if not candidates:
            return []

        # Ensure sorted by score descending
        sorted_cands = sorted(candidates, key=lambda x: x.score, reverse=True)
        
        # Take top N
        final_cands = sorted_cands[:top_n]
        
        logger.info("rerank.complete", input_count=len(candidates), output_count=len(final_cands))
        return final_cands
