"""
Reciprocal Rank Fusion (RRF) algorithm.
Combines multiple ranked search lists (Dense, BM25, Graph) into a single unified ranking.
Formula: score(item) = sum(1 / (k + rank_i)) for each list i where item appears.
"""
from app.models.retrieval import SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
    top_n: int = 20,
) -> list[SearchResult]:
    """
    Fuse multiple ranked lists of SearchResult objects using RRF.

    Args:
        ranked_lists: List of candidate result lists (e.g. [dense_results, bm25_results]).
        k: Constant bias factor (default 60).
        top_n: Max number of fused results to return.

    Returns:
        Deduplicated list of SearchResult items ordered by combined RRF score.
    """
    scores: dict[str, float] = {}
    item_map: dict[str, SearchResult] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            cid = item.chunk_id
            if not cid:
                continue

            rrf_contrib = 1.0 / (k + rank)
            scores[cid] = scores.get(cid, 0.0) + rrf_contrib

            # Keep item with richest context/metadata
            if cid not in item_map or (item.context_prefix and not item_map[cid].context_prefix):
                item_map[cid] = item

    # Sort candidates by combined score in descending order
    sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_results: list[SearchResult] = []
    for cid in sorted_cids[:top_n]:
        res = item_map[cid]
        # Update score to RRF score
        res_copy = res.model_copy(update={"score": round(scores[cid], 5)})
        fused_results.append(res_copy)

    return fused_results
