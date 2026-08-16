"""
Unit tests for Reciprocal Rank Fusion (RRF).
"""
from app.models.retrieval import SearchResult
from app.utils.fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion():
    res_a = SearchResult(chunk_id="chunk_1", score=0.9, content="First chunk")
    res_b = SearchResult(chunk_id="chunk_2", score=0.8, content="Second chunk")
    res_c = SearchResult(chunk_id="chunk_1", score=0.85, content="First chunk")
    res_d = SearchResult(chunk_id="chunk_3", score=0.7, content="Third chunk")

    list1 = [res_a, res_b]
    list2 = [res_c, res_d]

    fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=10)

    assert len(fused) == 3
    # chunk_1 appears in both lists, should have highest score
    assert fused[0].chunk_id == "chunk_1"
    assert fused[0].score > fused[1].score
