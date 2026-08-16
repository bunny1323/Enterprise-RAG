"""
Unit tests for ConfidenceScoringService.
"""
from app.models.retrieval import SearchResult
from app.services.confidence.service import ConfidenceScoringService


def test_confidence_scoring():
    scorer = ConfidenceScoringService()

    # High confidence test
    res_high = [SearchResult(chunk_id="c1", score=0.85, content="High match content")]
    conf_high = scorer.score("Query", res_high)
    assert conf_high.level == "HIGH"
    assert conf_high.score == 0.85

    # Medium confidence test
    res_med = [SearchResult(chunk_id="c1", score=0.55, content="Med match content")]
    conf_med = scorer.score("Query", res_med)
    assert conf_med.level == "MEDIUM"

    # Low confidence test
    res_low = [SearchResult(chunk_id="c1", score=0.2, content="Low match content")]
    conf_low = scorer.score("Query", res_low)
    assert conf_low.level == "LOW"
