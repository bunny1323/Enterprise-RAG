from app.models.retrieval import SearchResult
from app.services.query.decomposition import QueryDecompositionService
from app.services.query.normalization import QueryNormalizationService
from app.services.retrieval.coverage import EvidenceCoverageService


SYMBOL_EVIDENCE = (
    "Symbol Item Remarks. Special safety precautions are necessary when performing the work. "
    "Extra special safety precautions are necessary when performing the work because it is under internal pressure. "
    "Special technical precautions or other precautions for preserving standards are necessary when performing the work. "
    "Safety. Caution."
)


def _evidence() -> list[SearchResult]:
    return [SearchResult(
        chunk_id="symbol-definition",
        content=SYMBOL_EVIDENCE,
        score=1.0,
        page_number=2,
        section="HOW TO READ THE SERVICE MANUAL",
    )]


def test_symbol_purpose_variants_route_to_focused_retrieval():
    normalizer = QueryNormalizationService()
    decomposer = QueryDecompositionService()

    for query in (
        "What are the three main reasons the manual provides symbols?",
        "What are the 3 main reasons the manual provides symbols?",
        "What are the reasons the manual provides symbols?",
        "Why does the manual provide symbols?",
    ):
        normalized = normalizer.normalize(query)
        assert normalized.intent == "SYMBOL_PURPOSE"
        assert "internal pressure" in normalized.retrieval_query
        assert any("preserving standards" in item.query for item in decomposer.decompose(normalized))

    for query in (
        "Why does the manual provide symbols?",
        "What precautions are indicated by the symbols?",
    ):
        normalized = normalizer.normalize(query)
        assert normalized.intent == "SYMBOL_PURPOSE"


def test_symbol_purpose_requires_all_three_categories():
    normalizer = QueryNormalizationService()
    normalized = normalizer.normalize("What are the three main reasons the manual provides symbols?")
    result = EvidenceCoverageService().evaluate_coverage(normalized, _evidence())

    assert result.answerable is True
    assert result.coverage_status == "COMPLETE"
    assert len(result.entities_found) == 3


def test_symbol_comparison_requires_safety_caution_and_pressure_evidence():
    normalizer = QueryNormalizationService()
    normalized = normalizer.normalize("What is the difference between the safety and caution symbols?")
    result = EvidenceCoverageService().evaluate_coverage(normalized, _evidence())

    assert normalized.intent == "COMPARISON"
    assert result.answerable is True
    assert result.coverage_status == "COMPLETE"
    assert result.relationship_supported is True