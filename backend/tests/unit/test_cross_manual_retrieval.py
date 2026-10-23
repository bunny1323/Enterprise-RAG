"""
Unit and regression tests for cross-manual evidence packing, system instructions,
and answerability gating.
"""
import pytest
from app.models.retrieval import SearchResult
from app.services.generation.citations import CitationService
from app.services.generation.llm import (
    DEFAULT_SYSTEM_INSTRUCTION,
    format_evidence_context,
    get_context_summary,
    get_intent_system_instruction,
)
from app.services.query.normalization import NormalizedQuery
from app.services.retrieval.coverage import EvidenceCoverageService


def test_evidence_context_includes_manual_names_and_pages():
    """Verify that source manual file names, page numbers, and sections appear in evidence blocks."""
    evidence = [
        SearchResult(
            chunk_id="chunk-hyd-1",
            content="Main pump delivers hydraulic pressure to the control valve spool.",
            page_number=12,
            file_name="hyundai-r215l-smart-hydraulic-system.pdf",
            section="SECTION 3 HYDRAULIC SYSTEM",
            section_title="HYDRAULIC CIRCUIT",
            score=0.92,
        ),
        SearchResult(
            chunk_id="chunk-mech-1",
            content="The CPU controller outputs EPPR valve control current to govern pump swash plate angle.",
            page_number=21,
            file_name="hyundai-r215l-smart-mechatronics-system-manual.pdf",
            section="SECTION 5 MECHATRONICS SYSTEM",
            section_title="CAPO SYSTEM",
            score=0.88,
        ),
    ]

    context = format_evidence_context(evidence)

    assert "[Source 1: hyundai-r215l-smart-hydraulic-system.pdf (Page 12) [Section: HYDRAULIC CIRCUIT]]" in context
    assert "Main pump delivers hydraulic pressure" in context
    assert "[Source 2: hyundai-r215l-smart-mechatronics-system-manual.pdf (Page 21) [Section: CAPO SYSTEM]]" in context
    assert "The CPU controller outputs EPPR valve" in context


def test_diversity_aware_packing_prevents_single_manual_starvation():
    """Verify that evidence from secondary manuals is not choked out by a dominant manual."""
    # Create 5 chunks from Hydraulic manual and 2 chunks from Mechatronics manual
    hydraulic_chunks = [
        SearchResult(
            chunk_id=f"hyd-{i}",
            content=f"Hydraulic operation detail {i}: " + ("x" * 200),
            page_number=i,
            file_name="hyundai-r215l-hydraulic.pdf",
            score=0.95 - (i * 0.01),
        )
        for i in range(5)
    ]
    mechatronic_chunks = [
        SearchResult(
            chunk_id=f"mech-{i}",
            content=f"Mechatronic controller signal {i}: " + ("y" * 200),
            page_number=i + 5,
            file_name="hyundai-r215l-mechatronics.pdf",
            score=0.85 - (i * 0.01),
        )
        for i in range(2)
    ]

    all_evidence = hydraulic_chunks + mechatronic_chunks

    # Pack with a tight budget
    packed = format_evidence_context(all_evidence, max_total_chars=2000, max_chunks_per_source=2)

    # Both manuals MUST be present in the packed context
    assert "hyundai-r215l-hydraulic.pdf" in packed
    assert "hyundai-r215l-mechatronics.pdf" in packed
    assert "Mechatronic controller signal 0" in packed


def test_context_summary_observability():
    """Verify get_context_summary returns correct diagnostic metadata."""
    evidence = [
        SearchResult(
            chunk_id="c1",
            content="Check hydraulic oil level.",
            page_number=4,
            file_name="hyundai-hydraulic.pdf",
            score=0.9,
        ),
        SearchResult(
            chunk_id="c2",
            content="Inspect electrical wiring harness.",
            page_number=8,
            file_name="hyundai-electrical.pdf",
            score=0.85,
        ),
    ]

    summary = get_context_summary(evidence, max_total_chars=10000)

    assert summary["context_chunks_count"] == 2
    assert summary["context_truncated"] is False
    assert "hyundai-hydraulic.pdf" in summary["source_manuals"]
    assert "hyundai-electrical.pdf" in summary["source_manuals"]
    assert summary["context_char_count"] > 0


def test_comparison_instructions_general_vs_symbol():
    """Verify comparison instructions dispatch general technical comparison for machine inquiries
    and symbol-specific directives only for symbol inquiries."""
    # 1. Technical comparison query
    tech_query = "Compare the operating principles described in the hydraulic system manual and the mechatronics system manual."
    tech_instruction = get_intent_system_instruction(intent="COMPARISON", query=tech_query)

    assert "SPECIFIC DIRECTIVE FOR TECHNICAL COMPARISONS" in tech_instruction
    assert "Clearly attribute each principle, specification, or procedure to its respective source manual" in tech_instruction
    assert "SPECIFIC DIRECTIVE FOR SYMBOL COMPARISONS" not in tech_instruction

    # 2. Safety symbol query
    sym_query = "What is the difference between the safety symbol and the caution symbol?"
    sym_instruction = get_intent_system_instruction(intent="COMPARISON", query=sym_query)

    assert "SPECIFIC DIRECTIVE FOR SYMBOL COMPARISONS" in sym_instruction
    assert "internal-pressure" in sym_instruction


def test_relationship_instructions():
    """Verify relationship instructions mandate manual attribution and grounded interactions."""
    rel_instruction = get_intent_system_instruction(intent="RELATIONSHIP")
    assert "SPECIFIC DIRECTIVE FOR RELATIONSHIP & INTERACTION QUESTIONS" in rel_instruction
    assert "Clearly identify the source manual" in rel_instruction


def test_answerability_gate_allows_valid_cross_manual_evidence():
    """Verify that cross-manual questions with relevant evidence pass answerability without requiring
    exact phrase matching."""
    norm = NormalizedQuery(
        raw_query="Compare the operating principles described in the hydraulic system manual and the mechatronics system manual.",
        clean_query="Compare the operating principles described in the hydraulic system manual and the mechatronics system manual.",
        retrieval_query="",
        typo_query="",
        intent="COMPARISON",
        extracted_entities=["hydraulic", "mechatronics"],
    )

    evidence = [
        SearchResult(
            chunk_id="hyd-1",
            content="The hydraulic circuit drives the swing motor and cylinders via pilot control.",
            page_number=3,
            file_name="hyundai-r215l-smart-hydraulic-system.pdf",
            score=0.9,
        ),
        SearchResult(
            chunk_id="mech-1",
            content="The mechatronics CAPO controller regulates engine speed and main pump discharge flow.",
            page_number=2,
            file_name="hyundai-r215l-smart-mechatronics-system-manual.pdf",
            score=0.88,
        ),
    ]

    service = EvidenceCoverageService()
    result = service.evaluate_coverage(norm, evidence)

    assert result.answerable is True
    assert result.coverage_status == "COMPLETE"
    assert "Both comparison subjects supported" in result.reason


def test_answerability_gate_rejects_unsupported_cross_manual_query():
    """Verify that queries where the evidence lacks the required systems are rejected with honest explanation."""
    norm = NormalizedQuery(
        raw_query="Compare the hydraulic system with the nuclear propulsion subsystem.",
        clean_query="Compare the hydraulic system with the nuclear propulsion subsystem.",
        retrieval_query="",
        typo_query="",
        intent="COMPARISON",
        extracted_entities=["hydraulic", "nuclear propulsion"],
    )

    # Evidence only contains hydraulic system data; no nuclear propulsion
    evidence = [
        SearchResult(
            chunk_id="hyd-1",
            content="The main pump supplies hydraulic oil to the cylinders.",
            page_number=2,
            file_name="hyundai-r215l-smart-hydraulic-system.pdf",
            score=0.9,
        ),
    ]

    service = EvidenceCoverageService()
    result = service.evaluate_coverage(norm, evidence)

    # Should detect partial or insufficient coverage and flag appropriately
    assert result.inferred_only is True or result.answerable is False


def test_answerability_gate_empty_evidence():
    """Verify that completely empty evidence returns answerable = False."""
    norm = NormalizedQuery(
        raw_query="How does hydraulic oil flow?",
        clean_query="How does hydraulic oil flow?",
        retrieval_query="",
        typo_query="",
        intent="GENERAL_QA",
    )

    service = EvidenceCoverageService()
    result = service.evaluate_coverage(norm, [])

    assert result.answerable is False
    assert result.coverage_status == "INSUFFICIENT"
    assert "No evidence" in result.reason


def test_citation_service_preserves_multiple_manual_sources():
    """Verify that CitationService preserves file names and page numbers for multiple manuals."""
    evidence = [
        SearchResult(
            chunk_id="c1",
            content="Hydraulic pilot valve function.",
            page_number=10,
            file_name="hyundai-hydraulic.pdf",
            section="SECTION 3",
            section_number=3,
            section_title="HYDRAULIC SYSTEM",
        ),
        SearchResult(
            chunk_id="c2",
            content="Electrical battery disconnect procedure.",
            page_number=5,
            file_name="hyundai-electrical.pdf",
            section="SECTION 4",
            section_number=4,
            section_title="ELECTRICAL SYSTEM",
        ),
    ]

    service = CitationService()
    citations, sources = service.map_citations("Test answer", evidence)

    assert len(citations) == 2
    assert citations[0].file_name == "hyundai-hydraulic.pdf"
    assert citations[0].page_number == 10
    assert citations[1].file_name == "hyundai-electrical.pdf"
    assert citations[1].page_number == 5

    assert len(sources) == 2
    source_files = {s.file_name for s in sources}
    assert "hyundai-hydraulic.pdf" in source_files
    assert "hyundai-electrical.pdf" in source_files
