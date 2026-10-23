import pytest
from app.models.retrieval import SearchResult
from app.services.generation.citations import CitationService
from app.services.retrieval.coverage import EvidenceCoverageService
from app.services.query.normalization import QueryNormalizationService, NormalizedQuery


class TestCitationAudit:
    """Test suite verifying citation accuracy, section mapping, and elimination of hardcoding."""

    def test_citation_file_name_not_hardcoded_to_hyundai(self):
        """Verify that missing file_name does not invent 'Hyundai_R215L_Manual.pdf'."""
        svc = CitationService()
        evidence = [
            SearchResult(
                chunk_id="chunk-generic-1",
                score=0.92,
                content="Operating temperature range is -20C to 50C.",
                page_number=14,
                document_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                section_title="Operating Conditions",
                file_name=None,
                metadata={},
            )
        ]
        citations, sources = svc.map_citations("Operating temperature is -20C to 50C.", evidence)

        assert len(citations) == 1
        assert citations[0].file_name != "Hyundai_R215L_Manual.pdf"
        assert citations[0].file_name == "Document_a1b2c3d4"
        assert citations[0].section == "Operating Conditions"
        assert citations[0].page_number == 14

        assert len(sources) == 1
        assert sources[0].file_name == "Document_a1b2c3d4"
        assert sources[0].page_number == 14

    def test_citation_prefers_section_title_over_generic_section(self):
        """Verify that human-readable section_title is preferred over raw OCR section string."""
        svc = CitationService()
        evidence = [
            SearchResult(
                chunk_id="chunk-sec-1",
                score=0.88,
                content="Hydraulic pump discharge pressure is 350 bar.",
                page_number=45,
                document_id="doc-hyd-1",
                section="Section 3",
                section_title="HYDRAULIC CIRCUIT DIAGRAM",
                file_name="industrial_manual.pdf",
                metadata={},
            )
        ]
        citations, _ = svc.map_citations("Pump discharge pressure is 350 bar.", evidence)
        assert citations[0].section == "HYDRAULIC CIRCUIT DIAGRAM"

    def test_citation_fallback_when_no_section_title(self):
        """Verify clean fallback to section when section_title is None."""
        svc = CitationService()
        evidence = [
            SearchResult(
                chunk_id="chunk-sec-2",
                score=0.85,
                content="Torque spec is 120 Nm.",
                page_number=8,
                document_id="doc-torque-1",
                section="Section 9",
                section_title=None,
                file_name="torque_manual.pdf",
                metadata={},
            )
        ]
        citations, _ = svc.map_citations("Torque spec is 120 Nm.", evidence)
        assert citations[0].section == "Section 9"

    def test_coverage_count_query_generalized_sections(self):
        """Verify COUNT_QUERY works with non-Hyundai section numbers and titles."""
        norm_svc = QueryNormalizationService()
        cov_svc = EvidenceCoverageService()

        norm = norm_svc.normalize("How many major sections are in this manual?")
        assert norm.intent == "COUNT_QUERY"

        # Evidence from an arbitrary aerospace manual with Chapter 1 to Chapter 4
        evidence = [
            SearchResult(
                chunk_id="chunk-aero-1",
                score=0.9,
                content="Table of Contents: Chapter 1 Flight Controls, Chapter 2 Avionics, Chapter 3 Propulsion, Chapter 4 Landing Gear.",
                page_number=2,
                document_id="doc-aero-1",
                file_name="aerospace_manual.pdf",
                metadata={"source": "document_structure"},
            )
        ]
        res = cov_svc.evaluate_coverage(norm, evidence)
        assert res.answerable is True
        assert res.coverage_status == "COMPLETE"

    def test_coverage_list_query_without_hardcoded_sections(self):
        """Verify LIST_QUERY handles arbitrary sections without requiring range(1, 10)."""
        norm_svc = QueryNormalizationService()
        cov_svc = EvidenceCoverageService()

        norm = norm_svc.normalize("List all the sections of the machine.")
        assert norm.intent == "LIST_QUERY"

        # Document with 14 sections
        evidence = [
            SearchResult(
                chunk_id="chunk-sec-14",
                score=0.95,
                content="Section 14: Emergency Power Shutdown and Battery Backup.",
                page_number=110,
                section_number=14,
                section_title="Emergency Power Shutdown",
                document_id="doc-power-1",
                file_name="power_system.pdf",
                metadata={},
            )
        ]
        res = cov_svc.evaluate_coverage(norm, evidence)
        assert res.answerable is True
        assert res.coverage_status == "COMPLETE"

    def test_coverage_page_number_format_without_hardcoded_notation(self):
        """Verify PAGE_NUMBER_FORMAT handles custom notations without falling back to '2-3'."""
        cov_svc = EvidenceCoverageService()
        norm = NormalizedQuery(
            raw_query="What does 5-12 mean in the page footer?",
            clean_query="what does 5-12 mean in the page footer",
            intent="PAGE_NUMBER_FORMAT",
            extracted_notation="5-12",
        )
        evidence = [
            SearchResult(
                chunk_id="chunk-pf-1",
                score=0.92,
                content="Page number 5-12 indicates Section 5 (Mechatronics) consecutive page 12.",
                page_number=3,
                document_id="doc-1",
                file_name="mechatronics.pdf",
            )
        ]
        res = cov_svc.evaluate_coverage(norm, evidence)
        assert res.answerable is True
        assert res.coverage_status == "COMPLETE"
        assert "5-12" in res.entities_found

    def test_intent_confidence_calibration(self):
        """Verify intent confidence is not hardcoded to 1.0 and is properly calibrated."""
        norm_svc = QueryNormalizationService()

        # Rule-matched intent
        n_count = norm_svc.normalize("How many major sections are in the manual?")
        assert n_count.intent == "COUNT_QUERY"
        assert n_count.intent_confidence >= 0.95

        # Heuristic comparison intent
        n_comp = norm_svc.normalize("Compare the hydraulic system and the electrical system.")
        assert n_comp.intent == "COMPARISON"
        assert n_comp.intent_confidence >= 0.90

        # Generic QA fallback
        n_gen = norm_svc.normalize("The sky is blue and machines work on fuel.")
        assert n_gen.intent == "GENERAL_QA"
        assert n_gen.intent_confidence == 0.60
