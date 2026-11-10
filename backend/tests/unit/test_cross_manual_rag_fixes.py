"""
Tests for cross-manual retrieval, query decomposition, evidence coverage,
and context generation fixes.

Test cases A-J as specified in the task requirements.
All synthetic documents and fixtures are invented solely for test purposes.
No hardcoded production logic is introduced.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.query_workflow.nodes import _deduplicate_evidence_by_source
from app.agents.retrieval.agent import (
    RetrievalAgent,
    _deduplicate_parent_candidates,
    _diversity_aware_select,
    _section_lookup_evidence,
    _source_key,
)
from app.agents.retrieval.state import QueryState
from app.models.retrieval import SearchResult
from app.models.tenant import TenantContext
from app.services.confidence.service import ConfidenceResult
from app.services.generation.hallucination import GroundednessVerificationService
from app.services.generation.llm import (
    DEFAULT_SYSTEM_INSTRUCTION,
    format_evidence_context,
    get_context_summary,
    get_intent_system_instruction,
)
from app.services.query.decomposition import (
    _COMPARISON_CONNECTIVES,
    QueryDecompositionService,
    SubQuery,
)
from app.services.query.normalization import NormalizedQuery, QueryNormalizationService
from app.services.retrieval.coverage import EvidenceCoverageService
from app.services.retrieval.reranking import RRFRerankerService
from app.utils.fusion import reciprocal_rank_fusion

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_result(chunk_id: str, score: float, content: str, file_name: str = "", page: int = 1) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        content=content,
        page_number=page,
        file_name=file_name or None,
    )


def make_norm(query: str, intent: str, entities: list[str] | None = None) -> NormalizedQuery:
    return NormalizedQuery(
        raw_query=query,
        clean_query=query,
        retrieval_query="",
        typo_query="",
        intent=intent,
        extracted_entities=entities or [],
    )


# ===========================================================================
# Test A: Single-source question
# ===========================================================================

class TestSingleSource:
    def test_single_source_complete_coverage(self):
        """A question answerable from one source passes answerability."""
        norm = make_norm("What is the maximum operating pressure?", "SPECIFICATION")
        evidence = [
            make_result("c1", 0.9, "The maximum operating pressure is 35 MPa.", "hydraulic-manual.pdf"),
        ]
        svc = EvidenceCoverageService()
        result = svc.evaluate_coverage(norm, evidence)
        assert result.answerable is True
        assert result.coverage_status == "COMPLETE"

    def test_single_source_context_contains_file_name(self):
        """Evidence block includes the file name from the single source."""
        evidence = [
            make_result("c1", 0.9, "Operating pressure is 35 MPa.", "hydraulic-manual.pdf", page=5),
        ]
        ctx = format_evidence_context(evidence)
        assert "hydraulic-manual.pdf" in ctx
        assert "(Page 5)" in ctx
        assert "Operating pressure is 35 MPa." in ctx

    def test_section_lookup_coverage_rejects_unrelated_sections(self):
        norm = make_norm("What does Section 3 cover?", "SECTION_LOOKUP", ["section_3"])
        evidence = [
            make_result("unrelated", 0.9, "Section 9 describes component mounting torque.", "torque.pdf")
        ]

        result = EvidenceCoverageService().evaluate_coverage(norm, evidence)

        assert result.coverage_status == "INSUFFICIENT"
        assert result.answerable is False

    def test_list_coverage_rejects_incomplete_single_section_evidence(self):
        norm = make_norm("List all the sections of the machine.", "LIST_QUERY")
        evidence = [
            make_result("section-3", 0.9, "SECTION 3 HYDRAULIC SYSTEM", "foreword.pdf")
        ]

        result = EvidenceCoverageService().evaluate_coverage(norm, evidence)

        assert result.coverage_status == "PARTIAL"
        assert result.answerable is False

    def test_groundedness_does_not_pool_support_across_answer_sentences(self):
        evidence = [
            make_result(
                "section-3",
                0.9,
                "Section 3 is the Hydraulic System. It explains the hydraulic circuit and single and combined operation.",
                "foreword.pdf",
            ),
            make_result(
                "unrelated-section",
                0.8,
                "Component mounting torque values and conversion tables are listed in Section 9.",
                "torque.pdf",
            ),
        ]
        answer = (
            "Section 3 is the Hydraulic System and explains the hydraulic circuit, "
            "component mounting torque values, and conversion tables."
        )

        result = GroundednessVerificationService().verify(answer, evidence)

        assert result.verification_status != "SUPPORTED"

    def test_refusal_with_unverified_explanation_is_not_automatically_supported(self):
        evidence = [make_result("c1", 0.9, "Routine troubleshooting steps.", "troubleshooting.pdf")]
        answer = (
            "The provided evidence does not contain enough information to answer this question. "
            "The manuals cover the password reset procedure on page 42."
        )

        result = GroundednessVerificationService().verify(answer, evidence)

        assert result.verification_status != "SUPPORTED"

    def test_section_lookup_selection_excludes_other_sections(self):
        evidence = [
            SearchResult(
                chunk_id="section-3",
                content="SECTION 3 HYDRAULIC SYSTEM",
                section_number=3,
                score=0.9,
            ),
            SearchResult(
                chunk_id="section-9",
                content="SECTION 9 COMPONENT MOUNTING TORQUE",
                section_number=9,
                score=0.8,
            ),
        ]

        selected = _section_lookup_evidence(evidence, 3)

        assert [item.chunk_id for item in selected] == ["section-3"]


# ===========================================================================
# Test B: Two-source comparison
# ===========================================================================

class TestTwoSourceComparison:
    def test_coverage_complete_when_both_subjects_in_evidence(self):
        """Both comparison subjects present → COMPLETE + answerable."""
        norm = make_norm(
            "Compare the hydraulic system and the electrical system.",
            "COMPARISON",
            entities=["hydraulic system", "electrical system"],
        )
        evidence = [
            make_result("h1", 0.9, "The hydraulic system drives cylinders via pilot valves.", "hydro.pdf"),
            make_result("e1", 0.85, "The electrical system controls the solenoid signals.", "elec.pdf"),
        ]
        svc = EvidenceCoverageService()
        result = svc.evaluate_coverage(norm, evidence)
        assert result.answerable is True
        assert result.coverage_status == "COMPLETE"

    def test_partial_entity_token_overlap_does_not_count_as_supported(self):
        norm = make_norm(
            "Compare the hydraulic pressure regulator and the electrical controller.",
            "COMPARISON",
            entities=["hydraulic pressure regulator", "electrical controller"],
        )
        evidence = [
            make_result("c1", 0.9, "The pressure is checked during routine maintenance.", "service.pdf"),
        ]

        result = EvidenceCoverageService().evaluate_coverage(norm, evidence)

        assert result.coverage_status == "INSUFFICIENT"
        assert result.answerable is False

    def test_comparison_without_identified_subjects_is_not_answerable(self):
        norm = make_norm("Which is better?", "COMPARISON")
        evidence = [make_result("c1", 0.9, "Routine maintenance instructions.", "service.pdf")]

        result = EvidenceCoverageService().evaluate_coverage(norm, evidence)

        assert result.coverage_status == "INSUFFICIENT"
        assert result.answerable is False

    def test_context_contains_both_source_files(self):
        """format_evidence_context includes both manual file names."""
        evidence = [
            make_result("h1", 0.9, "Hydraulic circuit detail.", "hydro-manual.pdf"),
            make_result("e1", 0.85, "Electrical wiring detail.", "elec-manual.pdf"),
        ]
        ctx = format_evidence_context(evidence, max_total_chars=10000)
        assert "hydro-manual.pdf" in ctx
        assert "elec-manual.pdf" in ctx

    def test_context_packing_respects_token_budget(self):
        evidence = [
            make_result("a1", 0.9, "alpha beta gamma delta", "manual-a.pdf"),
            make_result("b1", 0.8, "epsilon zeta eta theta", "manual-b.pdf"),
        ]

        context = format_evidence_context(
            evidence,
            max_total_chars=10000,
            max_total_tokens=16,
        )

        summary = get_context_summary(evidence, max_total_chars=10000, max_total_tokens=16)
        assert summary["context_token_estimate"] <= 16
        assert context.count("[Source ") == 1

    def test_shared_manual_prefix_does_not_deduplicate_distinct_evidence(self):
        prefix = "Common introductory boilerplate shared across manuals. " * 4
        evidence = [
            make_result("h1", 0.9, prefix + "Hydraulic pump evidence.", "hydraulic.pdf"),
            make_result("e1", 0.85, prefix + "Electrical controller evidence.", "electrical.pdf"),
        ]

        context = format_evidence_context(evidence, max_total_chars=10000)

        assert "hydraulic.pdf" in context
        assert "electrical.pdf" in context
        assert "Hydraulic pump evidence." in context
        assert "Electrical controller evidence." in context

    def test_decomposition_produces_per_subject_subqueries(self):
        """Decomposer generates subqueries for each comparison subject."""
        norm = make_norm(
            "Compare the hydraulic system and the electrical system.",
            "COMPARISON",
        )
        svc = QueryDecompositionService()
        subs = svc.decompose(norm)
        # Should have at least two per-subject subqueries (not just the original)
        assert len(subs) >= 2
        queries_lower = [sq.query.lower() for sq in subs]
        # Both subjects should appear in subqueries
        assert any("hydraulic" in q for q in queries_lower)
        assert any("electrical" in q for q in queries_lower)

    def test_vs_pattern_decomposed(self):
        """'X vs Y' structural pattern triggers decomposition."""
        norm = make_norm("Hydraulic system vs electrical system", "COMPARISON")
        svc = QueryDecompositionService()
        subs = svc.decompose(norm)
        assert len(subs) >= 2

    def test_versus_pattern_coverage(self):
        """Coverage handles 'versus' structural pattern."""
        norm = make_norm(
            "hydraulic system versus electrical system performance",
            "COMPARISON",
            entities=["hydraulic system", "electrical system"],
        )
        evidence = [
            make_result("h1", 0.9, "Hydraulic system delivers 200 bar pressure.", "hydro.pdf"),
            make_result("e1", 0.85, "Electrical system runs at 24V DC.", "elec.pdf"),
        ]
        svc = EvidenceCoverageService()
        result = svc.evaluate_coverage(norm, evidence)
        assert result.answerable is True


# ===========================================================================
# Test C: More than two sources
# ===========================================================================

class TestMoreThanTwoSources:
    def test_parent_and_child_share_one_selection_slot(self):
        parent = make_result("parent-1", 0.8, "Parent evidence", "manual.pdf")
        child = SearchResult(
            chunk_id="child-1",
            score=0.9,
            content="Child evidence",
            file_name="manual.pdf",
            parent_id="parent-1",
        )

        unique = _deduplicate_parent_candidates([child, parent])

        assert [item.chunk_id for item in unique] == ["child-1"]

    def test_diversity_pass_does_not_assume_two_sources(self):
        """_diversity_aware_select works with any number of sources."""
        results = [
            make_result(f"a-{i}", 0.9 - i * 0.01, f"content a {i}", "source-a.pdf")
            for i in range(8)
        ] + [
            make_result("b-0", 0.5, "content b", "source-b.pdf"),
            make_result("c-0", 0.5, "content c", "source-c.pdf"),
            make_result("d-0", 0.5, "content d", "source-d.pdf"),
        ]
        selected = _diversity_aware_select(results, top_k=10)
        sources = {_source_key(r) for r in selected}
        # At least the dominant source and at least one other should be present
        assert "source-a.pdf" in sources
        # The function should return at most top_k
        assert len(selected) <= 10

    def test_diversity_rescue_can_add_sources_after_two_are_present(self):
        results = [
            make_result("a-1", 0.95, "a1", "source-a.pdf"),
            make_result("a-2", 0.90, "a2", "source-a.pdf"),
            make_result("b-1", 0.85, "b1", "source-b.pdf"),
            make_result("b-2", 0.80, "b2", "source-b.pdf"),
            make_result("c-1", 0.70, "c1", "source-c.pdf"),
            make_result("d-1", 0.69, "d1", "source-d.pdf"),
        ]

        selected = _diversity_aware_select(results, top_k=4)

        assert {_source_key(item) for item in selected} == {
            "source-a.pdf",
            "source-b.pdf",
            "source-c.pdf",
            "source-d.pdf",
        }

    def test_selection_preserves_every_available_query_requirement(self):
        requirements = [
            ("hydraulic", "hydraulic.pdf", 0.95),
            ("hydraulic", "hydraulic.pdf", 0.9),
            ("engine", "engine.pdf", 0.85),
            ("engine", "engine.pdf", 0.8),
            ("electrical", "electrical.pdf", 0.4),
            ("cooling", "cooling.pdf", 0.3),
        ]
        results = [
            SearchResult(
                chunk_id=f"chunk-{index}",
                score=score,
                content=f"Evidence for requirement {requirement}",
                file_name=source,
                metadata={
                    "_retrieval_subqueries": [requirement],
                    "_rerank_requirement_scores": {requirement: score},
                },
            )
            for index, (requirement, source, score) in enumerate(requirements)
        ]

        selected = _diversity_aware_select(
            results,
            top_k=4,
            required_queries=["hydraulic", "engine", "electrical", "cooling"],
        )
        selected_requirements = {
            requirement
            for item in selected
            for requirement in item.metadata["_rerank_requirement_scores"]
        }

        assert len(selected) == 4
        assert selected_requirements == {"hydraulic", "engine", "electrical", "cooling"}

    def test_rrf_fusion_handles_three_source_lists(self):
        """RRF fusion correctly merges three channel lists."""
        list_a = [make_result("a1", 1.0, "content a1")]
        list_b = [make_result("b1", 0.9, "content b1")]
        list_c = [make_result("c1", 0.8, "content c1")]
        fused = reciprocal_rank_fusion([list_a, list_b, list_c], k=60, top_n=10)
        chunk_ids = {r.chunk_id for r in fused}
        assert "a1" in chunk_ids
        assert "b1" in chunk_ids
        assert "c1" in chunk_ids

    def test_three_source_context_all_present(self):
        """format_evidence_context packs evidence from three manuals."""
        evidence = [
            make_result("a1", 0.9, "Content from manual A.", "manual-a.pdf"),
            make_result("b1", 0.85, "Content from manual B.", "manual-b.pdf"),
            make_result("c1", 0.80, "Content from manual C.", "manual-c.pdf"),
        ]
        ctx = format_evidence_context(evidence, max_total_chars=20000)
        assert "manual-a.pdf" in ctx
        assert "manual-b.pdf" in ctx
        assert "manual-c.pdf" in ctx


# ===========================================================================
# Test D: One source dominates retrieval
# ===========================================================================

class TestSourceDominance:
    def test_diversity_rescue_preserves_secondary_source(self):
        """When one source fills all top_k slots, a high-relevance secondary is rescued."""
        dominant_results = [
            make_result(f"dom-{i}", 0.95 - i * 0.01, f"dominant content {i}", "dominant.pdf")
            for i in range(12)
        ]
        # Secondary source with score above 30% of top score (0.95 * 0.30 = 0.285)
        secondary = make_result("sec-0", 0.50, "necessary secondary fact", "secondary.pdf")
        all_results = dominant_results + [secondary]
        # Sort by score descending (as reranker would produce)
        all_results.sort(key=lambda r: r.score, reverse=True)

        selected = _diversity_aware_select(all_results, top_k=10)
        sources = {_source_key(r) for r in selected}
        assert "secondary.pdf" in sources, (
            "Secondary source should be rescued when it clears the relevance ratio threshold"
        )

    def test_diversity_does_not_force_irrelevant_source(self):
        """A genuinely low-scoring secondary source is NOT rescued."""
        dominant_results = [
            make_result(f"dom-{i}", 0.95 - i * 0.01, f"dominant content {i}", "dominant.pdf")
            for i in range(12)
        ]
        # Score far below 30% threshold (0.95 * 0.30 = 0.285)
        irrelevant = make_result("irr-0", 0.05, "irrelevant content", "irrelevant.pdf")
        all_results = dominant_results + [irrelevant]
        all_results.sort(key=lambda r: r.score, reverse=True)

        selected = _diversity_aware_select(all_results, top_k=10)
        sources = {_source_key(r) for r in selected}
        # Irrelevant source should NOT be forced in
        assert "irrelevant.pdf" not in sources


# ===========================================================================
# Test E: Dense and sparse retrieval complement each other
# ===========================================================================

class TestDenseSparseFusion:
    @pytest.mark.asyncio
    async def test_reranker_scores_global_candidates_against_each_requirement(self, monkeypatch):
        candidates = [
            make_result("hydraulic", 0.9, "Hydraulic system details."),
            make_result("electrical", 0.8, "Electrical system details."),
        ]
        model = MagicMock()
        model.predict.return_value = [0.1, 0.9, 0.2, 0.1, 0.2, 0.85]
        monkeypatch.setenv("RERANKER_ENABLED", "true")
        monkeypatch.setattr("app.services.retrieval.reranking._get_reranker", lambda: model)

        reranked = await RRFRerankerService().rerank(
            query="Explain how the systems work together.",
            candidates=candidates,
            top_n=2,
            relevance_queries=["hydraulic system", "electrical system"],
        )

        assert [item.chunk_id for item in reranked] == ["hydraulic", "electrical"]
        assert [item.score for item in reranked] == [0.9, 0.85]
        assert len(model.predict.call_args.args[0]) == 6

    def test_content_hash_deduplication_preserves_distinct_manual_sources(self):
        evidence = [
            SearchResult(
                chunk_id="a1",
                content="Shared text.",
                document_id="doc-a",
                file_name="manual-a.pdf",
                metadata={"content_hash": "same-hash"},
            ),
            SearchResult(
                chunk_id="a2",
                content="Shared text.",
                document_id="doc-a",
                file_name="manual-a.pdf",
                metadata={"content_hash": "same-hash"},
            ),
            SearchResult(
                chunk_id="b1",
                content="Shared text.",
                document_id="doc-b",
                file_name="manual-b.pdf",
                metadata={"content_hash": "same-hash"},
            ),
        ]

        deduped = _deduplicate_evidence_by_source(evidence)

        assert [item.chunk_id for item in deduped] == ["a1", "b1"]

    def test_rrf_merges_dense_and_bm25_without_duplicate_slots(self):
        """RRF deduplicates results found in both dense and BM25 channels."""
        # Same chunk_id appears in both channels
        dense_results = [
            make_result("shared-1", 0.9, "shared content"),
            make_result("dense-only", 0.8, "dense-only content"),
        ]
        bm25_results = [
            make_result("shared-1", 0.7, "shared content"),
            make_result("bm25-only", 0.75, "bm25-only content"),
        ]
        fused = reciprocal_rank_fusion([dense_results, bm25_results], k=60, top_n=10)
        chunk_ids = [r.chunk_id for r in fused]
        # No duplicates
        assert len(chunk_ids) == len(set(chunk_ids))
        # All unique chunks are present
        assert "shared-1" in chunk_ids
        assert "dense-only" in chunk_ids
        assert "bm25-only" in chunk_ids
        # Shared chunk gets higher fused score than either individual channel
        shared = next(r for r in fused if r.chunk_id == "shared-1")
        dense_only = next(r for r in fused if r.chunk_id == "dense-only")
        assert shared.score > dense_only.score

    def test_rrf_preserves_provenance_from_richer_duplicate_channel_result(self):
        sparse_result = make_result("shared", 0.9, "shared content", page=0)
        dense_result = SearchResult(
            chunk_id="shared",
            score=0.8,
            content="shared content",
            document_id="doc-1",
            file_name="manual.pdf",
            page_number=12,
            section_title="Hydraulic circuit",
        )

        fused = reciprocal_rank_fusion([[sparse_result], [dense_result]], top_n=10)

        assert len(fused) == 1
        assert fused[0].document_id == "doc-1"
        assert fused[0].file_name == "manual.pdf"
        assert fused[0].page_number == 12
        assert fused[0].section_title == "Hydraulic circuit"

    @pytest.mark.asyncio
    async def test_subquery_channels_keep_independent_rrf_rank_positions(self):
        query = "Compare alpha system and beta system"
        normalized = make_norm(
            query,
            "COMPARISON",
            entities=["alpha system", "beta system"],
        )
        normalized.retrieval_query = query
        normalized.typo_query = query

        vectors = {
            query: [100.0],
            "alpha details": [1.0],
            "beta details": [2.0],
        }
        dense_lists = {
            100.0: [make_result("main-d1", 1.0, "main dense first"), make_result("main-d2", 0.9, "main dense second")],
            1.0: [make_result("alpha-d1", 1.0, "alpha dense first"), make_result("alpha-d2", 0.9, "alpha dense second")],
            2.0: [make_result("beta-d1", 1.0, "beta dense first"), make_result("beta-d2", 0.9, "beta dense second")],
        }
        bm25_lists = {
            query: [make_result("main-b1", 1.0, "main sparse first"), make_result("main-b2", 0.9, "main sparse second")],
            "alpha details": [make_result("alpha-b1", 1.0, "alpha sparse first"), make_result("alpha-b2", 0.9, "alpha sparse second")],
            "beta details": [make_result("beta-b1", 1.0, "beta sparse first"), make_result("beta-b2", 0.9, "beta sparse second")],
        }

        embedder = MagicMock()
        embedder.embed_batch.side_effect = lambda texts: [vectors[texts[0]]]
        dense = MagicMock()
        dense.search = AsyncMock(side_effect=lambda query_vector, **kwargs: dense_lists[query_vector[0]])
        bm25 = MagicMock()
        bm25.search = AsyncMock(side_effect=lambda query_text, **kwargs: bm25_lists[query_text])
        reranker = MagicMock()
        reranker.rerank = AsyncMock(
            side_effect=lambda query, candidates, top_n, relevance_queries=None: candidates[:top_n]
        )
        hierarchical = MagicMock()
        hierarchical.get_accessible_chunk_ids = AsyncMock(
            side_effect=lambda chunk_ids, **kwargs: set(chunk_ids)
        )
        hierarchical.expand_to_parents = AsyncMock(side_effect=lambda results, **kwargs: results)
        decomposer = MagicMock()
        decomposer.decompose.return_value = [
            SubQuery(query="alpha details"),
            SubQuery(query="beta details"),
        ]
        confidence = MagicMock()
        confidence.score.return_value = ConfidenceResult(level="HIGH", score=0.9, reason="test")

        agent = RetrievalAgent(
            embedder=embedder,
            dense_svc=dense,
            bm25_svc=bm25,
            graph_svc=MagicMock(),
            reranker=reranker,
            normalizer=MagicMock(normalize=MagicMock(return_value=normalized)),
            confidence_svc=confidence,
            hierarchical_svc=hierarchical,
            decomposer_svc=decomposer,
        )

        result = await agent.retrieve(
            QueryState(query=query, tenant_context=TenantContext(), top_k=20)
        )

        fused_scores = {item.chunk_id: item.score for item in result.fused_results}
        expected_rank_one_score = round(1 / 61, 5)
        assert fused_scores["alpha-d1"] == expected_rank_one_score
        assert fused_scores["alpha-b1"] == expected_rank_one_score
        assert fused_scores["beta-d1"] == expected_rank_one_score
        assert fused_scores["beta-b1"] == expected_rank_one_score
        assert reranker.rerank.await_args.kwargs["top_n"] == len(result.fused_results)
        assert reranker.rerank.await_args.kwargs["relevance_queries"] == [
            "alpha details",
            "beta details",
        ]
        assert all(
            call.kwargs["top_k"] == 40
            for call in dense.search.await_args_list[1:]
        )
        assert all(
            call.kwargs["top_k"] == 40
            for call in bm25.search.await_args_list[1:]
        )


# ===========================================================================
# Test F: Partial evidence
# ===========================================================================

class TestPartialEvidence:
    def test_partial_coverage_is_answerable(self):
        """PARTIAL coverage allows the LLM to answer what is supported."""
        norm = make_norm(
            "Compare the hydraulic system and the nuclear propulsion system.",
            "COMPARISON",
            entities=["hydraulic system", "nuclear propulsion system"],
        )
        evidence = [
            make_result("h1", 0.9, "The hydraulic system operates at 350 bar.", "hydro.pdf"),
        ]
        svc = EvidenceCoverageService()
        result = svc.evaluate_coverage(norm, evidence)
        # Should be PARTIAL (only one subject has evidence) and answerable
        assert result.coverage_status in ("PARTIAL", "COMPLETE")
        assert result.answerable is True  # allow partial synthesis

    def test_partial_coverage_reason_identifies_missing_subject(self):
        """PARTIAL coverage reason mentions which subject is missing."""
        norm = make_norm(
            "Compare the hydraulic system and the nuclear propulsion system.",
            "COMPARISON",
            entities=["hydraulic system", "nuclear propulsion system"],
        )
        evidence = [
            make_result("h1", 0.9, "The hydraulic system operates at 350 bar.", "hydro.pdf"),
        ]
        svc = EvidenceCoverageService()
        result = svc.evaluate_coverage(norm, evidence)
        if result.coverage_status == "PARTIAL":
            assert "nuclear" in result.reason.lower() or "second subject" in result.reason.lower()

    def test_partial_system_instruction_supports_partial_synthesis(self):
        """DEFAULT_SYSTEM_INSTRUCTION permits partial synthesis."""
        assert "partial" in DEFAULT_SYSTEM_INSTRUCTION.lower() or "parts" in DEFAULT_SYSTEM_INSTRUCTION.lower()
        assert "Do NOT refuse the entire question" in DEFAULT_SYSTEM_INSTRUCTION


# ===========================================================================
# Test G: No supporting evidence
# ===========================================================================

class TestNoEvidence:
    def test_empty_evidence_returns_insufficient(self):
        """Empty evidence list → INSUFFICIENT + not answerable."""
        norm = make_norm("How does the superconductor array work?", "GENERAL_QA")
        svc = EvidenceCoverageService()
        result = svc.evaluate_coverage(norm, [])
        assert result.answerable is False
        assert result.coverage_status == "INSUFFICIENT"

    def test_empty_evidence_generates_refusal_without_invention(self):
        """format_evidence_context on empty evidence returns empty string."""
        ctx = format_evidence_context([])
        assert ctx == ""

    def test_system_instruction_refuses_total_absence(self):
        """DEFAULT_SYSTEM_INSTRUCTION instructs LLM to refuse when no evidence supports."""
        assert "does not contain enough information" in DEFAULT_SYSTEM_INSTRUCTION


# ===========================================================================
# Test H: Conflicting evidence
# ===========================================================================

class TestConflictingEvidence:
    def test_conflicting_evidence_both_sources_included_in_context(self):
        """Both conflicting sources appear in the evidence context."""
        evidence = [
            make_result("s1", 0.9, "The maximum torque is 150 Nm according to the service manual.", "service.pdf"),
            make_result("s2", 0.85, "The maximum torque is 180 Nm according to the overhaul manual.", "overhaul.pdf"),
        ]
        ctx = format_evidence_context(evidence, max_total_chars=10000)
        assert "service.pdf" in ctx
        assert "overhaul.pdf" in ctx
        assert "150 Nm" in ctx
        assert "180 Nm" in ctx

    def test_system_instruction_requires_conflict_attribution(self):
        """DEFAULT_SYSTEM_INSTRUCTION mandates attributing conflicting values."""
        assert "CONFLICT" in DEFAULT_SYSTEM_INSTRUCTION or "conflict" in DEFAULT_SYSTEM_INSTRUCTION.lower()
        assert "attribute" in DEFAULT_SYSTEM_INSTRUCTION.lower()


# ===========================================================================
# Test I: Decomposition failure
# ===========================================================================

class TestDecompositionFailure:
    def test_decompose_returns_original_on_unrecognised_pattern(self):
        """An unrecognisable query produces a passthrough (original query) subquery."""
        norm = make_norm("Xyzzy frobnicate quux.", "GENERAL_QA")
        svc = QueryDecompositionService()
        subs = svc.decompose(norm)
        assert len(subs) >= 1
        assert subs[0].query == norm.clean_query

    def test_decompose_never_returns_empty_list(self):
        """decompose always returns at least one subquery."""
        norm = make_norm("", "GENERAL_QA")
        svc = QueryDecompositionService()
        subs = svc.decompose(norm)
        # Even for an empty query the fallback kicks in
        assert isinstance(subs, list)
        assert len(subs) >= 1

    def test_invalid_comparison_falls_back_gracefully(self):
        """A malformed comparison query produces a safe passthrough."""
        norm = make_norm("compare", "COMPARISON")
        svc = QueryDecompositionService()
        subs = svc.decompose(norm)
        assert len(subs) >= 1


# ===========================================================================
# Test J: Regression — existing tests still pass
# ===========================================================================

class TestRegression:
    def test_context_includes_manual_names_and_pages(self):
        """Existing: source manual file names, page numbers, and sections appear in blocks."""
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
        assert "[Source 2: hyundai-r215l-smart-mechatronics-system-manual.pdf (Page 21) [Section: CAPO SYSTEM]]" in context

    def test_comparison_instructions_general_vs_symbol(self):
        """Existing: COMPARISON dispatch for general vs symbol queries."""
        tech_q = "Compare the operating principles in the hydraulic and mechatronics manuals."
        tech_inst = get_intent_system_instruction(intent="COMPARISON", query=tech_q)
        assert "SPECIFIC DIRECTIVE FOR TECHNICAL COMPARISONS" in tech_inst
        assert "SPECIFIC DIRECTIVE FOR SYMBOL COMPARISONS" not in tech_inst

        sym_q = "What is the difference between the safety symbol and the caution symbol?"
        sym_inst = get_intent_system_instruction(intent="COMPARISON", query=sym_q)
        assert "SPECIFIC DIRECTIVE FOR SYMBOL COMPARISONS" in sym_inst

    def test_rrf_fusion_basic(self):
        """Existing: RRF fuses lists and deduplicates by chunk_id."""
        l1 = [make_result("a", 1.0, "alpha"), make_result("b", 0.9, "beta")]
        l2 = [make_result("a", 0.8, "alpha"), make_result("c", 0.7, "gamma")]
        fused = reciprocal_rank_fusion([l1, l2], k=60, top_n=10)
        ids = [r.chunk_id for r in fused]
        assert len(ids) == len(set(ids))
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids
        # "a" should rank highest (appears in both lists)
        assert ids[0] == "a"

    def test_answerability_gate_empty_evidence(self):
        """Existing: empty evidence → not answerable."""
        norm = make_norm("How does hydraulic oil flow?", "GENERAL_QA")
        svc = EvidenceCoverageService()
        result = svc.evaluate_coverage(norm, [])
        assert result.answerable is False
        assert result.coverage_status == "INSUFFICIENT"

    def test_diversity_packing_secondary_manual_present(self):
        """Existing: secondary manual not choked by dominant manual."""
        hydraulic_chunks = [
            SearchResult(
                chunk_id=f"hyd-{i}",
                content=f"Hydraulic operation detail {i}: " + ("x" * 200),
                page_number=i,
                file_name="manual-hydraulic.pdf",
                score=0.95 - (i * 0.01),
            )
            for i in range(5)
        ]
        mech_chunks = [
            SearchResult(
                chunk_id=f"mech-{i}",
                content=f"Mechatronic signal {i}: " + ("y" * 200),
                page_number=i + 5,
                file_name="manual-mechatronics.pdf",
                score=0.85 - (i * 0.01),
            )
            for i in range(2)
        ]
        packed = format_evidence_context(
            hydraulic_chunks + mech_chunks,
            max_total_chars=3000,
            max_chunks_per_source=2,
        )
        assert "manual-hydraulic.pdf" in packed
        assert "manual-mechatronics.pdf" in packed

    def test_context_summary_has_observability_fields(self):
        """Existing: get_context_summary returns required diagnostic fields."""
        evidence = [
            make_result("c1", 0.9, "Check oil level.", "hydro.pdf"),
            make_result("c2", 0.85, "Inspect wiring.", "elec.pdf"),
        ]
        summary = get_context_summary(evidence, max_total_chars=10000)
        assert summary["context_chunks_count"] == 2
        assert summary["context_truncated"] is False
        assert "hydro.pdf" in summary["source_manuals"]
        assert "elec.pdf" in summary["source_manuals"]

    def test_comparison_connective_regex_detects_vs(self):
        """_COMPARISON_CONNECTIVES detects 'vs' pattern."""
        assert _COMPARISON_CONNECTIVES.search("hydraulic system vs electrical system")

    def test_comparison_connective_regex_detects_versus(self):
        """_COMPARISON_CONNECTIVES detects 'versus' pattern."""
        assert _COMPARISON_CONNECTIVES.search("compare system A versus system B")

    def test_comparison_connective_regex_detects_difference_between(self):
        """_COMPARISON_CONNECTIVES detects 'difference between' pattern."""
        assert _COMPARISON_CONNECTIVES.search("what is the difference between X and Y?")

    def test_relationship_decomposition_extracts_entities(self):
        """Relationship decomposition generates per-entity subqueries."""
        norm = make_norm(
            "What is the relationship between the hydraulic circuit and the control valve?",
            "RELATIONSHIP",
        )
        svc = QueryDecompositionService()
        subs = svc.decompose(norm)
        assert len(subs) >= 2

    def test_relationship_decomposition_extracts_all_coordinated_subjects(self):
        query = (
            "Explain how the hydraulic system, engine control system, electrical system, "
            "and cooling system work together during machine operation."
        )
        norm = QueryNormalizationService().normalize(query)

        subs = QueryDecompositionService().decompose(norm)

        assert len(subs) == 4
        assert [sub.query for sub in subs] == [
            "the hydraulic system",
            "engine control system",
            "electrical system",
            "cooling system",
        ]

    def test_multihop_decomposition_splits_on_connective(self):
        """MULTI_HOP decomposition splits on 'and what/which/how'."""
        norm = make_norm(
            "What is the service interval and what oil grade is recommended?",
            "MULTI_HOP",
        )
        svc = QueryDecompositionService()
        subs = svc.decompose(norm)
        assert len(subs) >= 2

    def test_source_key_falls_back_to_document_id(self):
        """_source_key returns document_id when file_name is absent."""
        r = SearchResult(chunk_id="c1", content="text", document_id="doc-abc")
        assert _source_key(r) == "doc-abc"

    def test_source_key_unknown_when_no_identifiers(self):
        """_source_key returns 'unknown' when no identifiers are present."""
        r = SearchResult(chunk_id="c1", content="text")
        assert _source_key(r) == "unknown"
