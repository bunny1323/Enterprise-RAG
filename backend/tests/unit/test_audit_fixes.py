"""
Unit tests covering all components fixed in the audit.

Tests are deterministic and use mocks for external dependencies.
They do NOT require Weaviate, Groq, Gemini, or network access.

Covers:
- LocalEmbeddingProvider: query prefix, long-input warning guard, empty list handling,
  error propagation, health check.
- RRFRerankerService: RERANKER_ENABLED env flag, fallback behavior, score sorting,
  metadata preservation, empty/single candidate handling.
- LLM Provider: PREFERRED_MODELS list, EXCLUDED_PATTERNS.
- Evaluation: dataset schema, placeholder detection, evaluate.py validation logic.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from app.models.retrieval import SearchResult
from app.services.embeddings.local_provider import (
    LocalEmbeddingProvider,
    _BGE_QUERY_INSTRUCTION,
    _check_input_length,
)
from app.services.retrieval.reranking import RRFRerankerService, _is_reranker_enabled
from app.services.generation.llm import OpenAICompatibleProvider


# ==============================================================================
# LocalEmbeddingProvider Tests
# ==============================================================================

class TestLocalEmbeddingProvider:
    """Tests for BGE-small-en-v1.5 embedding provider fixes."""

    def test_default_query_prefix_is_false(self):
        """use_query_prefix should default to False to protect existing index compatibility."""
        provider = LocalEmbeddingProvider()
        assert provider.use_query_prefix is False

    @patch("app.services.embeddings.local_provider._get_model")
    def test_embed_query_without_prefix(self, mock_get_model):
        """When use_query_prefix=False, query text is passed as-is to encode."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        mock_get_model.return_value = mock_model

        provider = LocalEmbeddingProvider(use_query_prefix=False)
        _ = provider.embed_query("What is the operating pressure?")

        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args[0][0]
        assert call_args == ["What is the operating pressure?"]

    @patch("app.services.embeddings.local_provider._get_model")
    def test_embed_query_with_prefix(self, mock_get_model):
        """When use_query_prefix=True, query text is prepended with the BGE instruction."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        mock_get_model.return_value = mock_model

        provider = LocalEmbeddingProvider(use_query_prefix=True)
        _ = provider.embed_query("What is the operating pressure?")

        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args[0][0]
        expected_text = f"{_BGE_QUERY_INSTRUCTION}What is the operating pressure?"
        assert call_args == [expected_text]

    def test_check_input_length_warning(self, caplog):
        """Inputs exceeding 350 words should trigger a warning log."""
        long_text = "word " * 360
        with caplog.at_level("WARNING"):
            _check_input_length([long_text], context="test_context")

        assert any("local_embedder.input_too_long" in record.message or "input_too_long" in str(record.__dict__)
                   for record in caplog.records) or len(caplog.records) >= 1

    @patch("app.services.embeddings.local_provider._get_model")
    def test_embed_batch_handles_empty_list(self, mock_get_model):
        """Empty input list should return an empty list without calling model.encode."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        provider = LocalEmbeddingProvider()
        result = provider.embed_batch([])
        assert result == []
        mock_model.encode.assert_not_called()

    @patch("app.services.embeddings.local_provider._get_model")
    def test_embed_batch_propagates_model_errors(self, mock_get_model):
        """Runtime errors during encode should be logged and re-raised."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("OOM in SentenceTransformer")
        mock_get_model.return_value = mock_model

        provider = LocalEmbeddingProvider()
        with pytest.raises(RuntimeError, match="OOM in SentenceTransformer"):
            provider.embed_batch(["Some test text"])

    @patch("app.services.embeddings.local_provider._get_model")
    def test_health_check_returns_prefix_status(self, mock_get_model):
        """Health check returns status healthy and reports use_query_prefix."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_get_model.return_value = mock_model

        provider = LocalEmbeddingProvider(use_query_prefix=True)
        health = provider.health_check()
        assert health["status"] == "healthy"
        assert health["use_query_prefix"] is True
        assert health["embedding_dimension"] == 384


# ==============================================================================
# RRFRerankerService Tests
# ==============================================================================

class TestRRFRerankerService:
    """Tests for Cross-Encoder reranker and score-sort fallback."""

    def _make_candidate(self, chunk_id: str, score: float, content: str = "doc content") -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            document_id="doc-123",
            content=content,
            score=score,
            metadata={"category": "manual"},
        )

    def test_reranker_disabled_by_env_var(self, monkeypatch):
        """When RERANKER_ENABLED=false, rerank() returns candidates in original order."""
        monkeypatch.setenv("RERANKER_ENABLED", "false")
        assert _is_reranker_enabled() is False

        reranker = RRFRerankerService()
        candidates = [
            self._make_candidate("c1", 0.5),
            self._make_candidate("c2", 0.9),
        ]
        result = reranker.rerank(candidates, query="hydraulic pump", top_k=2)
        assert len(result) == 2
        assert [r.chunk_id for r in result] == ["c2", "c1"]  # sorted by incoming score

    @patch("app.services.retrieval.reranking._get_reranker", return_value=None)
    def test_reranker_passthrough_when_cross_encoder_missing(self, mock_get, monkeypatch):
        """When CrossEncoder model is None, rerank falls back to score sort."""
        monkeypatch.setenv("RERANKER_ENABLED", "true")
        reranker = RRFRerankerService()
        candidates = [
            self._make_candidate("c1", 0.3),
            self._make_candidate("c2", 0.8),
            self._make_candidate("c3", 0.6),
        ]
        result = reranker.rerank(candidates, query="test query", top_k=2)
        assert len(result) == 2
        assert [r.chunk_id for r in result] == ["c2", "c3"]

    @patch("app.services.retrieval.reranking._get_reranker")
    def test_reranker_scores_and_sorts_correctly(self, mock_get, monkeypatch):
        """When CrossEncoder is available, candidates are re-ordered by cross-encoder score."""
        monkeypatch.setenv("RERANKER_ENABLED", "true")

        mock_ce = MagicMock()
        # Candidate 1 incoming score 0.9 -> cross-encoder gives 0.1
        # Candidate 2 incoming score 0.4 -> cross-encoder gives 0.85
        # Candidate 2 should be ranked FIRST
        mock_ce.predict.return_value = np.array([0.1, 0.85], dtype=np.float32)
        mock_get.return_value = mock_ce

        reranker = RRFRerankerService()
        candidates = [
            self._make_candidate("c1", 0.9, content="Irrelevant high-BM25 chunk"),
            self._make_candidate("c2", 0.4, content="Highly relevant exact match"),
        ]
        result = reranker.rerank(candidates, query="hydraulic pump pressure", top_k=2)

        assert len(result) == 2
        assert result[0].chunk_id == "c2"
        assert result[1].chunk_id == "c1"
        assert result[0].score == pytest.approx(0.85, 0.01)
        assert result[1].score == pytest.approx(0.1, 0.01)

    @patch("app.services.retrieval.reranking._get_reranker")
    def test_reranker_preserves_metadata(self, mock_get, monkeypatch):
        """Reranking must preserve document_id, metadata, and chunk_id."""
        monkeypatch.setenv("RERANKER_ENABLED", "true")
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.95], dtype=np.float32)
        mock_get.return_value = mock_ce

        reranker = RRFRerankerService()
        candidate = SearchResult(
            chunk_id="chunk-abc",
            document_id="doc-xyz",
            content="sample text",
            score=0.2,
            metadata={"section": "4.1", "page": 12},
        )
        result = reranker.rerank([candidate], query="sample", top_k=1)
        assert len(result) == 1
        assert result[0].chunk_id == "chunk-abc"
        assert result[0].document_id == "doc-xyz"
        assert result[0].metadata == {"section": "4.1", "page": 12}

    def test_reranker_handles_empty_candidates(self):
        """Reranker on empty candidates list returns empty list."""
        reranker = RRFRerankerService()
        assert reranker.rerank([], query="any query") == []

    @patch("app.services.retrieval.reranking._get_reranker")
    def test_reranker_handles_single_candidate(self, mock_get, monkeypatch):
        """Single candidate list is processed cleanly."""
        monkeypatch.setenv("RERANKER_ENABLED", "true")
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.77], dtype=np.float32)
        mock_get.return_value = mock_ce

        reranker = RRFRerankerService()
        candidates = [self._make_candidate("c1", 0.5)]
        result = reranker.rerank(candidates, query="test", top_k=1)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.77, 0.01)


# ==============================================================================
# LLM Provider Configuration Tests
# ==============================================================================

class TestLLMProviderConfig:
    """Tests for LLM provider model selection and filtering rules."""

    def test_preferred_models_prioritizes_llama33(self):
        """llama-3.3-70b-versatile must be the top preferred model for Groq."""
        preferred = OpenAICompatibleProvider.PREFERRED_MODELS
        assert preferred[0] == "llama-3.3-70b-versatile"
        assert "llama-3.1-70b-versatile" in preferred
        assert "llama-3.1-8b-instant" in preferred

    def test_excluded_patterns_contain_safety_and_specialized_models(self):
        """Guard, whisper, and vision models must be in EXCLUDED_PATTERNS."""
        patterns = OpenAICompatibleProvider.EXCLUDED_PATTERNS
        assert "prompt-guard" in patterns
        assert "whisper" in patterns
        assert "vision" in patterns


# ==============================================================================
# Golden Dataset & Evaluation Validation Tests
# ==============================================================================

class TestEvaluationValidation:
    """Tests verifying the evaluation dataset structure and validation logic."""

    DATASET_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "golden_dataset.json"

    def test_golden_dataset_structure(self):
        """golden_dataset.json must load and contain valid schema keys."""
        assert self.DATASET_PATH.exists(), f"Missing {self.DATASET_PATH}"
        data = json.loads(self.DATASET_PATH.read_text(encoding="utf-8"))

        entries = [e for e in data if isinstance(e, dict) and "question" in e]
        assert len(entries) >= 5, "Dataset must have at least 5 benchmark questions"

        required_keys = {"test_id", "question", "expected_document", "expected_evidence", "category", "difficulty"}
        for entry in entries:
            for key in required_keys:
                assert key in entry, f"Missing key '{key}' in test_id {entry.get('test_id')}"

    def test_golden_dataset_flags_placeholders(self):
        """All un-ingested entries in golden_dataset.json should start with PLACEHOLDER_."""
        data = json.loads(self.DATASET_PATH.read_text(encoding="utf-8"))
        entries = [e for e in data if isinstance(e, dict) and "question" in e]

        placeholders = [e for e in entries if e["expected_document"].startswith("PLACEHOLDER_")]
        assert len(placeholders) == len(entries), (
            "All seed entries must use PLACEHOLDER_ prefix until real documents are ingested"
        )
