"""
Unit & Integration Tests for LLM Provider Hierarchy (Groq Primary + Ollama Fallback).

Verifies:
1. Groq succeeds -> answer generated directly.
2. Groq fails -> Ollama fallback is attempted.
3. Both fail -> LLMProviderError raised, returning clean 503 error details.
"""
import pytest
from unittest.mock import AsyncMock, patch
import httpx

from app.models.retrieval import SearchResult
from app.services.generation.llm import (
    GenerationResult,
    LLMProviderError,
    FallbackLLMProvider,
    GroqLLMProvider,
    OllamaProvider,
    format_evidence_context,
)


@pytest.fixture
def dummy_evidence():
    return [
        SearchResult(
            chunk_id="chunk_1",
            content="Section 3 covers the Hydraulic System of the excavator.",
            score=1.0,
            page_number=1,
            section="SECTION 3 HYDRAULIC SYSTEM",
        )
    ]


@pytest.mark.asyncio
async def test_groq_success(dummy_evidence):
    """When Groq succeeds, primary answer is returned and fallback is not called."""
    mock_primary = AsyncMock()
    mock_primary.generate.return_value = GenerationResult(
        answer="Section 3 covers the Hydraulic System.",
        model_name="qwen/qwen3.8-27b",
        prompt_tokens=50,
        completion_tokens=20,
    )
    mock_fallback = AsyncMock()

    orchestrator = FallbackLLMProvider(
        primary=mock_primary,
        fallback=mock_fallback,
        primary_name="groq",
        fallback_name="ollama",
    )

    result = await orchestrator.generate(
        prompt="What does Section 3 cover?",
        evidence=dummy_evidence,
    )

    assert "Hydraulic System" in result.answer
    assert result.model_name == "qwen/qwen3.8-27b"
    mock_primary.generate.assert_called_once()
    mock_fallback.generate.assert_not_called()


@pytest.mark.asyncio
async def test_groq_fails_ollama_fallback_succeeds(dummy_evidence):
    """When Groq fails (e.g. 400 or network error), Ollama fallback is executed."""
    mock_primary = AsyncMock()
    mock_primary.generate.side_effect = LLMProviderError("Groq 400: terms required")

    mock_fallback = AsyncMock()
    mock_fallback.generate.return_value = GenerationResult(
        answer="Section 3 covers the Hydraulic System from Ollama.",
        model_name="qwen2.5:7b",
        prompt_tokens=40,
        completion_tokens=15,
    )

    orchestrator = FallbackLLMProvider(
        primary=mock_primary,
        fallback=mock_fallback,
        primary_name="groq",
        fallback_name="ollama",
    )

    result = await orchestrator.generate(
        prompt="What does Section 3 cover?",
        evidence=dummy_evidence,
    )

    assert "from Ollama" in result.answer
    assert result.model_name == "qwen2.5:7b"
    mock_primary.generate.assert_called_once()
    mock_fallback.generate.assert_called_once()


@pytest.mark.asyncio
async def test_both_providers_fail_raises_clear_error(dummy_evidence):
    """When both Groq and Ollama fail, LLMProviderError with both details is raised."""
    mock_primary = AsyncMock()
    mock_primary.generate.side_effect = LLMProviderError("Groq failed with 400 Bad Request")

    mock_fallback = AsyncMock()
    mock_fallback.generate.side_effect = LLMProviderError("Ollama generation timed out after 60s")

    orchestrator = FallbackLLMProvider(
        primary=mock_primary,
        fallback=mock_fallback,
        primary_name="groq",
        fallback_name="ollama",
    )

    with pytest.raises(LLMProviderError) as exc_info:
        await orchestrator.generate(
            prompt="What does Section 3 cover?",
            evidence=dummy_evidence,
        )

    err_msg = str(exc_info.value)
    assert "Primary provider 'groq' failed" in err_msg
    assert "Fallback provider 'ollama' also failed" in err_msg


@pytest.mark.asyncio
async def test_groq_rate_limit_retries_once_then_succeeds(dummy_evidence):
    provider = GroqLLMProvider(api_key="test-key", max_tokens=128)
    first = httpx.Response(429, headers={"Retry-After": "0"})
    second = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "The documented answer."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        },
    )
    provider._http.post = AsyncMock(side_effect=[first, second])

    result = await provider.generate("What is documented?", dummy_evidence)

    assert result.answer == "The documented answer."
    assert provider._http.post.await_count == 2
    await provider.close()


@pytest.mark.asyncio
async def test_groq_rate_limit_is_bounded_and_raises(dummy_evidence):
    provider = GroqLLMProvider(api_key="test-key", max_tokens=128)
    provider._http.post = AsyncMock(return_value=httpx.Response(429))

    with pytest.raises(LLMProviderError):
        await provider.generate("What is documented?", dummy_evidence)

    assert provider._http.post.await_count == 2
    await provider.close()


def test_evidence_context_skips_large_early_chunks():
    evidence = [
        SearchResult(chunk_id="large", content="x " * 1900, score=1.0),
        SearchResult(chunk_id="relevant", content="1 kg = 2.2046 lb", score=0.9),
    ]

    context = format_evidence_context(evidence, max_total_chars=100)

    assert "1 kg = 2.2046 lb" in context
