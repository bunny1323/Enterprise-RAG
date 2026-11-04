"""
Hallucination and Groundedness Verification Service.
Evaluates generated answer against retrieved evidence to verify faithfulness.
"""
import re
from pydantic import BaseModel
from app.models.retrieval import SearchResult


class GroundednessResult(BaseModel):
    verification_status: str  # "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED"
    score: float
    reason: str
    supported_chunk_ids: list[str] = []
    supported_sentences: list[str] = []


class GroundednessVerificationService:
    """
    Stateless groundedness verifier.
    """

    def verify(
        self,
        answer: str,
        evidence: list[SearchResult],
    ) -> GroundednessResult:
        if not evidence:
            return GroundednessResult(
                verification_status="UNSUPPORTED",
                score=0.0,
                reason="No evidence available to ground the answer.",
            )

        normalized_answer = re.sub(r"\s+", " ", answer.lower()).strip()
        if normalized_answer in {
            "no relevant evidence was found to answer your question.",
            "the provided evidence does not contain enough information to answer this question.",
            "i don't have enough verified information in the knowledge base to answer this question.",
        }:
            return GroundednessResult(
                verification_status="SUPPORTED",
                score=1.0,
                reason="System correctly acknowledged lack of evidence.",
            )

        stop_words = {
            "also", "that", "this", "with", "from", "into", "than", "then",
            "when", "where", "which", "their", "there", "they", "them",
            "does", "have", "been", "were", "will", "would", "should",
            "about", "covers", "cover", "section", "system",
        }
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer)
            if sentence.strip()
        ]
        if not sentences:
            return GroundednessResult(
                verification_status="UNSUPPORTED",
                score=0.0,
                reason="No verifiable answer claims were found.",
            )

        sentence_scores: list[float] = []
        supported_chunk_ids: set[str] = set()
        supported_sentences: list[str] = []
        unsupported_sentences: list[str] = []
        for sentence in sentences:
            terms = {
                word for word in re.findall(r"\w+", sentence.lower())
                if len(word) > 3 and word not in stop_words
            }
            if not terms:
                unsupported_sentences.append(sentence)
                sentence_scores.append(0.0)
                continue

            best_score = 0.0
            best_chunks: list[SearchResult] = []
            for item in evidence:
                evidence_terms = set(re.findall(r"\w+", item.content.lower()))
                score = len(terms & evidence_terms) / len(terms)
                if score > best_score:
                    best_score = score
                    best_chunks = [item]
                elif score == best_score and score > 0:
                    best_chunks.append(item)

            sentence_scores.append(best_score)
            if best_score >= 0.75:
                supported_chunk_ids.update(item.chunk_id for item in best_chunks)
                supported_sentences.append(sentence)
            else:
                unsupported_sentences.append(sentence)

        overlap_ratio = sum(sentence_scores) / len(sentence_scores)
        if not unsupported_sentences:
            status = "SUPPORTED"
        elif supported_chunk_ids:
            status = "PARTIALLY_SUPPORTED"
        else:
            status = "UNSUPPORTED"

        return GroundednessResult(
            verification_status=status,
            score=round(overlap_ratio, 2),
            reason=(
                f"{len(sentences) - len(unsupported_sentences)} of {len(sentences)} answer claims "
                f"matched individual evidence chunks; {len(unsupported_sentences)} claim(s) lack "
                "sufficient support."
            ),
            supported_chunk_ids=sorted(supported_chunk_ids),
            supported_sentences=supported_sentences,
        )
