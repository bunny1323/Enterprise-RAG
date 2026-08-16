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

        if "no relevant evidence" in answer.lower() or "insufficient evidence" in answer.lower():
            return GroundednessResult(
                verification_status="SUPPORTED",
                score=1.0,
                reason="System correctly acknowledged lack of evidence.",
            )

        # Word overlap verification between answer and evidence text
        evidence_text = " ".join(item.content.lower() for item in evidence)
        answer_words = [w for w in re.findall(r"\w+", answer.lower()) if len(w) > 3]

        if not answer_words:
            return GroundednessResult(verification_status="SUPPORTED", score=1.0, reason="Short answer")

        matches = sum(1 for w in answer_words if w in evidence_text)
        overlap_ratio = matches / len(answer_words)

        if overlap_ratio >= 0.6:
            status = "SUPPORTED"
        elif overlap_ratio >= 0.35:
            status = "PARTIALLY_SUPPORTED"
        else:
            status = "UNSUPPORTED"

        return GroundednessResult(
            verification_status=status,
            score=round(overlap_ratio, 2),
            reason=f"Word overlap grounding score: {overlap_ratio:.2f}",
        )
