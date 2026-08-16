"""
Confidence Scoring Service.
Evaluates the quality and agreement of retrieved evidence chunks.
"""
from pydantic import BaseModel
from app.models.retrieval import SearchResult


class ConfidenceResult(BaseModel):
    level: str  # "HIGH" | "MEDIUM" | "LOW"
    score: float
    reason: str


class ConfidenceScoringService:
    """
    Stateless confidence scoring engine.
    """

    def score(
        self,
        query: str,
        evidence: list[SearchResult],
    ) -> ConfidenceResult:
        if not evidence:
            return ConfidenceResult(level="LOW", score=0.0, reason="No evidence retrieved")

        top_score = evidence[0].score

        if top_score >= 0.75:
            level = "HIGH"
            reason = f"Strong top evidence score ({top_score:.2f})"
        elif top_score >= 0.45:
            level = "MEDIUM"
            reason = f"Moderate top evidence score ({top_score:.2f})"
        else:
            level = "LOW"
            reason = f"Weak evidence score ({top_score:.2f})"

        return ConfidenceResult(
            level=level,
            score=round(top_score, 3),
            reason=reason,
        )
