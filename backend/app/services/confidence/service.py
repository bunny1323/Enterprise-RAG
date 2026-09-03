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

        # If score is in RRF fusion scale (typically 0.005 to 0.035 for k=60)
        if top_score <= 0.1:
            # Scale against optimal top-rank multi-channel RRF score (approx 0.0328)
            norm_score = min(1.0, top_score / 0.0328)
            if norm_score >= 0.65:
                level = "HIGH"
                reason = f"Strong multi-channel RRF agreement (raw={top_score:.4f}, norm={norm_score:.2f})"
            elif norm_score >= 0.35:
                level = "MEDIUM"
                reason = f"Moderate RRF candidate rank (raw={top_score:.4f}, norm={norm_score:.2f})"
            else:
                level = "LOW"
                reason = f"Weak RRF candidate rank (raw={top_score:.4f}, norm={norm_score:.2f})"
            
            return ConfidenceResult(
                level=level,
                score=round(norm_score, 3),
                reason=reason,
            )

        # Standard 0.0 - 1.0 similarity score scale
        if top_score >= 0.70:
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

