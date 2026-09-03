"""
Query Decomposition Service for Multi-hop and Relationship reasoning.
Breaks down complex queries into focused sub-queries for independent retrieval.
"""
import re
from pydantic import BaseModel
from app.services.query.normalization import NormalizedQuery


class SubQuery(BaseModel):
    query: str
    target_entity: str | None = None
    expected_type: str = "TEXT"


class QueryDecompositionService:
    """
    Decomposes queries into parallel sub-queries for multi-hop or relationship verification.
    """

    def decompose(self, norm: NormalizedQuery) -> list[SubQuery]:
        lowered = norm.clean_query.lower()
        sub_queries: list[SubQuery] = []

        if norm.intent == "RELATIONSHIP":
            # Extract Entity A and Entity B
            # Examples:
            # "relationship between troubleshooting and structure and function"
            # "how is section 2 related to troubleshooting"
            match = re.search(r"between\s+(.+?)\s+and\s+(.+)", lowered)
            if match:
                ent_a = match.group(1).strip()
                ent_b = match.group(2).strip()
                sub_queries.append(SubQuery(query=f"What is {ent_a}?", target_entity=ent_a))
                sub_queries.append(SubQuery(query=f"What is {ent_b}?", target_entity=ent_b))
                sub_queries.append(SubQuery(query=f"{ent_a} related to {ent_b}", target_entity="relationship"))
            else:
                # Fallback for "how is X related to Y"
                match2 = re.search(r"how\s+is\s+(.+?)\s+related\s+to\s+(.+)", lowered)
                if match2:
                    ent_a = match2.group(1).strip()
                    ent_b = match2.group(2).strip()
                    sub_queries.append(SubQuery(query=f"What is {ent_a}?", target_entity=ent_a))
                    sub_queries.append(SubQuery(query=f"What is {ent_b}?", target_entity=ent_b))
                    sub_queries.append(SubQuery(query=f"{ent_a} {ent_b} reference connection", target_entity="relationship"))
                else:
                    sub_queries.append(SubQuery(query=norm.clean_query))

        elif norm.intent == "MULTI_HOP":
            # Split clauses by "and what", "and which", "and how"
            parts = re.split(r"\s+and\s+(?=what|which|how)", norm.clean_query, flags=re.IGNORECASE)
            for part in parts:
                cleaned_part = part.strip().rstrip("?,.")
                if cleaned_part:
                    sub_queries.append(SubQuery(query=cleaned_part))

        if not sub_queries:
            sub_queries.append(SubQuery(query=norm.clean_query))

        return sub_queries
