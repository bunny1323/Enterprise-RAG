"""
PII Detection and Sensitive Data Redaction Hook.
Optional service interface called before chunk indexing when PII_DETECTION_ENABLED is set.
"""
from abc import ABC, abstractmethod
import re
from pydantic import BaseModel


class PIIResult(BaseModel):
    contains_pii: bool
    detected_types: list[str] = []
    redacted_text: str


class PIIDetector(ABC):
    """Abstract interface for PII detection."""

    @abstractmethod
    def detect_and_redact(self, text: str) -> PIIResult:
        """Detect and redact sensitive PII patterns."""
        pass


class SimpleRegexPIIDetector(PIIDetector):
    """
    Default lightweight regex-based PII detector.
    Redacts emails, phone numbers, and SSN patterns.
    """

    def detect_and_redact(self, text: str) -> PIIResult:
        detected: list[str] = []
        redacted = text

        # Email regex
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text):
            detected.append("EMAIL")
            redacted = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]", redacted)

        # Phone regex
        if re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", text):
            detected.append("PHONE")
            redacted = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", redacted)

        # SSN regex
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", text):
            detected.append("SSN")
            redacted = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", redacted)

        return PIIResult(
            contains_pii=len(detected) > 0,
            detected_types=detected,
            redacted_text=redacted,
        )
