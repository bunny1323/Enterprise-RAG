"""
Step 01 — Validate.
Checks file existence, size limits, and MIME type (must be application/pdf).
"""
from pathlib import Path
from typing import Any

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.models.document import DocumentStatus

logger = get_logger(__name__)

_MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Validate the uploaded file before any processing begins.

    Checks:
    1. File exists at state.storage_path
    2. File size ≤ 100 MB
    3. MIME type is application/pdf

    Args:
        state: Current ingestion state.
        services: Injected service registry (not used in this step).

    Returns:
        Updated state with status=VALIDATING.

    Raises:
        ValueError: If any validation check fails.
    """
    logger.info("step.validate.start", document_id=str(state.document_id))

    path = Path(state.storage_path)

    # ── Check file existence ───────────────────────────────────────────────────
    if not path.exists():
        raise ValueError(f"File not found at storage path: {state.storage_path}")

    # ── Check file size ────────────────────────────────────────────────────────
    file_size = path.stat().st_size
    if file_size > _MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File size {file_size / 1024 / 1024:.1f} MB exceeds 100 MB limit"
        )

    # ── Check MIME type ────────────────────────────────────────────────────────
    mime_type = _detect_mime(str(path))
    if mime_type != "application/pdf":
        raise ValueError(
            f"Invalid file type: expected 'application/pdf', got '{mime_type}'. "
            f"File: {state.filename}"
        )

    logger.info(
        "step.validate.complete",
        document_id=str(state.document_id),
        size_mb=round(file_size / 1024 / 1024, 2),
        mime=mime_type,
    )

    return state.model_copy(update={"status": DocumentStatus.VALIDATING})


def _detect_mime(file_path: str) -> str:
    """Detect MIME type using python-magic, with extension fallback."""
    try:
        import magic  # type: ignore[import-untyped]

        return magic.from_file(file_path, mime=True)
    except ImportError:
        logger.warning("step.validate.magic_unavailable", fallback="extension check")
        # Fallback: check file extension and PDF magic bytes
        return _check_pdf_magic_bytes(file_path)
    except Exception as err:
        logger.warning("step.validate.magic_error", error=str(err))
        return _check_pdf_magic_bytes(file_path)


def _check_pdf_magic_bytes(file_path: str) -> str:
    """Check first 4 bytes for PDF magic signature (%PDF)."""
    try:
        with open(file_path, "rb") as fh:
            header = fh.read(4)
            if header == b"%PDF":
                return "application/pdf"
    except OSError:
        pass
    return "application/octet-stream"
