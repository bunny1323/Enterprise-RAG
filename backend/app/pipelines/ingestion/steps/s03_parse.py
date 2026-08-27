"""
Step 03 — Parse.
Invokes DocumentParserService to extract structured content from the PDF.
"""
from typing import Any

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.models.document import DocumentStatus
from app.services.document_parser.service import DocumentParserService
from app.config.settings import get_settings

logger = get_logger(__name__)


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Parse the PDF file into structured pages with text blocks, tables, and figures.

    Args:
        state: Current ingestion state (storage_path must be set).
        services: Must contain 'parser' key → DocumentParserService.

    Returns:
        Updated state with parsed_doc, page_count, and status=PARSING.

    Raises:
        Exception: Re-raises any parser error after logging.
    """
    logger.info("step.parse.start", document_id=str(state.document_id))

    parser: DocumentParserService = services["parser"]

    # Run parser (synchronous Docling/PyMuPDF call wrapped in thread pool by pipeline)
    settings = get_settings()

    parsed_doc = await parser.parse(
        state.storage_path,
        timeout=settings.ingestion_timeout_parse,
    )

    page_count = len(parsed_doc.get("pages", []))

    logger.info(
        "step.parse.complete",
        document_id=str(state.document_id),
        pages=page_count,
        tables=sum(len(p.get("tables", [])) for p in parsed_doc.get("pages", [])),
        figures=sum(len(p.get("figures", [])) for p in parsed_doc.get("pages", [])),
    )

    return state.model_copy(
        update={
            "parsed_doc": parsed_doc,
            "page_count": page_count,
            "status": DocumentStatus.PARSING,
        }
    )
