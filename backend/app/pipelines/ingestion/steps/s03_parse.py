"""
Step 03 — Parse.

Invokes DocumentParserService to extract structured content from the PDF.
"""

from typing import Any

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.models.document import DocumentStatus
from app.services.document_parser.service import DocumentParserService

logger = get_logger(__name__)


async def step(
    state: IngestionState,
    services: dict[str, Any],
) -> IngestionState:
    """
    Parse the PDF file into structured pages with text blocks,
    tables, and figures.

    Args:
        state: Current ingestion state.
        services: Must contain a 'parser' key with a
            DocumentParserService instance.

    Returns:
        Updated ingestion state containing parsed_doc,
        page_count, and PARSING status.
    """

    logger.info(
        "step.parse.start",
        document_id=str(state.document_id),
    )

    # ---------------------------------------------------------
    # 1. Get parser service
    # ---------------------------------------------------------
    parser = services.get("parser")

    if parser is None:
        raise RuntimeError(
            "DocumentParserService is not available in ingestion services"
        )

    if not isinstance(parser, DocumentParserService):
        raise TypeError(
            f"Expected DocumentParserService, got {type(parser).__name__}"
        )

    # ---------------------------------------------------------
    # 2. Parse document
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # DocumentParserService.parse() is async.
    # Therefore we MUST await it directly.
    #
    parsed_doc = await parser.parse(state.storage_path)

    # ---------------------------------------------------------
    # 3. Validate parser output
    # ---------------------------------------------------------
    if not isinstance(parsed_doc, dict):
        raise TypeError(
            "DocumentParserService.parse() must return a dict, "
            f"got {type(parsed_doc).__name__}"
        )

    pages = parsed_doc.get("pages", [])

    if pages is None:
        pages = []

    if not isinstance(pages, list):
        raise TypeError(
            "Parsed document 'pages' must be a list, "
            f"got {type(pages).__name__}"
        )

    # ---------------------------------------------------------
    # 4. Calculate document statistics
    # ---------------------------------------------------------
    page_count = len(pages)

    table_count = 0
    figure_count = 0

    for page in pages:
        if not isinstance(page, dict):
            continue

        tables = page.get("tables", [])
        figures = page.get("figures", [])

        if isinstance(tables, list):
            table_count += len(tables)

        if isinstance(figures, list):
            figure_count += len(figures)

    # ---------------------------------------------------------
    # 5. Log successful parsing
    # ---------------------------------------------------------
    logger.info(
        "step.parse.complete",
        document_id=str(state.document_id),
        pages=page_count,
        tables=table_count,
        figures=figure_count,
    )

    # ---------------------------------------------------------
    # 6. Update ingestion state
    # ---------------------------------------------------------
    return state.model_copy(
        update={
            "parsed_doc": parsed_doc,
            "page_count": page_count,
            "status": DocumentStatus.PARSING,
        }
    )