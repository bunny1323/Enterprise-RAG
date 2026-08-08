"""
Step 04 — Vision Analysis.
For each figure with an image_path, calls VisionService to analyze diagrams.
Results are stored back into the figure dict for use by the chunking step.
"""
import asyncio
from typing import Any

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.services.vision.service import VisionService

logger = get_logger(__name__)


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Analyze all figures/diagrams found during parsing using local Ollama llava.

    Vision analysis results (functional_summary, components, relationships)
    are stored in each figure's 'vision_analysis' key for downstream chunking.

    Args:
        state: Current ingestion state (parsed_doc must be set).
        services: Must contain 'vision' key → VisionService.

    Returns:
        Updated state with vision_analysis data embedded in parsed_doc figures.
    """
    logger.info("step.vision.start", document_id=str(state.document_id))

    if state.parsed_doc is None:
        logger.warning("step.vision.skipped", reason="parsed_doc is None")
        return state

    vision: VisionService = services["vision"]
    loop = asyncio.get_event_loop()

    pages = state.parsed_doc.get("pages", [])
    total_figures = 0
    analyzed_figures = 0

    for page_data in pages:
        for figure in page_data.get("figures", []):
            image_path: str = figure.get("image_path", "")
            total_figures += 1

            if not image_path:
                logger.debug(
                    "step.vision.skip_figure",
                    reason="no image_path",
                    page=page_data.get("page_num"),
                )
                continue

            try:
                # Run blocking Ollama HTTP call in thread pool
                analysis = await loop.run_in_executor(
                    None, vision.analyze_diagram, image_path
                )
                figure["vision_analysis"] = analysis
                analyzed_figures += 1

                logger.debug(
                    "step.vision.figure_analyzed",
                    image=image_path,
                    components=len(analysis.get("components", [])),
                )
            except Exception as err:
                logger.warning(
                    "step.vision.figure_error",
                    image=image_path,
                    error=str(err),
                )
                # Non-fatal: continue processing remaining figures
                figure["vision_analysis"] = {}

    logger.info(
        "step.vision.complete",
        document_id=str(state.document_id),
        total=total_figures,
        analyzed=analyzed_figures,
    )

    return state.model_copy(update={"parsed_doc": state.parsed_doc})
