"""
Step 04 — Vision Analysis.
For each figure with an image_path, calls VisionService to analyze diagrams.
Results are stored back into the figure dict for use by the chunking step.
"""
import asyncio
from typing import Any

from app.agents.supervisor.state import IngestionState
from app.config.logging import get_logger
from app.services.vision.vision_provider import BaseVisionProvider

logger = get_logger(__name__)


async def step(state: IngestionState, services: dict[str, Any]) -> IngestionState:
    """
    Analyze all figures/diagrams found during parsing using configured vision provider.

    Vision analysis results (functional_summary, components, relationships)
    are stored in each figure's 'vision_analysis' key for downstream chunking.

    Args:
        state: Current ingestion state (parsed_doc must be set).
        services: Must contain 'vision' key → BaseVisionProvider.

    Returns:
        Updated state with vision_analysis data embedded in parsed_doc figures.
    """
    logger.info("step.vision.start", document_id=str(state.document_id))

    if state.parsed_doc is None:
        logger.warning("step.vision.skipped", reason="parsed_doc is None")
        return state

    vision: BaseVisionProvider = services["vision"]
    provider_name = getattr(vision, "provider_name", vision.__class__.__name__.lower())
    model_name = getattr(vision, "model_name", "n/a")
    loop = asyncio.get_running_loop()

    pages = state.parsed_doc.get("pages", [])
    total_figures = 0
    analyzed_figures = 0
    skipped_figures = 0
    failed_figures = 0

    for page_data in pages:
        for figure in page_data.get("figures", []):
            image_path: str = figure.get("image_path", "")
            total_figures += 1

            if not image_path:
                logger.info(
                    "step.vision.figure_skip",
                    provider=provider_name,
                    model=model_name,
                    page=page_data.get("page_num"),
                    reason="no image_path",
                )
                continue

            if provider_name == "disabled":
                skipped_figures += 1
                figure["vision_analysis"] = {
                    "functional_summary": "Vision analysis disabled",
                    "components": [],
                    "relationships": [],
                    "spatial_layout": "Unknown",
                    "status": "skipped",
                }
                logger.info(
                    "step.vision.figure_skipped",
                    provider=provider_name,
                    model=model_name,
                    image=image_path,
                    reason="VISION_PROVIDER=disabled",
                )
                continue

            try:
                analysis = await loop.run_in_executor(
                    None, vision.analyze_diagram, image_path
                )
                figure["vision_analysis"] = analysis
                analyzed_figures += 1

                logger.info(
                    "step.vision.figure_analyzed",
                    provider=provider_name,
                    model=model_name,
                    image=image_path,
                    components=len(analysis.get("components", [])),
                )
            except Exception as err:
                failed_figures += 1
                logger.error(
                    "step.vision.figure_failed",
                    provider=provider_name,
                    model=model_name,
                    image=image_path,
                    error=str(err),
                )
                figure["vision_analysis"] = {
                    "functional_summary": "Vision analysis failed",
                    "components": [],
                    "relationships": [],
                    "spatial_layout": "Unknown",
                    "status": "failed",
                    "error": str(err),
                }

    logger.info(
        "step.vision.complete",
        document_id=str(state.document_id),
        total=total_figures,
        analyzed=analyzed_figures,
        skipped=skipped_figures,
        failed=failed_figures,
    )

    return state.model_copy(update={"parsed_doc": state.parsed_doc})
