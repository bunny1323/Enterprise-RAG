"""
Ingestion Agent — Phase 1 Placeholder.

In Phase 2+, this agent will make decisions about:
- Routing documents to specialized parsers based on content type
- Selecting chunking strategies based on document structure complexity
- Adapting embedding modality per chunk type (text vs. image vs. table)
- Orchestrating multi-step reasoning for complex document structures

For Phase 1, all decisions are handled by the pipeline steps directly.
The supervisor agent routes to the sequential pipeline without agent logic.
"""
from app.config.logging import get_logger

logger = get_logger(__name__)


class IngestionAgent:
    """
    Future: LangGraph-based ingestion decision agent.

    Will use local Ollama for zero-cost routing decisions.
    Phase 1: No-op placeholder. Phase 2+: Full LangGraph state machine.
    """

    def __init__(self) -> None:
        logger.info("ingestion_agent.initialized", phase="1_placeholder")

    async def decide_parser_strategy(self, file_metadata: dict) -> str:
        """Phase 2: Route to specialized parser based on document metadata."""
        # Phase 1: Always use default sequential pipeline
        return "sequential"

    async def decide_chunk_strategy(self, parsed_doc: dict) -> str:
        """Phase 2: Choose chunking strategy based on document structure."""
        # Phase 1: Always use hierarchical chunking
        return "hierarchical"
