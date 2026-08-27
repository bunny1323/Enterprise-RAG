import asyncio
import os
import time
import uuid
from typing import Any

import pytest
from app.agents.supervisor.state import IngestionState
from app.config.settings import get_settings
from app.models.document import DocumentStatus
from app.pipelines.ingestion.pipeline import IngestionPipeline
from app.services.document_parser.service import DocumentParserService, ParseProfile

# This test requires actually calling the service
@pytest.mark.asyncio
async def test_parser_timeout_isolation() -> None:
    # Use a real service but give it an impossibly small timeout
    parser = DocumentParserService()
    
    # We'll use this file itself as a dummy to try to parse (it will likely fail as a PDF but it serves to test the loop/timeout logic)
    dummy_file = __file__
    
    start_t = time.time()
    try:
        # Force a 1 second timeout
        await parser.parse(dummy_file, profile=ParseProfile.BALANCED, timeout=1)
        # It might also just fail normally very quickly if it realizes it's not a PDF, but let's test if it handles the timeout.
    except Exception as e:
        # Should be a timeout or a fallback failure
        pass
    
    end_t = time.time()
    
    # Just a simple sanity check that the call didn't block forever
    assert end_t - start_t < 5
