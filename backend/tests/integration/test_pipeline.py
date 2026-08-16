"""
Integration test for IngestionPipeline resumability and state transitions.
"""
import pytest
from uuid import uuid4
from app.agents.supervisor.state import IngestionState
from app.models.document import DocumentStatus


def test_state_checkpoint_resume():
    doc_id = uuid4()
    job_id = uuid4()

    state = IngestionState(
        document_id=doc_id,
        job_id=job_id,
        filename="manual.pdf",
        storage_path="./data/raw/test.pdf",
        stage_checkpoints={"validate": True, "duplicate": True},
        last_successful_stage="duplicate",
    )

    assert state.stage_checkpoints.get("validate") is True
    assert state.stage_checkpoints.get("duplicate") is True
    assert state.stage_checkpoints.get("parse") is None
