"""
Unit and integration tests for StorageClient, Supervisor upload handling,
and Ingestion Step 01 Validate.

Covers:
- Authoritative storage path resolution (absolute, relative, container shifts).
- Pre-write and post-write verification (rejecting empty files, ensuring readability).
- Safe recovery of missing files (preventing ephemeral Render restarts from crashing).
- Step 01 validation checks (existence, size, PDF MIME).
- Failure recording in database without false successes.
"""
from __future__ import annotations

import asyncio
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.agents.supervisor.agent import IngestionSupervisor
from app.agents.supervisor.state import IngestionState
from app.infrastructure.storage.client import StorageClient
from app.models.document import DocumentStatus
from app.pipelines.ingestion.steps import s01_validate
from app.services.storage.service import StorageService


# Minimal valid PDF header bytes
_VALID_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


# ==============================================================================
# 1. StorageClient Unit Tests
# ==============================================================================

class TestStorageClient:
    def test_save_upload_writes_verified_readable_file(self, tmp_path: Path):
        """save_upload creates a real file, matches size, and returns an absolute path."""
        client = StorageClient(
            raw_path=str(tmp_path / "raw"),
            processed_path=str(tmp_path / "processed"),
            chunks_path=str(tmp_path / "chunks"),
        )
        data = _VALID_PDF_BYTES
        saved_path = client.save_upload(data, "manual.pdf")

        p = Path(saved_path)
        assert p.is_file()
        assert p.is_absolute()
        assert p.name.endswith("_manual.pdf")
        assert p.stat().st_size == len(data)
        assert p.read_bytes() == data

    def test_save_upload_rejects_empty_bytes(self, tmp_path: Path):
        """save_upload raises ValueError if file bytes are empty (0 bytes)."""
        client = StorageClient(
            raw_path=str(tmp_path / "raw"),
            processed_path=str(tmp_path / "processed"),
            chunks_path=str(tmp_path / "chunks"),
        )
        with pytest.raises(ValueError, match="empty \\(0 bytes\\)"):
            client.save_upload(b"", "empty.pdf")

    def test_save_upload_path_traversal_protection(self, tmp_path: Path):
        """save_upload strips path traversal characters and stays inside raw dir."""
        client = StorageClient(
            raw_path=str(tmp_path / "raw"),
            processed_path=str(tmp_path / "processed"),
            chunks_path=str(tmp_path / "chunks"),
        )
        saved_path = client.save_upload(_VALID_PDF_BYTES, "../../../etc/passwd.pdf")
        p = Path(saved_path)
        assert p.is_file()
        assert p.parent == (tmp_path / "raw").resolve()

    def test_get_raw_path_resolves_existing_file(self, tmp_path: Path):
        """get_raw_path returns canonical path for existing file."""
        client = StorageClient(
            raw_path=str(tmp_path / "raw"),
            processed_path=str(tmp_path / "processed"),
            chunks_path=str(tmp_path / "chunks"),
        )
        saved_path = client.save_upload(_VALID_PDF_BYTES, "test.pdf")
        resolved = client.get_raw_path(saved_path)
        assert resolved.is_file()
        assert str(resolved) == saved_path

    def test_get_raw_path_resolves_shifted_container_path(self, tmp_path: Path):
        """If an old container path is provided, get_raw_path recovers the file by filename in raw_dir."""
        client = StorageClient(
            raw_path=str(tmp_path / "raw"),
            processed_path=str(tmp_path / "processed"),
            chunks_path=str(tmp_path / "chunks"),
        )
        saved_path = client.save_upload(_VALID_PDF_BYTES, "hydraulics.pdf")
        filename = Path(saved_path).name

        # Simulate path from an old Render container
        fake_old_render_path = f"/opt/render/project/src/backend/data/raw/{filename}"
        resolved = client.get_raw_path(fake_old_render_path)

        assert resolved.is_file()
        assert resolved == Path(saved_path)


# ==============================================================================
# 2. IngestionSupervisor Tests
# ==============================================================================

class TestIngestionSupervisor:
    @pytest.mark.asyncio
    async def test_handle_upload_saves_file_and_enqueues(self, tmp_path: Path):
        """handle_upload writes the PDF, inserts into DB, and queues the job."""
        client = StorageClient(
            raw_path=str(tmp_path / "raw"),
            processed_path=str(tmp_path / "processed"),
            chunks_path=str(tmp_path / "chunks"),
        )
        storage_svc = StorageService(client)

        mock_postgres = AsyncMock()
        mock_postgres.fetchrow.side_effect = [
            {"id": uuid4()},       # doc row
            {"job_id": uuid4()},   # job row
        ]
        mock_pipeline = MagicMock()

        supervisor = IngestionSupervisor(
            pipeline=mock_pipeline,
            storage=storage_svc,
            postgres=mock_postgres,
        )

        mock_file = AsyncMock()
        mock_file.filename = "schematic.pdf"
        mock_file.read.return_value = _VALID_PDF_BYTES

        res = await supervisor.handle_upload(mock_file, industry="manufacturing")

        assert "document_id" in res
        assert "job_id" in res
        assert res["status"] == "RECEIVED"
        assert supervisor._queue.qsize() == 1

        queued_state = await supervisor._queue.get()
        assert Path(queued_state.storage_path).is_file()
        assert queued_state.filename == "schematic.pdf"

    @pytest.mark.asyncio
    async def test_handle_upload_rejects_empty_file(self, tmp_path: Path):
        """handle_upload raises ValueError when uploaded file has 0 bytes."""
        client = StorageClient(
            raw_path=str(tmp_path / "raw"),
            processed_path=str(tmp_path / "processed"),
            chunks_path=str(tmp_path / "chunks"),
        )
        storage_svc = StorageService(client)
        mock_postgres = AsyncMock()
        mock_pipeline = MagicMock()

        supervisor = IngestionSupervisor(
            pipeline=mock_pipeline,
            storage=storage_svc,
            postgres=mock_postgres,
        )

        mock_file = AsyncMock()
        mock_file.filename = "empty.pdf"
        mock_file.read.return_value = b""

        with pytest.raises(ValueError, match="empty \\(0 bytes\\)"):
            await supervisor.handle_upload(mock_file)

        assert supervisor._queue.qsize() == 0
        mock_postgres.fetchrow.assert_not_called()

    @pytest.mark.asyncio
    async def test_recovery_marks_missing_ephemeral_file_failed(self, tmp_path: Path):
        """On container restart, missing files are marked FAILED in DB and not queued."""
        client = StorageClient(
            raw_path=str(tmp_path / "raw"),
            processed_path=str(tmp_path / "processed"),
            chunks_path=str(tmp_path / "chunks"),
        )
        storage_svc = StorageService(client)

        missing_doc_id = uuid4()
        mock_postgres = AsyncMock()
        # Return one pending doc whose file does not exist on this container
        mock_postgres.fetch.return_value = [
            {
                "id": missing_doc_id,
                "file_name": "wiped_file.pdf",
                "storage_path": "/opt/render/project/src/backend/data/raw/wiped_file.pdf",
                "industry": "manufacturing",
                "tenant_id": "tenant-1",
                "assistant_id": "asst-1",
                "knowledge_base_id": "kb-1",
            }
        ]

        mock_pipeline = MagicMock()
        supervisor = IngestionSupervisor(
            pipeline=mock_pipeline,
            storage=storage_svc,
            postgres=mock_postgres,
        )

        # Trigger process_queue recovery logic with task cancellation
        worker_task = asyncio.create_task(supervisor.process_queue())
        await asyncio.sleep(0.05)
        worker_task.cancel()

        # The missing document must NOT be put in queue
        assert supervisor._queue.qsize() == 0

        # It must have executed an UPDATE setting status = 'FAILED'
        execute_calls = mock_postgres.execute.call_args_list
        assert len(execute_calls) >= 2
        assert any("FAILED" in str(call) for call in execute_calls)


# ==============================================================================
# 3. Step 01 Validate Tests
# ==============================================================================

class TestStepValidate:
    @pytest.mark.asyncio
    async def test_step_validate_success_with_valid_pdf(self, tmp_path: Path):
        """A valid existing PDF passes Step 01 validation."""
        pdf_file = tmp_path / "test_manual.pdf"
        pdf_file.write_bytes(_VALID_PDF_BYTES)

        state = IngestionState(
            document_id=uuid4(),
            filename="test_manual.pdf",
            storage_path=str(pdf_file),
        )

        res_state = await s01_validate.step(state, services={})
        assert res_state.status == DocumentStatus.VALIDATING

    @pytest.mark.asyncio
    async def test_step_validate_missing_file_raises_clear_error(self):
        """A missing file raises a descriptive ValueError with persistent disk advice."""
        missing_path = "/opt/render/project/src/backend/data/raw/missing_doc.pdf"
        state = IngestionState(
            document_id=uuid4(),
            filename="missing_doc.pdf",
            storage_path=missing_path,
        )

        with pytest.raises(ValueError) as excinfo:
            await s01_validate.step(state, services={})

        err_text = str(excinfo.value)
        assert "File not found at storage path" in err_text
        assert "missing_doc.pdf" in err_text
        assert "Render" in err_text

    @pytest.mark.asyncio
    async def test_step_validate_empty_file_raises_error(self, tmp_path: Path):
        """A 0-byte file raises ValueError."""
        empty_file = tmp_path / "empty.pdf"
        empty_file.write_bytes(b"")

        state = IngestionState(
            document_id=uuid4(),
            filename="empty.pdf",
            storage_path=str(empty_file),
        )

        with pytest.raises(ValueError, match="empty \\(0 bytes\\)"):
            await s01_validate.step(state, services={})

    @pytest.mark.asyncio
    async def test_step_validate_resolves_path_via_storage_service(self, tmp_path: Path):
        """Step 01 uses storage service get_raw_path if container path shifted."""
        real_file = tmp_path / "real_file.pdf"
        real_file.write_bytes(_VALID_PDF_BYTES)

        mock_storage = MagicMock()
        mock_storage.get_raw_path.return_value = real_file

        state = IngestionState(
            document_id=uuid4(),
            filename="real_file.pdf",
            storage_path="/old/path/real_file.pdf",
        )

        res_state = await s01_validate.step(state, services={"storage": mock_storage})
        assert res_state.status == DocumentStatus.VALIDATING
        assert res_state.storage_path == str(real_file)
