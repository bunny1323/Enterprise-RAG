"""
Unit tests for ChunkingService.
"""
from uuid import uuid4
from app.services.chunking.service import ChunkingService


def test_chunking_deterministic_table_ids():
    chunker = ChunkingService()
    doc_id = uuid4()
    parsed_doc = {
        "pages": [
            {
                "page_num": 1,
                "text_blocks": [{"text": "Intro block", "bbox": [0, 0, 10, 10]}],
                "tables": [{"markdown": "| Col1 | Col2 |\n|---|---|\n| Val1 | Val2 |", "bbox": [10, 10, 50, 50]}],
                "figures": [],
            }
        ]
    }

    chunks1 = chunker.chunk(parsed_doc, doc_id, tenant_id="tenant_a")
    chunks2 = chunker.chunk(parsed_doc, doc_id, tenant_id="tenant_a")

    table_chunks1 = [c for c in chunks1 if c.chunk_type.value == "TABLE"]
    table_chunks2 = [c for c in chunks2 if c.chunk_type.value == "TABLE"]

    assert len(table_chunks1) == 1
    assert table_chunks1[0].chunk_id == table_chunks2[0].chunk_id
    assert table_chunks1[0].tenant_id == "tenant_a"
    assert table_chunks1[0].content_hash is not None
    assert "Page 1" in table_chunks1[0].context_prefix
