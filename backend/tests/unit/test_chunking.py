"""
Unit tests for ChunkingService.
"""
from uuid import uuid4
from app.services.chunking.service import ChunkingService


def test_flattened_numeric_table_preserves_title_and_rows():
    chunker = ChunkingService()
    doc_id = uuid4()
    parsed_doc = {
        "pages": [{
            "page_num": 4,
            "text_blocks": [
                {"text": "Kilogram to Pound 1 kg = 2.2046 lb", "bbox": [0, 0, 10, 10]},
                {"text": "0 2.20 4.41 6.61 8.82", "bbox": [0, 10, 10, 20]},
                {"text": "10 22.05 24.25 26.46 28.66", "bbox": [0, 20, 10, 30]},
                {"text": "The next section explains maintenance procedures.", "bbox": [0, 30, 10, 40]},
            ],
            "tables": [],
            "figures": [],
        }]
    }

    chunks, _ = chunker.chunk(parsed_doc, doc_id, filename="manual.pdf")

    table_chunks = [chunk for chunk in chunks if chunk.chunk_type.value == "TABLE"]
    assert len(table_chunks) == 1
    assert "Kilogram to Pound 1 kg = 2.2046 lb" in table_chunks[0].content
    assert "0 2.20 4.41 6.61 8.82" in table_chunks[0].content
    assert "10 22.05 24.25 26.46 28.66" in table_chunks[0].content
    assert table_chunks[0].metadata["source"] == "flattened_text"
    assert any("maintenance" in chunk.content.lower() for chunk in chunks if chunk.chunk_type.value == "TEXT")


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

    chunks1, _ = chunker.chunk(parsed_doc, doc_id, tenant_id="tenant_a")
    chunks2, _ = chunker.chunk(parsed_doc, doc_id, tenant_id="tenant_a")

    table_chunks1 = [c for c in chunks1 if c.chunk_type.value == "TABLE"]
    table_chunks2 = [c for c in chunks2 if c.chunk_type.value == "TABLE"]

    assert len(table_chunks1) == 1
    assert table_chunks1[0].chunk_id == table_chunks2[0].chunk_id
    assert table_chunks1[0].tenant_id == "tenant_a"
    assert table_chunks1[0].content_hash is not None
    assert "Page 1" in table_chunks1[0].context_prefix
