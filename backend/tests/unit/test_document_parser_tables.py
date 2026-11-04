from app.services.document_parser.service import DocumentParserService


def test_pymupdf_fallback_recovers_aligned_table_with_caption_and_units():
    blocks = [
        {
            "text": "COMPONENT GROUP A",
            "bbox": [10.0, 100.0, 190.0, 114.0],
            "item_type": "SectionHeaderItem",
        },
        {"text": "Item", "bbox": [10.0, 170.0, 35.0, 180.0], "item_type": "TextItem"},
        {"text": "Size", "bbox": [50.0, 170.0, 80.0, 180.0], "item_type": "TextItem"},
        {"text": "kgf.m", "bbox": [110.0, 170.0, 145.0, 180.0], "item_type": "TextItem"},
        {"text": "lbf.ft", "bbox": [170.0, 170.0, 205.0, 180.0], "item_type": "TextItem"},
        {"text": "4", "bbox": [10.0, 190.0, 18.0, 200.0], "item_type": "TextItem"},
        {"text": "M12 x 1.75", "bbox": [50.0, 190.0, 100.0, 200.0], "item_type": "TextItem"},
        {"text": "12.8 +/- 3.0", "bbox": [110.0, 190.0, 160.0, 200.0], "item_type": "TextItem"},
        {"text": "92.6 +/- 21.7", "bbox": [170.0, 190.0, 225.0, 200.0], "item_type": "TextItem"},
        {"text": "5", "bbox": [10.0, 210.0, 18.0, 220.0], "item_type": "TextItem"},
        {"text": "M16 x 2.0", "bbox": [50.0, 210.0, 100.0, 220.0], "item_type": "TextItem"},
        {"text": "29.7 +/- 4.5", "bbox": [110.0, 210.0, 160.0, 220.0], "item_type": "TextItem"},
        {"text": "215 +/- 32.5", "bbox": [170.0, 210.0, 225.0, 220.0], "item_type": "TextItem"},
    ]

    tables, residual = DocumentParserService._extract_aligned_text_tables(blocks, page_width=600)

    assert len(tables) == 1
    assert tables[0]["title"] == "COMPONENT GROUP A"
    assert "Item | Size | kgf.m | lbf.ft" in tables[0]["rows"][0]
    assert "M12 x 1.75" in tables[0]["rows"][1]
    assert "12.8 +/- 3.0" in tables[0]["rows"][1]
    assert "92.6 +/- 21.7" in tables[0]["rows"][1]
    assert "M16 x 2.0" in tables[0]["rows"][2]
    assert residual == [blocks[0]]