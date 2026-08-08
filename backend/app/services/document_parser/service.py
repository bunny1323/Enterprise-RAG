"""
Document Parser Service — primary: IBM Docling, fallback: PyMuPDF.

Extracts structured content from PDFs:
  - Text blocks with bounding boxes
  - Tables (as Markdown)
  - Figures/images with captions
"""
import tempfile
from pathlib import Path
from typing import Any

from app.config.logging import get_logger

logger = get_logger(__name__)


class DocumentParserService:
    """
    Stateless PDF parsing service.

    Primary path: IBM Docling (TableFormer + DocLayNet layout model).
    Fallback path: PyMuPDF (fitz) for text extraction when Docling fails.
    """

    def parse(self, file_path: str) -> dict[str, Any]:
        """
        Parse a PDF file into structured page content.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            Dictionary with structure:
            {
                "pages": [
                    {
                        "page_num": int,
                        "text_blocks": [{"text": str, "bbox": list[float], "page_num": int}],
                        "tables":      [{"markdown": str, "bbox": list[float], "page_num": int}],
                        "figures":     [{"image_path": str, "bbox": list[float], "page_num": int,
                                         "caption": str}],
                    }
                ]
            }
        """
        logger.info("parser.start", path=file_path)
        try:
            return self._parse_with_docling(file_path)
        except Exception as docling_err:
            logger.warning(
                "parser.docling_failed_fallback",
                error=str(docling_err),
                path=file_path,
            )
            return self._parse_with_pymupdf(file_path)

    # ── Docling (primary) ──────────────────────────────────────────────────────

    def _parse_with_docling(self, file_path: str) -> dict[str, Any]:
        """Use IBM Docling with TableFormer + DocLayNet for structured extraction."""
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(file_path)
        doc = result.document

        pages: list[dict[str, Any]] = []
        # Group items by page
        page_map: dict[int, dict[str, Any]] = {}

        def _get_page(page_num: int) -> dict[str, Any]:
            if page_num not in page_map:
                page_map[page_num] = {
                    "page_num": page_num,
                    "text_blocks": [],
                    "tables": [],
                    "figures": [],
                }
            return page_map[page_num]

        # Extract text elements
        for item, _ in doc.iterate_items():
            try:
                page_num = item.prov[0].page_no if item.prov else 1
            except (IndexError, AttributeError):
                page_num = 1

            item_type = type(item).__name__

            if item_type in ("TextItem", "SectionHeaderItem", "ListItem"):
                text = getattr(item, "text", "") or ""
                bbox = self._extract_bbox(item)
                if text.strip():
                    _get_page(page_num)["text_blocks"].append(
                        {"text": text, "bbox": bbox, "page_num": page_num}
                    )

            elif item_type == "TableItem":
                try:
                    md = item.export_to_markdown()
                except Exception:
                    md = str(item)
                bbox = self._extract_bbox(item)
                _get_page(page_num)["tables"].append(
                    {"markdown": md, "bbox": bbox, "page_num": page_num}
                )

            elif item_type == "PictureItem":
                bbox = self._extract_bbox(item)
                caption = ""
                try:
                    caption = item.caption_text(doc) or ""
                except Exception:
                    pass
                # Save figure image to temp path
                img_path = self._save_figure(item, file_path, page_num)
                _get_page(page_num)["figures"].append(
                    {
                        "image_path": img_path,
                        "bbox": bbox,
                        "page_num": page_num,
                        "caption": caption,
                    }
                )

        pages = sorted(page_map.values(), key=lambda p: p["page_num"])
        logger.info("parser.docling_complete", pages=len(pages), file=file_path)
        return {"pages": pages}

    def _extract_bbox(self, item: Any) -> list[float]:
        """Extract normalized bounding box [x0,y0,x1,y1] from a Docling item."""
        try:
            prov = item.prov[0]
            bbox = prov.bbox
            return [
                float(bbox.l),
                float(bbox.t),
                float(bbox.r),
                float(bbox.b),
            ]
        except (AttributeError, IndexError, TypeError):
            return []

    def _save_figure(self, item: Any, source_path: str, page_num: int) -> str:
        """Attempt to save a Docling PictureItem as PNG; return path or empty string."""
        try:
            # Create temp directory based on source file
            out_dir = Path(tempfile.gettempdir()) / "rag_figures" / Path(source_path).stem
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"page{page_num}_{id(item)}.png"
            image = item.get_image()
            if image:
                image.save(str(out_path))
                return str(out_path)
        except Exception:
            pass
        return ""

    # ── PyMuPDF (fallback) ─────────────────────────────────────────────────────

    def _parse_with_pymupdf(self, file_path: str) -> dict[str, Any]:
        """Fallback parser using PyMuPDF for text block extraction."""
        import fitz  # type: ignore[import-untyped]

        doc = fitz.open(file_path)
        pages: list[dict[str, Any]] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            text_blocks: list[dict[str, Any]] = []

            raw = page.get_text("dict")
            for block in raw.get("blocks", []):
                if block.get("type") != 0:  # 0 = text block
                    continue
                lines = block.get("lines", [])
                text = " ".join(
                    span.get("text", "")
                    for line in lines
                    for span in line.get("spans", [])
                ).strip()
                if text:
                    b = block.get("bbox", [0, 0, 0, 0])
                    text_blocks.append(
                        {
                            "text": text,
                            "bbox": list(b),
                            "page_num": page_num,
                        }
                    )

            pages.append(
                {
                    "page_num": page_num,
                    "text_blocks": text_blocks,
                    "tables": [],
                    "figures": [],
                }
            )

        doc.close()
        logger.info("parser.pymupdf_complete", pages=len(pages), file=file_path)
        return {"pages": pages}
