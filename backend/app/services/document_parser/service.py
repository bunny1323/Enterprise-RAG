"""
Document Parser Service — primary: IBM Docling, fallback: PyMuPDF.

Extracts structured content from PDFs:
  - Text blocks with bounding boxes
  - Tables (as Markdown)
  - Figures/images with captions
"""
import asyncio
import multiprocessing as mp
import queue
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any

from app.config.logging import get_logger

logger = get_logger(__name__)


class ParseProfile(str, Enum):
    FAST = "FAST"
    BALANCED = "BALANCED"
    HIGH_ACCURACY = "HIGH_ACCURACY"


def _run_docling_in_process_queue(file_path: str, profile: str, result_queue: mp.Queue) -> None:
    """
    Runs IBM Docling extraction in a separate process.
    Communicates success/failure via the provided multiprocessing.Queue.
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PipelineOptions, PdfPipelineOptions
        
        # Configure Pipeline Options based on profile
        pipeline_options = PdfPipelineOptions()
        if profile == ParseProfile.FAST.value:
            pipeline_options.do_table_structure = False
            pipeline_options.do_ocr = False
            pipeline_options.generate_page_images = False
        elif profile == ParseProfile.BALANCED.value:
            pipeline_options.do_table_structure = True
            pipeline_options.do_ocr = False
            pipeline_options.generate_page_images = False
        else:  # HIGH_ACCURACY
            pipeline_options.do_table_structure = True
            pipeline_options.do_ocr = True
            pipeline_options.generate_page_images = True

        converter = DocumentConverter(
            format_options={
                "pdf": pipeline_options
            }
        )
        result = converter.convert(file_path)
        doc = result.document

        pages: list[dict[str, Any]] = []
        page_map: dict[int, dict[str, Any]] = {}
        
        stats = {
            "page_count": 0,
            "ocr_pages": set(),
            "table_pages": set(),
            "vision_pages": set(),
        }

        def _get_page(page_num: int) -> dict[str, Any]:
            if page_num not in page_map:
                page_map[page_num] = {
                    "page_num": page_num,
                    "text_blocks": [],
                    "tables": [],
                    "figures": [],
                }
                stats["page_count"] = max(stats["page_count"], page_num)
            return page_map[page_num]

        # Helper functions within the isolated process
        def _extract_bbox(item: Any) -> list[float]:
            try:
                prov = item.prov[0]
                bbox = prov.bbox
                return [float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b)]
            except (AttributeError, IndexError, TypeError):
                return []

        def _save_figure(item: Any, source_path: str, page_num: int) -> str:
            try:
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

        for item, _ in doc.iterate_items():
            try:
                page_num = item.prov[0].page_no if item.prov else 1
            except (IndexError, AttributeError):
                page_num = 1

            item_type = type(item).__name__

            if item_type in ("TextItem", "SectionHeaderItem", "ListItem"):
                text = getattr(item, "text", "") or ""
                bbox = _extract_bbox(item)
                if text.strip():
                    _get_page(page_num)["text_blocks"].append({"text": text, "bbox": bbox, "page_num": page_num})

            elif item_type == "TableItem":
                stats["table_pages"].add(page_num)
                try:
                    md = item.export_to_markdown()
                except Exception:
                    md = str(item)
                bbox = _extract_bbox(item)
                _get_page(page_num)["tables"].append({"markdown": md, "bbox": bbox, "page_num": page_num})

            elif item_type == "PictureItem":
                stats["vision_pages"].add(page_num)
                bbox = _extract_bbox(item)
                caption = ""
                try:
                    caption = item.caption_text(doc) or ""
                except Exception:
                    pass
                img_path = _save_figure(item, file_path, page_num)
                _get_page(page_num)["figures"].append({
                    "image_path": img_path, "bbox": bbox, "page_num": page_num, "caption": caption
                })

        pages = sorted(page_map.values(), key=lambda p: p["page_num"])
        
        final_result = {
            "pages": pages,
            "metadata": {
                "page_count": stats["page_count"],
                "ocr_required_count": len(stats["ocr_pages"]) if profile == ParseProfile.HIGH_ACCURACY.value else 0,
                "table_required_count": len(stats["table_pages"]),
                "vision_required_count": len(stats["vision_pages"]),
                "parser_profile": profile
            }
        }
        result_queue.put({"status": "success", "data": final_result})
    except Exception as err:
        result_queue.put({"status": "error", "error": str(err)})


class DocumentParserService:
    """
    Stateless PDF parsing service.
    
    Primary path: IBM Docling executed in a dedicated, bounded multiprocessing.Process 
    that can be safely killed on timeout to avoid leaking CPU work.
    Fallback path: PyMuPDF (fitz) for text extraction when Docling fails.
    """
    def __init__(self, max_workers: int = 4):
        # We no longer use ProcessPoolExecutor here because it doesn't allow hard cancellation.
        # Instead, we will manage isolated processes manually per request.
        self._default_timeout = 300  # Default timeout in seconds

    async def parse(
        self, 
        file_path: str, 
        profile: ParseProfile = ParseProfile.BALANCED,
        timeout: int | None = None
    ) -> dict[str, Any]:
        """
        Parse a PDF file asynchronously using a bounded isolated process.
        """
        start_time = time.time()
        active_timeout = timeout or self._default_timeout
        logger.info("parser.start", path=file_path, profile=profile.value)
        
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(
            target=_run_docling_in_process_queue, 
            args=(file_path, profile.value, q)
        )
        
        p.start()
        
        try:
            # Poll the queue without blocking the asyncio loop indefinitely
            while True:
                # 1. Check if we've exceeded the timeout manually
                if time.time() - start_time > active_timeout:
                    raise asyncio.TimeoutError(f"Docling parse exceeded {active_timeout}s")
                
                # 2. Try to read from queue
                try:
                    res = q.get_nowait()
                    if res["status"] == "success":
                        duration = time.time() - start_time
                        logger.info(
                            "parser.docling_complete", 
                            file=file_path, 
                            duration=duration,
                            metadata=res["data"].get("metadata")
                        )
                        return res["data"]
                    else:
                        raise Exception(res["error"])
                except queue.Empty:
                    # 3. Check if process died unexpectedly without writing to queue
                    if not p.is_alive():
                        raise Exception("Parser process died unexpectedly before returning results.")
                    
                    # Wait briefly before polling again
                    await asyncio.sleep(0.5)

        except asyncio.TimeoutError:
            logger.error("parser.docling_timeout", path=file_path, timeout=active_timeout)
            # Re-raise it so the pipeline catches it and sets TIMEOUT status!
            raise

        except Exception as docling_err:
            logger.warning("parser.docling_failed_fallback", error=str(docling_err), path=file_path)
            return await self._parse_with_pymupdf(file_path)
            
        finally:
            # Cleanup process if it's still alive (crucial for timeout/cancellation)
            if p.is_alive():
                logger.warning("parser.cleanup_terminating_process", pid=p.pid)
                p.terminate()
                p.join(timeout=1.0)
                if p.is_alive():
                    p.kill()
            q.close()

    async def _parse_with_pymupdf(self, file_path: str) -> dict[str, Any]:
        """Fallback parser using PyMuPDF (run in default thread pool to avoid blocking)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_parse_pymupdf, file_path)

    def _sync_parse_pymupdf(self, file_path: str) -> dict[str, Any]:
        import fitz  # type: ignore[import-untyped]
        start_time = time.time()
        doc = fitz.open(file_path)
        pages: list[dict[str, Any]] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            text_blocks: list[dict[str, Any]] = []

            raw = page.get_text("dict")
            for block in raw.get("blocks", []):
                if block.get("type") != 0:
                    continue
                lines = block.get("lines", [])
                text = " ".join(
                    span.get("text", "") for line in lines for span in line.get("spans", [])
                ).strip()
                if text:
                    b = block.get("bbox", [0, 0, 0, 0])
                    text_blocks.append({"text": text, "bbox": list(b), "page_num": page_num})

            pages.append({"page_num": page_num, "text_blocks": text_blocks, "tables": [], "figures": []})

        doc.close()
        duration = time.time() - start_time
        logger.info("parser.pymupdf_complete", pages=len(pages), file=file_path, duration=duration)
        return {
            "pages": pages,
            "metadata": {
                "page_count": len(pages),
                "ocr_required_count": 0,
                "table_required_count": 0,
                "vision_required_count": 0,
                "parser_profile": "FALLBACK_PYMUPDF"
            }
        }
