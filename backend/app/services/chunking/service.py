"""
Chunking Service — structure-aware hierarchical chunking.

Produces parent (section-level ~1024 tokens) and child (paragraph-level ~256 tokens)
chunks with proper IDs, hierarchy paths, and bounding boxes.
"""
import uuid
from typing import Any
from uuid import UUID

from app.config.logging import get_logger
from app.models.chunk import Chunk, ChunkType

logger = get_logger(__name__)

# Token budget constants (word-count approximation: 1 token ≈ 0.75 words)
_PARENT_TARGET_WORDS = 768   # ≈ 1024 tokens
_CHILD_TARGET_WORDS = 192    # ≈ 256 tokens
_WORDS_PER_TOKEN = 0.75


def _word_count(text: str) -> int:
    return len(text.split())


def _token_estimate(text: str) -> int:
    return int(_word_count(text) / _WORDS_PER_TOKEN)


class ChunkingService:
    """
    Stateless structure-aware chunker.

    Strategy:
    1. Group text blocks from the same page into parent sections (~1024 tokens).
    2. Split each parent into child chunks (~256 tokens).
    3. Tables are single chunks (TABLE type).
    4. Images/figures become single chunks with vision summary as content (IMAGE/DIAGRAM type).
    """

    def chunk(
        self,
        parsed_doc: dict[str, Any],
        document_id: UUID,
        industry: str,
    ) -> list[Chunk]:
        """
        Produce hierarchical chunks from a parsed document.

        Args:
            parsed_doc: Output from DocumentParserService ({"pages": [...]}).
            document_id: UUID of the parent document.
            industry: Industry domain label for metadata.

        Returns:
            List of Chunk objects (parents first, then children).
        """
        all_chunks: list[Chunk] = []
        doc_str = str(document_id)

        for page_data in parsed_doc.get("pages", []):
            page_num: int = page_data.get("page_num", 0)
            text_blocks: list[dict] = page_data.get("text_blocks", [])
            tables: list[dict] = page_data.get("tables", [])
            figures: list[dict] = page_data.get("figures", [])

            # ── Text chunking (parent → children) ─────────────────────────────
            parent_chunks = self._make_parent_chunks(
                text_blocks, page_num, doc_str, document_id, industry
            )
            for parent in parent_chunks:
                all_chunks.append(parent)
                children = self._make_child_chunks(parent, document_id, industry)
                all_chunks.extend(children)

            # ── Table chunks ───────────────────────────────────────────────────
            for table in tables:
                chunk = self._make_table_chunk(table, page_num, doc_str, document_id, industry)
                all_chunks.append(chunk)

            # ── Figure/image chunks ────────────────────────────────────────────
            for figure in figures:
                chunk = self._make_figure_chunk(figure, page_num, doc_str, document_id, industry)
                if chunk:
                    all_chunks.append(chunk)

        logger.info(
            "chunking.complete",
            document_id=doc_str,
            total_chunks=len(all_chunks),
        )
        return all_chunks

    # ── Parent chunk creation ──────────────────────────────────────────────────

    def _make_parent_chunks(
        self,
        text_blocks: list[dict],
        page_num: int,
        doc_str: str,
        document_id: UUID,
        industry: str,
    ) -> list[Chunk]:
        """Group text blocks into parent sections up to ~1024 tokens each."""
        parents: list[Chunk] = []
        current_texts: list[str] = []
        current_bboxes: list[list[float]] = []
        section_idx = 0

        for block in text_blocks:
            text = block.get("text", "").strip()
            bbox = block.get("bbox", [])
            if not text:
                continue

            current_texts.append(text)
            if bbox:
                current_bboxes.append(bbox)

            # Flush when we hit parent token target
            if _word_count(" ".join(current_texts)) >= _PARENT_TARGET_WORDS:
                parent = self._flush_parent(
                    current_texts,
                    current_bboxes,
                    page_num,
                    doc_str,
                    document_id,
                    industry,
                    section_idx,
                )
                parents.append(parent)
                current_texts = []
                current_bboxes = []
                section_idx += 1

        # Flush remainder
        if current_texts:
            parent = self._flush_parent(
                current_texts,
                current_bboxes,
                page_num,
                doc_str,
                document_id,
                industry,
                section_idx,
            )
            parents.append(parent)

        return parents

    def _flush_parent(
        self,
        texts: list[str],
        bboxes: list[list[float]],
        page_num: int,
        doc_str: str,
        document_id: UUID,
        industry: str,
        section_idx: int,
    ) -> Chunk:
        content = " ".join(texts)
        merged_bbox = self._merge_bboxes(bboxes) if bboxes else None
        chunk_id = f"{doc_str}::p{page_num}::s{section_idx}"
        return Chunk(
            chunk_id=chunk_id,
            parent_id=None,
            document_id=document_id,
            content=content,
            page_number=page_num,
            bounding_box=merged_bbox,
            chunk_type=ChunkType.TEXT,
            industry_domain=industry,
            hierarchy_path=f"doc.page{page_num}.section{section_idx}",
        )

    # ── Child chunk creation ───────────────────────────────────────────────────

    def _make_child_chunks(
        self,
        parent: Chunk,
        document_id: UUID,
        industry: str,
    ) -> list[Chunk]:
        """Split a parent chunk into ~256-token child chunks."""
        words = parent.content.split()
        child_chunks: list[Chunk] = []
        child_size_words = int(_CHILD_TARGET_WORDS * _WORDS_PER_TOKEN)
        child_idx = 0

        for i in range(0, len(words), child_size_words):
            window = words[i : i + child_size_words]
            if not window:
                break
            content = " ".join(window)
            chunk_id = f"{parent.chunk_id}::c{child_idx}"
            child = Chunk(
                chunk_id=chunk_id,
                parent_id=parent.chunk_id,
                document_id=document_id,
                content=content,
                page_number=parent.page_number,
                bounding_box=parent.bounding_box,
                chunk_type=ChunkType.TEXT,
                industry_domain=industry,
                hierarchy_path=f"{parent.hierarchy_path}.child{child_idx}",
            )
            child_chunks.append(child)
            child_idx += 1

        return child_chunks

    # ── Table chunks ───────────────────────────────────────────────────────────

    def _make_table_chunk(
        self,
        table: dict,
        page_num: int,
        doc_str: str,
        document_id: UUID,
        industry: str,
    ) -> Chunk:
        """Create a single TABLE chunk from a parsed table dict."""
        uid = uuid.uuid4().hex[:8]
        chunk_id = f"{doc_str}::p{page_num}::t{uid}"
        markdown = table.get("markdown", "")
        return Chunk(
            chunk_id=chunk_id,
            parent_id=None,
            document_id=document_id,
            content=markdown,
            page_number=page_num,
            bounding_box=table.get("bbox"),
            chunk_type=ChunkType.TABLE,
            industry_domain=industry,
            hierarchy_path=f"doc.page{page_num}.table_{uid}",
        )

    # ── Figure chunks ──────────────────────────────────────────────────────────

    def _make_figure_chunk(
        self,
        figure: dict,
        page_num: int,
        doc_str: str,
        document_id: UUID,
        industry: str,
    ) -> Chunk | None:
        """
        Create an IMAGE or DIAGRAM chunk from a figure dict.

        The content is built from the vision analysis (if available) and caption.
        """
        vision_data: dict = figure.get("vision_analysis", {})
        caption: str = figure.get("caption", "")
        image_path: str = figure.get("image_path", "")

        # Build meaningful text content from vision analysis
        parts: list[str] = []
        if caption:
            parts.append(f"Caption: {caption}")
        if vision_data:
            summary = vision_data.get("functional_summary", "")
            if summary:
                parts.append(f"Description: {summary}")
            components = vision_data.get("components", [])
            if components:
                parts.append(f"Components: {', '.join(str(c) for c in components)}")
        if not parts:
            if not image_path:
                return None
            parts.append(f"Figure from page {page_num}")

        uid = uuid.uuid4().hex[:8]
        chunk_id = f"{doc_str}::p{page_num}::f{uid}"
        chunk_type = ChunkType.DIAGRAM if vision_data else ChunkType.IMAGE

        return Chunk(
            chunk_id=chunk_id,
            parent_id=None,
            document_id=document_id,
            content="\n".join(parts),
            page_number=page_num,
            bounding_box=figure.get("bbox"),
            chunk_type=chunk_type,
            industry_domain=industry,
            hierarchy_path=f"doc.page{page_num}.figure_{uid}",
            metadata={"image_path": image_path, "vision_analysis": vision_data},
        )

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_bboxes(bboxes: list[list[float]]) -> list[float]:
        """Merge multiple bounding boxes into a single enclosing bbox."""
        x0 = min(b[0] for b in bboxes if len(b) >= 4)
        y0 = min(b[1] for b in bboxes if len(b) >= 4)
        x1 = max(b[2] for b in bboxes if len(b) >= 4)
        y1 = max(b[3] for b in bboxes if len(b) >= 4)
        return [x0, y0, x1, y1]
