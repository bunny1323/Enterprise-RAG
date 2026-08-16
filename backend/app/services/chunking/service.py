"""
Chunking Service — structure-aware hierarchical chunking.

Produces parent (section-level ~1024 tokens) and child (paragraph-level ~256 tokens)
chunks with proper deterministic IDs, hierarchy paths, bounding boxes, content_hash,
context_prefix, and multi-tenancy annotations.
"""
import hashlib
from typing import Any
from uuid import UUID

from app.config.logging import get_logger
from app.models.chunk import Chunk, ChunkType
from app.utils.hashing import compute_chunk_hash

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
    3. Tables are single chunks (TABLE type) with deterministic content-hash IDs.
    4. Images/figures become single chunks with vision summary as content (IMAGE/DIAGRAM type).
    """

    def chunk(
        self,
        parsed_doc: dict[str, Any],
        document_id: UUID,
        industry: str = "manufacturing",
        tenant_id: str = "default",
        assistant_id: str = "default",
        knowledge_base_id: str = "default",
        filename: str = "document.pdf",
    ) -> list[Chunk]:
        """
        Produce hierarchical chunks from a parsed document with multi-tenancy and context_prefix.
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
                text_blocks=text_blocks,
                page_num=page_num,
                doc_str=doc_str,
                document_id=document_id,
                industry=industry,
                tenant_id=tenant_id,
                assistant_id=assistant_id,
                knowledge_base_id=knowledge_base_id,
                filename=filename,
            )
            for parent in parent_chunks:
                all_chunks.append(parent)
                children = self._make_child_chunks(
                    parent=parent,
                    document_id=document_id,
                    industry=industry,
                    tenant_id=tenant_id,
                    assistant_id=assistant_id,
                    knowledge_base_id=knowledge_base_id,
                    filename=filename,
                )
                all_chunks.extend(children)

            # ── Table chunks ───────────────────────────────────────────────────
            for table in tables:
                chunk = self._make_table_chunk(
                    table=table,
                    page_num=page_num,
                    doc_str=doc_str,
                    document_id=document_id,
                    industry=industry,
                    tenant_id=tenant_id,
                    assistant_id=assistant_id,
                    knowledge_base_id=knowledge_base_id,
                    filename=filename,
                )
                all_chunks.append(chunk)

            # ── Figure/image chunks ────────────────────────────────────────────
            for figure in figures:
                chunk = self._make_figure_chunk(
                    figure=figure,
                    page_num=page_num,
                    doc_str=doc_str,
                    document_id=document_id,
                    industry=industry,
                    tenant_id=tenant_id,
                    assistant_id=assistant_id,
                    knowledge_base_id=knowledge_base_id,
                    filename=filename,
                )
                if chunk:
                    all_chunks.append(chunk)

        logger.info(
            "chunking.complete",
            document_id=doc_str,
            tenant_id=tenant_id,
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
        tenant_id: str,
        assistant_id: str,
        knowledge_base_id: str,
        filename: str,
    ) -> list[Chunk]:
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

            if _word_count(" ".join(current_texts)) >= _PARENT_TARGET_WORDS:
                parent = self._flush_parent(
                    texts=current_texts,
                    bboxes=current_bboxes,
                    page_num=page_num,
                    doc_str=doc_str,
                    document_id=document_id,
                    industry=industry,
                    tenant_id=tenant_id,
                    assistant_id=assistant_id,
                    knowledge_base_id=knowledge_base_id,
                    filename=filename,
                    section_idx=section_idx,
                )
                parents.append(parent)
                current_texts = []
                current_bboxes = []
                section_idx += 1

        if current_texts:
            parent = self._flush_parent(
                texts=current_texts,
                bboxes=current_bboxes,
                page_num=page_num,
                doc_str=doc_str,
                document_id=document_id,
                industry=industry,
                tenant_id=tenant_id,
                assistant_id=assistant_id,
                knowledge_base_id=knowledge_base_id,
                filename=filename,
                section_idx=section_idx,
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
        tenant_id: str,
        assistant_id: str,
        knowledge_base_id: str,
        filename: str,
        section_idx: int,
    ) -> Chunk:
        content = " ".join(texts)
        merged_bbox = self._merge_bboxes(bboxes) if bboxes else None
        section_name = f"Section {section_idx + 1}"
        chunk_id = f"{doc_str}::p{page_num}::s{section_idx}"
        c_hash = compute_chunk_hash(content)
        ctx_prefix = f"{filename} > Page {page_num} > {section_name}"

        return Chunk(
            chunk_id=chunk_id,
            parent_id=None,
            document_id=document_id,
            tenant_id=tenant_id,
            assistant_id=assistant_id,
            knowledge_base_id=knowledge_base_id,
            content=content,
            content_hash=c_hash,
            section=section_name,
            context_prefix=ctx_prefix,
            embedding_representation="text",
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
        tenant_id: str,
        assistant_id: str,
        knowledge_base_id: str,
        filename: str,
    ) -> list[Chunk]:
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
            c_hash = compute_chunk_hash(content)
            ctx_prefix = f"{filename} > Page {parent.page_number} > {parent.section or 'Section'} > Chunk {child_idx}"

            child = Chunk(
                chunk_id=chunk_id,
                parent_id=parent.chunk_id,
                document_id=document_id,
                tenant_id=tenant_id,
                assistant_id=assistant_id,
                knowledge_base_id=knowledge_base_id,
                content=content,
                content_hash=c_hash,
                section=parent.section,
                context_prefix=ctx_prefix,
                embedding_representation="text",
                page_number=parent.page_number,
                bounding_box=parent.bounding_box,
                chunk_type=ChunkType.TEXT,
                industry_domain=industry,
                hierarchy_path=f"{parent.hierarchy_path}.child{child_idx}",
            )
            child_chunks.append(child)
            child_idx += 1

        return child_chunks

    # ── Table chunks (Deterministic IDs) ───────────────────────────────────────

    def _make_table_chunk(
        self,
        table: dict,
        page_num: int,
        doc_str: str,
        document_id: UUID,
        industry: str,
        tenant_id: str,
        assistant_id: str,
        knowledge_base_id: str,
        filename: str,
    ) -> Chunk:
        markdown = table.get("markdown", "")
        c_hash = compute_chunk_hash(markdown)
        # Deterministic table chunk ID
        table_hash_id = c_hash[:12]
        chunk_id = f"{doc_str}::p{page_num}::t{table_hash_id}"
        ctx_prefix = f"{filename} > Page {page_num} > Table {table_hash_id}"

        return Chunk(
            chunk_id=chunk_id,
            parent_id=None,
            document_id=document_id,
            tenant_id=tenant_id,
            assistant_id=assistant_id,
            knowledge_base_id=knowledge_base_id,
            content=markdown,
            content_hash=c_hash,
            section=f"Table Page {page_num}",
            context_prefix=ctx_prefix,
            embedding_representation="text",
            page_number=page_num,
            bounding_box=table.get("bbox"),
            chunk_type=ChunkType.TABLE,
            industry_domain=industry,
            hierarchy_path=f"doc.page{page_num}.table_{table_hash_id}",
        )

    # ── Figure chunks (Deterministic IDs) ──────────────────────────────────────

    def _make_figure_chunk(
        self,
        figure: dict,
        page_num: int,
        doc_str: str,
        document_id: UUID,
        industry: str,
        tenant_id: str,
        assistant_id: str,
        knowledge_base_id: str,
        filename: str,
    ) -> Chunk | None:
        vision_data: dict = figure.get("vision_analysis", {})
        caption: str = figure.get("caption", "")
        image_path: str = figure.get("image_path", "")

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

        content = "\n".join(parts)
        c_hash = compute_chunk_hash(content)
        fig_hash_id = c_hash[:12]
        chunk_id = f"{doc_str}::p{page_num}::f{fig_hash_id}"

        chunk_type = ChunkType.DIAGRAM if vision_data else ChunkType.IMAGE
        representation = "image" if image_path else "text_summary_of_image"
        ctx_prefix = f"{filename} > Page {page_num} > Figure {fig_hash_id}"

        return Chunk(
            chunk_id=chunk_id,
            parent_id=None,
            document_id=document_id,
            tenant_id=tenant_id,
            assistant_id=assistant_id,
            knowledge_base_id=knowledge_base_id,
            content=content,
            content_hash=c_hash,
            section=f"Figure Page {page_num}",
            context_prefix=ctx_prefix,
            embedding_representation=representation,
            page_number=page_num,
            bounding_box=figure.get("bbox"),
            chunk_type=chunk_type,
            industry_domain=industry,
            hierarchy_path=f"doc.page{page_num}.figure_{fig_hash_id}",
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
