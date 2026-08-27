"""
Chunking Service — structure-aware hierarchical chunking.

Produces parent (section-level) and child chunks with proper deterministic IDs, 
hierarchy paths, bounding boxes, content_hash, context_prefix, and multi-tenancy annotations.
Implements token-aware refinement and content-type-aware handling.
"""
import hashlib
import re
from typing import Any
from uuid import UUID

from app.config.logging import get_logger
from app.models.chunk import Chunk, ChunkType
from app.utils.hashing import compute_chunk_hash

logger = get_logger(__name__)

# Token budget constants 
_TOKEN_TARGET = 512
_TOKEN_MAX = 1024
_WORDS_PER_TOKEN = 0.75

def _word_count(text: str) -> int:
    return len(text.split())

def _token_estimate(text: str) -> int:
    # A simple fallback proxy for token estimation when tiktoken is unavailable
    return int(_word_count(text) / _WORDS_PER_TOKEN)


class ChunkingService:
    """
    Stateless structure-aware chunker with content-type awareness.
    
    1. Group text blocks respecting semantic boundaries (paragraphs, headings).
    2. Split parents into child chunks respecting token limits and sentence boundaries.
    3. Tables and Figures remain intact with deterministic hashing.
    4. Validates chunk quality objectively.
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
        Produce hierarchical chunks from a parsed document with multi-tenancy.
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
                if self._validate_chunk(parent):
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
                    all_chunks.extend([c for c in children if self._validate_chunk(c)])

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
                if self._validate_chunk(chunk):
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
                if chunk and self._validate_chunk(chunk):
                    all_chunks.append(chunk)

        logger.info(
            "chunking.complete",
            document_id=doc_str,
            tenant_id=tenant_id,
            total_chunks=len(all_chunks),
        )
        return all_chunks

    def _validate_chunk(self, chunk: Chunk) -> bool:
        """Objective chunk quality validation without fake semantic scores."""
        tokens = _token_estimate(chunk.content)
        if tokens < 3 and chunk.chunk_type not in (ChunkType.IMAGE, ChunkType.DIAGRAM):
            return False  # Noise / Orphan
        if not chunk.document_id or not chunk.chunk_id:
            return False
        return True

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
        current_tokens = 0
        current_section_name = f"Page {page_num} Section {section_idx + 1}"

        for block in text_blocks:
            text = block.get("text", "").strip()
            bbox = block.get("bbox", [])
            if not text:
                continue

            # Detect content types (Warnings, Procedures, Specs)
            if re.match(r"^(warning|caution|note|danger):?", text, re.IGNORECASE):
                # Flush current and start a new block for warning to preserve it
                if current_texts:
                    parents.append(self._flush_parent(
                        current_texts, current_bboxes, page_num, doc_str, document_id, 
                        industry, tenant_id, assistant_id, knowledge_base_id, filename, current_section_name, section_idx
                    ))
                    current_texts, current_bboxes, current_tokens = [], [], 0
                    section_idx += 1
                current_section_name = "Warning/Note"

            block_tokens = _token_estimate(text)
            
            if current_tokens + block_tokens > _TOKEN_MAX and current_texts:
                parents.append(self._flush_parent(
                    current_texts, current_bboxes, page_num, doc_str, document_id, 
                    industry, tenant_id, assistant_id, knowledge_base_id, filename, current_section_name, section_idx
                ))
                current_texts, current_bboxes, current_tokens = [], [], 0
                section_idx += 1
                current_section_name = f"Page {page_num} Section {section_idx + 1}"

            current_texts.append(text)
            if bbox:
                current_bboxes.append(bbox)
            current_tokens += block_tokens

            if current_tokens >= _TOKEN_TARGET:
                parents.append(self._flush_parent(
                    current_texts, current_bboxes, page_num, doc_str, document_id, 
                    industry, tenant_id, assistant_id, knowledge_base_id, filename, current_section_name, section_idx
                ))
                current_texts, current_bboxes, current_tokens = [], [], 0
                section_idx += 1
                current_section_name = f"Page {page_num} Section {section_idx + 1}"

        if current_texts:
            parents.append(self._flush_parent(
                current_texts, current_bboxes, page_num, doc_str, document_id, 
                industry, tenant_id, assistant_id, knowledge_base_id, filename, current_section_name, section_idx
            ))

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
        section_name: str,
        section_idx: int,
    ) -> Chunk:
        content = "\n".join(texts)
        merged_bbox = self._merge_bboxes(bboxes) if bboxes else None
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
        # Token-aware splitting at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', parent.content)
        child_chunks: list[Chunk] = []
        child_idx = 0
        current_sentences = []
        current_tokens = 0
        CHILD_TARGET = 256

        for sentence in sentences:
            if not sentence.strip():
                continue
            s_tokens = _token_estimate(sentence)
            if current_tokens + s_tokens > CHILD_TARGET and current_sentences:
                child = self._flush_child(
                    current_sentences, parent, document_id, industry, tenant_id, assistant_id, knowledge_base_id, filename, child_idx
                )
                child_chunks.append(child)
                child_idx += 1
                current_sentences = []
                current_tokens = 0
            
            current_sentences.append(sentence)
            current_tokens += s_tokens

        if current_sentences:
            child = self._flush_child(
                current_sentences, parent, document_id, industry, tenant_id, assistant_id, knowledge_base_id, filename, child_idx
            )
            child_chunks.append(child)

        return child_chunks

    def _flush_child(
        self, 
        sentences: list[str], 
        parent: Chunk, 
        document_id: UUID, 
        industry: str, 
        tenant_id: str, 
        assistant_id: str, 
        knowledge_base_id: str, 
        filename: str, 
        child_idx: int
    ) -> Chunk:
        content = " ".join(sentences)
        chunk_id = f"{parent.chunk_id}::c{child_idx}"
        c_hash = compute_chunk_hash(content)
        ctx_prefix = f"{filename} > Page {parent.page_number} > {parent.section or 'Section'} > Chunk {child_idx}"

        return Chunk(
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
        x0 = min(b[0] for b in bboxes if len(b) >= 4)
        y0 = min(b[1] for b in bboxes if len(b) >= 4)
        x1 = max(b[2] for b in bboxes if len(b) >= 4)
        y1 = max(b[3] for b in bboxes if len(b) >= 4)
        return [x0, y0, x1, y1]
