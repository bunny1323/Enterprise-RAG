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
from app.models.structure import StructureEntry
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
    ) -> tuple[list[Chunk], list[StructureEntry]]:
        """
        Produce hierarchical chunks from a parsed document with multi-tenancy.

        Returns:
            (chunks, structure_entries) where structure_entries contains
            section headers and page-format explanations for the structure index.
        """
        all_chunks: list[Chunk] = []
        structure_entries: list[StructureEntry] = []
        doc_str = str(document_id)

        # Mutable section context that persists across ALL pages (sections span pages)
        section_ctx: dict = {
            "number": None,   # int | None — current section number
            "title": None,    # str | None — e.g. "HYDRAULIC SYSTEM"
            "label": None,    # str — human-readable label
            "idx": 0,         # monotonic chunk index within document
        }

        for page_data in parsed_doc.get("pages", []):
            page_num: int = page_data.get("page_num", 0)
            text_blocks: list[dict] = page_data.get("text_blocks", [])
            tables: list[dict] = page_data.get("tables", [])
            figures: list[dict] = page_data.get("figures", [])

            # ── Text chunking (parent → children) ─────────────────────────────
            new_sections, parent_chunks = self._make_parent_chunks_with_structure(
                text_blocks=text_blocks,
                page_num=page_num,
                doc_str=doc_str,
                document_id=document_id,
                industry=industry,
                tenant_id=tenant_id,
                assistant_id=assistant_id,
                knowledge_base_id=knowledge_base_id,
                filename=filename,
                section_ctx=section_ctx,
            )
            structure_entries.extend(new_sections)
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

            # ── Page-format detection (first 3 pages only) ─────────────────────
            # Look for patterns like "2-3" near text explaining "item number" /
            # "consecutive page". Only scan early pages to stay efficient.
            if page_num <= 3:
                pf_entries = self._detect_page_format_entries(text_blocks, page_num)
                structure_entries.extend(pf_entries)

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
                    section_ctx=section_ctx,
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
                    section_ctx=section_ctx,
                )
                if chunk and self._validate_chunk(chunk):
                    all_chunks.append(chunk)

        logger.info(
            "chunking.complete",
            document_id=doc_str,
            tenant_id=tenant_id,
            total_chunks=len(all_chunks),
            structure_entries=len(structure_entries),
        )
        # Deduplicate structure entries (same section number seen on multiple pages
        # is only stored once — keep the first occurrence with lowest page_number)
        structure_entries = self._deduplicate_structure_entries(structure_entries)
        return all_chunks, structure_entries

    # ── Structure-aware parent chunk creation (emits structure entries too) ──────

    def _make_parent_chunks_with_structure(
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
        section_ctx: dict,
    ) -> tuple[list[StructureEntry], list[Chunk]]:
        """Same as _make_parent_chunks but also returns StructureEntry items."""
        structure_entries: list[StructureEntry] = []
        parents: list[Chunk] = []
        current_texts: list[str] = []
        current_bboxes: list[list[float]] = []
        current_tokens = 0

        def _flush():
            nonlocal current_texts, current_bboxes, current_tokens
            if not current_texts:
                return
            parents.append(self._flush_parent(
                current_texts, current_bboxes, page_num, doc_str, document_id,
                industry, tenant_id, assistant_id, knowledge_base_id, filename,
                section_ctx,
            ))
            section_ctx["idx"] += 1
            current_texts, current_bboxes, current_tokens = [], [], 0

        for block in text_blocks:
            text = block.get("text", "").strip()
            bbox = block.get("bbox", [])
            item_type = block.get("item_type", "TextItem")
            if not text:
                continue

            # ── Section boundary detection ────────────────────────────────────
            section_match = re.match(
                r"^SECTION\s+(\d+)\s*(.*?)$", text.strip(), re.IGNORECASE
            )
            is_section_heading = (
                item_type == "SectionHeaderItem" or section_match is not None
            )

            if is_section_heading and section_match:
                _flush()
                sec_num = int(section_match.group(1))
                sec_title = section_match.group(2).strip().upper() or f"SECTION {sec_num}"
                section_ctx["number"] = sec_num
                section_ctx["title"] = sec_title
                section_ctx["label"] = f"SECTION {sec_num} {sec_title}".strip()
                # Emit structure entry for this section
                structure_entries.append(StructureEntry(
                    structure_type="section",
                    number=sec_num,
                    title=sec_title,
                    raw_text=text.strip(),
                    page_number=page_num,
                ))
                current_texts.append(text)
                if bbox:
                    current_bboxes.append(bbox)
                current_tokens += _token_estimate(text)
                continue

            if is_section_heading and item_type == "SectionHeaderItem":
                _flush()
                current_texts.append(text)
                if bbox:
                    current_bboxes.append(bbox)
                current_tokens += _token_estimate(text)
                continue

            if re.match(r"^(warning|caution|note|danger):?", text, re.IGNORECASE):
                _flush()
                current_texts.append(text)
                if bbox:
                    current_bboxes.append(bbox)
                current_tokens += _token_estimate(text)
                _flush()
                continue

            block_tokens = _token_estimate(text)
            if current_tokens + block_tokens > _TOKEN_MAX and current_texts:
                _flush()
            current_texts.append(text)
            if bbox:
                current_bboxes.append(bbox)
            current_tokens += block_tokens
            if current_tokens >= _TOKEN_TARGET:
                _flush()

        _flush()
        return structure_entries, parents

    def _detect_page_format_entries(
        self,
        text_blocks: list[dict],
        page_num: int,
    ) -> list[StructureEntry]:
        """
        Detect page-number-format explanation blocks in early pages.

        Matches passages that explain notation like "2-3" where:
          '2' = item number
          '3' = consecutive page number

        Heuristic: must contain a digit-dash-digit pattern AND one of
        the keywords 'item number', 'consecutive', 'page number'.
        """
        entries: list[StructureEntry] = []
        # Collect all text on this page
        page_text = " ".join(b.get("text", "") for b in text_blocks).strip()
        if not page_text:
            return entries

        # Find all X-Y notation patterns
        notations = re.findall(r"\b(\d+)-(\d+)\b", page_text)
        keywords = ["item number", "consecutive", "page number", "each item"]
        has_keyword = any(kw in page_text.lower() for kw in keywords)

        if notations and has_keyword:
            for x, y in notations:
                notation = f"{x}-{y}"
                entries.append(StructureEntry(
                    structure_type="page_format",
                    number=None,
                    title=None,
                    raw_text=page_text[:2000],  # store full page context (capped)
                    page_number=page_num,
                    metadata={
                        "notation": notation,
                        "item_number": x,
                        "page_number_part": y,
                    },
                ))
        return entries

    def _deduplicate_structure_entries(
        self, entries: list[StructureEntry]
    ) -> list[StructureEntry]:
        """Keep first occurrence of each (structure_type, number/notation) pair."""
        seen: set[tuple] = set()
        deduped: list[StructureEntry] = []
        for e in entries:
            if e.structure_type == "section":
                key = ("section", e.number)
            elif e.structure_type == "page_format":
                key = ("page_format", e.metadata.get("notation"))
            else:
                key = (e.structure_type, e.title)
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        return deduped

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
        # Section context is passed in and updated in place via a mutable dict
        section_ctx: dict,
    ) -> list[Chunk]:
        """Build parent chunks with structure-aware section detection.

        section_ctx is a mutable dict with keys:
            number: int | None   — current section number
            title:  str | None   — current section title (e.g. "HYDRAULIC SYSTEM")
            label:  str          — human-readable label for context_prefix
            idx:    int          — monotonic parent chunk index within document
        """
        parents: list[Chunk] = []
        current_texts: list[str] = []
        current_bboxes: list[list[float]] = []
        current_tokens = 0

        def _flush():
            nonlocal current_texts, current_bboxes, current_tokens
            if not current_texts:
                return
            parents.append(self._flush_parent(
                current_texts, current_bboxes, page_num, doc_str, document_id,
                industry, tenant_id, assistant_id, knowledge_base_id, filename,
                section_ctx,
            ))
            section_ctx["idx"] += 1
            current_texts, current_bboxes, current_tokens = [], [], 0

        for block in text_blocks:
            text = block.get("text", "").strip()
            bbox = block.get("bbox", [])
            item_type = block.get("item_type", "TextItem")
            if not text:
                continue

            # ── Section boundary detection ────────────────────────────────────
            # ONLY trigger on SectionHeaderItem OR explicit "SECTION N TITLE" pattern.
            # "3. CONVERSION TABLE" (numbered list item) is intentionally NOT matched.
            section_match = re.match(
                r"^SECTION\s+(\d+)\s*(.*?)$", text.strip(), re.IGNORECASE
            )
            is_section_heading = (
                item_type == "SectionHeaderItem" or section_match is not None
            )

            if is_section_heading and section_match:
                # Flush accumulated text before starting a new section
                _flush()
                sec_num = int(section_match.group(1))
                sec_title = section_match.group(2).strip().upper() or f"SECTION {sec_num}"
                section_ctx["number"] = sec_num
                section_ctx["title"] = sec_title
                section_ctx["label"] = f"SECTION {sec_num} {sec_title}".strip()
                # Include the heading text itself in the new chunk
                current_texts.append(text)
                if bbox:
                    current_bboxes.append(bbox)
                current_tokens += _token_estimate(text)
                continue

            if is_section_heading and item_type == "SectionHeaderItem":
                # SectionHeaderItem but didn't match our SECTION N pattern —
                # treat as a subsection/sub-heading: flush and start a new parent
                # but keep current section_ctx.number / title unchanged.
                _flush()
                current_texts.append(text)
                if bbox:
                    current_bboxes.append(bbox)
                current_tokens += _token_estimate(text)
                continue

            # ── Warning / special block ───────────────────────────────────────
            if re.match(r"^(warning|caution|note|danger):?", text, re.IGNORECASE):
                _flush()
                current_texts.append(text)
                if bbox:
                    current_bboxes.append(bbox)
                current_tokens += _token_estimate(text)
                _flush()
                continue

            # ── Normal body text accumulation ─────────────────────────────────
            block_tokens = _token_estimate(text)

            if current_tokens + block_tokens > _TOKEN_MAX and current_texts:
                _flush()

            current_texts.append(text)
            if bbox:
                current_bboxes.append(bbox)
            current_tokens += block_tokens

            if current_tokens >= _TOKEN_TARGET:
                _flush()

        _flush()
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
        section_ctx: dict,
    ) -> Chunk:
        content = "\n".join(texts)
        merged_bbox = self._merge_bboxes(bboxes) if bboxes else None
        chunk_idx = section_ctx.get("idx", 0)
        chunk_id = f"{doc_str}::p{page_num}::s{chunk_idx}"
        c_hash = compute_chunk_hash(content)

        sec_num = section_ctx.get("number")
        sec_title = section_ctx.get("title")
        sec_label = section_ctx.get("label") or (
            f"SECTION {sec_num} {sec_title}".strip() if sec_num else f"Page {page_num}"
        )
        ctx_prefix = f"{filename} > {sec_label} > Page {page_num}"

        return Chunk(
            chunk_id=chunk_id,
            parent_id=None,
            document_id=document_id,
            tenant_id=tenant_id,
            assistant_id=assistant_id,
            knowledge_base_id=knowledge_base_id,
            content=content,
            content_hash=c_hash,
            section=sec_label,
            section_number=sec_num,
            section_title=sec_title,
            file_name=filename,
            context_prefix=ctx_prefix,
            embedding_representation="text",
            page_number=page_num,
            bounding_box=merged_bbox,
            chunk_type=ChunkType.TEXT,
            industry_domain=industry,
            hierarchy_path=f"doc.section{sec_num or 0}.page{page_num}.chunk{chunk_idx}",
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
            section_number=parent.section_number,
            section_title=parent.section_title,
            file_name=parent.file_name,
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
        section_ctx: dict | None = None,
    ) -> Chunk:
        markdown = table.get("markdown", "")
        c_hash = compute_chunk_hash(markdown)
        table_hash_id = c_hash[:12]
        chunk_id = f"{doc_str}::p{page_num}::t{table_hash_id}"
        sec_ctx = section_ctx or {}
        sec_num = sec_ctx.get("number")
        sec_title = sec_ctx.get("title")
        sec_label = sec_ctx.get("label") or f"Page {page_num}"
        ctx_prefix = f"{filename} > {sec_label} > Page {page_num} > Table"

        return Chunk(
            chunk_id=chunk_id,
            parent_id=None,
            document_id=document_id,
            tenant_id=tenant_id,
            assistant_id=assistant_id,
            knowledge_base_id=knowledge_base_id,
            content=markdown,
            content_hash=c_hash,
            section=sec_label,
            section_number=sec_num,
            section_title=sec_title,
            file_name=filename,
            context_prefix=ctx_prefix,
            embedding_representation="text",
            page_number=page_num,
            bounding_box=table.get("bbox"),
            chunk_type=ChunkType.TABLE,
            industry_domain=industry,
            hierarchy_path=f"doc.section{sec_num or 0}.page{page_num}.table_{table_hash_id}",
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
        section_ctx: dict | None = None,
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
        sec_ctx = section_ctx or {}
        sec_num = sec_ctx.get("number")
        sec_title = sec_ctx.get("title")
        sec_label = sec_ctx.get("label") or f"Page {page_num}"
        ctx_prefix = f"{filename} > {sec_label} > Page {page_num} > Figure"

        return Chunk(
            chunk_id=chunk_id,
            parent_id=None,
            document_id=document_id,
            tenant_id=tenant_id,
            assistant_id=assistant_id,
            knowledge_base_id=knowledge_base_id,
            content=content,
            content_hash=c_hash,
            section=sec_label,
            section_number=sec_num,
            section_title=sec_title,
            file_name=filename,
            context_prefix=ctx_prefix,
            embedding_representation=representation,
            page_number=page_num,
            bounding_box=figure.get("bbox"),
            chunk_type=chunk_type,
            industry_domain=industry,
            hierarchy_path=f"doc.section{sec_num or 0}.page{page_num}.figure_{fig_hash_id}",
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
