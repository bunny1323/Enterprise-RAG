"""
Document structure domain models.

Represent structural metadata extracted from parsed documents:
- Section headers (SECTION 1 GENERAL, etc.)
- Page-format explanations (e.g. "2-3 means item 2, page 3")
- Figure/table references (future)
"""
from pydantic import BaseModel, Field


class StructureEntry(BaseModel):
    """
    Atomic structural element extracted during ingestion.
    Stored in Postgres `document_structure` table.
    """
    structure_type: str = Field(
        ...,
        description="Type: 'section' | 'page_format' | 'figure_ref' | 'table_ref'"
    )
    number: int | None = Field(
        default=None,
        description="Section number (e.g. 3 for 'SECTION 3 HYDRAULIC SYSTEM')"
    )
    title: str | None = Field(
        default=None,
        description="Section title (e.g. 'HYDRAULIC SYSTEM')"
    )
    raw_text: str = Field(
        default="",
        description="Verbatim source text that produced this entry"
    )
    page_number: int = Field(
        default=0,
        description="1-based page number where entry was found"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Extra fields (e.g. notation='2-3', item_label='Structure and Function')"
    )
