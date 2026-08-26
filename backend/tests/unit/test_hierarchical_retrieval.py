from unittest.mock import AsyncMock

import pytest

from app.models.retrieval import SearchResult
from app.services.retrieval.hierarchical import (
    HierarchicalRetrievalService,
)


@pytest.mark.asyncio
async def test_child_expands_to_parent() -> None:
    """A retrieved child should be replaced with its parent."""

    postgres = AsyncMock()

    postgres.fetch.return_value = [
        {
            "chunk_id": "parent-1",
            "parent_id": None,
            "document_id": "doc-1",
            "content": "This is the full parent context.",
            "page_number": 1,
            "chunk_type": "PARENT",
            "section": "Introduction",
            "subsection": None,
            "context_prefix": None,
            "metadata": {},
        }
    ]

    service = HierarchicalRetrievalService(postgres)

    child = SearchResult(
        chunk_id="child-1",
        score=0.95,
        content="This is a small child chunk.",
        page_number=1,
        document_id="doc-1",
        chunk_type="CHILD",
        parent_id="parent-1",
    )

    results = await service.expand_to_parents(
        [child],
        tenant_id="tenant-1",
        knowledge_base_id="kb-1",
    )

    assert len(results) == 1
    assert results[0].chunk_id == "parent-1"
    assert results[0].parent_id is None
    assert results[0].content == "This is the full parent context."
    assert results[0].metadata["retrieved_child_id"] == "child-1"
    assert results[0].metadata["hierarchical_expansion"] is True


@pytest.mark.asyncio
async def test_missing_parent_keeps_child() -> None:
    """A valid child should remain if its parent cannot be found."""

    postgres = AsyncMock()

    # PostgreSQL returns no matching parent.
    postgres.fetch.return_value = []

    service = HierarchicalRetrievalService(postgres)

    child = SearchResult(
        chunk_id="child-1",
        score=0.95,
        content="Important child content.",
        page_number=2,
        document_id="doc-1",
        chunk_type="CHILD",
        parent_id="missing-parent",
    )

    results = await service.expand_to_parents(
        [child],
        tenant_id="tenant-1",
        knowledge_base_id="kb-1",
    )

    assert len(results) == 1
    assert results[0].chunk_id == "child-1"
    assert results[0].parent_id == "missing-parent"
    assert results[0].content == "Important child content."


@pytest.mark.asyncio
async def test_duplicate_children_return_one_parent() -> None:
    """Multiple children with the same parent should produce one parent."""

    postgres = AsyncMock()

    postgres.fetch.return_value = [
        {
            "chunk_id": "parent-1",
            "parent_id": None,
            "document_id": "doc-1",
            "content": "Shared parent context.",
            "page_number": 1,
            "chunk_type": "PARENT",
            "section": "Chapter 1",
            "subsection": None,
            "context_prefix": None,
            "metadata": {},
        }
    ]

    service = HierarchicalRetrievalService(postgres)

    children = [
        SearchResult(
            chunk_id="child-1",
            score=0.95,
            content="Child one.",
            document_id="doc-1",
            parent_id="parent-1",
        ),
        SearchResult(
            chunk_id="child-2",
            score=0.92,
            content="Child two.",
            document_id="doc-1",
            parent_id="parent-1",
        ),
        SearchResult(
            chunk_id="child-3",
            score=0.89,
            content="Child three.",
            document_id="doc-1",
            parent_id="parent-1",
        ),
    ]

    results = await service.expand_to_parents(
        children,
        tenant_id="tenant-1",
        knowledge_base_id="kb-1",
    )

    assert len(results) == 1
    assert results[0].chunk_id == "parent-1"
    assert results[0].content == "Shared parent context."
