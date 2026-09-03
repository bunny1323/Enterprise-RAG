"""
End-to-End Regression Test Suite for Enterprise RAG.

Validates:
1. Document Structure & Major Section Count (COUNT_QUERY -> 9)
2. Canonical Section List (LIST_QUERY -> 9 Sections)
3. Page Number Format Notation ("2-3" explanation -> item number 2, page 3)
4. Existing working Section Retrieval (Section 3 -> Hydraulic System, Section 4 -> Electrical System, Section 5 -> Mechatronics System)
5. Relationship & Multi-hop Reasoning
"""
import pytest
import httpx

BASE_URL = "http://localhost:8001/api/v1"
HEADERS = {
    "Content-Type": "application/json",
    "X-Tenant-ID": "default",
    "X-Knowledge-Base-ID": "default",
    "X-Access-Level": "INTERNAL",
}


@pytest.mark.asyncio
async def test_health():
    """Verify backend is healthy."""
    async with httpx.AsyncClient(base_url="http://localhost:8001") as client:
        resp = await client.get("/health/live")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_query_count_major_sections():
    """Test A: How many major sections are described in the service manual? (Expected: 9)"""
    query = "How many major sections are described in the service manual?"
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        resp = await client.post("/chat", json={"query": query}, headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        answer = data["answer"]
        assert "9" in answer or "nine" in answer.lower(), f"Expected '9' in answer: {answer}"


@pytest.mark.asyncio
async def test_query_list_major_sections():
    """Test B: What are the major sections in the service manual? (Expected: Sections 1-9)"""
    query = "What are the major sections in the service manual?"
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        resp = await client.post("/chat", json={"query": query}, headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        answer = data["answer"].lower()
        # Verify key major sections appear
        assert "general" in answer
        assert "structure and function" in answer
        assert "hydraulic" in answer


@pytest.mark.asyncio
async def test_query_page_number_format_2_3():
    """Test C/D: What does 2-3 mean / what does the 2 represent?"""
    query = "In the example page number '2-3', what does the '2' represent?"
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        resp = await client.post("/chat", json={"query": query}, headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        answer = data["answer"].lower()
        assert "item" in answer or "structure and function" in answer, f"Expected item number in answer: {answer}"


@pytest.mark.asyncio
async def test_regression_section_3():
    """Regression Test: What does Section 3 cover? (Expected: HYDRAULIC SYSTEM)"""
    query = "What does Section 3 cover?"
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        resp = await client.post("/chat", json={"query": query}, headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        answer = data["answer"].lower()
        assert "hydraulic" in answer, f"Expected Hydraulic System in answer: {answer}"


@pytest.mark.asyncio
async def test_regression_section_4():
    """Regression Test: What does Section 4 cover? (Expected: ELECTRICAL SYSTEM)"""
    query = "What does Section 4 cover?"
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        resp = await client.post("/chat", json={"query": query}, headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        answer = data["answer"].lower()
        assert "electrical" in answer, f"Expected Electrical System in answer: {answer}"


@pytest.mark.asyncio
async def test_regression_section_5():
    """Regression Test: What does Section 5 cover? (Expected: MECHATRONICS SYSTEM)"""
    query = "What does Section 5 cover?"
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        resp = await client.post("/chat", json={"query": query}, headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        answer = data["answer"].lower()
        assert "mechatronics" in answer, f"Expected Mechatronics System in answer: {answer}"


@pytest.mark.asyncio
async def test_relationship_section_2_troubleshooting():
    """Relationship Test: Which section provides reference material for troubleshooting?"""
    query = "Which section serves as reference material for troubleshooting?"
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        resp = await client.post("/chat", json={"query": query}, headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        answer = data["answer"].lower()
        assert "section 2" in answer or "structure and function" in answer, f"Expected Section 2 in answer: {answer}"
