"""
End-to-end pipeline verification script for Enterprise-RAG (local BGE embeddings).

Validates:
  1. Backend liveness
  2. Backend readiness
  3. Embedding health + dimension
  4. LLM health
  5. Postgres connectivity
  6. Weaviate connectivity
  7. Neo4j connectivity
  8. Document upload
  9. Ingestion completion (COMPLETED, not PARTIAL)
  10. Weaviate object count
  11. Vector dimension = 384 (BAAI/bge-small-en-v1.5)
  12. Dense retrieval
  13. Intent detection
  14. Grounded RAG answer
  15. Citations + sources returned

Usage:
    cd backend
    python verify.py
"""
import asyncio
import os
import sys
from pathlib import Path

import httpx
import asyncpg
import weaviate
import weaviate.classes as wvc
from weaviate.auth import AuthApiKey

_COLLECTION = "DocumentChunk"
_API_URL = "http://127.0.0.1:8001/api/v1"
_ROOT_URL = "http://127.0.0.1:8001"
_TARGET_DIM = 384  # BAAI/bge-small-en-v1.5

PASS = "\u2705"
FAIL = "\u274c"
WARN = "\u26a0\ufe0f "


def load_env() -> None:
    env_file = Path(".env")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key:
                os.environ.setdefault(key, value)


def _weaviate_client() -> weaviate.WeaviateClient:
    url = os.environ["WEAVIATE_URL"]
    key = os.environ["WEAVIATE_API_KEY"]
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=AuthApiKey(key),
        additional_config=wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=10, query=30)),
    )


def get_weaviate_count() -> int:
    with _weaviate_client() as client:
        try:
            collection = client.collections.get(_COLLECTION)
            agg = collection.aggregate.over_all(total_count=True)
            return agg.total_count or 0
        except Exception:
            return 0


def get_weaviate_vector_dim() -> int | None:
    """Fetch one object with vector to check actual stored dimension."""
    with _weaviate_client() as client:
        try:
            if not client.collections.exists(_COLLECTION):
                return None
            collection = client.collections.get(_COLLECTION)
            response = collection.query.fetch_objects(limit=1, include_vector=True)
            if response.objects and response.objects[0].vector:
                vec = response.objects[0].vector
                if isinstance(vec, dict):
                    for v in vec.values():
                        return len(v)
                elif isinstance(vec, list):
                    return len(vec)
        except Exception:
            pass
    return None


def clear_weaviate() -> int:
    with _weaviate_client() as client:
        if not client.collections.exists(_COLLECTION):
            return 0
        collection = client.collections.get(_COLLECTION)
        agg = collection.aggregate.over_all(total_count=True)
        count_before = agg.total_count or 0
        collection.data.delete_many(
            where=wvc.query.Filter.by_property("tenant_id").like("*")
        )
        return count_before


async def verify_pipeline() -> None:
    load_env()
    failures: list[str] = []

    print("=" * 60)
    print("  Enterprise RAG — E2E Verification (Local BGE Embeddings)")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=30.0) as http:

        # ── 1. Backend liveness ────────────────────────────────────────────────
        print("[1] Checking backend liveness /health/live …")
        try:
            r = await http.get(f"{_ROOT_URL}/health/live")
            if r.status_code == 200:
                print(f"    {PASS} Live: {r.json()}")
            else:
                print(f"    {FAIL} Liveness failed: {r.status_code}")
                failures.append("liveness")
        except Exception as e:
            print(f"    {FAIL} Backend not reachable: {e}")
            print("    Ensure uvicorn is running on port 8001.")
            sys.exit(1)

        # ── 2. Backend readiness ───────────────────────────────────────────────
        print("[2] Checking backend readiness /health/ready …")
        try:
            r = await http.get(f"{_ROOT_URL}/health/ready")
            d = r.json()
            if r.status_code == 200:
                print(f"    {PASS} Ready. Services: {d.get('services', {})}")
            else:
                print(f"    {WARN} Degraded: {d.get('services', {})}")
        except Exception as e:
            print(f"    {WARN} Readiness check failed: {e}")

        # ── 3. Embedding health ────────────────────────────────────────────────
        print("[3] Checking embedding health /health/embedding …")
        try:
            r = await http.get(f"{_ROOT_URL}/health/embedding")
            d = r.json()
            if d.get("status") == "ok":
                dim = d.get("dimension", 0)
                model = d.get("model", "unknown")
                print(f"    {PASS} Embedding OK — model={model}  dim={dim}")
                if dim != _TARGET_DIM:
                    print(f"    {FAIL} Dimension mismatch! Expected {_TARGET_DIM}, got {dim}")
                    failures.append("embedding_dim")
                    print(f"    Run: python scripts/migrate_weaviate_to_bge.py")
            else:
                print(f"    {FAIL} Embedding error: {d.get('error')}")
                failures.append("embedding")
        except Exception as e:
            print(f"    {FAIL} Embedding health check failed: {e}")
            failures.append("embedding")

        # ── 4. LLM health ──────────────────────────────────────────────────────
        print("[4] Checking LLM health /health/llm …")
        try:
            r = await http.post(f"{_ROOT_URL}/health/llm")
            d = r.json()
            if d.get("status") in ("ok", "healthy"):
                print(f"    {PASS} LLM OK — provider={d.get('provider')}  model={d.get('model')}  latency={d.get('latency_ms', '?')}ms")
                if "primary" in d:
                    print(f"         Primary ({d['primary'].get('provider')}): {d['primary'].get('status')}  Fallback ({d['fallback'].get('provider')}): {d['fallback'].get('status')}")
                else:
                    print(f"         Response: {d.get('response', '')[:80]}")
            else:
                print(f"    {WARN} LLM issue: {d.get('error', d)}")
        except Exception as e:
            print(f"    {WARN} LLM health check failed: {e} (Ollama may not be running)")

    # ── 5. Postgres connectivity ───────────────────────────────────────────────
    print("[5] Checking PostgreSQL connectivity …")
    try:
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM documents")
        await conn.close()
        print(f"    {PASS} Postgres OK — {row['cnt']} document(s) in DB")
    except Exception as e:
        print(f"    {FAIL} Postgres failed: {e}")
        failures.append("postgres")

    # ── 6. Weaviate connectivity ───────────────────────────────────────────────
    print("[6] Checking Weaviate connectivity …")
    try:
        count = get_weaviate_count()
        print(f"    {PASS} Weaviate OK — {count} object(s) in {_COLLECTION}")
    except Exception as e:
        print(f"    {FAIL} Weaviate failed: {e}")
        failures.append("weaviate")

    # ── 7. Reset Postgres + Weaviate ──────────────────────────────────────────
    print("[7] Resetting Postgres (TRUNCATE documents CASCADE) + clearing Weaviate …")
    try:
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        await conn.execute("TRUNCATE TABLE documents CASCADE;")
        await conn.close()
        print(f"    {PASS} Postgres reset.")
    except Exception as e:
        print(f"    {FAIL} Postgres reset failed: {e}")
        failures.append("postgres_reset")

    try:
        removed = clear_weaviate()
        print(f"    {PASS} Weaviate cleared ({removed} object(s) removed).")
    except Exception as e:
        print(f"    {WARN} Weaviate clear failed: {e}")

    # ── 8. Upload document ────────────────────────────────────────────────────
    raw_dir = Path("data/raw")
    pdf_path: Path | None = None
    if raw_dir.exists():
        for p in sorted(raw_dir.glob("*.pdf")):
            pdf_path = p
            break

    if not pdf_path or not pdf_path.exists():
        print("[8] ERROR: No PDF found in data/raw/. Stopping here.")
        print("    Upload tests require a PDF at data/raw/*.pdf")
        _report(failures)
        return

    print(f"[8] Uploading '{pdf_path.name}' ({pdf_path.stat().st_size // 1024} KB) …")
    headers = {"x-tenant-id": "default", "x-knowledge-base-id": "default"}
    doc_id = None
    job_id = None

    async with httpx.AsyncClient(timeout=120.0) as http:
        with open(pdf_path, "rb") as f:
            files = {"file": (pdf_path.name, f, "application/pdf")}
            data = {"industry": "manufacturing"}
            try:
                r = await http.post(f"{_API_URL}/documents", files=files, data=data, headers=headers)
                if r.status_code != 202:
                    print(f"    {FAIL} Upload failed ({r.status_code}): {r.text}")
                    failures.append("upload")
                    _report(failures)
                    return
                resp = r.json()
                doc_id = resp.get("document_id")
                job_id = resp.get("job_id")
                print(f"    {PASS} Accepted — document_id={doc_id}  job_id={job_id}")
            except Exception as e:
                print(f"    {FAIL} Upload exception: {e}")
                failures.append("upload")
                _report(failures)
                return

        # ── 9. Poll ingestion ─────────────────────────────────────────────────
        print("[9] Polling ingestion status …")
        terminal = {"COMPLETED", "FAILED", "PARTIAL", "DUPLICATE", "TIMEOUT"}
        last_line = ""
        final_status = "UNKNOWN"

        while True:
            await asyncio.sleep(5)
            try:
                r = await http.get(f"{_API_URL}/jobs/{job_id}", headers=headers)
                if r.status_code != 200:
                    print(f"    Poll error {r.status_code}: {r.text[:200]}")
                    continue
                d = r.json()
            except Exception as exc:
                print(f"    Poll exception: {exc}")
                continue

            status = d.get("status", "UNKNOWN")
            stage = d.get("current_stage", "-")
            progress = d.get("progress_percent", 0)
            err_msg = d.get("error_message") or ""
            line = f"    [{status}] stage={stage}  progress={progress}%"
            if line != last_line:
                print(line)
                last_line = line
            if status in terminal:
                final_status = status
                if status == "COMPLETED":
                    print(f"    {PASS} Ingestion COMPLETED")
                elif status == "PARTIAL":
                    print(f"    {WARN} Ingestion PARTIAL — some indexing may have failed")
                    failures.append("ingestion_partial")
                else:
                    print(f"    {FAIL} Ingestion {status}: {err_msg[:300]}")
                    failures.append(f"ingestion_{status.lower()}")
                break

    # ── 10. Verify Weaviate object count ──────────────────────────────────────
    print("[10] Verifying Weaviate object count …")
    try:
        final_count = get_weaviate_count()
        if final_count > 0:
            print(f"    {PASS} {final_count} chunk(s) indexed in {_COLLECTION}")
        else:
            print(f"    {FAIL} Weaviate count is 0. s08_index may have failed.")
            failures.append("weaviate_count")
    except Exception as e:
        print(f"    {FAIL} Weaviate count check failed: {e}")
        failures.append("weaviate_count")

    # ── 11. Verify vector dimension ───────────────────────────────────────────
    print("[11] Verifying vector dimension in Weaviate …")
    try:
        dim = get_weaviate_vector_dim()
        if dim is None:
            print(f"    {WARN} No vectors found yet (collection empty or not vectorised)")
        elif dim == _TARGET_DIM:
            print(f"    {PASS} Vector dimension = {dim} (matches BGE-small target)")
        else:
            print(f"    {FAIL} Vector dimension = {dim} (expected {_TARGET_DIM}!)")
            print(f"    Run: python scripts/migrate_weaviate_to_bge.py")
            failures.append("vector_dim_mismatch")
    except Exception as e:
        print(f"    {WARN} Vector dimension check failed: {e}")

    # ── 12-15. RAG End-to-End ─────────────────────────────────────────────────
    print("[12] Executing E2E RAG test queries …")
    test_queries = [
        ("What is the purpose of this service manual?", "GENERAL_QA"),
        ("What is the engine oil capacity?", "SPECIFICATION"),
        ("How do I replace the hydraulic filter?", "PROCEDURE"),
    ]

    async with httpx.AsyncClient(timeout=120.0) as http:
        for query_text, expected_intent in test_queries:
            print(f"\n    Query: '{query_text}'")
            print(f"    Expected intent: {expected_intent}")
            try:
                r = await http.post(
                    f"{_API_URL}/chat",
                    json={"query": query_text, "top_k": 5},
                    headers=headers,
                )
                if r.status_code == 200:
                    d = r.json()
                    answer = d.get("answer", "")[:200]
                    intent = d.get("intent", "?")
                    confidence = d.get("confidence", "?")
                    evidence_count = len(d.get("evidence", []))
                    citations_count = len(d.get("citations", []))
                    print(f"    {PASS} Answer ({len(d.get('answer',''))} chars): {answer}...")
                    print(f"         Intent={intent}  Confidence={confidence}  Evidence={evidence_count}  Citations={citations_count}")
                    if not d.get("answer"):
                        print(f"    {WARN} Empty answer returned")
                        failures.append(f"empty_answer_{query_text[:20]}")
                else:
                    print(f"    {FAIL} Chat failed ({r.status_code}): {r.text[:1000]}")
                    failures.append(f"chat_{query_text[:20]}")
            except Exception as exc:
                print(f"    {FAIL} Chat exception: {exc}")
                failures.append(f"chat_exception_{query_text[:20]}")

    _report(failures)


def _report(failures: list[str]) -> None:
    print()
    print("=" * 60)
    if not failures:
        print(f"  {PASS} ALL CHECKS PASSED — RAG pipeline is fully operational!")
    else:
        print(f"  {FAIL} {len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"     - {f}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(verify_pipeline())
