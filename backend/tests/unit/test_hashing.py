"""
Unit tests for hashing utilities.
"""
from app.utils.hashing import compute_content_hash, compute_chunk_hash, compute_sha256_bytes


def test_compute_sha256_bytes():
    data = b"Hello Enterprise RAG"
    h1 = compute_sha256_bytes(data)
    h2 = compute_sha256_bytes(data)
    assert h1 == h2
    assert len(h1) == 64


def test_compute_content_hash_normalization():
    text1 = "  Hello   World!\nThis is A Test.  "
    text2 = "hello world! this is a test."
    h1 = compute_content_hash(text1)
    h2 = compute_content_hash(text2)
    assert h1 == h2


def test_compute_chunk_hash():
    chunk1 = "Section 1: Hydraulic Pump Specification"
    h1 = compute_chunk_hash(chunk1)
    assert len(h1) == 64
