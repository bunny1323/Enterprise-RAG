"""
Unit tests for MinHash and LSH deduplication logic.
"""
from app.services.deduplication.minhash import LSHIndex, MinHasher


def test_minhash_and_lsh():
    hasher = MinHasher(num_perm=64)
    index = LSHIndex(num_perm=64, bands=8)

    text1 = "Hydraulic pressure specification for main pump model X100 is 250 bar."
    text2 = "Hydraulic pressure spec for main pump model X100 is 250 bar."
    text3 = "Electrical wiring diagram for safety switch relay."

    sig1 = hasher.compute_signature(text1)
    sig2 = hasher.compute_signature(text2)
    sig3 = hasher.compute_signature(text3)

    index.insert("chunk_1", sig1)
    index.insert("chunk_3", sig3)

    candidates = index.query(sig2)
    assert "chunk_1" in candidates
    assert "chunk_3" not in candidates
