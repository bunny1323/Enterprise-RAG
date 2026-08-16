"""
MinHash and LSH implementation for Level 4 near-duplicate detection.
Calculates MinHash signatures for text chunks and indexes them into LSH buckets to quickly find near-duplicate candidate pairs.
"""
import hashlib
import re
from typing import Any


def _get_shingles(text: str, k: int = 3) -> set[str]:
    """Extract character k-shingles from text."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    if len(normalized) < k:
        return {normalized}
    return {normalized[i : i + k] for i in range(len(normalized) - k + 1)}


class MinHasher:
    """
    MinHash generator for calculating Jaccard similarity estimations.
    """

    def __init__(self, num_perm: int = 64, seed: int = 42) -> None:
        self.num_perm = num_perm
        # Generate pseudo-random hash coefficients (a * x + b) % prime
        self._prime = 4294967311
        import random

        rnd = random.Random(seed)
        self._a = [rnd.randint(1, self._prime - 1) for _ in range(num_perm)]
        self._b = [rnd.randint(0, self._prime - 1) for _ in range(num_perm)]

    def compute_signature(self, text: str) -> list[int]:
        """Compute MinHash signature vector for input text."""
        shingles = _get_shingles(text)
        if not shingles:
            return [0] * self.num_perm

        signature = [self._prime] * self.num_perm

        for shingle in shingles:
            # Hash shingle string to 32-bit int
            h = int(hashlib.md5(shingle.encode("utf-8")).hexdigest()[:8], 16)
            for i in range(self.num_perm):
                hash_val = (self._a[i] * h + self._b[i]) % self._prime
                if hash_val < signature[i]:
                    signature[i] = hash_val

        return signature


class LSHIndex:
    """
    Locality-Sensitive Hashing index for fast candidate retrieval.
    """

    def __init__(self, num_perm: int = 64, bands: int = 8) -> None:
        self.num_perm = num_perm
        self.bands = bands
        self.rows_per_band = num_perm // bands
        # band_idx -> (band_hash -> list[item_id])
        self._buckets: list[dict[str, list[str]]] = [{} for _ in range(bands)]

    def insert(self, item_id: str, signature: list[int]) -> None:
        """Insert item_id with its MinHash signature into the LSH index."""
        for b in range(self.bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_tuple = tuple(signature[start:end])
            band_hash = hashlib.md5(str(band_tuple).encode("utf-8")).hexdigest()

            if band_hash not in self._buckets[b]:
                self._buckets[b][band_hash] = []
            self._buckets[b][band_hash].append(item_id)

    def query(self, signature: list[int]) -> set[str]:
        """Find all candidate item_ids that share at least one band hash with signature."""
        candidates: set[str] = set()
        for b in range(self.bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_tuple = tuple(signature[start:end])
            band_hash = hashlib.md5(str(band_tuple).encode("utf-8")).hexdigest()

            if band_hash in self._buckets[b]:
                candidates.update(self._buckets[b][band_hash])
        return candidates
