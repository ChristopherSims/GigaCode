"""Duplicate / near-duplicate code detection using MinHash + LSH.

Lightweight implementation with no external dependencies beyond Python stdlib.
For large codebases (>100K chunks) consider ``datasketch`` for speed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Default parameters tuned for code chunks
_DEFAULT_NGRAM = 4
_DEFAULT_NUM_HASHES = 128
_DEFAULT_BANDS = 16
_DEFAULT_ROWS = _DEFAULT_NUM_HASHES // _DEFAULT_BANDS
_DEFAULT_JACCARD_THRESHOLD = 0.85


def _tokenize(text: str) -> list[str]:
    """Simple token list for n-gram generation."""
    tokens = re.findall(r"[A-Za-z0-9_]+", text)
    return [t.lower() for t in tokens]


def _shingles(text: str, n: int = _DEFAULT_NGRAM) -> set[str]:
    """Return a set of n-gram shingles."""
    tokens = _tokenize(text)
    if len(tokens) < n:
        # If too short, treat the whole text as one shingle
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class MinHash:
    """Simple MinHash signature generator."""

    def __init__(self, num_hashes: int = _DEFAULT_NUM_HASHES) -> None:
        self.num_hashes = num_hashes
        # Deterministic hash seeds
        self._seeds = [hashlib.sha256(str(i).encode()).digest()[:4] for i in range(num_hashes)]

    def signature(self, shingles: set[str]) -> list[int]:
        """Compute min-hash signature for a set of shingles."""
        sig: list[int] = []
        for seed_bytes in self._seeds:
            seed = int.from_bytes(seed_bytes, "little")
            min_val = 2**32
            for shingle in shingles:
                h = hash((seed, shingle)) & 0xFFFFFFFF
                if h < min_val:
                    min_val = h
            sig.append(min_val)
        return sig


class LshIndex:
    """Locality-Sensitive Hashing index for approximate near-duplicate detection."""

    def __init__(self, num_hashes: int = _DEFAULT_NUM_HASHES, bands: int = _DEFAULT_BANDS) -> None:
        self.num_hashes = num_hashes
        self.bands = bands
        self.rows = num_hashes // bands
        self._buckets: list[dict[str, set[int]]] = [{} for _ in range(bands)]

    def add(self, doc_id: int, signature: list[int]) -> None:
        """Insert a document signature into the LSH index."""
        for b in range(self.bands):
            band_sig = tuple(signature[b * self.rows : (b + 1) * self.rows])
            key = str(band_sig)
            self._buckets[b].setdefault(key, set()).add(doc_id)

    def query(self, doc_id: int, signature: list[int]) -> set[int]:
        """Return candidate doc_ids that share at least one band with *doc_id*."""
        candidates: set[int] = set()
        for b in range(self.bands):
            band_sig = tuple(signature[b * self.rows : (b + 1) * self.rows])
            key = str(band_sig)
            candidates.update(self._buckets[b].get(key, set()))
        candidates.discard(doc_id)
        return candidates


def find_duplicates(
    chunks: list[Any],
    threshold: float = _DEFAULT_JACCARD_THRESHOLD,
    ngram: int = _DEFAULT_NGRAM,
    num_hashes: int = _DEFAULT_NUM_HASHES,
    bands: int = _DEFAULT_BANDS,
) -> list[dict[str, Any]]:
    """Find near-duplicate code chunks within a buffer.

    Args:
        chunks: List of CodeChunk objects (or any object with ``.text``, ``.file``, ``.start_line``, ``.end_line``).
        threshold: Jaccard similarity threshold (0.0–1.0).
        ngram: Shingle size.
        num_hashes: Number of MinHash hashes.
        bands: Number of LSH bands.

    Returns:
        List of duplicate pair dicts with file/line metadata and similarity score.
    """
    minhash = MinHash(num_hashes)
    lsh = LshIndex(num_hashes, bands)
    sigs: dict[int, list[int]] = {}
    shingle_sets: dict[int, set[str]] = {}

    logger.info("Building MinHash + LSH for %d chunks", len(chunks))
    for i, ch in enumerate(chunks):
        shs = _shingles(ch.text, ngram)
        shingle_sets[i] = shs
        sig = minhash.signature(shs)
        sigs[i] = sig
        lsh.add(i, sig)

    seen: set[tuple[int, int]] = set()
    duplicates: list[dict[str, Any]] = []

    for i in range(len(chunks)):
        candidates = lsh.query(i, sigs[i])
        for j in candidates:
            if j <= i:
                continue
            pair = (i, j)
            if pair in seen:
                continue
            seen.add(pair)

            # Exact Jaccard on shingles
            inter = len(shingle_sets[i] & shingle_sets[j])
            union = len(shingle_sets[i] | shingle_sets[j])
            if union == 0:
                continue
            sim = inter / union
            if sim >= threshold:
                a, b = chunks[i], chunks[j]
                duplicates.append({
                    "file_a": a.file,
                    "start_line_a": a.start_line,
                    "end_line_a": a.end_line,
                    "file_b": b.file,
                    "start_line_b": b.start_line,
                    "end_line_b": b.end_line,
                    "similarity": round(sim, 4),
                })

    # Sort by descending similarity
    duplicates.sort(key=lambda x: x["similarity"], reverse=True)
    logger.info("Found %d duplicate pairs above threshold %f", len(duplicates), threshold)
    return duplicates
