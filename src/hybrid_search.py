"""Hybrid semantic + lexical search with Reciprocal Rank Fusion (RRF).

Merges results from a FAISS semantic index and a BM25 lexical index into a
single ranked list.  No external dependencies.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# RRF constant — standard value from Cormack et al.
_RRF_K = 60


def reciprocal_rank_fusion(
    semantic_results: list[dict[str, Any]],
    lexical_results: list[dict[str, Any]],
    semantic_weight: float = 1.0,
    lexical_weight: float = 1.0,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Merge two ranked lists via weighted Reciprocal Rank Fusion.

    Args:
        semantic_results: List of ``{doc_id, score, ...}`` from FAISS.
        lexical_results: List of ``{doc_id, score, ...}`` from BM25.
        semantic_weight: Weight for semantic ranks (default 1.0).
        lexical_weight: Weight for lexical ranks (default 1.0).
        top_k: Number of results to return.

    Returns:
        Merged list sorted by RRF score descending.
    """
    rrf: dict[int, float] = {}
    meta: dict[int, dict[str, Any]] = {}

    for rank, item in enumerate(semantic_results):
        doc_id = item["doc_id"]
        rrf[doc_id] = rrf.get(doc_id, 0.0) + semantic_weight / (_RRF_K + rank + 1)
        meta.setdefault(doc_id, {}).update(item)
        meta[doc_id]["semantic_rank"] = rank + 1

    for rank, item in enumerate(lexical_results):
        doc_id = item["doc_id"]
        rrf[doc_id] = rrf.get(doc_id, 0.0) + lexical_weight / (_RRF_K + rank + 1)
        meta.setdefault(doc_id, {}).update(item)
        meta[doc_id]["lexical_rank"] = rank + 1

    merged = []
    for doc_id, score in sorted(rrf.items(), key=lambda x: x[1], reverse=True):
        entry = dict(meta[doc_id])
        entry["rrf_score"] = round(score, 6)
        merged.append(entry)

    return merged[:top_k]
