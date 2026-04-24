"""Tests for src.hybrid_search."""

from src.hybrid_search import reciprocal_rank_fusion


def test_rrf_merge():
    semantic = [{"doc_id": 0}, {"doc_id": 1}, {"doc_id": 2}]
    lexical = [{"doc_id": 1}, {"doc_id": 3}, {"doc_id": 0}]
    merged = reciprocal_rank_fusion(semantic, lexical, top_k=10)
    doc_ids = [m["doc_id"] for m in merged]
    # doc 1 appears in both lists, doc 0 also appears in both
    assert 1 in doc_ids
    assert 0 in doc_ids
