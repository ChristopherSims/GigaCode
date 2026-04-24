"""Tests for src.context_packer."""

from src.context_packer import pack_context
from src.chunker import CodeChunk


def test_pack_within_budget():
    chunks = [
        CodeChunk(id=0, file="a.py", start_line=1, end_line=2, type="function", name="f1", text="def f1(): pass"),
        CodeChunk(id=1, file="a.py", start_line=3, end_line=4, type="function", name="f2", text="def f2(): pass"),
    ]
    scores = [1.0, 0.5]
    result = pack_context(chunks, scores, max_tokens=10)
    assert result["status"] == "ok"
    assert result["total_tokens"] <= 10
    assert result["count"] >= 1
