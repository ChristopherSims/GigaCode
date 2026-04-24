"""Tests for src.duplicate_detector."""

from src.duplicate_detector import find_duplicates
from src.chunker import CodeChunk


def test_exact_duplicate():
    chunks = [
        CodeChunk(id=0, file="a.py", start_line=1, end_line=3, type="function", name="foo", text="def foo():\n    pass"),
        CodeChunk(id=1, file="b.py", start_line=1, end_line=3, type="function", name="bar", text="def foo():\n    pass"),
        CodeChunk(id=2, file="c.py", start_line=1, end_line=3, type="function", name="baz", text="def baz():\n    return 1"),
    ]
    dups = find_duplicates(chunks, threshold=0.9)
    assert len(dups) >= 1
    assert dups[0]["similarity"] >= 0.9
