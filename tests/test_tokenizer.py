"""Tests for src.tokenizer."""

from __future__ import annotations

from pathlib import Path

from src.tokenizer import tokenize_file


def test_tokenize_python_file(tmp_path: Path) -> None:
    code = """def hello():
    x = 1 + 2
    return x
"""
    f = tmp_path / "sample.py"
    f.write_text(code, encoding="utf-8")
    result = tokenize_file(f, language_hint="python")
    assert len(result) == 3
    assert result[0]["line_num"] == 1
    assert "def" in result[0]["tokens"]
    assert result[1]["line_num"] == 2

def test_tokenize_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "empty.py"
    f.write_text("", encoding="utf-8")
    result = tokenize_file(f)
    assert result == []
