"""End-to-end tests for the agent tool (CPU fallback mode)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent_tool import CodeEmbeddingTool


def test_embed_and_search(tmp_path: Path) -> None:
    # Create a tiny example codebase
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "math.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu")
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    assert "buffer_id" in result
    buf_id = result["buffer_id"]

    # Semantic search
    search = tool.semantic_search(buf_id, "addition function", top_k=2)
    assert search["status"] == "ok"
    assert len(search["matches"]) == 2
    # Best match should be line 1 (def add)
    assert search["matches"][0]["line"] == 1

    # Clustering
    clusters = tool.cluster_code(buf_id, threshold=0.5)
    assert clusters["status"] == "ok"
    assert len(clusters["clusters"]) > 0

    tool.close()


def test_size_guard_preflight(tmp_path: Path) -> None:
    # Use a tiny codebase with a very low threshold to trigger the guard
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", threshold_mb=0.0001)

    # Pre-flight check should warn without creating large files
    preflight = tool.check_codebase(code_dir, pattern="*.py")
    assert preflight["status"] == "exceeds_threshold"
    assert "estimated_mb" in preflight

    # embed_codebase should also bail early
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "warning"
    assert "too large" in result["message"].lower()
    tool.close()


def test_list_and_delete(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu")
    result = tool.embed_codebase(code_dir, pattern="*.py")
    buf_id = result["buffer_id"]

    listed = tool.list_buffers()
    assert any(b["buffer_id"] == buf_id for b in listed["buffers"])

    deleted = tool.delete_buffer(buf_id)
    assert deleted["status"] == "ok"

    listed2 = tool.list_buffers()
    assert not any(b["buffer_id"] == buf_id for b in listed2["buffers"])

    tool.close()
