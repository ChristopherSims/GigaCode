"""Tests for the agent read/write/commit/discard workflow (chunk-level)."""

from __future__ import annotations

from pathlib import Path

import pytest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.gigacode_tool import CodeEmbeddingTool


def test_read_code_single_file(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "math.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    read = tool.read_code(buf_id, file="math.py")
    assert read["status"] == "ok"
    assert read["file"] == "math.py"
    assert read["lines"] == [
        "def add(a, b):",
        "    return a + b",
        "",
        "def sub(a, b):",
        "    return a - b",
    ]

    read2 = tool.read_code(buf_id, file="math.py", start_line=1, end_line=3)
    assert read2["lines"] == ["def add(a, b):", "    return a + b"]

    tool.close()


def test_read_code_all_files(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
    (code_dir / "b.py").write_text("y = 2\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    buf_id = result["buffer_id"]

    read = tool.read_code(buf_id)
    assert read["status"] == "ok"
    assert "files" in read
    assert set(read["files"].keys()) == {"a.py", "b.py"}
    assert read["files"]["a.py"] == ["x = 1"]
    assert read["files"]["b.py"] == ["y = 2"]

    tool.close()


def test_write_code_and_diff(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "math.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    buf_id = result["buffer_id"]

    diff_before = tool.diff(buf_id)
    assert diff_before["changed_files"] == []

    write = tool.write_code(
        buf_id,
        file="math.py",
        start_line=1,
        new_lines=["def add(a: int, b: int) -> int:", "    return a + b"],
    )
    assert write["status"] == "ok"
    assert write["changed_lines"] == 2

    diff_after = tool.diff(buf_id)
    assert len(diff_after["changed_files"]) == 1
    assert diff_after["changed_files"][0]["file"] == "math.py"
    assert diff_after["changed_files"][0]["dirty"] is True

    read = tool.read_code(buf_id, file="math.py")
    assert read["lines"] == ["def add(a: int, b: int) -> int:", "    return a + b"]

    tool.close()


def test_commit_overwrites_original(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    original = code_dir / "math.py"
    original.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    buf_id = result["buffer_id"]

    tool.write_code(
        buf_id,
        file="math.py",
        start_line=1,
        new_lines=["def add(a: int, b: int) -> int:", "    return a + b"],
    )

    dry = tool.commit(buf_id, dry_run=True)
    assert dry["status"] == "ok"
    assert dry["dry_run"] is True
    assert "math.py" in dry["written_files"]
    assert original.read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"

    commit = tool.commit(buf_id, dry_run=False)
    assert commit["status"] == "ok"
    assert commit["dry_run"] is False
    assert "math.py" in commit["written_files"]

    new_text = original.read_text(encoding="utf-8")
    assert "def add(a: int, b: int) -> int:" in new_text

    diff_after = tool.diff(buf_id)
    assert diff_after["changed_files"] == []

    tool.close()


def test_discard_reverts_changes(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    original = code_dir / "math.py"
    original.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    buf_id = result["buffer_id"]

    tool.write_code(
        buf_id,
        file="math.py",
        start_line=1,
        new_lines=["def add(a: int, b: int) -> int:"],
    )

    discard = tool.discard(buf_id, file="math.py")
    assert discard["status"] == "ok"
    assert "math.py" in discard["reverted_files"]

    read = tool.read_code(buf_id, file="math.py")
    assert read["lines"] == ["def add(a, b):", "    return a + b"]

    diff = tool.diff(buf_id)
    assert diff["changed_files"] == []

    tool.close()


def test_commit_aborts_on_disk_change(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    original = code_dir / "math.py"
    original.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    buf_id = result["buffer_id"]

    tool.write_code(
        buf_id,
        file="math.py",
        start_line=1,
        new_lines=["def add(a: int, b: int) -> int:"],
    )

    original.write_text("# modified externally\n", encoding="utf-8")

    commit = tool.commit(buf_id, dry_run=False)
    assert commit["status"] == "error"
    assert "hash mismatch" in commit["message"].lower()

    tool.close()


def test_full_round_trip(tmp_path: Path) -> None:
    """Embed -> read -> write -> commit -> verify on disk."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    original = code_dir / "demo.py"
    original.write_text(
        "def greet(name):\n    print('Hello', name)\n\ndef bye():\n    print('Goodbye')\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    embed = tool.embed_codebase(code_dir, pattern="*.py")
    assert embed["status"] == "ok"
    buf_id = embed["buffer_id"]

    read = tool.read_code(buf_id, file="demo.py")
    assert read["lines"][0] == "def greet(name):"

    tool.write_code(
        buf_id,
        file="demo.py",
        start_line=1,
        end_line=2,
        new_lines=["def greet(name: str) -> None:", "    print('Hello', name)"],
    )

    commit = tool.commit(buf_id)
    assert commit["status"] == "ok"

    text = original.read_text(encoding="utf-8")
    assert "def greet(name: str) -> None:" in text
    assert "def bye():" in text

    search = tool.semantic_search(buf_id, "greeting function", top_k=2)
    assert search["status"] == "ok"
    assert len(search["matches"]) >= 1

    tool.close()
