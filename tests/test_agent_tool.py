"""End-to-end tests for the agent tool (chunk-level, FAISS backend)."""

from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.gigacode_tool import CodeEmbeddingTool


def test_embed_and_search(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "math.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    assert "buffer_id" in result
    buf_id = result["buffer_id"]

    # Semantic search returns chunks now (start_line / end_line)
    search = tool.semantic_search(buf_id, "addition function", top_k=2)
    assert search["status"] == "ok"
    assert 1 <= len(search["matches"]) <= 2
    assert "start_line" in search["matches"][0]
    assert "end_line" in search["matches"][0]
    assert "type" in search["matches"][0]

    # Clustering
    clusters = tool.cluster_code(buf_id, threshold=0.5)
    assert clusters["status"] == "ok"
    assert len(clusters["clusters"]) >= 0

    tool.close()


def test_size_guard_preflight(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", threshold_mb=0.0001, use_gpu=False)

    preflight = tool.check_codebase(code_dir, pattern="*.py")
    assert preflight["status"] == "exceeds_threshold"
    assert "estimated_mb" in preflight

    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "warning"
    assert "too large" in result["message"].lower()
    tool.close()


def test_list_and_delete(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    buf_id = result["buffer_id"]

    listed = tool.list_buffers()
    assert any(b["buffer_id"] == buf_id for b in listed["buffers"])

    deleted = tool.delete_buffer(buf_id)
    assert deleted["status"] == "ok"

    listed2 = tool.list_buffers()
    assert not any(b["buffer_id"] == buf_id for b in listed2["buffers"])

    tool.close()


def test_reload_codebase_with_matching_hash(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    reload_result = tool.reload_codebase(buf_id)
    assert reload_result["status"] == "ok"
    assert "reloaded without re-embedding" in reload_result["message"]
    assert reload_result["buffer_id"] == buf_id

    tool.close()


def test_reload_codebase_with_changed_hash(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    (code_dir / "a.py").write_text("x = 2\ny = 3\n", encoding="utf-8")

    reload_result = tool.reload_codebase(buf_id)
    assert reload_result["status"] == "ok"
    assert reload_result.get("message", "").lower() != "hashes match; reloaded without re-embedding."

    tool.close()


def test_gcbuff_directory_naming(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"

    buf_id = result["buffer_id"]
    buffer_dir = work_dir / f"{buf_id}.gcbuff"
    assert buffer_dir.exists()
    assert (buffer_dir / "embeddings.npy").exists()
    assert (buffer_dir / "chunks.json").exists()
    assert (buffer_dir / "source_snapshot.json").exists()

    tool.close()


def test_tool_schemas_exposed() -> None:
    schemas = CodeEmbeddingTool.get_tool_schemas()
    assert len(schemas) >= 13
    names = {s["name"] for s in schemas}
    assert "embed_codebase" in names
    assert "semantic_search" in names
    assert "search_for" in names
    assert "search_symbols" in names
    assert "cluster_code" in names
    assert "read_code" in names
    assert "write_code" in names
    assert "commit" in names
    assert "diff" in names
    assert "discard" in names


def test_search_for_literal(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "math.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    (code_dir / "utils.py").write_text(
        "def helper():\n    return 42\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Case-insensitive literal search
    search = tool.search_for(buf_id, "return")
    assert search["status"] == "ok"
    assert search["total"] == 3
    files = {m["file"] for m in search["matches"]}
    assert files == {"math.py", "utils.py"}

    # Case-sensitive search (should miss lowercase)
    search_cs = tool.search_for(buf_id, "Return", case_sensitive=True)
    assert search_cs["total"] == 0

    # Specific function name
    search_name = tool.search_for(buf_id, "subtract")
    assert search_name["total"] == 1
    assert search_name["matches"][0]["line"] == 4

    tool.close()


def test_search_symbols(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "api.py").write_text(
        "def fetch_data(url):\n    pass\n\ndef post_data(url):\n    pass\n\nclass DataClient:\n    def get(self):\n        pass\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Name-based search
    sym = tool.search_symbols(buf_id, "fetch_data")
    assert sym["status"] == "ok"
    assert any(m["name"] == "fetch_data" and m["match_type"] == "name" for m in sym["matches"])

    # Semantic search for related concept
    sym2 = tool.search_symbols(buf_id, "get request", top_k=5)
    assert sym2["status"] == "ok"
    names = {m["name"] for m in sym2["matches"]}
    assert "get" in names or "fetch_data" in names or "DataClient" in names

    tool.close()
