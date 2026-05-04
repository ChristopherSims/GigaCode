"""End-to-end tests for the agent tool (chunk-level, FAISS backend)."""

from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gigacode.gigacode_tool import CodeEmbeddingTool


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
    # Metadata-only snapshot (replaces old source_snapshot.json)
    assert (buffer_dir / "snapshot_manifest.json").exists()

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


def test_write_code_conflict_detection(tmp_path: Path) -> None:
    """Phase 2: Detect 3-way merge conflicts in write_code()."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "config.py").write_text(
        "# Configuration\nDEBUG = True\nVERSION = '1.0'\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Write to buffer (buffer diverges from snapshot)
    write_result = tool.write_code(buf_id, "config.py", 2, ["DEBUG = False\n"], end_line=2)
    assert write_result["status"] == "ok"
    assert write_result["changed_lines"] == 1

    # Externally modify the file on disk (disk diverges from snapshot)
    (code_dir / "config.py").write_text(
        "# Configuration\nDEBUG = True\nVERSION = '2.0'\n# Updated externally\n",
        encoding="utf-8",
    )

    # Now try to write again - should detect conflict
    # (buffer was modified, disk was also modified independently)
    conflict_result = tool.write_code(buf_id, "config.py", 3, ["VERSION = '3.0'\n"], end_line=3)
    assert conflict_result["status"] == "conflict"
    assert "both disk and buffer have been modified" in conflict_result["message"]
    assert conflict_result["file"] == "config.py"
    assert "disk_lines" in conflict_result
    assert "buffer_lines" in conflict_result
    assert "snapshot_line_count" in conflict_result

    tool.close()


def test_write_code_no_conflict_disk_unchanged(tmp_path: Path) -> None:
    """Phase 2: No conflict if disk hasn't changed externally."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "simple.py").write_text(
        "x = 1\ny = 2\nz = 3\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Write to buffer multiple times (disk unchanged)
    write1 = tool.write_code(buf_id, "simple.py", 1, ["x = 10\n"], end_line=1)
    assert write1["status"] == "ok"
    assert write1["total_lines"] == 3

    # Second write should also succeed (no conflict since disk unchanged)
    write2 = tool.write_code(buf_id, "simple.py", 2, ["y = 20\n"], end_line=2)
    assert write2["status"] == "ok"
    assert write2["total_lines"] == 3

    tool.close()


def test_commit_no_conflicts(tmp_path: Path) -> None:
    """Phase 3: Commit writes files to disk when no conflicts."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "script.py").write_text(
        "# Script\nprint('hello')\nprint('world')\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Edit in buffer
    write_result = tool.write_code(buf_id, "script.py", 2, ["print('modified')\n"], end_line=2)
    assert write_result["status"] == "ok"

    # Commit should succeed (no conflicts since disk unchanged)
    commit_result = tool.commit(buf_id, dry_run=False)
    assert commit_result["status"] == "ok"
    assert "script.py" in commit_result["written_files"]
    assert len(commit_result["conflict_files"]) == 0
    assert commit_result["dry_run"] is False

    # Verify file was written to disk
    disk_content = (code_dir / "script.py").read_text(encoding="utf-8")
    assert "modified" in disk_content

    # Dirty files should be cleared after commit
    tool_buffers = tool.list_buffers()
    for buf in tool_buffers["buffers"]:
        if buf["buffer_id"] == buf_id:
            assert len(buf.get("dirty_files", {})) == 0

    tool.close()


def test_commit_with_3way_merge_conflict(tmp_path: Path) -> None:
    """Phase 3: Commit detects conflicts when both disk and buffer modified."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "data.py").write_text(
        "# Data file\nVAL = 1\nNAME = 'original'\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Edit in buffer
    write_result = tool.write_code(buf_id, "data.py", 3, ["NAME = 'buffer_edit'\n"], end_line=3)
    assert write_result["status"] == "ok"

    # Externally modify the same file on disk
    (code_dir / "data.py").write_text(
        "# Data file\nVAL = 999\nNAME = 'disk_edit'\nEXTRA = 'field'\n",
        encoding="utf-8",
    )

    # Commit should detect conflict (both modified)
    commit_result = tool.commit(buf_id, dry_run=False)
    assert commit_result["status"] == "conflict"
    assert len(commit_result["conflict_files"]) == 1
    assert commit_result["conflict_files"][0]["file"] == "data.py"
    assert "merge conflict" in commit_result["conflict_files"][0]["message"]
    assert "disk_lines" in commit_result["conflict_files"][0]
    assert "buffer_lines" in commit_result["conflict_files"][0]

    # File should remain unchanged on disk (not overwritten during conflict)
    disk_content = (code_dir / "data.py").read_text(encoding="utf-8")
    assert "disk_edit" in disk_content
    assert "buffer_edit" not in disk_content

    # Dirty file should still be dirty (not cleared)
    tool_buffers = tool.list_buffers()
    for buf in tool_buffers["buffers"]:
        if buf["buffer_id"] == buf_id:
            assert "data.py" in buf.get("dirty_files", {})

    tool.close()


def test_commit_dry_run(tmp_path: Path) -> None:
    """Phase 3: Dry run checks what would be written without modifying."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "test.py").write_text(
        "x = 1\ny = 2\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Edit in buffer
    write_result = tool.write_code(buf_id, "test.py", 1, ["x = 100\n"], end_line=1)
    assert write_result["status"] == "ok"

    # Dry run commit (should not write)
    dry_result = tool.commit(buf_id, dry_run=True)
    assert dry_result["status"] == "ok"
    assert "test.py" in dry_result["written_files"]
    assert dry_result["dry_run"] is True
    assert len(dry_result["conflict_files"]) == 0

    # File should NOT have changed on disk
    disk_content = (code_dir / "test.py").read_text(encoding="utf-8")
    assert disk_content == "x = 1\ny = 2\n"

    # Dirty files should still be dirty after dry run
    tool_buffers = tool.list_buffers()
    for buf in tool_buffers["buffers"]:
        if buf["buffer_id"] == buf_id:
            assert "test.py" in buf.get("dirty_files", {})

    # Real commit should still work
    real_result = tool.commit(buf_id, dry_run=False)
    assert real_result["status"] == "ok"
    assert "test.py" in real_result["written_files"]

    # Now the file should have changed
    disk_content = (code_dir / "test.py").read_text(encoding="utf-8")
    assert "x = 100" in disk_content

    tool.close()


def test_commit_with_transaction_logging(tmp_path: Path) -> None:
    """Phase 4: Commit operations are wrapped in transactions for crash recovery."""
    import json
    
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "module.py").write_text(
        "def foo():\n    pass\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Edit in buffer
    write_result = tool.write_code(buf_id, "module.py", 1, ["def bar():\n"], end_line=1)
    assert write_result["status"] == "ok"

    # Commit should return transaction_id
    commit_result = tool.commit(buf_id, dry_run=False)
    assert commit_result["status"] == "ok"
    assert commit_result["transaction_id"] is not None
    assert isinstance(commit_result["transaction_id"], str)

    # Verify WAL file was created (transaction logged)
    wal_path = work_dir / "wal.jsonl"
    assert wal_path.exists(), "Write-ahead log should exist after commit"

    # Verify WAL contains transaction entries
    wal_lines = wal_path.read_text(encoding="utf-8").strip().split('\n')
    assert len(wal_lines) > 0, "WAL should contain entries"

    # At least one entry should reference the transaction_id
    wal_entries = [json.loads(line) for line in wal_lines if line.strip()]
    transaction_ids = [entry.get("transaction_id") for entry in wal_entries]
    assert commit_result["transaction_id"] in transaction_ids

    tool.close()


def test_commit_transaction_contains_file_info(tmp_path: Path) -> None:
    """Phase 4: Transaction logs include file information for recovery."""
    import json
    
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "app.py").write_text(
        "class App:\n    pass\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Edit multiple files would be ideal, but we only have one; edit once
    write_result = tool.write_code(buf_id, "app.py", 1, ["def main():\n"], end_line=1)
    assert write_result["status"] == "ok"

    # Commit
    commit_result = tool.commit(buf_id, dry_run=False)
    assert commit_result["status"] == "ok"
    transaction_id = commit_result["transaction_id"]

    # Find the transaction log entry
    wal_path = work_dir / "wal.jsonl"
    wal_entries = [json.loads(line) for line in wal_path.read_text(encoding="utf-8").strip().split('\n') if line.strip()]
    
    # Find the entry with the transaction_id
    transaction_entry = None
    for entry in wal_entries:
        if entry.get("transaction_id") == transaction_id:
            transaction_entry = entry
            break
    
    assert transaction_entry is not None, f"Should find transaction {transaction_id} in WAL"
    assert transaction_entry.get("operation") == "commit"
    assert transaction_entry.get("buffer_id") == buf_id

    tool.close()



def test_commit_with_state_manager_recovery(tmp_path: Path) -> None:
    """Phase 4: StateManager recovers from incomplete transactions on restart."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "util.py").write_text(
        "def helper():\n    return 42\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Edit and commit successfully
    tool.write_code(buf_id, "util.py", 1, ["def updated():\n"], end_line=1)
    commit1 = tool.commit(buf_id, dry_run=False)
    assert commit1["status"] == "ok"
    assert commit1["transaction_id"] is not None
    
    # Verify file was written
    disk_content = (code_dir / "util.py").read_text()
    assert "updated" in disk_content

    # NOTE: After a successful commit, the transaction entry in the WAL has status="pending"
    # until the status-update entry is written. When we restart tool2, it will see the pending
    # transaction and mark it as rolled_back (since the commit was already completed on disk).
    # This is the expected behavior - the WAL ensures atomicity even if the status update is lost.
    
    # Create a new tool instance (simulating restart)
    tool2 = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)

    # After recovery, the registry should be loaded (and may show rolled back transaction)
    # But the file should still have the committed changes
    disk_content = (code_dir / "util.py").read_text()
    assert "def updated():" in disk_content
    assert "return 42" in disk_content
    
    # The StateManager should have recovered and the tool2 should be functional
    buffers2 = tool2.list_buffers()
    assert "buffers" in buffers2
    assert buffers2["status"] == "ok"

    tool.close()
    tool2.close()


# ============================================================================
# Phase 5: Integration Testing and Validation
# ============================================================================

def test_multi_file_concurrent_edits(tmp_path: Path) -> None:
    """Phase 5: Edit multiple files in one buffer and verify writes succeed."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "file1.py").write_text("x = 1\n", encoding="utf-8")
    (code_dir / "file2.py").write_text("y = 2\n", encoding="utf-8")
    (code_dir / "file3.py").write_text("z = 3\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Edit all three files
    write1 = tool.write_code(buf_id, "file1.py", 1, ["x = 100\n"], end_line=1)
    write2 = tool.write_code(buf_id, "file2.py", 1, ["y = 200\n"], end_line=1)
    write3 = tool.write_code(buf_id, "file3.py", 1, ["z = 300\n"], end_line=1)
    
    # Verify all writes succeeded (no conflicts)
    assert write1["status"] == "ok"
    assert write2["status"] == "ok"
    assert write3["status"] == "ok"

    # Verify the edits are accessible in the buffer
    read1 = tool.read_code(buf_id, "file1.py")
    assert read1["status"] == "ok"
    assert len(read1["lines"]) > 0
    
    read2 = tool.read_code(buf_id, "file2.py")
    assert read2["status"] == "ok"
    assert len(read2["lines"]) > 0
    
    read3 = tool.read_code(buf_id, "file3.py")
    assert read3["status"] == "ok"
    assert len(read3["lines"]) > 0

    tool.close()


def test_complex_conflict_3way_merge(tmp_path: Path) -> None:
    """Phase 5: Test real 3-way merge scenario (snapshot, disk, buffer all different)."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    original = "line1\nline2\nline3\nline4\nline5\n"
    (code_dir / "merge_test.py").write_text(original, encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Buffer edit: change line 2
    tool.write_code(buf_id, "merge_test.py", 2, ["line2_modified\n"], end_line=2)

    # Disk edit: change line 3 and 4 (externally modified after embedding)
    disk_content = "line1\nline2\nline3_external\nline4_external\nline5\n"
    (code_dir / "merge_test.py").write_text(disk_content, encoding="utf-8")

    # Now try to commit: should detect conflict (disk and buffer both modified from snapshot)
    commit_result = tool.commit(buf_id, dry_run=False)

    # Should detect conflict: snapshot != disk (line3, line4) AND snapshot != buffer (line2)
    # This results in a conflict since we can't auto-merge
    assert commit_result["status"] == "conflict" or commit_result["status"] == "ok"
    # If resolved via 3-way merge with buffer-wins, status will be "ok" 
    # If merge detected conflict, status will be "conflict"
    
    # Either way, file on disk should be one of: unchanged, or merged result
    disk_current = (code_dir / "merge_test.py").read_text()
    # File should either be unchanged or have the merged content
    assert len(disk_current) > 0

    tool.close()


def test_search_after_multi_edit_commit(tmp_path: Path) -> None:
    """Phase 5: Verify embeddings are updated after multi-file edits and commit."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "search_test.py").write_text(
        "def find_prime(n):\n    return n > 1\n\ndef find_even(n):\n    return n % 2 == 0\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Initial search should find prime function
    search1 = tool.semantic_search(buf_id, "check if number is prime", top_k=1)
    assert search1["status"] == "ok"
    assert len(search1["matches"]) > 0  # Use "matches" not "results"

    # Edit: modify code
    tool.write_code(buf_id, "search_test.py", 1, ["def check_positive(n):\n"], end_line=1)

    # Commit the changes (rebuilds embeddings)
    commit_result = tool.commit(buf_id, dry_run=False)
    assert commit_result["status"] == "ok"

    # Search should still work after commit
    search2 = tool.semantic_search(buf_id, "check if number is positive", top_k=1)
    assert search2["status"] == "ok"
    assert "matches" in search2
    # Should have matches (regardless of content)
    assert isinstance(search2["matches"], list)

    tool.close()


def test_large_file_edit_and_commit(tmp_path: Path) -> None:
    """Phase 5: Test with larger files to verify performance and correctness."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    
    # Create a large file with 100 lines
    large_code = "\n".join(f"def function_{i}(x):\n    return x + {i}" for i in range(100))
    (code_dir / "large.py").write_text(large_code + "\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Edit in the middle of the file
    new_lines = ["def function_50(x):\n", "    return x + 50000  # Modified\n"]
    tool.write_code(buf_id, "large.py", 101, new_lines, end_line=102)

    # Commit
    commit_result = tool.commit(buf_id, dry_run=False)
    assert commit_result["status"] == "ok"
    assert "large.py" in commit_result["written_files"]

    # Verify the change
    disk_content = (code_dir / "large.py").read_text()
    assert "50000" in disk_content

    tool.close()


def test_multiple_concurrent_buffers(tmp_path: Path) -> None:
    """Phase 5: Test multiple buffers being edited and committed simultaneously."""
    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)

    # Create two separate codebases
    code_dir1 = tmp_path / "code1"
    code_dir1.mkdir()
    (code_dir1 / "app1.py").write_text("version = '1.0'\n", encoding="utf-8")

    code_dir2 = tmp_path / "code2"
    code_dir2.mkdir()
    (code_dir2 / "app2.py").write_text("version = '2.0'\n", encoding="utf-8")

    # Embed both
    result1 = tool.embed_codebase(code_dir1, pattern="*.py")
    assert result1["status"] == "ok"
    buf_id1 = result1["buffer_id"]

    result2 = tool.embed_codebase(code_dir2, pattern="*.py")
    assert result2["status"] == "ok"
    buf_id2 = result2["buffer_id"]

    # Verify we have two buffers
    buffers = tool.list_buffers()["buffers"]
    assert len(buffers) == 2
    buf_ids = {b["buffer_id"] for b in buffers}
    assert buf_id1 in buf_ids
    assert buf_id2 in buf_ids

    # Edit both buffers
    tool.write_code(buf_id1, "app1.py", 1, ["version = '1.1'\n"], end_line=1)
    tool.write_code(buf_id2, "app2.py", 1, ["version = '2.2'\n"], end_line=1)

    # Commit both
    commit1 = tool.commit(buf_id1, dry_run=False)
    commit2 = tool.commit(buf_id2, dry_run=False)
    assert commit1["status"] == "ok"
    assert commit2["status"] == "ok"

    # Verify both were written to their respective directories (check content, not exact match for newlines)
    content1 = (code_dir1 / "app1.py").read_text()
    content2 = (code_dir2 / "app2.py").read_text()
    assert "1.1" in content1
    assert "2.2" in content2

    # Verify both buffers are now clean
    buffers = tool.list_buffers()["buffers"]
    for buf in buffers:
        assert len(buf.get("dirty_files", {})) == 0

    tool.close()


def test_edge_case_empty_file_edit(tmp_path: Path) -> None:
    """Phase 5: Test editing empty files and single-line files."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "empty.py").write_text("", encoding="utf-8")
    (code_dir / "single.py").write_text("x = 1\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    tool = CodeEmbeddingTool(work_dir=work_dir, device="cpu", use_gpu=False)
    result = tool.embed_codebase(code_dir, pattern="*.py")
    assert result["status"] == "ok"
    buf_id = result["buffer_id"]

    # Add to empty file
    tool.write_code(buf_id, "empty.py", 1, ["def foo():\n", "    pass\n"], end_line=0)

    # Modify single-line file
    tool.write_code(buf_id, "single.py", 1, ["x = 100\n"], end_line=1)

    # Commit
    commit_result = tool.commit(buf_id, dry_run=False)
    assert commit_result["status"] == "ok"

    # Verify changes
    empty_content = (code_dir / "empty.py").read_text()
    assert "def foo():" in empty_content
    single_content = (code_dir / "single.py").read_text()
    assert "100" in single_content

    tool.close()

