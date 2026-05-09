"""Tests for BufferManager class.

Tests buffer registry, lifecycle, and file I/O operations.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gigacode.buffer_manager import BufferManager
from gigacode.state_manager import StateManager


@pytest.fixture
def temp_work_dir():
    """Create temporary working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def state_manager(temp_work_dir):
    """Create StateManager instance."""
    return StateManager(temp_work_dir)


@pytest.fixture
def buffer_manager(temp_work_dir, state_manager):
    """Create BufferManager instance with AuditLogger."""
    from gigacode.audit_logger import AuditLogger
    
    audit_logger = AuditLogger(temp_work_dir / "audit.jsonl")
    return BufferManager(
        work_dir=temp_work_dir,
        state_manager=state_manager,
        embedding_dim=384,
        threshold_mb=500.0,
        audit_logger=audit_logger,
        user_id="test_user",
    )


class TestBufferManagerInit:
    """Test BufferManager initialization."""
    
    def test_init_creates_work_dir(self, temp_work_dir, state_manager):
        """Test that __init__ creates work_dir if needed."""
        new_dir = temp_work_dir / "subdir"
        assert not new_dir.exists()
        
        BufferManager(new_dir, state_manager, 384)
        
        assert new_dir.exists()
    
    def test_init_loads_existing_registry(self, temp_work_dir, state_manager):
        """Test that __init__ loads existing registry.json."""
        registry_path = temp_work_dir / "registry.json"
        existing_registry = {"buffer1": {"root": "/test/path"}}
        registry_path.write_text(json.dumps(existing_registry))
        
        bm = BufferManager(temp_work_dir, state_manager, 384)
        
        assert bm._registry == existing_registry
    
    def test_init_empty_registry_if_not_exists(self, temp_work_dir, state_manager):
        """Test that __init__ creates empty registry if file doesn't exist."""
        bm = BufferManager(temp_work_dir, state_manager, 384)
        
        assert bm._registry == {}


class TestAuditLogging:
    """Test audit logging functionality."""
    
    def test_audit_log_creates_entries(self, buffer_manager):
        """Test that _audit_log creates audit entries."""
        buffer_manager._audit_log(
            operation="test_op",
            buffer_id="test_id",
            status="ok",
            details={"key": "value"}
        )
        
        # Check that audit logger's log file was created
        audit_log_path = buffer_manager._audit_logger.log_file
        assert audit_log_path.exists()
        
        lines = audit_log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        
        entry = json.loads(lines[0])
        assert entry["operation"] == "test_op"
        assert entry["buffer_id"] == "test_id"
        assert entry["status"] == "success"
        assert entry["details"]["key"] == "value"
    
    def test_audit_log_handles_errors_gracefully(self, buffer_manager):
        """Test that audit logging doesn't crash on write errors."""
        # Mock the file write to raise an error
        with patch.object(Path, "open", side_effect=IOError("Disk full")):
            # Should not raise
            buffer_manager._audit_log("test_op", "test_id")


class TestRegistryManagement:
    """Test registry operations."""
    
    def test_get_buffer_info_returns_none_for_unknown(self, buffer_manager):
        """Test that _get_buffer_info returns None for unknown buffer."""
        result = buffer_manager._get_buffer_info("nonexistent")
        assert result is None
    
    def test_get_buffer_info_returns_info(self, buffer_manager):
        """Test that _get_buffer_info returns existing buffer info."""
        buffer_manager._registry["test_id"] = {
            "root": "/test/path",
            "buffer_dir": "/test/buffer"
        }
        
        result = buffer_manager._get_buffer_info("test_id")
        
        assert result["root"] == "/test/path"
        assert result["buffer_dir"] == "/test/buffer"
    
    def test_save_registry_persists_to_disk(self, buffer_manager):
        """Test that _save_registry writes registry.json."""
        buffer_manager._registry["test_id"] = {
            "root": "/test/path"
        }
        
        buffer_manager._save_registry()
        
        registry_path = buffer_manager._registry_path
        assert registry_path.exists()
        
        persisted = json.loads(registry_path.read_text())
        assert "test_id" in persisted


class TestBufferLifecycle:
    """Test buffer creation and deletion."""
    
    def test_list_buffers_empty(self, buffer_manager):
        """Test list_buffers when no buffers exist."""
        result = buffer_manager.list_buffers()
        
        assert result["status"] == "ok"
        assert result["buffers"] == []
    
    def test_list_buffers_shows_all(self, buffer_manager):
        """Test list_buffers returns all registered buffers."""
        buffer_manager._registry["buf1"] = {"root": "/path1"}
        buffer_manager._registry["buf2"] = {"root": "/path2"}
        
        result = buffer_manager.list_buffers()
        
        assert len(result["buffers"]) == 2
        assert any(b["buffer_id"] == "buf1" for b in result["buffers"])
        assert any(b["buffer_id"] == "buf2" for b in result["buffers"])
    
    def test_delete_buffer_removes_from_registry(self, buffer_manager):
        """Test delete_buffer removes from registry."""
        buffer_manager._registry["buf1"] = {
            "root": "/path1",
            "buffer_dir": "/tmp/nonexistent"
        }
        buffer_manager._save_registry()
        
        result = buffer_manager.delete_buffer("buf1")
        
        assert result["status"] == "ok"
        assert "buf1" not in buffer_manager._registry
    
    def test_delete_buffer_unknown_returns_error(self, buffer_manager):
        """Test delete_buffer with unknown ID returns error."""
        result = buffer_manager.delete_buffer("nonexistent")
        
        assert result["status"] == "error"
        assert "Unknown buffer_id" in result["message"]
    
    def test_delete_buffer_logs_audit(self, buffer_manager):
        """Test that delete_buffer logs to audit trail."""
        buffer_manager._registry["buf1"] = {
            "root": "/path",
            "buffer_dir": "/tmp/nonexistent"
        }
        
        buffer_manager.delete_buffer("buf1")
        
        # Check that audit logger wrote an entry
        audit_path = buffer_manager._audit_logger.log_file
        assert audit_path.exists()
        
        lines = audit_path.read_text().strip().split("\n")
        entry = json.loads(lines[-1])
        assert entry["operation"] == "delete_buffer"
        assert entry["buffer_id"] == "buf1"


class TestSourceSnapshot:
    """Test source snapshot persistence."""
    
    def test_save_and_load_source_snapshot(self, buffer_manager, temp_work_dir):
        """Test saving and loading source snapshots."""
        buffer_id = "test_buf"
        buffer_dir = temp_work_dir / f"{buffer_id}.gcbuff"
        buffer_dir.mkdir()
        
        buffer_manager._registry[buffer_id] = {
            "root": "/test",
            "buffer_dir": str(buffer_dir),
        }
        
        snapshot = {"file1.py": ["line1", "line2"], "file2.py": ["line3"]}
        
        buffer_manager._save_source_snapshot(buffer_id, snapshot)
        
        loaded = buffer_manager._load_source_snapshot(buffer_id)
        
        assert loaded == snapshot
    
    def test_load_source_snapshot_returns_none_for_unknown(self, buffer_manager):
        """Test _load_source_snapshot returns None for unknown buffer."""
        result = buffer_manager._load_source_snapshot("nonexistent")
        assert result is None


class TestCheckCodebase:
    """Test codebase size validation."""
    
    def test_check_codebase_nonexistent_returns_ok(self, buffer_manager):
        """Test check_codebase returns ok for empty dir."""
        result = buffer_manager.check_codebase("/nonexistent/path", "*.py")
        
        assert result["status"] == "ok"
        assert result["file_count"] == 0
        assert result["estimated_mb"] == 0.0


class TestDirHash:
    """Tests for _compute_dir_hash static method."""

    def test_compute_dir_hash_file(self, temp_work_dir):
        """Hashing a single file should be stable."""
        f = temp_work_dir / "a.py"
        f.write_text("hello")

        h1 = BufferManager._compute_dir_hash(f)
        h2 = BufferManager._compute_dir_hash(f)

        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex digest length

    def test_compute_dir_hash_directory(self, temp_work_dir):
        """Hashing a directory should be stable and change when files change."""
        (temp_work_dir / "sub").mkdir()
        (temp_work_dir / "sub" / "a.py").write_text("x")
        (temp_work_dir / "sub" / "b.py").write_text("y")

        h1 = BufferManager._compute_dir_hash(temp_work_dir / "sub")
        h2 = BufferManager._compute_dir_hash(temp_work_dir / "sub")

        assert h1 == h2

        # Modify a file -> hash should change
        (temp_work_dir / "sub" / "a.py").write_text("xx")
        h3 = BufferManager._compute_dir_hash(temp_work_dir / "sub")
        assert h3 != h1

    def test_compute_dir_hash_respects_pattern(self, temp_work_dir):
        """Only files matching pattern should contribute to hash."""
        (temp_work_dir / "foo.py").write_text("x")
        (temp_work_dir / "bar.txt").write_text("y")

        h_py = BufferManager._compute_dir_hash(temp_work_dir, "*.py")
        h_all = BufferManager._compute_dir_hash(temp_work_dir, "*")

        assert h_py != h_all


class TestCheckExistingBuffer:
    """Tests for check_existing_buffer method."""

    def test_not_found_when_registry_empty(self, buffer_manager, temp_work_dir):
        """Empty registry should always return not_found."""
        f = temp_work_dir / "a.py"
        f.write_text("x")
        result = buffer_manager.check_existing_buffer(f)
        assert result == {"status": "not_found"}

    def test_found_when_source_hash_matches(self, buffer_manager, temp_work_dir):
        """Matching source_hash in registry should resume."""
        f = temp_work_dir / "a.py"
        f.write_text("x")
        source_hash = BufferManager._compute_dir_hash(f)
        buffer_manager._registry["buf-1"] = {
            "root": str(f),
            "buffer_dir": str(temp_work_dir / "buf-1.gcbuff"),
            "chunk_count": 42,
            "source_hash": source_hash,
        }
        result = buffer_manager.check_existing_buffer(f)
        assert result["status"] == "resumed"
        assert result["buffer_id"] == "buf-1"
        assert result["num_chunks"] == 42

    def test_not_found_when_source_hash_differs(self, buffer_manager, temp_work_dir):
        """Mismatching source_hash should return not_found."""
        f = temp_work_dir / "a.py"
        f.write_text("x")
        buffer_manager._registry["buf-1"] = {
            "root": str(f),
            "buffer_dir": str(temp_work_dir / "buf-1.gcbuff"),
            "chunk_count": 42,
            "source_hash": "different-hash",
        }
        result = buffer_manager.check_existing_buffer(f)
        assert result == {"status": "not_found"}


class TestSessionPersistence:
    """Tests for save_session, load_session, and list_sessions."""

    def test_save_session_creates_file(self, buffer_manager, temp_work_dir):
        """save_session should write a JSON file under .sessions/."""
        result = buffer_manager.save_session("my-session", ["buf-a", "buf-b"])
        assert result["status"] == "ok"
        assert result["alias"] == "my-session"

        session_path = temp_work_dir / ".sessions" / "my-session.json"
        assert session_path.exists()
        payload = json.loads(session_path.read_text())
        assert payload["alias"] == "my-session"
        assert payload["buffer_ids"] == ["buf-a", "buf-b"]
        assert "saved_at" in payload
        assert payload["bookmarks"] == {}

    def test_load_session_ok(self, buffer_manager, temp_work_dir):
        """load_session should restore a previously saved session."""
        buffer_manager.save_session("sess", ["b1", "b2"])
        result = buffer_manager.load_session("sess")
        assert result["status"] == "ok"
        assert result["buffer_ids"] == ["b1", "b2"]
        assert result["bookmarks"] == {}

    def test_load_session_missing(self, buffer_manager):
        """load_session for nonexistent alias should return error."""
        result = buffer_manager.load_session("no-such-session")
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    def test_load_session_reports_missing_buffer_ids(self, buffer_manager, temp_work_dir):
        """load_session should note buffer_ids absent from registry."""
        buffer_manager.save_session("sess", ["missing-buf"])
        result = buffer_manager.load_session("sess")
        assert result["status"] == "ok"
        assert result["buffer_ids"] == ["missing-buf"]
        assert result["missing_buffer_ids"] == ["missing-buf"]

    def test_list_sessions_empty(self, buffer_manager):
        """list_sessions should return empty list when no sessions saved."""
        result = buffer_manager.list_sessions()
        assert result == {"sessions": []}

    def test_list_sessions_returns_metadata(self, buffer_manager):
        """list_sessions should reflect saved sessions with metadata."""
        buffer_manager.save_session("alpha", ["b1"])
        buffer_manager.save_session("beta", ["b2", "b3"])

        result = buffer_manager.list_sessions()
        assert len(result["sessions"]) == 2

        aliases = {s["alias"] for s in result["sessions"]}
        assert aliases == {"alpha", "beta"}

        for s in result["sessions"]:
            if s["alias"] == "alpha":
                assert s["buffer_count"] == 1
            else:
                assert s["buffer_count"] == 2
            assert "saved_at" in s
    
    def test_check_codebase_small_dir_returns_ok(self, temp_work_dir):
        """Test check_codebase returns ok for small codebase."""
        # Create small files
        (temp_work_dir / "file1.py").write_text("print('hello')")
        (temp_work_dir / "file2.py").write_text("print('world')")
        
        bm = BufferManager(temp_work_dir / "work", StateManager(temp_work_dir / "work"), 384)
        result = bm.check_codebase(temp_work_dir, "*.py")
        
        assert result["status"] == "ok"
        assert result["file_count"] == 2
    
    def test_check_codebase_exceeds_threshold(self, temp_work_dir):
        """Test check_codebase detects oversized codebase."""
        bm = BufferManager(
            temp_work_dir / "work",
            StateManager(temp_work_dir / "work"),
            384,
            threshold_mb=0.0001  # Very small threshold
        )
        
        (temp_work_dir / "large.py").write_text("x" * 10000)
        
        result = bm.check_codebase(temp_work_dir, "*.py")
        
        assert result["status"] == "exceeds_threshold"


class TestReadCode:
    """Test reading code from buffer."""
    
    def test_read_code_unknown_buffer(self, buffer_manager):
        """Test read_code returns error for unknown buffer."""
        result = buffer_manager.read_code("nonexistent")
        
        assert result["status"] == "error"
        assert "Unknown buffer_id" in result["message"]
    
    def test_read_code_missing_snapshot(self, buffer_manager):
        """Test read_code returns error if snapshot missing."""
        buffer_manager._registry["test_buf"] = {
            "root": "/test",
            "buffer_dir": "/nonexistent"
        }
        
        result = buffer_manager.read_code("test_buf")
        
        assert result["status"] == "error"
        assert "Snapshot" in result["message"]


class TestWriteCode:
    """Test writing code to buffer."""
    
    def test_write_code_unknown_buffer(self, buffer_manager):
        """Test write_code returns error for unknown buffer."""
        result = buffer_manager.write_code(
            "nonexistent",
            "file.py",
            1,
            ["new_line"]
        )
        
        assert result["status"] == "error"
        assert "Unknown buffer_id" in result["message"]
    
    def test_write_code_missing_snapshot(self, buffer_manager):
        """Test write_code returns error if snapshot missing."""
        buffer_manager._registry["test_buf"] = {
            "root": "/test",
            "buffer_dir": "/nonexistent"
        }
        
        result = buffer_manager.write_code(
            "test_buf",
            "file.py",
            1,
            ["new_line"]
        )
        
        assert result["status"] == "error"
        assert "snapshot" in result["message"].lower()


class TestCommit:
    """Test commit operation."""
    
    def test_commit_unknown_buffer(self, buffer_manager):
        """Test commit returns error for unknown buffer."""
        result = buffer_manager.commit("nonexistent", None)
        
        assert result["status"] == "error"
        assert "Unknown buffer_id" in result["message"]
    
    def test_commit_no_dirty_files(self, buffer_manager, temp_work_dir):
        """Test commit with no dirty files returns ok."""
        import json
        
        buffer_dir = temp_work_dir / "test_buffer"
        buffer_dir.mkdir()
        
        # Create source snapshot file
        snapshot = {"file.py": ["line1", "line2"]}
        snapshot_path = buffer_dir / "source_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot))
        
        buffer_manager._registry["test_buf"] = {
            "root": str(temp_work_dir),
            "buffer_dir": str(buffer_dir),
            "dirty_files": {},
            "file_hashes": {}
        }
        
        result = buffer_manager.commit("test_buf", None, dry_run=True)
        
        assert result["status"] == "ok"
        assert result["written_files"] == []
        assert result["conflict_files"] == []
