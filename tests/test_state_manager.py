"""Tests for state management with file locking and transactions."""

import json
import tempfile
from pathlib import Path
from threading import Thread
import time

import pytest

from gigacode.state_manager import StateManager, FileLocker, TransactionLog


class TestFileLocker:
    """Test file locking mechanism."""
    
    def test_lock_context_manager(self):
        """Test lock as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / "test.json"
            lock_file.write_text("test")
            
            with FileLocker(lock_file) as locker:
                # Lock held within context
                pass
            # Lock released after context


class TestTransactionLog:
    """Test transaction log entries."""
    
    def test_transaction_log_serialization(self):
        """Test TransactionLog to/from dict."""
        tx = TransactionLog(
            transaction_id="tx-123",
            timestamp="2026-05-04T10:00:00Z",
            operation="write_code",
            buffer_id="buf-1",
            file_path="module.py",
            start_line=5,
            end_line=10,
            new_lines=["new line 1", "new line 2"],
            status="pending"
        )
        
        data = tx.to_dict()
        assert data["transaction_id"] == "tx-123"
        assert data["operation"] == "write_code"
        
        tx2 = TransactionLog.from_dict(data)
        assert tx2.transaction_id == tx.transaction_id
        assert tx2.new_lines == tx.new_lines


class TestStateManager:
    """Test state management with ACID guarantees."""
    
    def test_initialization(self):
        """Test StateManager initializes with empty registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            assert manager.registry == {}
            assert manager.registry_path.exists() is False
    
    def test_save_and_load_registry(self):
        """Test atomic registry save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            
            # Set buffer info
            manager.set_buffer_info("buf-1", {
                "path": "/code",
                "chunk_count": 100,
                "dirty_files": {}
            })
            
            # Load in new manager instance
            manager2 = StateManager(Path(tmpdir))
            assert "buf-1" in manager2.registry
            assert manager2.registry["buf-1"]["chunk_count"] == 100
    
    def test_dirty_file_tracking(self):
        """Test marking and clearing dirty files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            manager.set_buffer_info("buf-1", {"path": "/code", "dirty_files": {}})
            
            # Mark file as dirty
            manager.mark_dirty_file("buf-1", "module.py")
            assert "module.py" in manager.registry["buf-1"]["dirty_files"]
            
            # Clear after commit
            manager.clear_dirty_files("buf-1")
            assert manager.registry["buf-1"]["dirty_files"] == {}
    
    def test_cache_invalidation_tracking(self):
        """Test cache invalidation flags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            manager.set_buffer_info("buf-1", {"path": "/code"})
            
            # Cache should be valid initially
            assert manager.is_cache_valid("buf-1", "index")
            
            # Invalidate index cache
            manager.invalidate_cache_for_buffer("buf-1", ["index"])
            assert not manager.is_cache_valid("buf-1", "index")
            assert manager.is_cache_valid("buf-1", "lexical")  # Other caches valid
            
            # Invalidate all caches
            manager.invalidate_cache_for_buffer("buf-1")
            assert not manager.is_cache_valid("buf-1", "index")
            assert not manager.is_cache_valid("buf-1", "lexical")
            assert not manager.is_cache_valid("buf-1", "query")
    
    def test_transaction_lifecycle(self):
        """Test transaction creation, commit, rollback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            manager.set_buffer_info("buf-1", {"path": "/code", "dirty_files": {}})
            
            # Start transaction
            tx_id = manager.start_transaction(
                buffer_id="buf-1",
                operation="write_code",
                file_path="module.py",
                start_line=5,
                end_line=10,
                new_lines=["new line"]
            )
            assert tx_id
            
            # Commit transaction
            manager.commit_transaction(tx_id)
            
            # Verify WAL has entry
            wal_content = manager.wal_path.read_text()
            assert "committed" in wal_content
    
    def test_wal_recovery_on_startup(self):
        """Test recovery from pending transactions in WAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            
            # Create manager and start transaction
            manager1 = StateManager(work_dir)
            manager1.set_buffer_info("buf-1", {
                "path": "/code",
                "dirty_files": {"module.py": {"modified_at": "2026-05-04T10:00:00Z"}}
            })
            
            # Start but don't commit transaction (simulate crash)
            tx_id = manager1.start_transaction(
                buffer_id="buf-1",
                operation="write_code",
                file_path="module.py",
                start_line=5,
                end_line=10,
                new_lines=["new line"]
            )
            
            # Simulate crash - create new manager
            manager2 = StateManager(work_dir)
            
            # Should have detected pending transaction and rolled back
            wal_content = manager2.wal_path.read_text()
            assert "rolled_back" in wal_content
            
            # Verify registry still has the buffer (WAL recovery doesn't lose data)
            assert "buf-1" in manager2.registry
    
    def test_concurrent_access_serialization(self):
        """Test that concurrent registry updates don't corrupt state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            manager = StateManager(work_dir)
            
            results = []
            
            def update_buffer(buffer_id, count):
                """Update same buffer from multiple threads."""
                mgr = StateManager(work_dir)
                for i in range(count):
                    mgr.set_buffer_info(buffer_id, {
                        "path": f"/code{i}",
                        "chunk_count": i,
                        "dirty_files": {}
                    })
                results.append(buffer_id)
            
            # Spawn multiple threads updating same buffer
            threads = [
                Thread(target=update_buffer, args=("buf-1", 3)),
                Thread(target=update_buffer, args=("buf-1", 3))
            ]
            
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # Registry should be consistent
            manager3 = StateManager(work_dir)
            assert "buf-1" in manager3.registry
            assert manager3.registry["buf-1"]["chunk_count"] in [0, 1, 2]  # Last write wins
    
    def test_corrupted_registry_recovery(self):
        """Test recovery from corrupted registry JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            
            # Create corrupted registry
            registry_path = work_dir / "registry.json"
            registry_path.write_text("{invalid json")
            
            # Should start fresh
            manager = StateManager(work_dir)
            assert manager.registry == {}
    
    def test_buffer_info_operations(self):
        """Test get/set buffer info operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            
            # Get non-existent buffer
            assert manager.get_buffer_info("nonexistent") is None
            
            # Set and get
            info = {"path": "/code", "chunk_count": 50}
            manager.set_buffer_info("buf-1", info)
            
            retrieved = manager.get_buffer_info("buf-1")
            assert retrieved == info


class TestStateManagerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_mark_dirty_nonexistent_buffer(self):
        """Test that marking dirty file on nonexistent buffer raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            
            with pytest.raises(ValueError):
                manager.mark_dirty_file("nonexistent", "file.py")
    
    def test_multiple_cache_types_invalidation(self):
        """Test invalidating multiple cache types at once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))
            manager.set_buffer_info("buf-1", {"path": "/code"})
            
            # Invalidate both index and lexical
            manager.invalidate_cache_for_buffer(
                "buf-1",
                ["index", "lexical"]
            )
            
            assert not manager.is_cache_valid("buf-1", "index")
            assert not manager.is_cache_valid("buf-1", "lexical")
            assert manager.is_cache_valid("buf-1", "query")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
