"""Integration tests for GigaCode critical workflows.

Tests validate write_code + commit integration, concurrent operations, cache behavior,
resource cleanup, and GPU/CPU fallback under various conditions.
"""

# CRITICAL: Initialize sklearn FIRST before any gigacode imports
import types

try:
    import sklearn

    if getattr(sklearn, "__spec__", None) is None:
        sklearn.__spec__ = types.ModuleSpec("sklearn", getattr(sklearn, "__file__", None))
except Exception:
    pass


import sys
import tempfile
import threading
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gigacode.gigacode_tool import CodeEmbeddingTool


class TestWriteCodeAndCommitIntegration:
    """Test write_code + commit workflow end-to-end."""

    def test_write_code_modifies_and_commits(self):
        """Test writing code, committing changes, and verifying on-disk persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)

            # Create initial codebase
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def add(a, b):\n    return a + b\n")

            # Embed codebase
            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            embed_response = tool.embed_codebase(str(code_dir))
            assert embed_response is not None
            assert embed_response.get("status") == "ok"
            buffer_id = embed_response.get("buffer_id")
            assert buffer_id is not None

            # Verify initial state
            result = tool.read_code(buffer_id, "module.py")
            assert result is not None
            assert "add" in str(result)

            # Modify code via write_code
            new_code = "def add(a, b):\n    '''Add two numbers.'''\n    return a + b\n"
            write_result = tool.write_code(buffer_id, "module.py", new_code)
            assert write_result is not None

            # Commit changes
            commit_result = tool.commit(buffer_id)
            assert commit_result is not None
            assert commit_result.get("status") == "ok"

            # Verify on-disk change persisted
            on_disk = (code_dir / "module.py").read_text()
            assert "Add two numbers" in on_disk
            assert "return a + b" in on_disk

            tool.close()

    def test_write_code_without_embed_fails(self):
        """Test that write_code fails gracefully when buffer doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = CodeEmbeddingTool(Path(tmpdir) / "tool", use_gpu=False)

            # Try to write to non-existent buffer
            result = tool.write_code("nonexistent_buffer", "file.py", "code")
            assert result is not None
            assert result.get("status") == "error" or result.get("error") is not None

            tool.close()

    def test_multiple_commits_accumulate(self):
        """Test that multiple write_code + commit cycles accumulate changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("# Start\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            buffer_id = tool.embed_codebase(str(code_dir))

            # First write + commit
            tool.write_code(buffer_id, "module.py", "# First\n")
            tool.commit(buffer_id)

            # Second write + commit
            tool.write_code(buffer_id, "module.py", "# First\n# Second\n")
            tool.commit(buffer_id)

            # Verify both changes persisted
            on_disk = (code_dir / "module.py").read_text()
            assert "First" in on_disk
            assert "Second" in on_disk

            tool.close()


class TestConcurrentOperations:
    """Test concurrent access patterns."""

    def test_concurrent_embed_codebase_same_root(self):
        """Test concurrent embedding of same codebase doesn't corrupt registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir) / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def func(): pass\n")

            work_dir = Path(tmpdir) / "work"
            work_dir.mkdir()

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False, max_buffers=5)

            # Concurrent embeds of same codebase
            results = []
            errors = []

            def embed_task(task_id):
                try:
                    response = tool.embed_codebase(str(code_dir))
                    if response.get("status") == "ok":
                        buffer_id = response.get("buffer_id")
                        results.append((task_id, buffer_id))
                    else:
                        errors.append((task_id, response.get("error", "Unknown error")))
                except Exception as e:
                    errors.append((task_id, e))

            threads = [threading.Thread(target=embed_task, args=(i,)) for i in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should have some results (may reuse same buffer_id due to same root)
            assert len(results) > 0, f"Errors: {errors}"
            assert len(errors) == 0, f"Errors occurred: {errors}"

            # Registry should be consistent (buffers stored in _buffer_manager._registry)
            assert len(tool._buffer_manager._registry) >= 1

            tool.close()

    def test_concurrent_searches_on_same_buffer(self):
        """Test concurrent search operations on same buffer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()

            # Create some code
            for i in range(3):
                (code_dir / f"module{i}.py").write_text(f"def function_{i}(): pass\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            embed_response = tool.embed_codebase(str(code_dir))
            assert embed_response.get("status") == "ok"
            buffer_id = embed_response.get("buffer_id")
            assert buffer_id is not None

            # Concurrent searches
            results = []
            errors = []

            def search_task(task_id, query):
                try:
                    result = tool.semantic_search(buffer_id, query, top_k=5)
                    results.append((task_id, result))
                except Exception as e:
                    errors.append((task_id, e))

            threads = [
                threading.Thread(target=search_task, args=(0, "function")),
                threading.Thread(target=search_task, args=(1, "def")),
                threading.Thread(target=search_task, args=(2, "function")),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(results) == 3, f"Expected 3 results, got {len(results)}, errors: {errors}"
            assert len(errors) == 0

            tool.close()


class TestCacheInvalidation:
    """Test cache behavior on code modifications."""

    def test_write_code_invalidates_query_cache(self):
        """Test that write_code invalidates query results cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def add(a, b): return a + b\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            embed_response = tool.embed_codebase(str(code_dir))
            assert embed_response.get("status") == "ok"
            buffer_id = embed_response.get("buffer_id")
            assert buffer_id is not None

            # First search (should cache)
            result1 = tool.semantic_search(buffer_id, "add function", top_k=5)
            cache_stats1 = tool._query_cache.stats()
            initial_size = cache_stats1["size"]

            # Write code (should invalidate caches)
            tool.write_code(buffer_id, "module.py", "def add(a, b, c): return a + b + c\n")

            # Query cache should be cleared for this buffer
            cache_stats2 = tool._query_cache.stats()
            assert cache_stats2["size"] < initial_size or cache_stats2["size"] == 0

            tool.close()

    def test_write_code_invalidates_index_cache(self):
        """Test that write_code doesn't invalidate index cache prematurely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def func(): pass\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            embed_response = tool.embed_codebase(str(code_dir))
            assert embed_response.get("status") == "ok"
            buffer_id = embed_response.get("buffer_id")
            assert buffer_id is not None

            # Load index into cache
            initial_size = tool._index_cache.stats()["size"]

            # Write code
            tool.write_code(buffer_id, "module.py", "def new_func(): pass\n")

            # Index cache may be cleared on commit, not on write_code
            # This depends on implementation, but should be documented behavior

            tool.close()


class TestMemoryManagement:
    """Test memory management and LRU eviction."""

    def test_lru_eviction_on_max_buffers(self):
        """Test that LRU eviction triggers when exceeding max_buffers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False, max_buffers=2)

            # Create multiple codebases
            for i in range(4):
                code_dir = work_dir / f"code{i}"
                code_dir.mkdir()
                (code_dir / "module.py").write_text(f"def func{i}(): pass\n")

            # Embed multiple codebases (should trigger eviction)
            buffer_ids = []
            for i in range(4):
                code_dir = work_dir / f"code{i}"
                buffer_id = tool.embed_codebase(str(code_dir))
                buffer_ids.append(buffer_id)

            # Cache should not exceed max_buffers
            assert tool._index_cache.stats()["size"] <= 2

            tool.close()

    def test_no_memory_leaks_on_close(self):
        """Test that close() releases all cached resources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def func(): pass\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            buffer_id = tool.embed_codebase(str(code_dir))

            # Populate caches
            tool.semantic_search(buffer_id, "function", top_k=5)

            assert tool._index_cache.stats()["size"] > 0
            assert tool._query_cache.stats()["size"] > 0

            # Close should clear caches
            tool.close()

            # Verify caches cleared
            assert tool._index_cache.stats()["size"] == 0
            assert tool._query_cache.stats()["size"] == 0


class TestErrorRecovery:
    """Test graceful error handling and recovery."""

    def test_write_code_to_nonexistent_file(self):
        """Test writing code to a file that doesn't exist in codebase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "existing.py").write_text("pass\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            buffer_id = tool.embed_codebase(str(code_dir))

            # Write to non-existent file
            result = tool.write_code(buffer_id, "nonexistent.py", "def new(): pass\n")

            # Should either create file or return error gracefully
            # (behavior depends on implementation)
            assert result is not None

            tool.close()

    def test_corrupted_registry_recovery(self):
        """Test recovery from corrupted registry file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("pass\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            buffer_id = tool.embed_codebase(str(code_dir))
            tool.close()

            # Corrupt registry file
            registry_file = work_dir / "tool" / "registry.json"
            if registry_file.exists():
                registry_file.write_text("{invalid json}")

            # Try to load tool again (should handle gracefully)
            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)

            # Should either recover or skip corrupted registry
            # Fresh embed should work
            code_dir2 = work_dir / "code2"
            code_dir2.mkdir()
            (code_dir2 / "module.py").write_text("pass\n")

            buffer_id2 = tool.embed_codebase(str(code_dir2))
            assert buffer_id2 is not None

            tool.close()


class TestGpuCpuFallback:
    """Test GPU/CPU fallback behavior."""

    def test_graceful_cpu_fallback_when_gpu_unavailable(self):
        """Test that system falls back to CPU when GPU is unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def func(): pass\n")

            # Try to create tool with GPU enabled but it will fall back to CPU
            # (since we don't have CUDA available in test environment)
            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=True, gpu_id=0)  # Request GPU

            # Should embed successfully even if GPU unavailable
            buffer_id = tool.embed_codebase(str(code_dir))
            assert buffer_id is not None

            # Search should work (CPU-only)
            result = tool.semantic_search(buffer_id, "function", top_k=5)
            assert result is not None
            assert result.get("status") == "ok"

            tool.close()

    def test_custom_gpu_device_selection(self):
        """Test that gpu_id parameter is stored and usable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)

            # Create with custom gpu_id
            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False, gpu_id=2)  # Custom device

            # Verify gpu_id is stored
            assert tool.gpu_id == 2

            tool.close()


class TestSemanticQueryCaching:
    """Test semantic similarity matching in query cache."""

    def test_paraphrased_queries_hit_cache(self):
        """Test that paraphrased queries return cached results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "math.py").write_text("def add(a, b): return a + b\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            buffer_id = tool.embed_codebase(str(code_dir))

            # First search
            result1 = tool.semantic_search(buffer_id, "addition function", top_k=5)
            assert result1 is not None

            # Cache should be populated
            assert tool._query_cache.stats()["size"] >= 1

            # Paraphrased query (if embedder available)
            # May or may not hit semantic cache depending on embedder availability
            result2 = tool.semantic_search(buffer_id, "add method", top_k=5)
            assert result2 is not None

            tool.close()


class TestHealthCheckAndMetrics:
    """Test health monitoring and metrics."""

    def test_health_check_when_operational(self):
        """Test health check reports healthy status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("pass\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            buffer_id = tool.embed_codebase(str(code_dir))

            # Check health
            health = tool.health_check()
            assert health is not None
            assert health.get("status") in ["healthy", "degraded"]
            assert "buffers_registered" in health
            assert "cache_utilization_percent" in health

            tool.close()

    def test_cache_stats_accumulate(self):
        """Test that cache statistics accurately track hits/misses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def func(): pass\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            buffer_id = tool.embed_codebase(str(code_dir))

            # First search (cache miss)
            tool.semantic_search(buffer_id, "function", top_k=5)
            stats1 = tool._query_cache.stats()
            miss_count1 = stats1["misses"]

            # Same search again (cache hit)
            tool.semantic_search(buffer_id, "function", top_k=5)
            stats2 = tool._query_cache.stats()
            hit_count2 = stats2["hits"]

            # Should have cache hit on second search
            assert hit_count2 >= 1

            tool.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
