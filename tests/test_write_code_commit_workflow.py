"""Integration tests for GigaCode critical workflows (simplified version).

Tests validate write_code + commit, cache behavior, and resource cleanup.
"""

# CRITICAL: Initialize sklearn FIRST before any gigacode imports
# This prevents torch._dynamo from encountering sklearn.__spec__ == None
import types

try:
    import sklearn

    if getattr(sklearn, "__spec__", None) is None:
        sklearn.__spec__ = types.ModuleSpec("sklearn", getattr(sklearn, "__file__", None))
except Exception:
    pass

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gigacode.gigacode_tool import CodeEmbeddingTool


class TestCoreWorkflows:
    """Test critical workflows: write_code + commit, caching, etc."""

    def test_write_code_and_commit_integration(self):
        """Test write_code + commit workflow end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def add(a, b):\n    return a + b\n")

            # Embed codebase
            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            response = tool.embed_codebase(str(code_dir))
            assert response.get("status") == "ok"
            buffer_id = response["buffer_id"]

            # Verify initial state
            result = tool.read_code(buffer_id, "module.py")
            assert result is not None

            # Modify code - insert doc string at line 1
            new_lines = ["def add(a, b):\n", '    """Add two numbers."""\n', "    return a + b\n"]
            write_result = tool.write_code(buffer_id, "module.py", 1, new_lines, end_line=3)
            assert write_result is not None

            # Commit changes
            commit_result = tool.commit(buffer_id)
            assert commit_result.get("status") == "ok"

            # Verify on-disk change
            on_disk = (code_dir / "module.py").read_text()
            assert "Add two numbers" in on_disk

            tool.close()

    def test_cache_stats_tracking(self):
        """Test cache hit/miss statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "math.py").write_text("def add(a, b): return a + b\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            response = tool.embed_codebase(str(code_dir))
            assert response.get("status") == "ok"
            buffer_id = response["buffer_id"]

            # First search (cache miss)
            result1 = tool.semantic_search(buffer_id, "add", top_k=5)
            assert result1.get("status") == "ok"
            stats1 = tool._query_cache.stats()

            # Same search (cache hit)
            result2 = tool.semantic_search(buffer_id, "add", top_k=5)
            assert result2.get("status") == "ok"
            stats2 = tool._query_cache.stats()
            assert stats2["hits"] >= 1

            tool.close()

    def test_gpu_id_parameter(self):
        """Test GPU device ID configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)

            # Create with custom GPU ID
            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False, gpu_id=2)

            # Verify GPU ID stored
            assert tool.gpu_id == 2

            tool.close()

    def test_health_check_operational(self):
        """Test health check when system is operational."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("pass\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            response = tool.embed_codebase(str(code_dir))
            assert response.get("status") == "ok"
            buffer_id = response["buffer_id"]

            # Check health
            health = tool.health_check()
            assert health is not None
            assert health.get("status") in ["healthy", "degraded"]
            assert "buffers_registered" in health

            tool.close()

    def test_lru_eviction_on_max_buffers(self):
        """Test LRU eviction when exceeding max_buffers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False, max_buffers=2)

            # Create and embed 4 codebases (should trigger eviction)
            for i in range(4):
                code_dir = work_dir / f"code{i}"
                code_dir.mkdir()
                (code_dir / "module.py").write_text(f"def func{i}(): pass\n")

                response = tool.embed_codebase(str(code_dir))
                assert response.get("status") == "ok"

            # Cache should not exceed max_buffers
            assert tool._index_cache.stats()["size"] <= 2

            tool.close()

    def test_cache_cleared_on_close(self):
        """Test that caches are cleared on tool.close()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("pass\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            response = tool.embed_codebase(str(code_dir))
            assert response.get("status") == "ok"
            buffer_id = response["buffer_id"]

            # Populate caches
            tool.semantic_search(buffer_id, "module", top_k=5)
            assert tool._index_cache.stats()["size"] > 0

            # Close should clear
            tool.close()
            assert tool._index_cache.stats()["size"] == 0
            assert tool._query_cache.stats()["size"] == 0

    def test_semantic_cache_matching(self):
        """Test semantic similarity in query cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def add(a, b): return a + b\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            response = tool.embed_codebase(str(code_dir))
            assert response.get("status") == "ok"
            buffer_id = response["buffer_id"]

            # Search and cache
            result1 = tool.semantic_search(buffer_id, "add function", top_k=5)
            assert result1.get("status") == "ok"

            # Semantic matching should work (may hit cache)
            result2 = tool.semantic_search(buffer_id, "addition", top_k=5)
            assert result2.get("status") == "ok"

            stats = tool._query_cache.stats()
            # Should have cache entries
            assert stats["size"] >= 1

            tool.close()

    def test_write_code_nonexistent_buffer_fails(self):
        """Test that write_code fails gracefully for non-existent buffers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = CodeEmbeddingTool(Path(tmpdir) / "tool", use_gpu=False)

            # Try to write to non-existent buffer
            result = tool.write_code("nonexistent", "file.py", 1, ["code"])
            assert result is not None
            assert result.get("status") == "error" or result.get("error")

            tool.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
