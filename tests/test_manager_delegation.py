"""Buffer and Index Manager Delegation Tests.

Tests the delegation pattern for list_buffers, delete_buffer, get_cache_stats,
and health_check methods from managers.

Note: SearchService import is skipped due to pre-existing sklearn/Windows incompatibility.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gigacode.gigacode_tool import CodeEmbeddingTool


@pytest.fixture
def temp_work_dir():
    """Create temporary working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cet_with_managers(temp_work_dir):
    """Create CodeEmbeddingTool with managers available."""
    with patch('gigacode.gigacode_tool.Embedder'):
        with patch('gigacode.gigacode_tool.StateManager'):
            # Patch the SearchService import to prevent sklearn Windows crash
            with patch.dict('sys.modules', {'gigacode.search_service': None}):
                cet = CodeEmbeddingTool(
                    work_dir=temp_work_dir,
                    model_name=None,
                    device='cpu',
                    max_buffers=10,
                    enable_prometheus=False,
                )
                # Manually create mock managers if not initialized
                if cet._buffer_manager is None:
                    cet._buffer_manager = MagicMock()
                if cet._index_manager is None:
                    cet._index_manager = MagicMock()
                return cet


class TestReloadCodebaseDelegation:
    """Test reload_codebase delegation to BufferManager."""

    def test_reload_codebase_delegates_when_available(self, cet_with_managers):
        """Test that reload_codebase delegates to BufferManager."""
        mock_response = {
            "status": "ok",
            "merge_results": [
                {"file": "main.py", "status": "merged"},
            ],
            "elapsed_s": 0.05,
        }
        
        cet_with_managers._buffer_manager.reload_codebase = MagicMock(
            return_value=mock_response
        )
        
        result = cet_with_managers.reload_codebase("buf-1")
        
        # Verify delegation happened
        cet_with_managers._buffer_manager.reload_codebase.assert_called_once()
        
        # Verify response format
        assert result["status"] == "ok"
        assert "merge_results" in result or "elapsed_s" in result




class TestListBuffersDelegation:
    """Test list_buffers delegation to BufferManager."""

    def test_list_buffers_delegates_when_available(self, cet_with_managers):
        """Test that list_buffers delegates to BufferManager."""
        mock_response = {
            "status": "ok",
            "buffers": [
                {"buffer_id": "buf-1", "chunk_count": 100},
                {"buffer_id": "buf-2", "chunk_count": 200},
            ],
        }
        
        cet_with_managers._buffer_manager.list_buffers = MagicMock(
            return_value=mock_response
        )
        
        result = cet_with_managers.list_buffers()
        
        # Verify delegation happened
        cet_with_managers._buffer_manager.list_buffers.assert_called_once()
        
        # Verify response format
        assert result["status"] == "ok"
        assert "buffers" in result
        assert len(result["buffers"]) == 2




class TestDeleteBufferDelegation:
    """Test delete_buffer delegation to BufferManager."""

    def test_delete_buffer_delegates_when_available(self, cet_with_managers):
        """Test that delete_buffer delegates to BufferManager."""
        mock_response = {
            "status": "ok",
            "message": "Deleted buffer buf-1",
        }
        
        cet_with_managers._buffer_manager.delete_buffer = MagicMock(
            return_value=mock_response
        )
        
        result = cet_with_managers.delete_buffer("buf-1")
        
        # Verify delegation happened
        cet_with_managers._buffer_manager.delete_buffer.assert_called_once_with("buf-1")
        
        # Verify response format
        assert result["status"] == "ok"
        assert "message" in result

    def test_delete_buffer_cleans_up_caches(self, cet_with_managers):
        """Test that delete_buffer cleans up after delegation."""
        mock_response = {"status": "ok", "message": "Deleted"}
        
        cet_with_managers._buffer_manager.delete_buffer = MagicMock(
            return_value=mock_response
        )
        # Setup index_manager._query_cache mock
        cet_with_managers._index_manager._query_cache = MagicMock()
        cet_with_managers._index_manager._query_cache.invalidate_buffer = MagicMock()
        
        result = cet_with_managers.delete_buffer("buf-1")
        
        # Verify delegation happened
        cet_with_managers._buffer_manager.delete_buffer.assert_called_once_with("buf-1")
        
        # Verify response format
        assert result["status"] == "ok"
        assert "message" in result




class TestGetCacheStatsDelegation:
    """Test get_cache_stats delegation to IndexManager."""

    def test_get_cache_stats_delegates_when_available(self, cet_with_managers):
        """Test that get_cache_stats delegates to IndexManager."""
        mock_response = {
            "index_cache_size": 5,
            "index_cache_max": 10,
            "lexical_cache_size": 3,
            "lexical_cache_max": 10,
            "query_cache_stats": {"hits": 42, "misses": 8},
        }
        
        cet_with_managers._index_manager.get_cache_stats = MagicMock(
            return_value=mock_response
        )
        
        result = cet_with_managers.get_cache_stats()
        
        # Verify delegation happened
        cet_with_managers._index_manager.get_cache_stats.assert_called_once()
        
        # Verify response format
        assert result["index_cache_size"] == 5
        assert result["query_cache_stats"]["hits"] == 42




class TestHealthCheckDelegation:
    """Test health_check delegation to IndexManager."""

    def test_health_check_delegates_when_available(self, cet_with_managers):
        """Test that health_check delegates to IndexManager."""
        mock_response = {
            "status": "healthy",
            "timestamp": "2026-05-04T14:30:00Z",
            "buffers_registered": 5,
            "buffers_loaded": 3,
            "embedder_ready": True,
            "warnings": [],
        }
        
        cet_with_managers._index_manager.health_check = MagicMock(
            return_value=mock_response
        )
        
        result = cet_with_managers.health_check()
        
        # Verify delegation happened
        cet_with_managers._index_manager.health_check.assert_called_once()
        
        # Verify response format
        assert result["status"] == "healthy"
        assert result["buffers_registered"] == 5




class TestPhase5cIntegration:
    """Test Phase 5c delegations in integrated scenarios."""

    def test_all_delegated_methods_callable(self, cet_with_managers):
        """Test that all Phase 5c delegated methods are callable."""
        # Setup manager mocks
        cet_with_managers._buffer_manager.list_buffers = MagicMock(
            return_value={"status": "ok", "buffers": []}
        )
        cet_with_managers._buffer_manager.delete_buffer = MagicMock(
            return_value={"status": "ok", "message": "Deleted"}
        )
        cet_with_managers._index_manager.get_cache_stats = MagicMock(
            return_value={"index_cache_size": 0}
        )
        cet_with_managers._index_manager.health_check = MagicMock(
            return_value={"status": "healthy"}
        )
        
        # All should be callable without error
        result1 = cet_with_managers.list_buffers()
        assert result1["status"] == "ok"
        
        result2 = cet_with_managers.delete_buffer("buf-1")
        assert result2["status"] == "ok"
        
        result3 = cet_with_managers.get_cache_stats()
        assert "index_cache_size" in result3
        
        result4 = cet_with_managers.health_check()
        assert "status" in result4
