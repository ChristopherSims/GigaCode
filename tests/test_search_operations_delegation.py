"""Search Operations Delegation Tests.

Tests the delegation pattern for search_for, look_for_file, search_symbols,
cluster_code, and find_duplicates.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gigacode.gigacode_tool import CodeEmbeddingTool
from gigacode.response_types import ResponseStatus, SearchResponse, SearchMatch


@pytest.fixture
def temp_work_dir():
    """Create temporary working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cet_with_service(temp_work_dir):
    """Create CodeEmbeddingTool with SearchService available."""
    with patch('gigacode.gigacode_tool.Embedder'):
        with patch('gigacode.gigacode_tool.StateManager'):
            with patch.dict('sys.modules', {'gigacode.search_service': MagicMock()}):
                cet = CodeEmbeddingTool(
                    work_dir=temp_work_dir,
                    model_name=None,
                    device='cpu',
                    max_buffers=10,
                    enable_prometheus=False,
                )
                # Manually create a mock SearchService
                if cet._search_service is None:
                    cet._search_service = MagicMock()
                return cet


class TestSearchForDelegation:
    """Test search_for delegation to SearchService."""

    def test_search_for_delegates_when_available(self, cet_with_service):
        """Test that search_for delegates to SearchService."""
        mock_response = SearchResponse(
            status=ResponseStatus.OK,
            matches=[
                SearchMatch(
                    file="test.py",
                    start_line=10,
                    end_line=20,
                    score=1.0,
                    doc_id=0,
                )
            ],
        )
        
        cet_with_service._search_service.search_for = MagicMock(
            return_value=mock_response
        )
        cet_with_service._get_buffer_info = MagicMock(return_value={"buffer_id": "test"})
        
        result = cet_with_service.search_for(
            buffer_id="test-buffer",
            query="test",
            case_sensitive=False,
        )
        
        # Verify delegation happened
        cet_with_service._search_service.search_for.assert_called_once()
        
        # Verify response format
        assert result["status"] == "ok"
        assert "matches" in result
        assert "total" in result


class TestLookForFileDelegation:
    """Test look_for_file delegation to SearchService."""

    def test_look_for_file_adapts_response(self, cet_with_service):
        """Test that look_for_file delegates and adapts response."""
        mock_response = {
            "status": "ok",
            "files": ["src/main.py", "src/utils.py"],
            "count": 2,
        }
        
        cet_with_service._search_service.look_for_file = MagicMock(
            return_value=mock_response
        )
        cet_with_service._get_buffer_info = MagicMock(return_value={"buffer_id": "test"})
        
        result = cet_with_service.look_for_file(
            buffer_id="test-buffer",
            file_name="main",
        )
        
        # Verify delegation happened
        cet_with_service._search_service.look_for_file.assert_called_once()
        
        # Verify response is adapted
        assert result["status"] == "ok"
        # Multiple matches should return candidates
        if len(mock_response["files"]) > 1:
            assert "candidates" in result or "file_location" in result


class TestSearchSymbolsDelegation:
    """Test search_symbols delegation."""

    def test_search_symbols_delegates(self, cet_with_service):
        """Test that search_symbols delegates to SearchService."""
        mock_response = SearchResponse(
            status=ResponseStatus.OK,
            matches=[
                SearchMatch(
                    file="test.py",
                    start_line=5,
                    end_line=50,
                    type="class",
                    name="MyClass",
                    score=1.0,
                    doc_id=0,
                )
            ],
        )
        
        cet_with_service._search_service.search_symbols = MagicMock(
            return_value=mock_response
        )
        cet_with_service._get_buffer_info = MagicMock(return_value={"buffer_id": "test"})
        
        result = cet_with_service.search_symbols(
            buffer_id="test-buffer",
            query="MyClass",
            top_k=10,
        )
        
        # Verify delegation happened
        cet_with_service._search_service.search_symbols.assert_called_once()
        
        # Verify response
        assert result["status"] == "ok"
        assert "matches" in result


class TestClusterCodeDelegation:
    """Test cluster_code delegation."""

    def test_cluster_code_delegates(self, cet_with_service):
        """Test that cluster_code delegates to SearchService."""
        mock_response = {
            "status": "ok",
            "clusters": {
                0: [
                    {"file": "test.py", "start_line": 10, "end_line": 20},
                    {"file": "test.py", "start_line": 30, "end_line": 40},
                ]
            },
        }
        
        cet_with_service._search_service.cluster_code = MagicMock(
            return_value=mock_response
        )
        cet_with_service._get_buffer_info = MagicMock(return_value={"buffer_id": "test"})
        
        result = cet_with_service.cluster_code(
            buffer_id="test-buffer",
            threshold=0.75,
        )
        
        # Verify delegation happened
        cet_with_service._search_service.cluster_code.assert_called_once()
        
        # Verify response
        assert result["status"] == "ok"
        assert "clusters" in result


class TestFindDuplicatesDelegation:
    """Test find_duplicates delegation."""

    def test_find_duplicates_delegates(self, cet_with_service):
        """Test that find_duplicates delegates to SearchService."""
        mock_response = {
            "status": "ok",
            "duplicates": [
                (
                    {"file": "a.py", "start_line": 10, "name": "func1"},
                    {"file": "b.py", "start_line": 20, "name": "func1"},
                )
            ],
        }
        
        cet_with_service._search_service.find_duplicates = MagicMock(
            return_value=mock_response
        )
        cet_with_service._get_buffer_info = MagicMock(return_value={"buffer_id": "test"})
        
        result = cet_with_service.find_duplicates(
            buffer_id="test-buffer",
            threshold=0.85,
        )
        
        # Verify delegation happened
        cet_with_service._search_service.find_duplicates.assert_called_once()
        
        # Verify response
        assert result["status"] == "ok"
        assert "duplicates" in result


class TestPhase5bResponseAdapters:
    """Test response adapters for Phase 5b methods."""

    def test_adapt_file_response_single_match(self, cet_with_service):
        """Test file response adaptation for single match."""
        service_response = {
            "status": "ok",
            "files": ["src/main.py"],
            "count": 1,
        }
        
        result = CodeEmbeddingTool._adapt_file_response(service_response)
        
        assert result["status"] == "ok"
        assert "file_location" in result
        assert result["file_location"] == "src/main.py"

    def test_adapt_file_response_multiple_matches(self, cet_with_service):
        """Test file response adaptation for multiple matches."""
        service_response = {
            "status": "ok",
            "files": ["src/main.py", "src/main_backup.py"],
            "count": 2,
        }
        
        result = CodeEmbeddingTool._adapt_file_response(service_response)
        
        assert result["status"] == "ok"
        assert result["match_type"] == "multiple"
        assert "candidates" in result

    def test_adapt_cluster_response(self, cet_with_service):
        """Test cluster response adaptation."""
        service_response = {
            "status": "ok",
            "clusters": {
                0: [
                    {"file": "test.py", "start_line": 10, "end_line": 20, "score": 0.9},
                    {"file": "test.py", "start_line": 30, "end_line": 40, "score": 0.8},
                ]
            },
        }
        
        result = CodeEmbeddingTool._adapt_cluster_response(service_response)
        
        assert result["status"] == "ok"
        assert "clusters" in result
        assert isinstance(result["clusters"], list)
        assert len(result["clusters"]) == 1
        # Verify avg_score is calculated (not hardcoded to 0.0)
        assert abs(result["clusters"][0]["avg_score"] - 0.85) < 0.0001  # (0.9 + 0.8) / 2

    def test_adapt_duplicate_response(self, cet_with_service):
        """Test duplicate response adaptation."""
        service_response = {
            "status": "ok",
            "duplicates": [
                (
                    {"file": "a.py", "start_line": 10},
                    {"file": "b.py", "start_line": 20},
                )
            ],
        }
        
        result = CodeEmbeddingTool._adapt_duplicate_response(service_response)
        
        assert result["status"] == "ok"
        assert "duplicates" in result


class TestPhase5bDelegationFallback:
    """Test fallback behavior when delegation fails."""

    def test_search_for_fallback_on_error(self, cet_with_service):
        """Test search_for falls back when delegation fails."""
        cet_with_service._search_service.search_for = MagicMock(
            side_effect=RuntimeError("Service error")
        )
        cet_with_service._get_buffer_info = MagicMock(return_value=None)
        
        result = cet_with_service.search_for(
            buffer_id="unknown",
            query="test",
        )
        
        # Should return error due to unknown buffer
        assert result is not None
        assert "status" in result


class TestPhase5bIntegration:
    """Test Phase 5b delegations in integrated scenarios."""

    def test_all_search_methods_callable(self, cet_with_service):
        """Test that all Phase 5b methods are callable."""
        cet_with_service._get_buffer_info = MagicMock(return_value={"buffer_id": "test"})
        cet_with_service._search_service.search_for = MagicMock(
            return_value={"status": "ok", "matches": []}
        )
        cet_with_service._search_service.look_for_file = MagicMock(
            return_value={"status": "ok", "files": []}
        )
        cet_with_service._search_service.search_symbols = MagicMock(
            return_value=SearchResponse(status=ResponseStatus.OK, matches=[])
        )
        cet_with_service._search_service.cluster_code = MagicMock(
            return_value={"status": "ok", "clusters": {}}
        )
        cet_with_service._search_service.find_duplicates = MagicMock(
            return_value={"status": "ok", "duplicates": []}
        )
        
        # All should be callable without error
        cet_with_service.search_for("test", "query")
        cet_with_service.look_for_file("test", "file")
        cet_with_service.search_symbols("test", "symbol")
        cet_with_service.cluster_code("test")
        cet_with_service.find_duplicates("test")
