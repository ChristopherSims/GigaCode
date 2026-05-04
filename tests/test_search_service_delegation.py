"""Semantic Search Service Delegation Tests.

Tests the delegation pattern where CodeEmbeddingTool methods call manager implementations
while maintaining backward compatibility.
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
            # Patch search_service import to succeed this time
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


class TestSemanticSearchDelegation:
    """Test semantic_search delegation to SearchService."""

    def test_semantic_search_delegates_when_service_available(self, cet_with_service):
        """Test that semantic_search delegates to SearchService when available."""
        # Setup mock SearchService response
        mock_response = {
            "status": "ok",
            "buffer_id": "test-buffer",
            "query": "test query",
            "matches": [
                {
                    "file": "test.py",
                    "start_line": 10,
                    "end_line": 20,
                    "type": "function",
                    "name": "test_func",
                    "score": 0.95,
                    "doc_id": 0,
                }
            ],
            "cache_hit": False,
            "mode": "semantic",
        }
        
        cet_with_service._search_service.semantic_search = MagicMock(
            return_value=mock_response
        )
        
        # Call semantic_search
        result = cet_with_service.semantic_search(
            buffer_id="test-buffer",
            query="test query",
            top_k=5,
        )
        
        # Verify delegation happened
        cet_with_service._search_service.semantic_search.assert_called_once()
        
        # Verify response format
        assert result["status"] == "ok"
        assert len(result["matches"]) == 1
        assert result["matches"][0]["file"] == "test.py"

    def test_semantic_search_falls_back_on_delegation_error(self, cet_with_service):
        """Test fallback when SearchService delegation fails."""
        # Make SearchService raise an exception
        cet_with_service._search_service.semantic_search = MagicMock(
            side_effect=RuntimeError("Service error")
        )
        
        # Mock the internal methods for fallback
        cet_with_service._get_buffer_info = MagicMock(return_value={"buffer_id": "test"})
        cet_with_service._query_cache = MagicMock()
        cet_with_service._query_cache.get = MagicMock(return_value=None)
        cet_with_service._get_index = MagicMock(return_value=MagicMock(
            is_gpu=False,
            search=MagicMock(return_value=([[0.95]], [[0]]))  # Return nested lists
        ))
        cet_with_service._load_chunks = MagicMock(return_value=[
            MagicMock(
                file="test.py",
                start_line=10,
                end_line=20,
                type="function",
                name="test",
                text="def test(): pass"
            )
        ])
        cet_with_service._embedder.encode = MagicMock(return_value=[[0.1, 0.2]])
        
        # Call semantic_search
        result = cet_with_service.semantic_search(
            buffer_id="test-buffer",
            query="test query",
            top_k=5,
        )
        
        # Verify fallback was used (result should not error out)
        assert result is not None
        assert "status" in result

    def test_semantic_search_validates_params(self, cet_with_service):
        """Test parameter validation in semantic_search."""
        result = cet_with_service.semantic_search(
            buffer_id="test-buffer",
            query="",  # Empty query
            top_k=5,
        )
        
        # Should return validation error before delegation
        assert result["status"] == "error"
        assert "non-empty" in result.get("message", "").lower()


class TestResponseAdaptation:
    """Test response type adaptation between managers and CodeEmbeddingTool."""

    def test_adapt_search_response_handles_error_dicts(self, cet_with_service):
        """Test that error dicts are passed through unchanged."""
        error_response = {"status": "error", "error": "Test error"}
        
        result = CodeEmbeddingTool._adapt_search_response(error_response)
        
        assert result == error_response
        assert result["status"] == "error"

    def test_adapt_search_response_converts_matches(self, cet_with_service):
        """Test that SearchService matches are adapted to CodeEmbeddingTool format."""
        service_response = {
            "status": "ok",
            "buffer_id": "test",
            "query": "test",
            "matches": [
                {
                    "file": "test.py",
                    "start_line": 10,
                    "end_line": 20,
                    "score": 0.95,
                    "type": "function",
                    "name": "test",
                }
            ],
            "cache_hit": False,
            "mode": "semantic",
        }
        
        result = CodeEmbeddingTool._adapt_search_response(service_response)
        
        assert result["status"] == "ok"
        assert len(result["matches"]) == 1
        assert result["matches"][0]["file"] == "test.py"
        assert "doc_id" in result["matches"][0]

    def test_adapt_search_response_with_offset(self, cet_with_service):
        """Test response adaptation with pagination offset."""
        service_response = {
            "status": "ok",
            "matches": [
                {"file": f"file{i}.py", "start_line": i, "end_line": i+10, "score": 0.9 - i*0.1}
                for i in range(10)
            ],
        }
        
        # Request with offset
        result = CodeEmbeddingTool._adapt_search_response(
            service_response,
            offset=5,
            top_k=3,
        )
        
        # Should return only 3 items starting from offset 5
        assert len(result["matches"]) == 3
        assert result["matches"][0]["file"] == "file5.py"


class TestDelegationFallback:
    """Test graceful fallback when manager isn't available."""

    def test_semantic_search_without_service(self, temp_work_dir):
        """Test semantic_search falls back when SearchService unavailable."""
        with patch('gigacode.gigacode_tool.Embedder'):
            with patch('gigacode.gigacode_tool.StateManager'):
                with patch.dict('sys.modules', {'gigacode.search_service': None}):
                    cet = CodeEmbeddingTool(
                        work_dir=temp_work_dir,
                        enable_prometheus=False,
                    )
                    
                    # Ensure SearchService is None
                    cet._search_service = None
                    
                    # Mock internal methods for fallback
                    cet._get_buffer_info = MagicMock(return_value=None)
                    
                    # Call semantic_search (should use fallback and return error)
                    result = cet.semantic_search(
                        buffer_id="unknown",
                        query="test",
                    )
                    
                    # Should return error due to unknown buffer
                    assert result["status"] == "error"


class TestDelegationIntegration:
    """Test full delegation workflows."""

    def test_delegation_chain_semantic_search(self, cet_with_service):
        """Test complete semantic search delegation chain."""
        # Setup mocks for delegation
        mock_response = {
            "status": "ok",
            "buffer_id": "buffer-1",
            "query": "search term",
            "matches": [
                {
                    "file": "app.py",
                    "start_line": 100,
                    "end_line": 110,
                    "type": "function",
                    "name": "process",
                    "score": 0.92,
                }
            ],
            "cache_hit": False,
        }
        
        cet_with_service._search_service.semantic_search = MagicMock(
            return_value=mock_response
        )
        
        # Call through CodeEmbeddingTool facade
        result = cet_with_service.semantic_search(
            buffer_id="buffer-1",
            query="search term",
            top_k=10,
        )
        
        # Verify delegation was called with correct parameters
        cet_with_service._search_service.semantic_search.assert_called_with(
            buffer_id="buffer-1",
            query="search term",
            top_k=10,  # top_k + offset (10 + 0)
        )
        
        # Verify response
        assert result["status"] == "ok"
        assert result["cached"] is False
        assert len(result["matches"]) == 1


class TestDelegationBackwardCompatibility:
    """Test that delegations maintain backward compatibility."""

    def test_search_response_format_unchanged(self, cet_with_service):
        """Test that SearchResponse format is compatible."""
        response = SearchResponse(
            status=ResponseStatus.OK,
            matches=[],
            cached=False,
        )
        
        response_dict = response.to_dict()
        
        # Verify expected keys
        assert "status" in response_dict
        assert "matches" in response_dict
        assert "cached" in response_dict
        
        # Verify types
        assert response_dict["status"] == "ok"
        assert isinstance(response_dict["matches"], list)
        assert isinstance(response_dict["cached"], bool)

    def test_search_match_format_unchanged(self, cet_with_service):
        """Test that SearchMatch format is compatible."""
        match = SearchMatch(
            file="test.py",
            start_line=1,
            end_line=10,
            score=0.95,
            doc_id=0,
        )
        
        match_dict = match.to_dict()
        
        # Verify expected keys
        assert match_dict["file"] == "test.py"
        assert match_dict["start_line"] == 1
        assert match_dict["end_line"] == 10
        assert match_dict["score"] == 0.95
        assert match_dict["doc_id"] == 0
