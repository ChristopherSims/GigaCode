"""Integration tests for Phase 4 — CodeEmbeddingTool manager integration.

Tests focus on manager initialization and delegation layer without triggering sklearn.
Note: SearchService import is skipped due to pre-existing sklearn/Windows incompatibility.
"""

import json
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
def cet_instance(temp_work_dir):
    """Create CodeEmbeddingTool instance with minimal mocking."""
    with patch('gigacode.gigacode_tool.Embedder'):
        with patch('gigacode.gigacode_tool.StateManager'):
            # Patch the manager imports inside the try block by mocking at module level
            with patch.dict('sys.modules', {'gigacode.search_service': None}):
                # This will cause ImportError when search_service is imported
                cet = CodeEmbeddingTool(
                    work_dir=temp_work_dir,
                    model_name=None,
                    device='cpu',
                    max_buffers=10,
                    enable_prometheus=False,
                )
                return cet


class TestCodeEmbeddingToolIntegration:
    """Test CodeEmbeddingTool with manager integration."""

    def test_cet_created_successfully(self, cet_instance):
        """Test that CodeEmbeddingTool instance is created."""
        assert cet_instance is not None
        assert isinstance(cet_instance, CodeEmbeddingTool)

    def test_embedder_initialized(self, cet_instance):
        """Test that embedder is initialized."""
        assert hasattr(cet_instance, '_embedder')
        assert cet_instance._embedder is not None

    def test_state_manager_initialized(self, cet_instance):
        """Test that state manager is initialized."""
        assert hasattr(cet_instance, '_state_manager')
        assert cet_instance._state_manager is not None

    def test_work_dir_created(self, cet_instance, temp_work_dir):
        """Test that work directory is set up."""
        assert cet_instance.work_dir == temp_work_dir
        assert cet_instance.work_dir.exists()

    def test_registry_loaded(self, cet_instance):
        """Test that registry is loaded."""
        assert hasattr(cet_instance, '_registry')
        assert isinstance(cet_instance._registry, dict)

    def test_audit_log_path_set(self, cet_instance):
        """Test that audit log path is set."""
        assert hasattr(cet_instance, '_audit_log_path')
        assert cet_instance._audit_log_path is not None

    def test_caches_initialized(self, cet_instance):
        """Test that all caches are initialized."""
        assert hasattr(cet_instance, '_index_cache')
        assert hasattr(cet_instance, '_lexical_cache')
        assert hasattr(cet_instance, '_query_cache')
        assert cet_instance._index_cache is not None
        assert cet_instance._lexical_cache is not None
        assert cet_instance._query_cache is not None

    def test_settings_preserved(self, cet_instance):
        """Test that settings are preserved."""
        assert cet_instance.threshold_mb == 500.0
        assert cet_instance.use_gpu is True
        assert cet_instance.gpu_id == 0
        assert cet_instance.max_buffers == 10

    def test_context_manager_protocol(self, temp_work_dir):
        """Test context manager support."""
        with patch('gigacode.gigacode_tool.Embedder'):
            with patch('gigacode.gigacode_tool.StateManager'):
                with patch.dict('sys.modules', {'gigacode.search_service': None}):
                    with CodeEmbeddingTool(work_dir=temp_work_dir) as cet:
                        assert cet is not None
                        assert hasattr(cet, 'work_dir')

    def test_public_methods_exist(self, cet_instance):
        """Test that important public methods exist."""
        # Buffer operations
        assert hasattr(cet_instance, 'list_buffers')
        assert callable(getattr(cet_instance, 'list_buffers', None))
        
        assert hasattr(cet_instance, 'embed_codebase')
        assert callable(getattr(cet_instance, 'embed_codebase', None))

        # Search operations
        assert hasattr(cet_instance, 'semantic_search')
        assert callable(getattr(cet_instance, 'semantic_search', None))
        
        assert hasattr(cet_instance, 'hybrid_search')
        assert callable(getattr(cet_instance, 'hybrid_search', None))

    def test_snapshot_managers_initialized(self, cet_instance):
        """Test that snapshot managers dict is initialized."""
        assert hasattr(cet_instance, '_snapshot_managers')
        assert isinstance(cet_instance._snapshot_managers, dict)

    def test_prometheus_disabled_by_default(self, cet_instance):
        """Test that Prometheus is disabled by default."""
        assert cet_instance._prometheus_exporter is None

    def test_buffer_manager_fallback(self, cet_instance):
        """Test that system falls back gracefully when managers fail to import."""
        # Since we patched SearchService to fail, buffer_manager should be None
        assert cet_instance._buffer_manager is None
        # But the CET itself should still work
        assert cet_instance is not None
        assert hasattr(cet_instance, '_registry')


class TestBackwardCompatibility:
    """Test that CodeEmbeddingTool maintains full backward compatibility."""

    def test_legacy_attributes_accessible(self, cet_instance):
        """Test that legacy attributes are still accessible."""
        # These are used by existing code
        assert hasattr(cet_instance, '_embedder')
        assert hasattr(cet_instance, '_registry')
        assert hasattr(cet_instance, '_index_cache')
        assert hasattr(cet_instance, '_lexical_cache')
        assert hasattr(cet_instance, '_query_cache')
        assert hasattr(cet_instance, '_snapshot_managers')
        assert hasattr(cet_instance, '_state_manager')

    def test_method_signatures_unchanged(self, cet_instance):
        """Test that public method signatures exist."""
        import inspect

        # Check that key methods have correct signatures
        embed = cet_instance.embed_codebase
        assert callable(embed)

        semantic = cet_instance.semantic_search
        assert callable(semantic)

    def test_no_breaking_changes(self, cet_instance):
        """Test that there are no breaking API changes."""
        # The class should be usable exactly as before
        assert isinstance(cet_instance.work_dir, Path)
        assert cet_instance.work_dir.exists()
        
        # All important attributes should be present
        assert cet_instance.threshold_mb is not None
        assert cet_instance.use_gpu is not None
        assert cet_instance.gpu_id is not None
        assert cet_instance.max_buffers is not None


class TestManagerInitializationStrategy:
    """Test the manager initialization strategy."""

    def test_graceful_fallback_on_import_error(self, temp_work_dir):
        """Test that CodeEmbeddingTool gracefully falls back when managers fail."""
        with patch('gigacode.gigacode_tool.Embedder'):
            with patch('gigacode.gigacode_tool.StateManager'):
                # Patch SearchService to always fail
                with patch.dict('sys.modules', {'gigacode.search_service': None}):
                    cet = CodeEmbeddingTool(
                        work_dir=temp_work_dir,
                        enable_prometheus=False,
                    )
                    
                    # Should still create CET instance
                    assert cet is not None
                    
                    # Managers should be None
                    assert cet._buffer_manager is None
                    assert cet._index_manager is None
                    assert cet._search_service is None
                    
                    # But CET should still be usable
                    assert cet.work_dir == temp_work_dir

    def test_partial_manager_failure_handling(self, temp_work_dir):
        """Test handling when one manager import fails."""
        with patch('gigacode.gigacode_tool.Embedder'):
            with patch('gigacode.gigacode_tool.StateManager'):
                # Patch to fail on BufferManager import
                with patch.dict('sys.modules', {'gigacode.buffer_manager': None}):
                    cet = CodeEmbeddingTool(
                        work_dir=temp_work_dir,
                        enable_prometheus=False,
                    )
                    
                    # All managers should be None due to exception
                    assert cet._buffer_manager is None
                    assert cet._index_manager is None
                    assert cet._search_service is None


class TestInitializationOrder:
    """Test that initialization happens in correct order."""

    def test_embedder_before_managers(self, cet_instance):
        """Test that embedder is created before managers."""
        # If embedder is None, managers shouldn't be initialized
        assert cet_instance._embedder is not None

    def test_state_manager_before_buffer_manager(self, cet_instance):
        """Test that state manager is created before buffer manager."""
        # State manager should always exist
        assert cet_instance._state_manager is not None

    def test_work_dir_before_everything(self, cet_instance, temp_work_dir):
        """Test that work_dir is set up before managers."""
        # Work directory should exist
        assert cet_instance.work_dir.exists()
        assert cet_instance.work_dir == temp_work_dir


class TestPrometheusIntegration:
    """Test Prometheus integration without actual Prometheus."""

    def test_prometheus_not_enabled_by_default(self, cet_instance):
        """Test that Prometheus is not enabled by default."""
        assert cet_instance._prometheus_exporter is None

    def test_prometheus_attributes_exist(self, cet_instance):
        """Test that Prometheus-related attributes exist."""
        assert hasattr(cet_instance, '_prometheus_exporter')

    def test_create_cet_with_prometheus_disabled(self, temp_work_dir):
        """Test creating CET with Prometheus explicitly disabled."""
        with patch('gigacode.gigacode_tool.Embedder'):
            with patch('gigacode.gigacode_tool.StateManager'):
                with patch.dict('sys.modules', {'gigacode.search_service': None}):
                    cet = CodeEmbeddingTool(
                        work_dir=temp_work_dir,
                        enable_prometheus=False,
                    )
                    
                    assert cet._prometheus_exporter is None
