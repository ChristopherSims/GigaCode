"""Tests for State Guards and Operation Configuration."""

import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

# Mock sklearn before importing gigacode modules
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.decomposition"] = MagicMock()
sys.modules["sklearn.feature_extraction"] = MagicMock()
sys.modules["sklearn.feature_extraction.text"] = MagicMock()

from gigacode.buffer_state import BufferState  # noqa: E402
from gigacode.gigacode_tool import CodeEmbeddingTool  # noqa: E402
from gigacode.health_status import HealthLevel, HealthStatus, HealthStatusTracker  # noqa: E402
from gigacode.operation_config import OperationConfig, OperationType  # noqa: E402


class TestOperationConfigTypes:
    """Test operation configuration and operation types."""

    def test_operation_types_defined(self):
        """Test that all operation types are defined."""
        assert OperationType.QUERY.value == "query"
        assert OperationType.WRITE.value == "write"
        assert OperationType.READ.value == "read"
        assert OperationType.REBUILD.value == "rebuild"

    def test_state_requirements_mapping(self):
        """Test state requirements for each operation type."""
        requirements = OperationConfig.get_state_requirements()

        # QUERY operations allowed in READY and DIRTY
        assert BufferState.READY in requirements[OperationType.QUERY]
        assert BufferState.DIRTY in requirements[OperationType.QUERY]
        assert BufferState.REBUILDING not in requirements[OperationType.QUERY]

        # WRITE operations allowed only in READY
        assert BufferState.READY in requirements[OperationType.WRITE]
        assert BufferState.DIRTY not in requirements[OperationType.WRITE]
        assert BufferState.REBUILDING not in requirements[OperationType.WRITE]

        # READ operations allowed in READY and DIRTY
        assert BufferState.READY in requirements[OperationType.READ]
        assert BufferState.DIRTY in requirements[OperationType.READ]

        # REBUILD operations allowed in READY and DIRTY
        assert BufferState.READY in requirements[OperationType.REBUILD]
        assert BufferState.DIRTY in requirements[OperationType.REBUILD]


class TestHealthStatusTracking:
    """Test health status tracking and warning levels."""

    def test_health_level_ok(self):
        """Test OK health level."""
        level = HealthStatus.compute_warning_level(
            dirty_file_count=0,
            index_age_seconds=0,
        )
        assert level == HealthLevel.OK

    def test_health_level_warning_dirty_files(self):
        """Test warning level from dirty files."""
        level = HealthStatus.compute_warning_level(
            dirty_file_count=5,
            index_age_seconds=0,
        )
        assert level == HealthLevel.WARNING

    def test_health_level_degraded_dirty_files(self):
        """Test degraded level from dirty files."""
        level = HealthStatus.compute_warning_level(
            dirty_file_count=20,
            index_age_seconds=0,
        )
        assert level == HealthLevel.DEGRADED

    def test_health_level_warning_index_age(self):
        """Test warning level from index age."""
        one_week = 7 * 24 * 3600
        level = HealthStatus.compute_warning_level(
            dirty_file_count=0,
            index_age_seconds=one_week,
        )
        assert level == HealthLevel.WARNING

    def test_health_level_degraded_index_age(self):
        """Test degraded level from index age."""
        one_month = 30 * 24 * 3600
        level = HealthStatus.compute_warning_level(
            dirty_file_count=0,
            index_age_seconds=one_month,
        )
        assert level == HealthLevel.DEGRADED

    def test_health_status_to_dict(self):
        """Test health status serialization."""
        health = HealthStatus(
            buffer_id="buf-123",
            state=BufferState.READY,
            last_state_change_timestamp=1000.0,
            dirty_file_count=0,
            index_age_seconds=100,
            warning_level=HealthLevel.OK,
        )

        health_dict = health.to_dict()
        assert health_dict["buffer_id"] == "buf-123"
        assert health_dict["state"] == "ready"
        assert health_dict["dirty_file_count"] == 0
        assert health_dict["warning_level"] == "ok"
        assert health_dict["is_rebuilding"] is False
        assert health_dict["has_uncommitted_changes"] is False


class TestHealthStatusTracker:
    """Test health status tracker."""

    def test_register_buffer(self):
        """Test buffer registration."""
        tracker = HealthStatusTracker()
        tracker.register_buffer("buf-123", BufferState.READY)

        health = tracker.get_health_status("buf-123")
        assert health is not None
        assert health.buffer_id == "buf-123"
        assert health.state == BufferState.READY

    def test_update_buffer_state(self):
        """Test buffer state update."""
        tracker = HealthStatusTracker()
        tracker.register_buffer("buf-123", BufferState.READY)

        tracker.update_buffer_state("buf-123", BufferState.DIRTY)

        health = tracker.get_health_status("buf-123")
        assert health.state == BufferState.DIRTY

    def test_update_dirty_file_count(self):
        """Test dirty file count update."""
        tracker = HealthStatusTracker()
        tracker.register_buffer("buf-123", BufferState.READY)

        tracker.update_dirty_file_count("buf-123", 5)

        health = tracker.get_health_status("buf-123")
        assert health.dirty_file_count == 5
        assert health.warning_level == HealthLevel.WARNING

    def test_update_index_age(self):
        """Test index age update."""
        tracker = HealthStatusTracker()
        tracker.register_buffer("buf-123", BufferState.READY)

        one_week = 7 * 24 * 3600
        tracker.update_index_age("buf-123", one_week)

        health = tracker.get_health_status("buf-123")
        assert health.index_age_seconds == one_week
        assert health.warning_level == HealthLevel.WARNING

    def test_increment_query_count(self):
        """Test query count increment."""
        tracker = HealthStatusTracker()
        tracker.register_buffer("buf-123", BufferState.READY)

        tracker.increment_query_count("buf-123")
        tracker.increment_query_count("buf-123")

        health = tracker.get_health_status("buf-123")
        assert health.query_count_since_rebuild == 2

    def test_query_count_reset_on_rebuild(self):
        """Test query count reset when buffer becomes READY."""
        tracker = HealthStatusTracker()
        tracker.register_buffer("buf-123", BufferState.READY)

        tracker.increment_query_count("buf-123")
        tracker.increment_query_count("buf-123")

        # Update to REBUILDING then back to READY
        tracker.update_buffer_state("buf-123", BufferState.REBUILDING)
        tracker.update_buffer_state("buf-123", BufferState.READY)

        health = tracker.get_health_status("buf-123")
        assert health.query_count_since_rebuild == 0

    def test_get_all_health_statuses(self):
        """Test getting all health statuses."""
        tracker = HealthStatusTracker()
        tracker.register_buffer("buf-123", BufferState.READY)
        tracker.register_buffer("buf-456", BufferState.DIRTY)

        all_health = tracker.get_all_health_statuses()
        assert len(all_health) == 2
        assert "buf-123" in all_health
        assert "buf-456" in all_health


class TestStateGuardsInCodeEmbeddingTool:
    """Test state guards in CodeEmbeddingTool."""

    @staticmethod
    def _create_cet(tmpdir):
        """Create a CodeEmbeddingTool with mocked dependencies."""
        with patch("gigacode.gigacode_tool.Embedder"):
            # Don't mock StateManager - we need real registry management
            with patch.dict(sys.modules, {"gigacode.search_service": None}):
                return CodeEmbeddingTool(work_dir=tmpdir)

    def test_check_state_for_query_ready(self):
        """Test query allowed in READY state."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            # Create test buffer with READY state
            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.READY.value,
                "state_changed_at": time.time(),
            }

            allowed, reason = cet._check_state_for_query(buffer_id)
            assert allowed is True
            assert reason == ""

    def test_check_state_for_query_rebuilding_blocked(self):
        """Test query blocked in REBUILDING state."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.REBUILDING.value,
                "state_changed_at": time.time(),
            }

            allowed, reason = cet._check_state_for_query(buffer_id)
            assert allowed is False
            assert "rebuilding" in reason.lower()

    def test_check_state_for_write_ready(self):
        """Test write allowed in READY state."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.READY.value,
                "state_changed_at": time.time(),
            }

            allowed, reason = cet._check_state_for_write(buffer_id)
            assert allowed is True

    def test_check_state_for_write_dirty_blocked(self):
        """Test write blocked in DIRTY state."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.DIRTY.value,
                "state_changed_at": time.time(),
            }

            allowed, reason = cet._check_state_for_write(buffer_id)
            assert allowed is False
            assert "cannot write" in reason.lower()

    def test_check_state_for_read_ready(self):
        """Test read allowed in READY state."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.READY.value,
                "state_changed_at": time.time(),
            }

            allowed, reason = cet._check_state_for_read(buffer_id)
            assert allowed is True

    def test_check_state_for_read_dirty_allowed(self):
        """Test read allowed in DIRTY state."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.DIRTY.value,
                "state_changed_at": time.time(),
            }

            allowed, reason = cet._check_state_for_read(buffer_id)
            assert allowed is True

    def test_check_state_for_rebuild_ready(self):
        """Test rebuild allowed in READY state."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.READY.value,
                "state_changed_at": time.time(),
            }

            allowed, reason = cet._check_state_for_rebuild(buffer_id)
            assert allowed is True

    def test_check_state_unknown_buffer(self):
        """Test error for unknown buffer."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            allowed, reason = cet._check_state_for_query("unknown-buf")
            assert allowed is False
            assert "unknown" in reason.lower()


class TestGetBufferState:
    """Test _get_buffer_state method."""

    @staticmethod
    def _create_cet(tmpdir):
        """Create a CodeEmbeddingTool with mocked dependencies."""
        with patch("gigacode.gigacode_tool.Embedder"):
            # Don't mock StateManager - we need real registry management
            with patch.dict(sys.modules, {"gigacode.search_service": None}):
                return CodeEmbeddingTool(work_dir=tmpdir)

    def test_get_buffer_state_ready(self):
        """Test retrieving READY state."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.READY.value,
            }

            state = cet._get_buffer_state(buffer_id)
            assert state == BufferState.READY

    def test_get_buffer_state_dirty(self):
        """Test retrieving DIRTY state."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.DIRTY.value,
            }

            state = cet._get_buffer_state(buffer_id)
            assert state == BufferState.DIRTY

    def test_get_buffer_state_default_ready(self):
        """Test default state is READY for backward compatibility."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
            }

            state = cet._get_buffer_state(buffer_id)
            assert state == BufferState.READY

    def test_get_buffer_state_unknown_raises(self):
        """Test error for unknown buffer."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            with pytest.raises(ValueError, match="unknown"):
                cet._get_buffer_state("unknown-buf")


class TestSetBufferState:
    """Test _set_buffer_state method."""

    @staticmethod
    def _create_cet(tmpdir):
        """Create a CodeEmbeddingTool with mocked dependencies."""
        with patch("gigacode.gigacode_tool.Embedder"):
            # Don't mock StateManager - we need real registry management
            with patch.dict(sys.modules, {"gigacode.search_service": None}):
                return CodeEmbeddingTool(work_dir=tmpdir)

    def test_set_buffer_state_ready_to_dirty(self):
        """Test transition from READY to DIRTY."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.READY.value,
                "state_changed_at": time.time(),
            }

            cet._set_buffer_state(buffer_id, BufferState.DIRTY)

            state = cet._get_buffer_state(buffer_id)
            assert state == BufferState.DIRTY
            assert "state_changed_at" in cet._registry[buffer_id]

    def test_set_buffer_state_invalid_transition_raises(self):
        """Test error on invalid transition."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.DIRTY.value,
                "state_changed_at": time.time(),
            }

            # DIRTY -> DIRTY is invalid
            with pytest.raises(ValueError):
                cet._set_buffer_state(buffer_id, BufferState.DIRTY)

    def test_set_buffer_state_persists_to_registry(self):
        """Test state persisted to registry file."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.READY.value,
                "state_changed_at": time.time(),
            }
            cet._save_registry()

            cet._set_buffer_state(buffer_id, BufferState.DIRTY)

            # Verify persisted
            registry_path = Path(tmpdir) / "registry.json"
            assert registry_path.exists()

            persisted = json.loads(registry_path.read_text())
            assert persisted[buffer_id]["state"] == BufferState.DIRTY.value

    def test_set_buffer_state_updates_health_tracker(self):
        """Test state change updates health tracker."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.READY.value,
                "state_changed_at": time.time(),
            }
            cet._health_tracker.register_buffer(buffer_id, BufferState.READY)

            cet._set_buffer_state(buffer_id, BufferState.DIRTY)

            health = cet._health_tracker.get_health_status(buffer_id)
            assert health.state == BufferState.DIRTY


class TestGetBufferHealth:
    """Test _get_buffer_health method."""

    @staticmethod
    def _create_cet(tmpdir):
        """Create a CodeEmbeddingTool with mocked dependencies."""
        with patch("gigacode.gigacode_tool.Embedder"):
            # Don't mock StateManager - we need real registry management
            with patch.dict(sys.modules, {"gigacode.search_service": None}):
                return CodeEmbeddingTool(work_dir=tmpdir)

    def test_get_buffer_health_ok(self):
        """Test health status for clean buffer."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.READY.value,
                "state_changed_at": time.time(),
            }
            cet._health_tracker.register_buffer(buffer_id, BufferState.READY)

            health = cet._get_buffer_health(buffer_id)
            assert health["warning_level"] == "ok"
            assert health["is_rebuilding"] is False

    def test_get_buffer_health_warning_dirty(self):
        """Test health status with dirty files."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.DIRTY.value,
                "state_changed_at": time.time(),
            }
            cet._health_tracker.register_buffer(buffer_id, BufferState.DIRTY)
            cet._health_tracker.update_dirty_file_count(buffer_id, 5)

            health = cet._get_buffer_health(buffer_id)
            assert health["warning_level"] == "warning"
            assert health["dirty_file_count"] == 5

    def test_get_buffer_health_unknown_buffer(self):
        """Test health for unknown buffer."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            health = cet._get_buffer_health("unknown-buf")
            assert "error" in health


class TestStateGuardsIntegration:
    """Integration tests for state guards and operation configuration."""

    @staticmethod
    def _create_cet(tmpdir):
        """Create a CodeEmbeddingTool with mocked dependencies."""
        with patch("gigacode.gigacode_tool.Embedder"):
            # Don't mock StateManager - we need real registry management
            with patch.dict(sys.modules, {"gigacode.search_service": None}):
                return CodeEmbeddingTool(work_dir=tmpdir)

    def test_state_operations_workflow(self):
        """Test complete state operation workflow."""
        with TemporaryDirectory() as tmpdir:
            cet = self._create_cet(tmpdir)

            buffer_id = "buf-123"

            # Initialize buffer in READY state
            cet._registry[buffer_id] = {
                "root": "/fake",
                "state": BufferState.READY.value,
                "state_changed_at": time.time(),
            }
            cet._health_tracker.register_buffer(buffer_id, BufferState.READY)

            # Should allow queries in READY
            allowed, _ = cet._check_state_for_query(buffer_id)
            assert allowed is True

            # Transition to DIRTY
            cet._set_buffer_state(buffer_id, BufferState.DIRTY)

            # Queries ARE allowed in DIRTY state (by design)
            allowed, reason = cet._check_state_for_query(buffer_id)
            assert allowed is True  # State guards allow DIRTY queries

            # Writes should be blocked
            allowed, _ = cet._check_state_for_write(buffer_id)
            assert allowed is False

            # Transition back to READY
            cet._set_buffer_state(buffer_id, BufferState.READY)

            # Should allow queries again
            allowed, _ = cet._check_state_for_query(buffer_id)
            assert allowed is True
