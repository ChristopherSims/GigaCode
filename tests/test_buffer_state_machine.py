"""Buffer State Machine Tests.

Verifies state transitions and lifecycle management.
"""
# CRITICAL: Initialize sklearn FIRST before any gigacode imports
import types
try:
    import sklearn
    if getattr(sklearn, "__spec__", None) is None:
        sklearn.__spec__ = types.ModuleSpec("sklearn", getattr(sklearn, "__file__", None))
except Exception:
    pass


import pytest

from gigacode.buffer_state import BufferState, BufferStateTransition


class TestBufferState:
    """Test BufferState enumeration."""

    def test_buffer_state_values(self):
        """Test that state values are correct."""
        assert BufferState.READY.value == "ready"
        assert BufferState.DIRTY.value == "dirty"
        assert BufferState.REBUILDING.value == "rebuilding"

    def test_buffer_state_str(self):
        """Test string representation."""
        assert str(BufferState.READY) == "ready"
        assert str(BufferState.DIRTY) == "dirty"
        assert str(BufferState.REBUILDING) == "rebuilding"


class TestBufferStateTransition:
    """Test state transition validation."""

    def test_valid_ready_to_dirty(self):
        """Test READY â†’ DIRTY transition."""
        assert BufferStateTransition.is_valid(BufferState.READY, BufferState.DIRTY)

    def test_valid_ready_to_rebuilding(self):
        """Test READY â†’ REBUILDING transition."""
        assert BufferStateTransition.is_valid(BufferState.READY, BufferState.REBUILDING)

    def test_valid_dirty_to_ready(self):
        """Test DIRTY â†’ READY transition."""
        assert BufferStateTransition.is_valid(BufferState.DIRTY, BufferState.READY)

    def test_valid_dirty_to_rebuilding(self):
        """Test DIRTY â†’ REBUILDING transition."""
        assert BufferStateTransition.is_valid(BufferState.DIRTY, BufferState.REBUILDING)

    def test_valid_rebuilding_to_ready(self):
        """Test REBUILDING â†’ READY transition."""
        assert BufferStateTransition.is_valid(BufferState.REBUILDING, BufferState.READY)

    def test_invalid_ready_to_ready(self):
        """Test READY â†’ READY is invalid."""
        assert not BufferStateTransition.is_valid(BufferState.READY, BufferState.READY)

    def test_invalid_dirty_to_dirty(self):
        """Test DIRTY â†’ DIRTY is invalid."""
        assert not BufferStateTransition.is_valid(BufferState.DIRTY, BufferState.DIRTY)

    def test_invalid_rebuilding_to_rebuilding(self):
        """Test REBUILDING â†’ REBUILDING is invalid."""
        assert not BufferStateTransition.is_valid(BufferState.REBUILDING, BufferState.REBUILDING)

    def test_invalid_rebuilding_to_dirty(self):
        """Test REBUILDING â†’ DIRTY is invalid."""
        assert not BufferStateTransition.is_valid(BufferState.REBUILDING, BufferState.DIRTY)

    def test_validate_or_raise_valid(self):
        """Test validate_or_raise with valid transition."""
        # Should not raise
        BufferStateTransition.validate_or_raise(BufferState.READY, BufferState.DIRTY)

    def test_validate_or_raise_invalid(self):
        """Test validate_or_raise with invalid transition."""
        with pytest.raises(ValueError) as exc_info:
            BufferStateTransition.validate_or_raise(BufferState.READY, BufferState.READY)

        assert "Invalid state transition" in str(exc_info.value)
        assert "ready \u2192 ready" in str(exc_info.value)

    def test_describe(self):
        """Test state machine description."""
        desc = BufferStateTransition.describe()
        assert "Buffer State Machine:" in desc
        assert "ready \u2192" in desc
        assert "dirty \u2192" in desc
        assert "rebuilding \u2192" in desc


class TestBufferStateLifecycle:
    """Test typical buffer state lifecycles."""

    def test_normal_workflow(self):
        """Test normal workflow: READY â†’ DIRTY â†’ READY."""
        # Start in READY
        state = BufferState.READY
        assert state == BufferState.READY

        # Write code â†’ DIRTY
        assert BufferStateTransition.is_valid(state, BufferState.DIRTY)
        state = BufferState.DIRTY
        assert state == BufferState.DIRTY

        # Commit â†’ READY
        assert BufferStateTransition.is_valid(state, BufferState.READY)
        state = BufferState.READY
        assert state == BufferState.READY

    def test_rebuild_during_ready(self):
        """Test rebuild can happen from READY state."""
        state = BufferState.READY

        # Rebuild from READY
        assert BufferStateTransition.is_valid(state, BufferState.REBUILDING)
        state = BufferState.REBUILDING

        # Back to READY
        assert BufferStateTransition.is_valid(state, BufferState.READY)
        state = BufferState.READY

    def test_rebuild_during_dirty(self):
        """Test rebuild can happen from DIRTY state."""
        state = BufferState.DIRTY

        # Rebuild from DIRTY
        assert BufferStateTransition.is_valid(state, BufferState.REBUILDING)
        state = BufferState.REBUILDING

        # Back to READY (not DIRTY)
        assert BufferStateTransition.is_valid(state, BufferState.READY)
        assert not BufferStateTransition.is_valid(state, BufferState.DIRTY)

    def test_discard_clears_dirty(self):
        """Test discard transitions DIRTY â†’ READY."""
        state = BufferState.DIRTY

        # Discard pending changes
        assert BufferStateTransition.is_valid(state, BufferState.READY)
        state = BufferState.READY
        assert state == BufferState.READY

    def test_invalid_sequences(self):
        """Test that invalid sequences are blocked."""
        # Can't go from REBUILDING to DIRTY
        assert not BufferStateTransition.is_valid(BufferState.REBUILDING, BufferState.DIRTY)

        # Can't stay in same state
        assert not BufferStateTransition.is_valid(BufferState.READY, BufferState.READY)
        assert not BufferStateTransition.is_valid(BufferState.DIRTY, BufferState.DIRTY)

