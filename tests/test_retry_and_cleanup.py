"""Tests for retry logic and resource cleanup utilities."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gigacode.cleanup_utils import ResourceTracker, cleanup_on_error
from gigacode.retry_utils import retry_on_exception, retry_on_io_error


def test_retry_on_io_error_success() -> None:
    """Test that successful calls don't retry."""
    call_count = 0

    @retry_on_io_error(max_attempts=3, delay_s=0.01)
    def always_succeeds():
        nonlocal call_count
        call_count += 1
        return "success"

    result = always_succeeds()
    assert result == "success"
    assert call_count == 1, "Should succeed on first try"


def test_retry_on_io_error_retries() -> None:
    """Test that IO errors trigger retries."""
    call_count = 0

    @retry_on_io_error(max_attempts=3, delay_s=0.01, jitter=False)
    def fails_twice_then_succeeds():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise OSError(f"Transient error #{call_count}")
        return "success"

    result = fails_twice_then_succeeds()
    assert result == "success"
    assert call_count == 3, "Should succeed on third try"


def test_retry_on_io_error_exhausts_attempts() -> None:
    """Test that retries are eventually exhausted."""
    call_count = 0

    @retry_on_io_error(max_attempts=2, delay_s=0.01)
    def always_fails():
        nonlocal call_count
        call_count += 1
        raise IOError(f"Persistent error #{call_count}")

    try:
        always_fails()
        raise AssertionError("Should have raised IOError")
    except IOError as exc:
        assert "Persistent error #2" in str(exc)
        assert call_count == 2, "Should attempt max_attempts times"


def test_retry_on_exception() -> None:
    """Test generic exception retry."""
    call_count = 0

    @retry_on_exception(exception_types=ValueError, max_attempts=3, delay_s=0.01)
    def fails_once_then_succeeds():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Transient value error")
        return "success"

    result = fails_once_then_succeeds()
    assert result == "success"
    assert call_count == 2


def test_resource_tracker() -> None:
    """Test resource tracker cleanup."""
    cleanup_order = []

    def cleanup_a():
        cleanup_order.append("a")

    def cleanup_b():
        cleanup_order.append("b")

    with ResourceTracker() as tracker:
        tracker.register("resource_a", cleanup_a)
        tracker.register("resource_b", cleanup_b)
        # Cleanup should happen on exit (no error)

    # Cleanup should happen in LIFO order: b, then a
    assert cleanup_order == ["b", "a"], f"Expected ['b', 'a'], got {cleanup_order}"


def test_resource_tracker_on_error() -> None:
    """Test that resource tracker cleans up even on error."""
    cleanup_order = []

    def cleanup_a():
        cleanup_order.append("a")

    def cleanup_b():
        cleanup_order.append("b")

    try:
        with ResourceTracker() as tracker:
            tracker.register("resource_a", cleanup_a)
            tracker.register("resource_b", cleanup_b)
            raise ValueError("Test error")
    except ValueError:
        pass  # Expected

    # Cleanup should still happen in LIFO order even on error
    assert cleanup_order == ["b", "a"], f"Expected ['b', 'a'], got {cleanup_order}"


def test_cleanup_on_error_context() -> None:
    """Test cleanup_on_error context manager."""
    cleanup_called = []

    def cleanup_resource():
        cleanup_called.append(True)

    try:
        with cleanup_on_error([("test_resource", cleanup_resource)]):
            raise ValueError("Test error")
    except ValueError:
        pass  # Expected

    # Cleanup should have been called
    assert cleanup_called == [True], "Cleanup should be called on error"


if __name__ == "__main__":
    print("Testing retry utilities...")
    test_retry_on_io_error_success()
    print("✓ test_retry_on_io_error_success")

    test_retry_on_io_error_retries()
    print("✓ test_retry_on_io_error_retries")

    test_retry_on_io_error_exhausts_attempts()
    print("✓ test_retry_on_io_error_exhausts_attempts")

    test_retry_on_exception()
    print("✓ test_retry_on_exception")

    test_resource_tracker()
    print("✓ test_resource_tracker")

    test_resource_tracker_on_error()
    print("✓ test_resource_tracker_on_error")

    test_cleanup_on_error_context()
    print("✓ test_cleanup_on_error_context")

    print("\n✅ All retry and cleanup tests passed!")
