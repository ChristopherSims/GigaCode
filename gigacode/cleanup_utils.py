"""Resource cleanup helpers for graceful error handling."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Iterable

logger = logging.getLogger(__name__)


__all__ = [
    "cleanup_on_error",
    "ensure_closed",
    "ResourceTracker",
]


@contextmanager
def cleanup_on_error(
    cleanup_funcs: Iterable[tuple[str, callable]] | None = None,
) -> Generator[None, None, None]:
    """Context manager that ensures cleanup functions run on error.

    Args:
        cleanup_funcs: List of (name, callable) tuples to run on error.
                      Called in reverse order (LIFO).

    Example:
        with cleanup_on_error([("close_file", f.close), ("free_gpu", gpu.free)]):
            # Do risky operations...
            # If error occurs, cleanup_funcs are called in reverse order
            ...
    """
    if cleanup_funcs is None:
        cleanup_funcs = []

    cleanup_list = list(cleanup_funcs)  # Convert to list for iteration

    try:
        yield
    except (OSError, RuntimeError, ValueError):
        # Run cleanups in reverse order (LIFO)
        for name, cleanup_func in reversed(cleanup_list):
            try:
                cleanup_func()
                logger.debug("Cleanup completed: %s", name)
            except (OSError, RuntimeError) as cleanup_exc:
                logger.error("Cleanup failed for %s: %s", name, cleanup_exc)
        raise


@contextmanager
def ensure_closed(file_handle: Any) -> Generator[None, None, None]:
    """Context manager that ensures file handle is closed on error or exit.

    Args:
        file_handle: File-like object with a close() method.

    Example:
        with open("file.txt") as f:
            with ensure_closed(f):
                # Do risky operations...
                # File will be closed even if exception occurs
                ...
    """
    try:
        yield
    finally:
        try:
            file_handle.close()
            logger.debug("File handle closed: %s", getattr(file_handle, "name", "unknown"))
        except (OSError, AttributeError) as exc:
            logger.error("Failed to close file handle: %s", exc)


class ResourceTracker:
    """Track resources that need cleanup on error."""

    def __init__(self) -> None:
        """Initialize resource tracker."""
        self._resources: list[tuple[str, callable]] = []

    def register(self, name: str, cleanup_func: callable) -> None:
        """Register a resource with cleanup function.

        Args:
            name: Name of the resource (for logging).
            cleanup_func: Callable to run on cleanup.
        """
        self._resources.append((name, cleanup_func))
        logger.debug("Registered resource: %s", name)

    def cleanup(self) -> None:
        """Run all cleanup functions in reverse order (LIFO)."""
        for name, cleanup_func in reversed(self._resources):
            try:
                cleanup_func()
                logger.debug("Cleanup completed: %s", name)
            except (OSError, RuntimeError, AttributeError) as exc:
                logger.error("Cleanup failed for %s: %s", name, exc)
        self._resources.clear()

    def __enter__(self) -> ResourceTracker:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and cleanup all resources."""
        self.cleanup()
