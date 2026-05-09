"""Retry logic for transient I/O failures."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


__all__ = [
    "retry_on_io_error",
    "retry_on_exception",
]


def retry_on_io_error(
    max_attempts: int = 3,
    delay_s: float = 0.5,
    backoff: float = 2.0,
    jitter: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry on transient I/O errors (timeout, connection, resource temp unavailable).

    Args:
        max_attempts: Max number of attempts (default 3).
        delay_s: Initial delay between retries in seconds (default 0.5).
        backoff: Exponential backoff multiplier (default 2.0 → 0.5s, 1s, 2s, ...).
        jitter: Add random jitter to retry delays (default True) to avoid thundering herd.

    Returns:
        Decorator that retries on transient I/O errors.

    Example:
        @retry_on_io_error(max_attempts=3, delay_s=0.5)
        def read_large_file(path: str) -> str:
            with open(path) as f:
                return f.read()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            current_delay = delay_s

            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except (OSError, IOError, TimeoutError) as exc:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(
                            "Giving up on %s after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            exc,
                        )
                        raise

                    # Add jitter to avoid thundering herd
                    if jitter:
                        import random

                        wait_time = current_delay * (0.5 + random.random())
                    else:
                        wait_time = current_delay

                    logger.warning(
                        "Transient I/O error in %s (attempt %d/%d): %s. Retrying in %.2fs...",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    current_delay *= backoff

        return wrapper

    return decorator


def retry_on_exception(
    exception_types: type | tuple[type, ...] = Exception,
    max_attempts: int = 3,
    delay_s: float = 0.1,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Generic retry decorator for any exception type.

    Args:
        exception_types: Exception class or tuple of classes to catch.
        max_attempts: Max number of attempts.
        delay_s: Delay between retries in seconds.

    Returns:
        Decorator that retries on specified exceptions.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exception_types as exc:
                    if attempt >= max_attempts:
                        logger.error(
                            "Giving up on %s after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            exc,
                        )
                        raise
                    logger.debug(
                        "%s failed (attempt %d/%d), retrying in %.2fs: %s",
                        func.__name__,
                        attempt,
                        max_attempts,
                        delay_s,
                        exc,
                    )
                    time.sleep(delay_s)

        return wrapper

    return decorator
