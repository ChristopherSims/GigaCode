"""LRU cache for repeated search queries.

Caches final result dicts keyed by (buffer_id, query, top_k, mode).  Stored
in memory only — not persisted across process restarts.
"""

from __future__ import annotations

from typing import Any


class QueryCache:
    """Simple in-memory LRU cache with a max size."""

    def __init__(self, maxsize: int = 128) -> None:
        self._maxsize = maxsize
        self._data: dict[tuple[str, ...], Any] = {}
        self._order: list[tuple[str, ...]] = []

    def _make_key(self, buffer_id: str, query: str, top_k: int, mode: str) -> tuple[str, str, int, str]:
        return (buffer_id, query.lower().strip(), top_k, mode)

    def get(self, buffer_id: str, query: str, top_k: int, mode: str) -> Any | None:
        key = self._make_key(buffer_id, query, top_k, mode)
        if key in self._data:
            # Move to end (most recently used)
            self._order.remove(key)
            self._order.append(key)
            return self._data[key]
        return None

    def set(self, buffer_id: str, query: str, top_k: int, mode: str, value: Any) -> None:
        key = self._make_key(buffer_id, query, top_k, mode)
        if key in self._data:
            self._order.remove(key)
        self._order.append(key)
        self._data[key] = value
        # Evict oldest if over capacity
        while len(self._order) > self._maxsize:
            oldest = self._order.pop(0)
            self._data.pop(oldest, None)

    def invalidate_buffer(self, buffer_id: str) -> None:
        """Drop all entries for a given buffer_id."""
        keys_to_drop = [k for k in self._order if k[0] == buffer_id]
        for k in keys_to_drop:
            self._order.remove(k)
            self._data.pop(k, None)
