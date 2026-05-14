"""Session-scoped LRU cache for LLM-assisted type inference confidence scores.

Only LLM-assisted confidence scores are cached; AST inference (~1-5ms)
is too cheap to warrant caching. Invalidation is write-driven: any file
modification evicts all cached entries for symbols defined in that file.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "InferredType",
    "TypeInferenceCache",
]


@dataclass
class InferredType:
    """A cached type inference result."""

    symbol_name: str
    file: str
    parameters: list[dict[str, str]]
    return_type: Optional[str]
    confidence: float
    method: str  # "llm" or "ast"
    reasoning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class TypeInferenceCache:
    """Session-scoped LRU cache for type inference confidence scores.

    Rules:
    - Scope: session-scoped per buffer_id. No persistence, no cross-session sharing.
    - Capacity: LRU eviction at 500 symbols.
    - Invalidation: write_code on a file evicts all cached entries for that file.
    - TTL: None. Session scope + write-invalidation removes the need.
    - Passthrough: cache miss = run inference immediately.
    - AST entries are NOT cached (too cheap to warrant it).
    """

    DEFAULT_MAX_ENTRIES = 500

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._cache: OrderedDict[str, InferredType] = OrderedDict()
        self._file_to_symbols: dict[str, set[str]] = {}
        self._max_entries = max_entries

    def _make_key(self, buffer_id: str, symbol_name: str) -> str:
        return f"{buffer_id}:{symbol_name}"

    def get(self, buffer_id: str, symbol_name: str) -> Optional[InferredType]:
        """Return cached inference or None."""
        key = self._make_key(buffer_id, symbol_name)
        entry = self._cache.get(key)
        if entry is not None:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry
        return None

    def put(self, buffer_id: str, inferred: InferredType) -> None:
        """Cache result. Evict LRU entry if at capacity. Track file->symbol mapping."""
        key = self._make_key(buffer_id, inferred.symbol_name)

        if key in self._cache:
            # Update existing entry
            old_entry = self._cache[key]
            if old_entry.file in self._file_to_symbols:
                self._file_to_symbols[old_entry.file].discard(key)
            self._cache[key] = inferred
            self._cache.move_to_end(key)
        else:
            # Evict LRU if at capacity
            while len(self._cache) >= self._max_entries:
                evicted_key, _ = self._cache.popitem(last=False)
                # Clean up file_to_symbols index
                for symbols_set in self._file_to_symbols.values():
                    symbols_set.discard(evicted_key)

            self._cache[key] = inferred

        # Track file -> symbol mapping for invalidation
        file = inferred.file
        if file not in self._file_to_symbols:
            self._file_to_symbols[file] = set()
        self._file_to_symbols[file].add(key)

    def invalidate_file(self, file: str) -> int:
        """Evict all entries for symbols defined in a modified file.

        Returns the number of entries evicted.
        """
        keys_to_evict = self._file_to_symbols.pop(file, set())
        count = 0
        for key in keys_to_evict:
            if key in self._cache:
                del self._cache[key]
                count += 1
        if count > 0:
            logger.debug(f"Invalidated {count} type cache entries for file: {file}")
        return count

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._file_to_symbols.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "files_tracked": len(self._file_to_symbols),
        }
