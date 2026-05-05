"""LRU (Least Recently Used) cache implementation.

Provides a bounded dictionary that automatically evicts the least recently used
item when the maximum size is exceeded.
"""

from collections import OrderedDict
from typing import Any

from gigacode.json_logger import StructuredJsonLogger

json_logger = StructuredJsonLogger(__name__)


class LRUDict(OrderedDict):
    """Bounded LRU dict. Evicts least-recently-used item when max_size exceeded.
    
    Usage:
        cache = LRUDict(max_size=10)
        cache['key'] = value  # Auto-evicts oldest if size exceeds max_size
        val = cache['key']    # Moves 'key' to end (most recently used)
    """

    def __init__(self, max_size: int = 10) -> None:
        super().__init__()
        self.max_size = max_size

    def __getitem__(self, key: str) -> Any:
        """Get item and move to end (mark as recently used)."""
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item and move to end. Evict LRU item if size exceeded."""
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            # Pop the oldest (first) item
            oldest_key = next(iter(self))
            self.pop(oldest_key)
            json_logger.debug(
                operation='lru_eviction',
                details={'removed_key': oldest_key, 'new_size': len(self)},
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get with default; moves to end if found."""
        if key in self:
            return self[key]
        return default

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": len(self),
            "maxsize": self.max_size,
            "utilization": len(self) / self.max_size if self.max_size > 0 else 0
        }
