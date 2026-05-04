"""GigaCode exception hierarchy.

Provides typed exceptions so callers can distinguish user-facing errors
from internal system failures without parsing status strings.
"""

from __future__ import annotations

from typing import Any


class GigaCodeError(Exception):
    """Base exception for all GigaCode errors."""

    def __init__(
        self,
        message: str,
        buffer_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.buffer_id = buffer_id
        self.context: dict[str, Any] = context or {}

    def __repr__(self) -> str:
        ctx = f", buffer_id={self.buffer_id!r}" if self.buffer_id else ""
        return f"{type(self).__name__}({str(self)!r}{ctx})"


class BufferNotFound(GigaCodeError):
    """Raised when a buffer_id does not exist in the registry."""


class CorruptedMetadata(GigaCodeError):
    """Raised when buffer metadata (chunks, embeddings, snapshot) is missing or corrupt."""


class QueryLimitExceeded(GigaCodeError):
    """Raised when top_k or another query parameter exceeds the allowed limits."""


class GPUMemoryExhausted(GigaCodeError):
    """Raised when GPU VRAM is insufficient for the requested index upload."""
