"""Public embedding-provider interface.

Applications may supply their own embedding backend — a locally hosted model,
a remote provider, or a lightweight test double — instead of the bundled
SentenceTransformers default.  The tool only relies on the small surface
defined here, which lets callers keep control of batching, credentials, and
provider configuration.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "EmbeddingProvider",
    "validate_embedding_provider",
]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal contract for an embedding backend.

    Only :attr:`embedding_dim` and :meth:`encode` are required.  A single-text
    :meth:`embed` helper is optional; the tool provides one when wrapping an
    encode-only provider.
    """

    @property
    def embedding_dim(self) -> int:
        """Dimension of the vectors returned by :meth:`encode`."""
        ...

    def encode(self, texts: list[str], batch_size: int = ...) -> np.ndarray:
        """Return an ``(len(texts), embedding_dim)`` float array."""
        ...


def validate_embedding_provider(provider: Any) -> None:
    """Validate that *provider* satisfies :class:`EmbeddingProvider`.

    Raises:
        TypeError: If the provider is missing required attributes or its
            reported dimension is not a positive integer.
    """
    if not hasattr(provider, "encode") or not callable(provider.encode):
        raise TypeError("Embedding provider must define a callable encode(texts) method")
    if not hasattr(provider, "embedding_dim"):
        raise TypeError("Embedding provider must expose an integer embedding_dim property")

    dim = provider.embedding_dim
    if not isinstance(dim, int) or dim <= 0:
        raise TypeError(f"Embedding provider embedding_dim must be a positive integer; got {dim!r}")
