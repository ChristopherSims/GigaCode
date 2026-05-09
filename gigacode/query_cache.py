"""LRU cache for repeated search queries.

Caches final result dicts keyed by (buffer_id, query, top_k, mode).  Stored
in memory only — not persisted across process restarts.

Supports semantic similarity matching to catch paraphrased queries
(e.g., "find the add function" vs "locate addition method").
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gigacode.embedder import Embedder


__all__ = [
    "QueryCache",
]


class QueryCache:
    """Simple in-memory LRU cache with optional semantic similarity matching.
    
    Attributes:
        maxsize: Maximum number of cached queries.
        similarity_threshold: Minimum cosine similarity (0-1) to consider queries equivalent.
                             Default 0.95. Set to 1.0 to disable semantic matching.
    """

    def __init__(self, maxsize: int = 128, embedder: Embedder | None = None, 
                 similarity_threshold: float = 0.95) -> None:
        """Initialize QueryCache.
        
        Args:
            maxsize: Maximum number of cached entries.
            embedder: Optional Embedder for semantic similarity matching. If provided,
                     cache will check semantic similarity before returning miss.
            similarity_threshold: Cosine similarity threshold for semantic matching (0-1).
                                 Default 0.95. Set to 1.0 to disable semantic matching.
        """
        self._maxsize = maxsize
        self._data: dict[tuple[str, ...], Any] = {}
        self._order: list[tuple[str, ...]] = []
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0
        
        # Semantic similarity configuration
        self._embedder = embedder
        self._similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self._query_embeddings: dict[tuple[str, ...], np.ndarray] = {}

    def _make_key(self, buffer_id: str, query: str, top_k: int, mode: str) -> tuple[str, str, int, str]:
        return (buffer_id, query.lower().strip(), top_k, mode)

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized embeddings.
        
        For L2-normalized vectors, cosine similarity = dot product.
        
        Args:
            emb1: First embedding (1D array).
            emb2: Second embedding (1D array).
            
        Returns:
            Cosine similarity score (0-1).
        """
        # Both embeddings should be L2-normalized from Embedder.encode()
        similarity = float(np.dot(emb1, emb2))
        # Clamp to [0, 1] in case of numerical errors
        return max(0.0, min(1.0, similarity))

    def _find_semantically_similar(self, query: str, buffer_id: str, top_k: int, 
                                   mode: str, query_embedding: np.ndarray) -> tuple[str, str, int, str] | None:
        """Find cached entry with semantically similar query.
        
        Args:
            query: Original query string.
            buffer_id: Buffer ID (must match for similarity check).
            top_k: Top-k results (must match).
            mode: Search mode (must match).
            query_embedding: Embedding of the query.
            
        Returns:
            Cache key of semantically similar match, or None if no match found.
        """
        for key in self._order:
            key_buffer_id, cached_query, key_top_k, key_mode = key
            
            # Only compare within same buffer_id, top_k, and mode
            if key_buffer_id != buffer_id or key_top_k != top_k or key_mode != mode:
                continue
            
            # Skip exact text match (already checked in get())
            if key in self._data and key[1] == query.lower().strip():
                continue
            
            # Check semantic similarity
            if key in self._query_embeddings:
                cached_embedding = self._query_embeddings[key]
                similarity = self._compute_similarity(query_embedding, cached_embedding)
                if similarity >= self._similarity_threshold:
                    return key
        
        return None

    def get(self, buffer_id: str, query: str, top_k: int, mode: str) -> Any | None:
        """Get cached result, checking exact match first, then semantic similarity.
        
        Args:
            buffer_id: Code buffer ID.
            query: Search query string.
            top_k: Number of results requested.
            mode: Search mode (e.g., "semantic", "hybrid", "lexical").
            
        Returns:
            Cached result dict, or None if not found.
        """
        key = self._make_key(buffer_id, query, top_k, mode)
        
        # Check exact match first
        if key in self._data:
            self._order.remove(key)
            self._order.append(key)
            self._hits += 1
            return self._data[key]
        
        # Check semantic similarity if embedder available
        if self._embedder is not None and self._similarity_threshold < 1.0:
            try:
                # Generate embedding for query
                query_emb = self._embedder.encode([query.lower().strip()], batch_size=1)
                if query_emb.shape[0] > 0:
                    query_embedding = query_emb[0]
                    similar_key = self._find_semantically_similar(query, buffer_id, top_k, mode, query_embedding)
                    if similar_key is not None:
                        # Move to end (mark recent) and return cached result
                        self._order.remove(similar_key)
                        self._order.append(similar_key)
                        self._semantic_hits += 1
                        return self._data[similar_key]
            except (RuntimeError, ValueError, TypeError):
                # Silently fall back to text-based key on embedding failure
                pass
        
        self._misses += 1
        return None

    def set(self, buffer_id: str, query: str, top_k: int, mode: str, value: Any) -> None:
        """Cache a search result and optionally its query embedding.
        
        Args:
            buffer_id: Code buffer ID.
            query: Search query string.
            top_k: Number of results requested.
            mode: Search mode.
            value: Result to cache.
        """
        key = self._make_key(buffer_id, query, top_k, mode)
        if key in self._data:
            self._order.remove(key)
        
        self._order.append(key)
        self._data[key] = value
        
        # Store query embedding for semantic similarity matching
        if self._embedder is not None and self._similarity_threshold < 1.0:
            try:
                query_emb = self._embedder.encode([query.lower().strip()], batch_size=1)
                if query_emb.shape[0] > 0:
                    self._query_embeddings[key] = query_emb[0]
            except (RuntimeError, ValueError, TypeError):
                # Silently skip embedding storage on failure
                pass
        
        # Evict oldest if over capacity
        while len(self._order) > self._maxsize:
            oldest = self._order.pop(0)
            self._data.pop(oldest, None)
            self._query_embeddings.pop(oldest, None)

    def invalidate_buffer(self, buffer_id: str) -> None:
        """Drop all entries for a given buffer_id."""
        keys_to_drop = [k for k in self._order if k[0] == buffer_id]
        for k in keys_to_drop:
            self._order.remove(k)
            self._data.pop(k, None)
            self._query_embeddings.pop(k, None)

    def clear(self) -> None:
        """Drop all cached entries."""
        self._data.clear()
        self._order.clear()
        self._query_embeddings.clear()
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0

    def stats(self) -> dict[str, int | float]:
        """Return cache statistics: size, max size, hits, misses, semantic hits.
        
        Returns:
            Dict with cache statistics including hit rate and semantic hit rate.
        """
        total_hits = self._hits + self._semantic_hits
        total_requests = total_hits + self._misses
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0
        semantic_rate = ((self._semantic_hits / total_requests) * 100) if total_requests > 0 else 0.0
        
        return {
            "size": len(self._data),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "semantic_hits": self._semantic_hits,
            "semantic_hit_rate_percent": round(semantic_rate, 2),
        }
