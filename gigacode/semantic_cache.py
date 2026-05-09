"""Semantic cache for query embeddings and search results.

Caches query embeddings and similar queries to accelerate repeated searches.
Detects paraphrased queries using semantic similarity to provide cache hits
even when queries are slightly different.
"""

import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


__all__ = [
    "CacheEntry",
    "SemanticQueryCache",
    "SearchResultCache",
]


@dataclass
class CacheEntry:
    """Entry in semantic cache."""
    query: str
    embedding: np.ndarray
    results: Any
    timestamp: float = field(default_factory=time.time)
    hits: int = 0
    
    def refresh(self) -> None:
        """Update timestamp and increment hit counter."""
        self.timestamp = time.time()
        self.hits += 1


class SemanticQueryCache:
    """Cache for query embeddings and search results.
    
    Features:
    - LRU eviction for memory management
    - Semantic similarity matching for paraphrased queries
    - Configurable cache size and similarity threshold
    - Hit/miss statistics
    """
    
    def __init__(
        self,
        max_entries: int = 1000,
        similarity_threshold: float = 0.95,
        embedder: Optional[Any] = None,
    ):
        """Initialize semantic cache.
        
        Args:
            max_entries: Maximum cache entries (LRU eviction if exceeded)
            similarity_threshold: Cosine similarity threshold for query matching
            embedder: Embedder for computing query embeddings
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._similarity_threshold = similarity_threshold
        self._embedder = embedder
        self._hits = 0
        self._misses = 0
    
    def _compute_embedding(self, query: str) -> np.ndarray:
        """Compute embedding for query.
        
        Args:
            query: Search query string
        
        Returns:
            Embedding vector (1, embedding_dim)
        """
        if self._embedder is None:
            raise ValueError("Embedder required for query caching")
        
        embedding = self._embedder.encode([query])
        return np.asarray(embedding[0], dtype=np.float32)
    
    def _compute_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
        
        Returns:
            Cosine similarity (0-1)
        """
        # Normalize
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        # Dot product
        return float(np.dot(emb1_norm, emb2_norm))
    
    def _get_query_key(self, query: str) -> str:
        """Get cache key for query.
        
        Args:
            query: Search query
        
        Returns:
            Cache key (normalized query hash)
        """
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        compute_embedding: bool = True,
    ) -> Optional[Tuple[Any, bool]]:
        """Get cached result for query.
        
        Searches for exact match first, then semantic matches.
        
        Args:
            query: Search query
            compute_embedding: Whether to compute embedding for semantic search
        
        Returns:
            Tuple of (cached_results, was_exact_match) or None
        """
        query_key = self._get_query_key(query)
        
        # Exact match
        if query_key in self._cache:
            entry = self._cache[query_key]
            entry.refresh()
            # Move to end (LRU)
            self._cache.move_to_end(query_key)
            self._hits += 1
            logger.debug(f"Cache HIT (exact): {query[:50]}")
            return entry.results, True
        
        # Semantic match
        if compute_embedding:
            try:
                query_emb = self._compute_embedding(query)
                
                # Search for similar queries
                for cache_key, entry in list(self._cache.items()):
                    similarity = self._compute_similarity(query_emb, entry.embedding)
                    
                    if similarity >= self._similarity_threshold:
                        entry.refresh()
                        self._cache.move_to_end(cache_key)
                        self._hits += 1
                        logger.debug(
                            f"Cache HIT (semantic, {similarity:.3f}): {query[:50]}"
                        )
                        return entry.results, False
            except (RuntimeError, ValueError, TypeError) as e:
                logger.warning(f"Failed to compute embedding for semantic cache: {e}")
        
        self._misses += 1
        logger.debug(f"Cache MISS: {query[:50]}")
        return None
    
    def put(
        self,
        query: str,
        results: Any,
    ) -> None:
        """Cache result for query.
        
        Args:
            query: Search query
            results: Search results to cache
        """
        query_key = self._get_query_key(query)
        
        # Compute embedding
        try:
            embedding = self._compute_embedding(query)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to compute embedding for caching: {e}")
            return
        
        # Create entry
        entry = CacheEntry(
            query=query,
            embedding=embedding,
            results=results,
        )
        
        # Add to cache
        self._cache[query_key] = entry
        self._cache.move_to_end(query_key)
        
        # Evict oldest if necessary
        if len(self._cache) > self._max_entries:
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key)
            logger.debug(f"Cache evicted (LRU): oldest entry removed")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Semantic cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with hits, misses, hit rate, size
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": f"{hit_rate:.1f}%",
            "size": len(self._cache),
            "max_entries": self._max_entries,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"SemanticQueryCache("
            f"size={stats['size']}/{self._max_entries}, "
            f"hit_rate={stats['hit_rate']}, "
            f"threshold={self._similarity_threshold})"
        )


class SearchResultCache:
    """Cache for search results (semantic + hybrid).
    
    Avoids re-running searches for identical or similar queries.
    """
    
    def __init__(
        self,
        embedder: Optional[Any] = None,
        max_entries: int = 500,
    ):
        """Initialize search result cache.
        
        Args:
            embedder: Embedder for computing query embeddings
            max_entries: Maximum cache entries
        """
        self._cache: Dict[str, Dict[str, Any]] = OrderedDict()
        self._embedder = embedder
        self._max_entries = max_entries
        self._semantic_cache = SemanticQueryCache(
            max_entries=max_entries,
            embedder=embedder,
        )
    
    def get_search_cache_key(
        self,
        buffer_id: str,
        query: str,
        search_type: str,
        top_k: int = 5,
    ) -> str:
        """Get cache key for search.
        
        Args:
            buffer_id: Buffer identifier
            query: Search query
            search_type: Type of search ("semantic", "hybrid", "lexical")
            top_k: Number of results
        
        Returns:
            Cache key
        """
        key_str = f"{buffer_id}#{query}#{search_type}#{top_k}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_search_result(
        self,
        buffer_id: str,
        query: str,
        search_type: str,
        top_k: int = 5,
    ) -> Optional[Any]:
        """Get cached search result.
        
        Args:
            buffer_id: Buffer identifier
            query: Search query
            search_type: Type of search
            top_k: Number of results
        
        Returns:
            Cached result or None
        """
        cache_key = self.get_search_cache_key(buffer_id, query, search_type, top_k)
        
        if cache_key in self._cache:
            result = self._cache[cache_key]
            result["cached"] = True
            result["cache_hits"] = result.get("cache_hits", 0) + 1
            logger.debug(f"Search result cache HIT: {search_type}")
            return result
        
        logger.debug(f"Search result cache MISS: {search_type}")
        return None
    
    def put_search_result(
        self,
        buffer_id: str,
        query: str,
        search_type: str,
        results: Any,
        top_k: int = 5,
    ) -> None:
        """Cache search result.
        
        Args:
            buffer_id: Buffer identifier
            query: Search query
            search_type: Type of search
            results: Search results
            top_k: Number of results
        """
        cache_key = self.get_search_cache_key(buffer_id, query, search_type, top_k)
        
        self._cache[cache_key] = {
            "results": results,
            "query": query,
            "search_type": search_type,
            "timestamp": time.time(),
            "cache_hits": 0,
        }
        
        # Evict oldest if necessary
        if len(self._cache) > self._max_entries:
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key)
    
    def clear(self) -> None:
        """Clear all caches."""
        self._cache.clear()
        self._semantic_cache.clear()
        logger.info("Search result cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "search_cache_size": len(self._cache),
            "max_entries": self._max_entries,
            "semantic_cache": self._semantic_cache.get_stats(),
        }
