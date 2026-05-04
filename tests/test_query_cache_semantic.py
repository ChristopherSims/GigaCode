"""Tests for semantic similarity matching in QueryCache."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from gigacode.query_cache import QueryCache


class MockEmbedder:
    """Mock embedder for testing semantic similarity."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        # Pre-defined L2-normalized embeddings for test queries
        # First two are 95% similar when L2-normalized and compared via dot product
        self._embeddings = {
            "find the add function": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "locate addition method": np.array([0.95, 0.312, 0.0, 0.0], dtype=np.float32),  # ~95% similar when normalized
            "search for multiply": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),      # orthogonal
            "find the add function ": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),   # same as first
            "query1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "query2": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "query3": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        }
    
    def encode(self, texts, batch_size=64):
        """Return L2-normalized embeddings for test queries."""
        result = []
        for text in texts:
            # Normalize text for lookup
            key = text.lower().strip()
            if key in self._embeddings:
                emb = self._embeddings[key]
            else:
                # Return default embedding for unknown queries
                emb = np.ones(4, dtype=np.float32) / 2.0
            
            # L2-normalize the embedding
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            result.append(emb)
        
        return np.array(result, dtype=np.float32)


def test_query_cache_without_semantic_matching():
    """Test QueryCache without semantic matching (backward compatible)."""
    cache = QueryCache(maxsize=10)
    
    # Add a result
    cache.set("buf1", "find add", 10, "semantic", {"matches": [1, 2, 3]})
    
    # Exact match should hit
    result = cache.get("buf1", "find add", 10, "semantic")
    assert result == {"matches": [1, 2, 3]}
    
    # Different query should miss
    result = cache.get("buf1", "find multiply", 10, "semantic")
    assert result is None
    
    # Stats should show only text-based hits
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["semantic_hits"] == 0


def test_query_cache_with_semantic_matching():
    """Test QueryCache with semantic similarity matching."""
    embedder = MockEmbedder()
    cache = QueryCache(maxsize=10, embedder=embedder, similarity_threshold=0.95)
    
    # Cache a result
    cache.set("buf1", "find the add function", 10, "semantic", {"matches": [1, 2]})
    
    # Exact match should hit (text-based)
    result = cache.get("buf1", "find the add function", 10, "semantic")
    assert result == {"matches": [1, 2]}
    assert cache.stats()["hits"] == 1
    assert cache.stats()["semantic_hits"] == 0
    
    # Paraphrased query (95% similar) should hit (semantic matching)
    result = cache.get("buf1", "locate addition method", 10, "semantic")
    assert result == {"matches": [1, 2]}
    
    # Check semantic hit was recorded
    stats = cache.stats()
    assert stats["hits"] == 1  # Text-based hits (exact match)
    assert stats["semantic_hits"] == 1  # One semantic hit
    assert stats["hit_rate_percent"] == 100.0  # Both were hits (no misses yet)


def test_query_cache_semantic_mismatch():
    """Test that dissimilar queries don't match semantically."""
    embedder = MockEmbedder()
    cache = QueryCache(maxsize=10, embedder=embedder, similarity_threshold=0.95)
    
    # Cache a result for "find add"
    cache.set("buf1", "find the add function", 10, "semantic", {"matches": [1, 2]})
    
    # Query for "multiply" (0% similar) should not hit
    result = cache.get("buf1", "search for multiply", 10, "semantic")
    assert result is None
    
    # Stats should show a miss
    stats = cache.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1


def test_query_cache_buffer_id_isolation():
    """Test that semantic matching only applies within same buffer_id."""
    embedder = MockEmbedder()
    cache = QueryCache(maxsize=10, embedder=embedder, similarity_threshold=0.95)
    
    # Cache in buffer1
    cache.set("buf1", "find the add function", 10, "semantic", {"matches": [1, 2]})
    
    # Query with same text but different buffer_id should miss
    result = cache.get("buf2", "locate addition method", 10, "semantic")
    assert result is None


def test_query_cache_mode_isolation():
    """Test that semantic matching only applies within same search mode."""
    embedder = MockEmbedder()
    cache = QueryCache(maxsize=10, embedder=embedder, similarity_threshold=0.95)
    
    # Cache in semantic mode
    cache.set("buf1", "find the add function", 10, "semantic", {"matches": [1, 2]})
    
    # Query in hybrid mode should miss even with similar text
    result = cache.get("buf1", "locate addition method", 10, "hybrid")
    assert result is None


def test_query_cache_semantic_disabled():
    """Test disabling semantic matching (similarity_threshold=1.0)."""
    embedder = MockEmbedder()
    cache = QueryCache(maxsize=10, embedder=embedder, similarity_threshold=1.0)
    
    # Cache a result
    cache.set("buf1", "find the add function", 10, "semantic", {"matches": [1, 2]})
    
    # Exact match should hit
    result = cache.get("buf1", "find the add function", 10, "semantic")
    assert result == {"matches": [1, 2]}
    
    # Even 95% similar query should miss (semantic matching disabled)
    result = cache.get("buf1", "locate addition method", 10, "semantic")
    assert result is None


def test_query_cache_lru_eviction_with_semantic():
    """Test that LRU eviction works correctly with semantic embeddings."""
    embedder = MockEmbedder()
    cache = QueryCache(maxsize=2, embedder=embedder, similarity_threshold=0.95)
    
    # Fill cache
    cache.set("buf1", "query1", 10, "semantic", {"result": 1})
    cache.set("buf1", "query2", 10, "semantic", {"result": 2})
    
    # Cache is full
    assert cache.stats()["size"] == 2
    
    # Add new query, should evict oldest
    cache.set("buf1", "query3", 10, "semantic", {"result": 3})
    assert cache.stats()["size"] == 2
    
    # First query should be evicted
    assert cache.get("buf1", "query1", 10, "semantic") is None


def test_query_cache_stats_with_semantic():
    """Test that stats correctly track semantic hits."""
    embedder = MockEmbedder()
    cache = QueryCache(maxsize=10, embedder=embedder, similarity_threshold=0.95)
    
    # Make some hits and misses
    cache.set("buf1", "find the add function", 10, "semantic", {"matches": [1]})
    
    # Text hit
    cache.get("buf1", "find the add function", 10, "semantic")
    
    # Semantic hit
    cache.get("buf1", "locate addition method", 10, "semantic")
    
    # Miss
    cache.get("buf1", "search for multiply", 10, "semantic")
    
    stats = cache.stats()
    assert stats["hits"] == 1  # Text-based hit
    assert stats["misses"] == 1  # Only actual miss
    assert stats["semantic_hits"] == 1  # One semantic hit
    
    # Total hit rate should be (1 text + 1 semantic) / (1 + 1 + 1) = 66.67%
    assert stats["hit_rate_percent"] == 66.67
    
    # Semantic hit rate should be 1 / (1 + 1 + 1) = 33.33%
    assert stats["semantic_hit_rate_percent"] == 33.33


def test_query_cache_clear_with_semantic():
    """Test that clear() removes embeddings as well."""
    embedder = MockEmbedder()
    cache = QueryCache(maxsize=10, embedder=embedder, similarity_threshold=0.95)
    
    cache.set("buf1", "find the add function", 10, "semantic", {"matches": [1]})
    assert cache.stats()["size"] == 1
    
    cache.clear()
    assert cache.stats()["size"] == 0
    
    # Semantic hit should not occur after clear
    result = cache.get("buf1", "locate addition method", 10, "semantic")
    assert result is None


def test_query_cache_invalidate_buffer_with_semantic():
    """Test that invalidate_buffer removes embeddings too."""
    embedder = MockEmbedder()
    cache = QueryCache(maxsize=10, embedder=embedder, similarity_threshold=0.95)
    
    cache.set("buf1", "find the add function", 10, "semantic", {"matches": [1]})
    cache.set("buf2", "query2", 10, "semantic", {"matches": [2]})
    
    assert cache.stats()["size"] == 2
    
    # Invalidate buffer1
    cache.invalidate_buffer("buf1")
    
    # buf2 should remain
    assert cache.stats()["size"] == 1
    assert cache.get("buf2", "query2", 10, "semantic") == {"matches": [2]}
    
    # buf1 should be gone
    assert cache.get("buf1", "find the add function", 10, "semantic") is None


def test_query_cache_embedding_failure_graceful():
    """Test that QueryCache gracefully handles embedding failures."""
    
    class FailingEmbedder:
        def encode(self, texts, batch_size=64):
            raise RuntimeError("Embedding service unavailable")
    
    cache = QueryCache(maxsize=10, embedder=FailingEmbedder(), similarity_threshold=0.95)
    
    # Should not crash even though embedder fails
    cache.set("buf1", "query1", 10, "semantic", {"result": 1})
    
    # Should still work (falls back to text-based)
    result = cache.get("buf1", "query1", 10, "semantic")
    assert result == {"result": 1}
    
    # Different query should miss (no semantic fallback)
    result = cache.get("buf1", "query2", 10, "semantic")
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
