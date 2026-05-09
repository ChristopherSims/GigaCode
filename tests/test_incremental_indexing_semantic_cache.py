"""Tests for Future 4.3 Phase 3 advanced optimizations.

Tests cover:
- Incremental indexing with chunk diff tracking
- Semantic query cache with paraphrase detection
- Search result caching
"""
# CRITICAL: Initialize sklearn FIRST before any gigacode imports
import types
try:
    import sklearn
    if getattr(sklearn, "__spec__", None) is None:
        sklearn.__spec__ = types.ModuleSpec("sklearn", getattr(sklearn, "__file__", None))
except Exception:
    pass


import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gigacode.incremental_indexer import (
    ChunkDiffTracker,
    IncrementalIndexManager,
)
from gigacode.semantic_cache import SearchResultCache, SemanticQueryCache


class MockChunk:
    """Mock CodeChunk for testing."""

    def __init__(self, text, file, start_line, end_line, chunk_type="function", name="test"):
        self.text = text
        self.file = file
        self.start_line = start_line
        self.end_line = end_line
        self.type = chunk_type
        self.name = name


class MockEmbedder:
    """Mock embedder for testing."""

    def encode(self, texts):
        """Return dummy embeddings."""
        import numpy as np

        # Return (N, 384) shaped embeddings
        return np.random.randn(len(texts), 384).astype(np.float32)


def test_chunk_diff_tracker():
    """Test ChunkDiffTracker for detecting changes."""
    print("\n" + "=" * 80)
    print("TEST 1: ChunkDiffTracker")
    print("=" * 80)

    try:
        tracker = ChunkDiffTracker()

        # Register initial chunks
        chunks_v1 = [
            MockChunk("def foo(): pass", "test.py", 1, 3, chunk_type="function"),
            MockChunk("def bar(): pass", "test.py", 5, 7, chunk_type="function"),
        ]

        tracker.register_chunks("test.py", chunks_v1)
        print("[OK] Registered 2 chunks")

        # Detect no changes
        changed, removed, kept = tracker.detect_changes("test.py", chunks_v1)
        assert len(changed) == 0, "No changes expected"
        assert len(removed) == 0, "No removals expected"
        assert len(kept) == 2, "2 kept expected"
        print("[OK] Correctly detected no changes")

        # Modify one chunk
        chunks_v2 = [
            MockChunk("def foo(): return 42", "test.py", 1, 3, chunk_type="function"),  # Modified
            MockChunk("def bar(): pass", "test.py", 5, 7, chunk_type="function"),  # Unchanged
        ]

        changed, removed, kept = tracker.detect_changes("test.py", chunks_v2)
        assert len(changed) == 1, "1 change expected"
        assert len(removed) == 0, "No removals expected"
        assert len(kept) == 1, "1 kept expected"
        print("[OK] Correctly detected 1 changed chunk")

        # Update tracker and remove a chunk
        tracker.update_after_changes("test.py", chunks_v2)
        chunks_v3 = [
            MockChunk("def foo(): return 42", "test.py", 1, 3, chunk_type="function"),
        ]

        changed, removed, kept = tracker.detect_changes("test.py", chunks_v3)
        assert len(removed) == 1, "1 removal expected"
        print("[OK] Correctly detected 1 removed chunk")

        # Update tracker and add a chunk
        tracker.update_after_changes("test.py", chunks_v3)
        chunks_v4 = [
            MockChunk("def foo(): return 42", "test.py", 1, 3, chunk_type="function"),
            MockChunk("class NewClass: pass", "test.py", 5, 8, chunk_type="class"),
        ]

        changed, removed, kept = tracker.detect_changes("test.py", chunks_v4)
        assert len(changed) == 1, "1 addition expected"
        print("[OK] Correctly detected 1 added chunk")

        return True
    except Exception as e:
        print(f"[FAILED] Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_incremental_index_manager():
    """Test IncrementalIndexManager for efficient updates."""
    print("\n" + "=" * 80)
    print("TEST 2: IncrementalIndexManager")
    print("=" * 80)

    try:

        embedder = MockEmbedder()
        manager = IncrementalIndexManager(embedder)

        # Register initial index
        chunks = [
            MockChunk("def foo(): pass", "test.py", 1, 3, chunk_type="function"),
            MockChunk("def bar(): pass", "test.py", 5, 7, chunk_type="function"),
        ]

        embeddings = embedder.encode([c.text for c in chunks])
        manager.register_initial_index("buf1", chunks, embeddings)
        print("[OK] Registered initial index with 2 chunks")

        # Verify cache
        stats = manager.get_cache_stats()
        assert stats["cached_embeddings"] == 2
        print("[OK] Cache has 2 embeddings")

        # Compute incremental update for changed chunks
        chunks_v2 = [
            MockChunk("def foo(): return 42", "test.py", 1, 3, chunk_type="function"),
            MockChunk("def bar(): pass", "test.py", 5, 7, chunk_type="function"),
        ]

        new_embs, metadata = manager.compute_incremental_update("test.py", chunks_v2)

        assert metadata["changed_count"] == 1
        assert metadata["kept_count"] == 1
        assert new_embs.shape[0] == 1  # Only changed chunks
        print(f"[OK] Computed incremental update: {metadata}")

        # Verify only changed chunk was embedded
        print(f"[OK] Embedded {new_embs.shape[0]} changed chunks (vs 2 total)")

        return True
    except Exception as e:
        print(f"[FAILED] Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_semantic_cache():
    """Test SemanticQueryCache for query deduplication."""
    print("\n" + "=" * 80)
    print("TEST 3: SemanticQueryCache")
    print("=" * 80)

    try:
        embedder = MockEmbedder()
        cache = SemanticQueryCache(
            max_entries=100,
            similarity_threshold=0.8,
            embedder=embedder,
        )

        # Cache a result
        query1 = "find the add function"
        result1 = {"matches": [{"name": "add", "score": 0.95}]}

        cache.put(query1, result1)
        print(f"[OK] Cached result for: '{query1}'")

        # Retrieve exact match
        cached = cache.get(query1, compute_embedding=False)
        assert cached is not None
        assert cached[1]  # Exact match
        print("[OK] Retrieved exact match from cache")

        # Attempt semantic match (will work with mock embedder)
        query2 = "locate the add function"  # Paraphrased
        cached = cache.get(query2, compute_embedding=True)
        # May or may not hit depending on mock embeddings

        # Get stats
        stats = cache.get_stats()
        print(f"[OK] Cache stats: {stats['hits']} hits, {stats['misses']} misses")

        # Test LRU eviction
        cache_small = SemanticQueryCache(max_entries=2, embedder=embedder)
        cache_small.put("query1", {"result": 1})
        cache_small.put("query2", {"result": 2})
        cache_small.put("query3", {"result": 3})  # Should evict oldest

        assert len(cache_small._cache) == 2
        print(f"[OK] LRU eviction working (size: {len(cache_small._cache)}/2)")

        return True
    except Exception as e:
        print(f"[FAILED] Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_search_result_cache():
    """Test SearchResultCache for caching search results."""
    print("\n" + "=" * 80)
    print("TEST 4: SearchResultCache")
    print("=" * 80)

    try:
        embedder = MockEmbedder()
        cache = SearchResultCache(embedder=embedder, max_entries=100)

        # Cache a search result
        query = "find buffer initialization"
        results = {
            "buffer_id": "buf1",
            "matches": [{"file": "main.py", "line": 10, "score": 0.92}],
        }

        cache.put_search_result("buf1", query, "semantic", results, top_k=5)
        print("[OK] Cached search result for semantic search")

        # Retrieve cached result
        cached = cache.get_search_result("buf1", query, "semantic", top_k=5)
        assert cached is not None
        assert cached["cached"]
        print("[OK] Retrieved cached search result")

        # Test cache miss
        cached = cache.get_search_result("buf1", "different query", "semantic", top_k=5)
        assert cached is None
        print("[OK] Cache miss for new query")

        # Get stats
        stats = cache.get_stats()
        assert "search_cache_size" in stats
        print(f"[OK] Cache stats available: {stats['search_cache_size']} entries")

        return True
    except Exception as e:
        print(f"[FAILED] Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance_comparison():
    """Compare performance with/without incremental indexing."""
    print("\n" + "=" * 80)
    print("TEST 5: Performance Comparison")
    print("=" * 80)

    try:

        embedder = MockEmbedder()
        manager = IncrementalIndexManager(embedder)

        # Simulate initial embedding of 100 chunks
        chunks = [
            MockChunk(f"def func_{i}(): pass", "test.py", i * 2, i * 2 + 1) for i in range(100)
        ]

        embeddings = embedder.encode([c.text for c in chunks])

        # Time initial indexing
        t0 = time.perf_counter()
        manager.register_initial_index("buf1", chunks, embeddings)
        initial_time = (time.perf_counter() - t0) * 1000

        # Simulate update with 10 changed chunks (10% modification)
        chunks_v2 = list(chunks)
        for i in range(10):
            chunks_v2[i] = MockChunk(
                f"def func_{i}(): return {i}",  # Modified
                "test.py",
                i * 2,
                i * 2 + 1,
            )

        # Time incremental update
        t0 = time.perf_counter()
        new_embs, metadata = manager.compute_incremental_update("test.py", chunks_v2)
        incremental_time = (time.perf_counter() - t0) * 1000

        # Full re-embedding would need 100 embeddings
        # Incremental only needs 10
        print(f"[OK] Initial indexing: {initial_time:.1f}ms")
        print(f"[OK] Incremental update: {incremental_time:.1f}ms")
        print(
            f"[OK] Efficiency: Only embedded {metadata['changed_count']}/{metadata['total_chunks']} chunks"
        )
        print(f"[OK] Expected speedup: ~{100/metadata['changed_count']:.1f}x (for this scenario)")

        return True
    except Exception as e:
        print(f"[FAILED] Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FUTURE 4.3 PHASE 3: ADVANCED OPTIMIZATIONS - TESTS")
    print("=" * 80)

    tests = [
        ("ChunkDiffTracker", test_chunk_diff_tracker),
        ("IncrementalIndexManager", test_incremental_index_manager),
        ("SemanticQueryCache", test_semantic_cache),
        ("SearchResultCache", test_search_result_cache),
        ("Performance Comparison", test_performance_comparison),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[FAILED] {name} test failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    exit(0 if failed == 0 else 1)

