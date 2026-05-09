"""Integration tests for Future 4.3 Phase 3b - Module Integration.

Tests cover:
- Incremental indexer integration into IndexManager
- Semantic cache integration into SearchService
- FAISS optimizer integration into GPUIndex
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from gigacode.gpu_index import GpuIndex
from gigacode.incremental_indexer import IncrementalIndexManager
from gigacode.index_manager import IndexManager
from gigacode.search_service import SearchService
from gigacode.semantic_cache import SemanticQueryCache


class MockEmbedder:
    """Mock embedder for testing."""

    def __init__(self):
        self.embedding_dim = 384

    def encode(self, texts):
        """Return dummy embeddings."""
        return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)

    def embed(self, text):
        """Embed single text."""
        return np.random.randn(self.embedding_dim).astype(np.float32)


def test_incremental_indexer_integration():
    """Test incremental indexer integrated into IndexManager."""
    print("\n" + "=" * 80)
    print("TEST 1: Incremental Indexer Integration (IndexManager)")
    print("=" * 80)

    try:
        embedder = MockEmbedder()
        work_dir = Path.home() / ".test_gigacode_phase3b" / "test_index"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Create IndexManager with incremental indexing
        manager = IndexManager(
            embedder=embedder,
            embedding_dim=384,
            max_buffers=5,
            work_dir=work_dir,
            use_gpu=False,
        )

        # Verify incremental manager is initialized
        assert hasattr(manager, "_incremental_manager")
        assert manager._incremental_manager is not None
        assert isinstance(manager._incremental_manager, IncrementalIndexManager)
        print(
            f"✅ IndexManager has _incremental_manager: {type(manager._incremental_manager).__name__}"
        )

        # Verify incremental manager has chunk tracker
        assert hasattr(manager._incremental_manager, "_chunk_tracker")
        print("✅ IncrementalIndexManager has _chunk_tracker")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_semantic_cache_integration():
    """Test semantic cache integrated into SearchService."""
    print("\n" + "=" * 80)
    print("TEST 2: Semantic Cache Integration (SearchService)")
    print("=" * 80)

    try:
        embedder = MockEmbedder()
        work_dir = Path.home() / ".test_gigacode_phase3b" / "test_search"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Create IndexManager and SearchService
        index_manager = IndexManager(
            embedder=embedder,
            embedding_dim=384,
            max_buffers=5,
            work_dir=work_dir,
            use_gpu=False,
        )

        search_service = SearchService(
            index_manager=index_manager,
            embedder=embedder,
        )

        # Verify semantic cache is initialized
        assert hasattr(search_service, "_semantic_query_cache")
        assert search_service._semantic_query_cache is not None
        assert isinstance(search_service._semantic_query_cache, SemanticQueryCache)
        print(
            f"✅ SearchService has _semantic_query_cache: {type(search_service._semantic_query_cache).__name__}"
        )

        # Verify semantic cache has expected methods
        cache = search_service._semantic_query_cache
        assert hasattr(cache, "get")
        assert hasattr(cache, "put")
        assert hasattr(cache, "get_stats")
        print("✅ SemanticQueryCache has expected methods (get, put, get_stats)")

        # Verify cache is empty initially
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        print(f"✅ Cache initialized empty: {stats}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gpu_index_faiss_integration():
    """Test FAISS optimizer integration into GPUIndex."""
    print("\n" + "=" * 80)
    print("TEST 3: FAISS Optimizer Integration (GPUIndex)")
    print("=" * 80)

    try:
        # Create GPUIndex
        index = GpuIndex(dim=384, use_gpu=False)

        # Verify index_type parameter is supported
        assert hasattr(index, "index_type")
        print("✅ GPUIndex supports index_type parameter")

        # Create with explicit index_type
        index2 = GpuIndex(dim=384, use_gpu=False, index_type="flat")
        assert index2.index_type == "flat"
        print(f"✅ GPUIndex accepts index_type override: {index2.index_type}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_end_to_end_integration():
    """Test end-to-end flow with all Phase 3b integrations."""
    print("\n" + "=" * 80)
    print("TEST 4: End-to-End Integration Flow")
    print("=" * 80)

    try:
        embedder = MockEmbedder()
        work_dir = Path.home() / ".test_gigacode_phase3b" / "test_e2e"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        index_manager = IndexManager(
            embedder=embedder,
            embedding_dim=384,
            max_buffers=5,
            work_dir=work_dir,
            use_gpu=False,
        )

        search_service = SearchService(
            index_manager=index_manager,
            embedder=embedder,
        )

        # Verify all integrations are present
        assert index_manager._incremental_manager is not None
        assert search_service._semantic_query_cache is not None
        print("✅ All Phase 3b components initialized")

        # Verify components work together
        cache_stats = search_service._semantic_query_cache.get_stats()
        assert "hits" in cache_stats
        assert "misses" in cache_stats
        print(f"✅ Components work together: cache_stats={cache_stats}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_incremental_tracking_flow():
    """Test incremental tracking in IndexManager workflow."""
    print("\n" + "=" * 80)
    print("TEST 5: Incremental Tracking Flow")
    print("=" * 80)

    try:
        embedder = MockEmbedder()
        work_dir = Path.home() / ".test_gigacode_phase3b" / "test_incremental"
        work_dir.mkdir(parents=True, exist_ok=True)

        index_manager = IndexManager(
            embedder=embedder,
            embedding_dim=384,
            max_buffers=5,
            work_dir=work_dir,
            use_gpu=False,
        )

        # Access incremental manager
        inc_manager = index_manager._incremental_manager
        assert inc_manager is not None
        print("✅ Incremental manager accessible from IndexManager")

        # Verify chunk tracker
        tracker = inc_manager._chunk_tracker
        assert tracker is not None
        print("✅ Chunk tracker accessible")

        # Create mock chunks
        from gigacode.chunker import CodeChunk

        chunks = [
            CodeChunk(
                id=1,
                text="def foo(): pass",
                file="test.py",
                start_line=1,
                end_line=3,
                type="function",
                name="foo",
            ),
            CodeChunk(
                id=2,
                text="def bar(): pass",
                file="test.py",
                start_line=5,
                end_line=7,
                type="function",
                name="bar",
            ),
        ]

        # Register chunks
        tracker.register_chunks("test.py", chunks)
        print("✅ Registered 2 chunks in tracker")

        # Verify chunk hashes created
        assert len(tracker.chunk_hashes) == 2
        print(f"✅ Chunk hashes created: {len(tracker.chunk_hashes)} chunks")

        # Detect no changes
        changed, removed, kept = tracker.detect_changes("test.py", chunks)
        assert len(changed) == 0
        assert len(removed) == 0
        assert len(kept) == 2
        print("✅ Change detection working: 0 changed, 0 removed, 2 kept")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_semantic_cache_flow():
    """Test semantic cache in SearchService workflow."""
    print("\n" + "=" * 80)
    print("TEST 6: Semantic Cache Flow")
    print("=" * 80)

    try:
        embedder = MockEmbedder()
        work_dir = Path.home() / ".test_gigacode_phase3b" / "test_cache"
        work_dir.mkdir(parents=True, exist_ok=True)

        index_manager = IndexManager(
            embedder=embedder,
            embedding_dim=384,
            max_buffers=5,
            work_dir=work_dir,
            use_gpu=False,
        )

        search_service = SearchService(
            index_manager=index_manager,
            embedder=embedder,
        )

        cache = search_service._semantic_query_cache

        # Put a result in cache
        query = "find initialization code"
        result = {"matches": [{"name": "init"}], "buffer_id": "buf1"}
        cache.put(query, result)
        print(f"✅ Put result in cache for: {query[:30]}")

        # Get result from cache
        cached, was_exact = cache.get(query, compute_embedding=False)
        assert cached is not None
        assert cached == result
        print("✅ Retrieved result from cache (exact match)")

        # Verify stats
        stats = cache.get_stats()
        assert stats["hits"] >= 1
        print(f"✅ Cache stats updated: {stats}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FUTURE 4.3 PHASE 3B: INTEGRATION TESTS")
    print("=" * 80)

    tests = [
        ("Incremental Indexer Integration", test_incremental_indexer_integration),
        ("Semantic Cache Integration", test_semantic_cache_integration),
        ("FAISS Optimizer Integration", test_gpu_index_faiss_integration),
        ("End-to-End Integration", test_end_to_end_integration),
        ("Incremental Tracking Flow", test_incremental_tracking_flow),
        ("Semantic Cache Flow", test_semantic_cache_flow),
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
            print(f"\n❌ {name} test failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)

    exit(0 if failed == 0 else 1)
