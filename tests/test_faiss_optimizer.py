"""Tests for FAISS optimizer and index type selection.

Tests cover:
- Index type selection logic
- Index creation for different types
- Search performance comparison
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

import numpy as np
import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gigacode.faiss_optimizer import FAISSIndexOptimizer


class MockFAISS:
    """Mock FAISS module when real one not available."""

    pass


def test_index_type_selection():
    """Test automatic index type selection."""
    print("\n" + "=" * 80)
    print("TEST 1: Index Type Selection")
    print("=" * 80)

    try:
        optimizer = FAISSIndexOptimizer()

        # Small dataset -> Flat
        index_type = optimizer.select_index_type(vector_count=5000)
        assert index_type == "flat", f"Expected flat, got {index_type}"
        print(f"[OK] Small dataset (5k vectors): {index_type}")

        # Medium dataset -> IVF
        index_type = optimizer.select_index_type(vector_count=50000)
        assert index_type == "ivf", f"Expected ivf, got {index_type}"
        print(f"[OK] Medium dataset (50k vectors): {index_type}")

        # Large dataset -> HNSW
        index_type = optimizer.select_index_type(vector_count=500000)
        assert index_type == "hnsw", f"Expected hnsw, got {index_type}"
        print(f"[OK] Large dataset (500k vectors): {index_type}")

        # Force type override
        index_type = optimizer.select_index_type(vector_count=5000, force_type="ivf")
        assert index_type == "ivf", f"Expected ivf override, got {index_type}"
        print(f"[OK] Force type override: {index_type}")

        return True
    except Exception as e:
        print(f"[FAILED] Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_index_parameters():
    """Test parameter generation for different index types."""
    print("\n" + "=" * 80)
    print("TEST 2: Index Parameters")
    print("=" * 80)

    try:
        optimizer = FAISSIndexOptimizer()

        # Flat parameters
        params = optimizer.get_index_params("flat", vector_count=1000)
        assert params["type"] == "flat"
        assert "metric" in params
        print(f"[OK] Flat parameters: {params}")

        # IVF parameters
        params = optimizer.get_index_params("ivf", vector_count=50000)
        assert params["type"] == "ivf"
        assert params["nlist"] > 0
        assert params["nprobe"] > 0
        assert params["nprobe"] < params["nlist"]
        print(f"[OK] IVF parameters: nlist={params['nlist']}, nprobe={params['nprobe']}")

        # HNSW parameters
        params = optimizer.get_index_params("hnsw", vector_count=100000)
        assert params["type"] == "hnsw"
        assert params["nlinks"] > 0
        print(
            f"[OK] HNSW parameters: nlinks={params['nlinks']}, efConstruction={params['efConstruction']}"
        )

        return True
    except Exception as e:
        print(f"[FAILED] Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_faiss_integration():
    """Test integration with actual FAISS if available."""
    print("\n" + "=" * 80)
    print("TEST 3: FAISS Integration")
    print("=" * 80)

    try:
        import faiss

        optimizer = FAISSIndexOptimizer()

        # Create sample vectors
        embedding_dim = 384
        vector_count = 1000
        vectors = np.random.randn(vector_count, embedding_dim).astype(np.float32)
        query = np.random.randn(1, embedding_dim).astype(np.float32)

        # Create flat index
        index = optimizer.create_optimized_index(
            vectors=vectors,
            index_type="flat",
            embedding_dim=embedding_dim,
        )

        if index is None:
            pytest.skip("FAISS not available, skipping index creation tests")

        assert index.ntotal == vector_count
        print(f"[OK] Created flat index with {index.ntotal} vectors")

        # Search
        distances, indices = optimizer.search_index(index, query[0], k=5)
        assert len(distances) == 5
        assert len(indices) == 5
        assert distances[0] <= distances[-1]  # Distances should be sorted
        print(f"[OK] Search returned {len(indices)} results")

        # Get index info
        info = optimizer.get_index_info(index)
        assert info["ntotal"] == vector_count
        print(f"[OK] Index info: {info}")

        return True
    except ImportError:
        print("[WARNING] FAISS not installed, skipping FAISS integration tests")
        return True
    except Exception as e:
        print(f"[FAILED] Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_index_creation_performance():
    """Test index creation performance characteristics."""
    print("\n" + "=" * 80)
    print("TEST 4: Index Creation Performance")
    print("=" * 80)

    try:
        import faiss

        optimizer = FAISSIndexOptimizer()
        embedding_dim = 384

        test_cases = [
            (1000, "flat"),
            (5000, "flat"),
            (50000, "ivf"),
        ]

        results = []

        for vector_count, expected_type in test_cases:
            vectors = np.random.randn(vector_count, embedding_dim).astype(np.float32)

            selected_type = optimizer.select_index_type(vector_count, embedding_dim)
            assert selected_type == expected_type, (
                f"Type mismatch: {selected_type} vs {expected_type}"
            )

            t0 = time.perf_counter()
            index = optimizer.create_optimized_index(vectors=vectors, embedding_dim=embedding_dim)
            creation_time = (time.perf_counter() - t0) * 1000

            if index is None:
                print(f"[WARNING] Failed to create {expected_type} index")
                continue

            # Measure search time
            query = np.random.randn(1, embedding_dim).astype(np.float32)
            t0 = time.perf_counter()
            for _ in range(10):
                distances, indices = optimizer.search_index(index, query[0], k=5)
            search_time = (time.perf_counter() - t0) * 1000 / 10

            results.append(
                {
                    "vectors": vector_count,
                    "type": selected_type,
                    "creation_ms": creation_time,
                    "search_ms": search_time,
                }
            )

            print(
                f"[OK] {selected_type:6s} ({vector_count:6d} vectors): "
                f"create={creation_time:6.1f}ms, search={search_time:5.2f}ms"
            )

        # Verify performance trend
        if len(results) >= 2:
            flat_times = [r["creation_ms"] for r in results if r["type"] == "flat"]
            assert flat_times[1] > flat_times[0], "Creation time should increase with vector count"
            print("[OK] Performance scaling verified")

        return True
    except ImportError:
        print("[WARNING] FAISS not installed, skipping performance tests")
        return True
    except Exception as e:
        print(f"[FAILED] Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_distance_metrics():
    """Test distance calculation correctness."""
    print("\n" + "=" * 80)
    print("TEST 5: Distance Metrics")
    print("=" * 80)

    try:
        import faiss

        optimizer = FAISSIndexOptimizer()
        embedding_dim = 384

        # Create known vectors for testing
        vectors = np.array(
            [
                np.ones(embedding_dim),
                np.ones(embedding_dim) * 2,
                np.zeros(embedding_dim),
            ],
            dtype=np.float32,
        )

        # Query vector same as first vector
        query = np.ones(embedding_dim).astype(np.float32)

        index = optimizer.create_optimized_index(vectors=vectors, embedding_dim=embedding_dim)

        if index is None:
            print("[WARNING] FAISS not available")
            return True

        distances, indices = optimizer.search_index(index, query, k=3)

        # First result should be the identical vector (distance 0)
        assert indices[0] == 0, f"Expected first result to be vector 0, got {indices[0]}"
        assert distances[0] < 0.01, f"Expected distance ~0, got {distances[0]}"
        print(f"[OK] Identical vectors: distance={distances[0]:.6f}")

        # Distances should be sorted
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1], "Distances should be sorted"
        print("[OK] Results sorted by distance")

        return True
    except ImportError:
        print("[WARNING] FAISS not installed, skipping distance metric tests")
        return True
    except Exception as e:
        print(f"[FAILED] Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FAISS OPTIMIZER - TESTS")
    print("=" * 80)

    tests = [
        ("Index Type Selection", test_index_type_selection),
        ("Index Parameters", test_index_parameters),
        ("FAISS Integration", test_faiss_integration),
        ("Creation Performance", test_index_creation_performance),
        ("Distance Metrics", test_distance_metrics),
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
