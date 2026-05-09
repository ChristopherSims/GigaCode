#!/usr/bin/env python3
"""Performance profiling suite for GigaCode.

This script profiles hot paths to identify optimization opportunities:
- Embedding performance (file I/O, chunking, model inference)
- Search performance (FAISS queries, scoring)
- Commit performance (diff generation, re-embedding)
- Memory usage with various buffer sizes

Usage:
    python scripts/profile_performance.py --help
"""

import argparse
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import psutil


@dataclass
class TimingResult:
    """Timing measurement result."""

    operation: str
    duration_ms: float
    memory_mb_start: float
    memory_mb_end: float
    memory_mb_delta: float

    def __str__(self) -> str:
        return (
            f"{self.operation:40} | "
            f"Time: {self.duration_ms:8.2f}ms | "
            f"Memory: {self.memory_mb_delta:+7.2f}MB "
            f"({self.memory_mb_start:7.1f}→{self.memory_mb_end:7.1f}MB)"
        )


class PerformanceProfiler:
    """Main profiling orchestrator."""

    def __init__(self):
        self.process = psutil.Process()
        self.results: List[TimingResult] = []

    def get_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    @contextmanager
    def profile_operation(self, name: str):
        """Context manager to profile an operation."""
        start_time = time.time()
        mem_start = self.get_memory_mb()

        try:
            yield
        finally:
            duration = (time.time() - start_time) * 1000  # ms
            mem_end = self.get_memory_mb()
            mem_delta = mem_end - mem_start

            result = TimingResult(
                operation=name,
                duration_ms=duration,
                memory_mb_start=mem_start,
                memory_mb_end=mem_end,
                memory_mb_delta=mem_delta,
            )
            self.results.append(result)
            print(result)

    def profile_import(self):
        """Profile module imports."""
        print("\n" + "=" * 100)
        print("IMPORT PERFORMANCE")
        print("=" * 100)

        # Profile gigacode imports
        with self.profile_operation("Import gigacode.gigacode_tool"):
            from gigacode.gigacode_tool import CodeEmbeddingTool

        with self.profile_operation("Import torch"):
            import torch

        with self.profile_operation("Import sentence_transformers"):
            from sentence_transformers import SentenceTransformer

        with self.profile_operation("Import faiss"):
            import faiss

    def profile_init(self):
        """Profile CodeEmbeddingTool initialization."""
        print("\n" + "=" * 100)
        print("INITIALIZATION PERFORMANCE")
        print("=" * 100)

        from gigacode.gigacode_tool import CodeEmbeddingTool

        # Create buffers directory if needed
        work_dir = Path("./test_buffers_profile")
        work_dir.mkdir(exist_ok=True)

        try:
            with self.profile_operation("CodeEmbeddingTool init (CPU)"):
                tool = CodeEmbeddingTool(work_dir=str(work_dir), device="cpu")

            with self.profile_operation("CodeEmbeddingTool context manager"):
                with tool:
                    pass

            return tool
        except Exception as e:
            print(f"Error during initialization profiling: {e}")
            return None

    def profile_chunking(self, tool=None):
        """Profile chunking operations."""
        print("\n" + "=" * 100)
        print("CHUNKING PERFORMANCE")
        print("=" * 100)

        from gigacode.chunker import CodeChunker

        if tool is None:
            from gigacode.gigacode_tool import CodeEmbeddingTool

            work_dir = Path("./test_buffers_profile")
            tool = CodeEmbeddingTool(work_dir=str(work_dir), device="cpu")

        try:
            chunker = CodeChunker()

            # Profile Python file chunking
            python_file = Path("gigacode/gigacode_tool.py")
            if python_file.exists():
                content = python_file.read_text()

                with self.profile_operation(f"Chunk Python file ({len(content)} bytes)"):
                    chunks = chunker.chunk_file(str(python_file), "python")

                print(f"  → Generated {len(chunks)} chunks")

            # Profile JSON file chunking (simple)
            import json

            test_json = {"key": "value", "data": [1, 2, 3] * 100}
            test_json_str = json.dumps(test_json) * 10

            with self.profile_operation(f"Chunk JSON content ({len(test_json_str)} bytes)"):
                # Note: CodeChunker might not handle JSON well, test with fallback
                from gigacode.chunker import fallback_chunk_text

                chunks = fallback_chunk_text(test_json_str)

            print(f"  → Generated {len(chunks)} chunks")

        except Exception as e:
            print(f"Error during chunking profiling: {e}")
            traceback.print_exc()

    def profile_embedding(self):
        """Profile embedding operations."""
        print("\n" + "=" * 100)
        print("EMBEDDING PERFORMANCE")
        print("=" * 100)

        try:
            from sentence_transformers import SentenceTransformer

            # Profile model loading
            with self.profile_operation("Load embedding model (all-MiniLM-L6-v2)"):
                model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

            # Profile embedding inference on small batch
            test_texts = [
                "def calculate_sum(a, b): return a + b",
                "class DataProcessor: pass",
                "if __name__ == '__main__': main()",
            ] * 10  # 30 texts

            with self.profile_operation(f"Embed {len(test_texts)} code snippets"):
                embeddings = model.encode(test_texts, show_progress_bar=False)

            print(f"  → Generated embeddings shape: {embeddings.shape}")

            # Profile larger batch
            large_batch = test_texts * 10  # 300 texts
            with self.profile_operation(f"Embed {len(large_batch)} code snippets (large batch)"):
                embeddings = model.encode(large_batch, show_progress_bar=False)

            print(f"  → Generated embeddings shape: {embeddings.shape}")

        except Exception as e:
            print(f"Error during embedding profiling: {e}")
            traceback.print_exc()

    def profile_faiss_search(self):
        """Profile FAISS search operations."""
        print("\n" + "=" * 100)
        print("FAISS SEARCH PERFORMANCE")
        print("=" * 100)

        try:
            import faiss
            import numpy as np

            # Create sample embeddings
            dimension = 384  # all-MiniLM-L6-v2 output dimension
            num_vectors = 10000

            with self.profile_operation(
                f"Generate {num_vectors} random embeddings (dim {dimension})"
            ):
                vectors = np.random.randn(num_vectors, dimension).astype("float32")
                faiss.normalize_L2(vectors)

            # Create and populate index
            with self.profile_operation("Create FAISS IndexFlatL2"):
                index = faiss.IndexFlatL2(dimension)
                index.add(vectors)

            print(f"  → Index contains {index.ntotal} vectors")

            # Profile search operations
            query_vectors = vectors[:10]  # Use first 10 as queries

            with self.profile_operation(f"Search for {len(query_vectors)} queries (k=10)"):
                distances, indices = index.search(query_vectors, 10)

            print(f"  → Found matches, distances shape: {distances.shape}")

            # Profile larger search
            query_vectors = vectors[:100]
            with self.profile_operation(f"Search for {len(query_vectors)} queries (k=100)"):
                distances, indices = index.search(query_vectors, 100)

            print(f"  → Found matches, distances shape: {distances.shape}")

        except Exception as e:
            print(f"Error during FAISS profiling: {e}")
            traceback.print_exc()

    def profile_memory_scaling(self):
        """Profile memory usage scaling with buffer size."""
        print("\n" + "=" * 100)
        print("MEMORY SCALING ANALYSIS")
        print("=" * 100)

        try:
            import numpy as np

            # Profile memory growth with embedding batches
            dimension = 384
            for batch_size in [100, 1000, 5000, 10000]:
                with self.profile_operation(f"Allocate {batch_size} embeddings ({dimension}D)"):
                    embeddings = np.random.randn(batch_size, dimension).astype("float32")
                    total_size_mb = embeddings.nbytes / 1024 / 1024

                print(f"  → Size: {total_size_mb:.2f}MB")

        except Exception as e:
            print(f"Error during memory scaling profiling: {e}")
            traceback.print_exc()

    def print_summary(self):
        """Print summary statistics."""
        if not self.results:
            return

        print("\n" + "=" * 100)
        print("SUMMARY STATISTICS")
        print("=" * 100)

        total_time = sum(r.duration_ms for r in self.results)
        max_memory = max(r.memory_mb_end for r in self.results)
        total_memory_delta = sum(r.memory_mb_delta for r in self.results)

        print(f"Total operations profiled: {len(self.results)}")
        print(f"Total time: {total_time:.2f}ms")
        print(f"Max memory used: {max_memory:.2f}MB")
        print(f"Total memory change: {total_memory_delta:+.2f}MB")

        # Top 5 slowest operations
        print("\nTop 5 Slowest Operations:")
        for i, result in enumerate(
            sorted(self.results, key=lambda r: r.duration_ms, reverse=True)[:5], 1
        ):
            print(f"  {i}. {result.operation}: {result.duration_ms:.2f}ms")

        # Top 5 memory consumers
        print("\nTop 5 Memory Consumers:")
        for i, result in enumerate(
            sorted(self.results, key=lambda r: r.memory_mb_delta, reverse=True)[:5], 1
        ):
            if result.memory_mb_delta > 0:
                print(f"  {i}. {result.operation}: +{result.memory_mb_delta:.2f}MB")


def main():
    parser = argparse.ArgumentParser(description="Profile GigaCode performance")
    parser.add_argument("--all", action="store_true", help="Run all profiling")
    parser.add_argument("--imports", action="store_true", help="Profile imports")
    parser.add_argument("--init", action="store_true", help="Profile initialization")
    parser.add_argument("--chunking", action="store_true", help="Profile chunking")
    parser.add_argument("--embedding", action="store_true", help="Profile embedding")
    parser.add_argument("--faiss", action="store_true", help="Profile FAISS search")
    parser.add_argument("--memory", action="store_true", help="Profile memory scaling")

    args = parser.parse_args()

    # Default to all if no specific option given
    if not any([args.imports, args.init, args.chunking, args.embedding, args.faiss, args.memory]):
        args.all = True

    profiler = PerformanceProfiler()

    try:
        if args.all or args.imports:
            profiler.profile_import()

        if args.all or args.init:
            tool = profiler.profile_init()

        if args.all or args.chunking:
            profiler.profile_chunking()

        if args.all or args.embedding:
            profiler.profile_embedding()

        if args.all or args.faiss:
            profiler.profile_faiss_search()

        if args.all or args.memory:
            profiler.profile_memory_scaling()

        profiler.print_summary()

    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import traceback

    sys.exit(main())
