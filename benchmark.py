"""Quick benchmark for GigaCode buffer operations.

Usage (from the repo root):
    source .venv/bin/activate
    python benchmark.py [--dir examplecode/]
"""

from __future__ import annotations

import argparse
import gc
import statistics
import tempfile
import time
from pathlib import Path

from src.agent_tool import CodeEmbeddingTool


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def benchmark_create(tool: CodeEmbeddingTool, code_dir: Path, pattern: str) -> dict:
    gc.collect()
    t0 = time.perf_counter()
    result = tool.embed_codebase(code_dir, pattern=pattern)
    elapsed = _elapsed(t0)
    return {
        "status": result["status"],
        "buffer_id": result.get("buffer_id"),
        "token_count": result.get("token_count", 0),
        "size_bytes": result.get("size_bytes", 0),
        "elapsed_sec": elapsed,
    }


def benchmark_search(tool: CodeEmbeddingTool, buffer_id: str, query: str, top_k: int = 5) -> dict:
    gc.collect()
    t0 = time.perf_counter()
    result = tool.semantic_search(buffer_id, query, top_k=top_k)
    elapsed = _elapsed(t0)
    return {
        "status": result["status"],
        "matches": len(result.get("matches", [])),
        "elapsed_sec": elapsed,
    }


def benchmark_cluster(tool: CodeEmbeddingTool, buffer_id: str, threshold: float = 0.75) -> dict:
    gc.collect()
    t0 = time.perf_counter()
    result = tool.cluster_code(buffer_id, threshold=threshold)
    elapsed = _elapsed(t0)
    return {
        "status": result["status"],
        "clusters": len(result.get("clusters", [])),
        "elapsed_sec": elapsed,
    }


def benchmark_edit(tool: CodeEmbeddingTool, buffer_id: str, file: str, iterations: int = 5) -> dict:
    times: list[float] = []
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter()
        tool.write_code(
            buffer_id,
            file=file,
            start_line=1,
            new_lines=["# benchmark edit placeholder"],
            end_line=2,
        )
        times.append(_elapsed(t0))
    return {
        "iterations": iterations,
        "elapsed_sec": {
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GigaCode buffer ops")
    parser.add_argument("--dir", type=Path, default=Path("examplecode"), help="Codebase to embed")
    parser.add_argument("--pattern", default="*.py", help="Glob pattern")
    parser.add_argument("--device", default="cpu", help="torch device for embedder")
    parser.add_argument("--edit-iters", type=int, default=5, help="write_code iterations")
    parser.add_argument("--search-iters", type=int, default=10, help="semantic_search iterations")
    args = parser.parse_args()

    code_dir = args.dir.resolve()
    if not code_dir.exists():
        raise SystemExit(f"Directory not found: {code_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        tool = CodeEmbeddingTool(work_dir=work_dir, device=args.device)

        print("=" * 60)
        print("GigaCode Benchmark")
        print(f"  codebase : {code_dir}")
        print(f"  pattern  : {args.pattern}")
        print(f"  device   : {args.device}")
        print("=" * 60)

        # 1. Create buffer
        print("\n[1] embed_codebase")
        create = benchmark_create(tool, code_dir, args.pattern)
        if create["status"] != "ok":
            raise SystemExit(f"Embedding failed: {create}")
        buf_id = create["buffer_id"]
        print(f"  tokens   : {create['token_count']:,}")
        print(f"  size     : {create['size_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  time     : {create['elapsed_sec']:.3f}s")

        # 2. Semantic search (multiple queries)
        print(f"\n[2] semantic_search (x{args.search_iters})")
        queries = ["sorting algorithm", "machine learning feature", "string utility", "math helper"]
        search_times: list[float] = []
        for i in range(args.search_iters):
            q = queries[i % len(queries)]
            res = benchmark_search(tool, buf_id, q, top_k=5)
            search_times.append(res["elapsed_sec"])
        print(f"  median   : {statistics.median(search_times):.4f}s")
        print(f"  mean     : {statistics.mean(search_times):.4f}s")
        print(f"  min/max  : {min(search_times):.4f}s / {max(search_times):.4f}s")

        # 3. Clustering
        print("\n[3] cluster_code")
        cluster = benchmark_cluster(tool, buf_id)
        print(f"  clusters : {cluster['clusters']}")
        print(f"  time     : {cluster['elapsed_sec']:.3f}s")

        # 4. Edit / rebuild (if at least one file exists)
        files = sorted(code_dir.rglob(args.pattern))
        if files:
            first_file = str(files[0].relative_to(code_dir))
            print(f"\n[4] write_code / _rebuild_file_region (x{args.edit_iters})")
            print(f"  target   : {first_file}")
            edit = benchmark_edit(tool, buf_id, first_file, iterations=args.edit_iters)
            e = edit["elapsed_sec"]
            print(f"  median   : {e['median']:.4f}s")
            print(f"  mean     : {e['mean']:.4f}s")
            print(f"  min/max  : {e['min']:.4f}s / {e['max']:.4f}s")

        print("\n" + "=" * 60)
        print("Benchmark complete.")
        print("=" * 60)

        tool.close()


if __name__ == "__main__":
    main()
