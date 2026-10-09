"""GigaCode MCP benchmark server entry point.

Wraps CodeEmbeddingTool with a fast deterministic numpy-hashing embedder so
the benchmark runs without sentence-transformers/torch.  Exposes the given
tool profile (default "editing") over stdio MCP in DEFERRED discovery mode:
only ``tool_search`` and ``tool_call`` are published as schema surface and
the agent discovers the remaining tools on demand.

GPU enablement is env-driven (``GIGACODE_BENCH_GPU``): auto (default),
on, off.  AUTO attempts the GPU mirror and transparently falls back to
CPU (faiss-cpu or brute-force numpy) when unavailable.  The resolved mode
is announced on stderr as ``GIGACODE_BENCH gpu_mode=...`` so the benchmark
can record it per run.

Usage:
    python scripts/mcp_bench_server.py <work-dir> [tool-profile]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


class HashEmbedder:
    """Deterministic feature-hashing embedder (no torch/huggingface needed).

    Tokenizes on non-alphanumerics, hashes each unigram + bigram feature into
    `dim` buckets (signed hashing), L2-normalizes.  Works well enough for
    lexical similarity and keeps semantic_search/hybrid_search functional.
    """

    def __init__(self, dim: int = 384) -> None:
        import hashlib
        self._dim = dim
        self._hashlib = hashlib

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def model_name(self) -> str:
        return "hashing-builtin-bench"

    def _featurize(self, text: str) -> dict[int, float]:
        feats: dict[int, float] = {}
        words = []
        cur = []
        for ch in text.lower():
            if ch.isalnum() or ch == "_":
                cur.append(ch)
            else:
                if cur:
                    words.append("".join(cur))
                    cur = []
        if cur:
            words.append("".join(cur))
        if not words:
            return feats

        def bump(feature: str, weight: float) -> None:
            h = int.from_bytes(
                self._hashlib.md5(feature.encode("utf-8")).digest()[:8], "little"
            )
            idx = h % self._dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            feats[idx] = feats.get(idx, 0.0) + sign * weight

        # unigrams
        for w in words:
            bump("u:" + w, 1.0)
        # bigrams
        for a, b in zip(words, words[1:], strict=False):
            bump("b:" + a + "_" + b, 0.7)
        return feats

    def _enc_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        feats = self._featurize(str(text))
        for idx, val in feats.items():
            vec[idx] = val
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, texts, batch_size: int = 32, **_kwargs) -> np.ndarray:
        rows = [self._enc_one(t) for t in texts]
        if not rows:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.vstack(rows).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        return self._enc_one(text)


GPU_MODE_ENV = "GIGACODE_BENCH_GPU"  # auto | on | off  (default: auto)


def _gpu_mode() -> str:
    """Resolve GPU enablement: default auto, fallback to CPU when unavailable."""
    mode = os.environ.get(GPU_MODE_ENV, "auto").strip().lower()
    return mode


def _gpu_status(use_gpu: bool) -> dict[str, Any]:
    """Report the effective vector-index residency for startup diagnostics."""
    status: dict[str, Any] = {"requested_use_gpu": use_gpu}
    try:
        import faiss  # noqa: F401

        status["faiss"] = True
        status["gpu_build"] = hasattr(faiss, "StandardGpuResources")
    except ImportError:
        status["faiss"] = False
        status["gpu_build"] = False
    if not use_gpu:
        status["mode"] = "off"
    elif not status["faiss"]:
        status["mode"] = "bruteforce-cpu"
    elif not status["gpu_build"]:
        status["mode"] = "faiss-cpu"
    else:
        # faiss-gpu build: mirror until proven otherwise; IndexManager logs
        # the fallback if the GPU device itself fails at runtime.
        status["mode"] = "gpu-build-available"
    return status


def main() -> int:
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "./buffers"
    profile = sys.argv[2] if len(sys.argv) > 2 else "editing"

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from gigacode.embedding_provider import validate_embedding_provider
    from gigacode.gigacode_tool import CodeEmbeddingTool

    embedder_mode = os.environ.get("GIGACODE_BENCH_EMBEDDER", "hashing")
    if embedder_mode not in {"hashing", "model"}:
        raise ValueError("GIGACODE_BENCH_EMBEDDER must be hashing or model")
    embedder = HashEmbedder() if embedder_mode == "hashing" else None
    if embedder is not None:
        validate_embedding_provider(embedder)

    # GPU enablement: default AUTO (attempt GPU mirror; the tool stack itself
    # falls back to CPU brute-force when faiss/GPU is unavailable).
    mode = _gpu_mode()
    use_gpu = mode != "off"
    tool = CodeEmbeddingTool(
        work_dir=work_dir,
        model_name=os.environ.get("GIGACODE_BENCH_EMBED_MODEL"),
        device=os.environ.get("GIGACODE_BENCH_DEVICE", "cpu"),
        use_gpu=use_gpu,
        embedder=embedder,
        tool_profile=profile,
    )
    tool.deferred_tools = True  # tool_search/tool_call only; never eager schemas
    tool.direct_tools = os.environ.get("GIGACODE_DIRECT_TOOLS", "on") != "off"

    print(
        f"GIGACODE_BENCH embedder={embedder_mode} gpu_mode={_gpu_status(use_gpu)['mode']} "
        f"{json.dumps(_gpu_status(use_gpu))}",
        file=sys.stderr,
        flush=True,
    )

    import asyncio

    from mcp.server.lowlevel.server import NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server

    from gigacode.mcp_server import _build_server, _server_version

    async def _run() -> None:
        server = _build_server(tool)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="gigacode",
                    server_version=_server_version(),
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    try:
        asyncio.run(_run())
    finally:
        tool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
