"""FAISS vector index with optional GPU residency.

Maintains a CPU IDMap+Flat index as the source of truth.  When a CUDA
device is available the index is mirrored to GPU for fast search.
Mutations (add / remove) happen on CPU; the GPU mirror is lazily rebuilt.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


__all__ = [
    "GpuIndex",
]

# faiss optional import handled gracefully
try:
    import faiss

    _HAS_FAISS = True
except (ImportError, ModuleNotFoundError) as exc:
    logger.warning("faiss not available (%s). Vector search will be brute-force.", exc)
    _HAS_FAISS = False
    faiss = None  # type: ignore[assignment]

# FAISS optimizer for index type selection
try:
    from gigacode.faiss_optimizer import FAISSIndexOptimizer

    _HAS_OPTIMIZER = True
except (ImportError, ModuleNotFoundError):
    _HAS_OPTIMIZER = False


class GpuIndex:
    """FAISS index manager with CPU source-of-truth + optional GPU mirror."""

    def __init__(
        self, dim: int, use_gpu: bool = True, gpu_id: int = 0, index_type: str | None = None
    ) -> None:
        """Initialize FAISS index with optional GPU support and auto-optimization.

        Args:
            dim: Vector dimension.
            use_gpu: Whether to attempt GPU mirroring (default True).
            gpu_id: GPU device ID to use for mirroring (default 0).
                    Only used if use_gpu=True and multiple GPUs available.
            index_type: Override index type ("flat", "ivf", "hnsw"). If None, will auto-select.
        """
        self.dim = dim
        self.gpu_id = gpu_id
        self.index_type = index_type  # For documentation/stats
        self._cpu_index: Any | None = None
        self._gpu_index: Any | None = None
        self._gpu_dirty = True
        self._gpu_available = False
        self._next_id = 0

        if _HAS_FAISS:
            # Build CPU IDMap over FlatIP (dot-product on L2-normalized vectors)
            # Note: We use FlatIP as the default for now, but index type selection
            # can be enhanced in future for more sophisticated selection
            base = faiss.IndexFlatIP(dim)
            self._cpu_index = faiss.IndexIDMap(base)
            self._gpu_available = self._init_gpu() if use_gpu else False
        else:
            self._cpu_index = _BruteForceIndex(dim)

    # ------------------------------------------------------------------
    # GPU helpers
    # ------------------------------------------------------------------
    def _init_gpu(self) -> bool:
        if faiss is None:
            return False
        try:
            ngpu = faiss.get_num_gpus()
        except (ImportError, AttributeError, RuntimeError):
            ngpu = 0
        if ngpu == 0:
            logger.info("No CUDA GPUs detected; using CPU FAISS.")
            return False
        try:
            self._gpu_res = faiss.StandardGpuResources()
            logger.info("CUDA GPU available for FAISS (%d device(s)).", ngpu)
            return True
        except (RuntimeError, ImportError, OSError) as exc:
            logger.warning("GPU init failed (%s); using CPU FAISS.", exc)
            return False

    def _sync_gpu(self) -> None:
        """Rebuild GPU mirror from CPU index if dirty."""
        if not self._gpu_available or not self._gpu_dirty or self._cpu_index is None:
            return
        try:
            # Estimate GPU memory requirement (4 bytes per float32)
            # Each vector: dim * 4 bytes (float32)
            # Plus index overhead (~20% for metadata)
            vectors_in_index = self._cpu_index.ntotal
            bytes_per_vector = self.dim * 4
            estimated_gpu_bytes = int(
                vectors_in_index * bytes_per_vector * 1.2
            )  # +20% for overhead

            # Check available GPU memory
            # Note: This is a soft check; exact allocation depends on FAISS internals
            available_bytes = self._get_available_gpu_memory()
            if available_bytes is not None and estimated_gpu_bytes > available_bytes:
                logger.warning(
                    "Insufficient GPU VRAM: need ~%.1f MB but only %.1f MB available. "
                    "Falling back to CPU search. Consider reducing index size or use CPU mode.",
                    estimated_gpu_bytes / 1024 / 1024,
                    available_bytes / 1024 / 1024,
                )
                self._gpu_index = None
                return

            self._gpu_index = faiss.index_cpu_to_gpu(self._gpu_res, self.gpu_id, self._cpu_index)
            self._gpu_dirty = False
            logger.debug(
                "GPU index synced to device %d (%d vectors).", self.gpu_id, self._cpu_index.ntotal
            )
        except (RuntimeError, OSError, ValueError) as exc:
            logger.warning(
                "GPU sync to device %d failed (%s); falling back to CPU search.", self.gpu_id, exc
            )
            self._gpu_index = None

    def _get_available_gpu_memory(self) -> int | None:
        """Return available GPU memory in bytes, or None if unable to determine."""
        if not self._gpu_available or self._gpu_res is None:
            return None
        try:
            # FAISS StandardGpuResources.getMemory() returns (used, total) in bytes
            mem_used, mem_total = self._gpu_res.getMemory()
            available = mem_total - mem_used
            logger.debug(
                "GPU memory: %.1f MB used / %.1f MB total",
                mem_used / 1024 / 1024,
                mem_total / 1024 / 1024,
            )
            return available
        except (RuntimeError, AttributeError, OSError) as exc:
            logger.debug("Could not query GPU memory (%s); skipping check", exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Add vectors with associated IDs.

        Args:
            ids: int64 array of shape ``(n,)``.
            vectors: float32 array of shape ``(n, dim)``.
        """
        if self._cpu_index is None:
            return
        if vectors.shape[0] == 0:
            return
        self._cpu_index.add_with_ids(vectors, ids)
        self._gpu_dirty = True
        self._next_id = max(self._next_id, int(ids.max()) + 1)

    def remove(self, ids: np.ndarray) -> None:
        """Remove vectors by ID."""
        if self._cpu_index is None or ids.size == 0:
            return
        self._cpu_index.remove_ids(ids)
        self._gpu_dirty = True

    def sync_gpu(self) -> None:
        """Explicitly sync GPU mirror from CPU index.

        Call this after bulk adds to pre-load the GPU before first search.
        Avoids lazy sync latency on first query.
        """
        self._sync_gpu()

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search the index.

        Returns:
            distances: float32 array ``(n_queries, k)``.
            indices: int64 array ``(n_queries, k)``.
        """
        n_queries = queries.shape[0]
        if self._cpu_index is None:
            return np.zeros((n_queries, k), dtype=np.float32), np.full(
                (n_queries, k), -1, dtype=np.int64
            )

        # Searching for more neighbours than the index holds is an error in
        # both FAISS and the brute-force fallback; clamp and pad with -1 so the
        # returned shape always matches the requested k.
        total = self.ntotal()
        if total == 0:
            return np.zeros((n_queries, k), dtype=np.float32), np.full(
                (n_queries, k), -1, dtype=np.int64
            )

        effective_k = min(k, total)
        # Use GPU if available and synced; no lazy sync (sync happens in embed_codebase)
        index = self._gpu_index if self._gpu_index is not None else self._cpu_index
        distances, indices = index.search(queries, effective_k)

        if effective_k < k:
            pad = k - effective_k
            distances = np.concatenate(
                [distances, np.zeros((n_queries, pad), dtype=np.float32)], axis=1
            )
            indices = np.concatenate(
                [indices, np.full((n_queries, pad), -1, dtype=np.int64)], axis=1
            )
        return distances, indices

    def reset(self) -> None:
        """Clear all vectors."""
        if self._cpu_index is not None:
            self._cpu_index.reset()
        self._gpu_index = None
        self._gpu_dirty = True
        self._next_id = 0

    def ntotal(self) -> int:
        if self._cpu_index is None:
            return 0
        return self._cpu_index.ntotal

    def new_ids(self, count: int) -> np.ndarray:
        """Allocate *count* fresh IDs."""
        ids = np.arange(self._next_id, self._next_id + count, dtype=np.int64)
        self._next_id += count
        return ids

    def save(self, path: str | Path) -> None:
        """Serialize CPU index to disk."""
        if self._cpu_index is None or faiss is None:
            return
        faiss.write_index(self._cpu_index, str(path))

    def load(self, path: str | Path) -> None:
        """Load CPU index from disk and wrap with IDMap."""
        if faiss is None:
            # Brute-force fallback: rebuild from the persisted embeddings so
            # searches work after a restart even without FAISS installed.
            embeddings_path = Path(path).with_name("embeddings.npy")
            if embeddings_path.exists():
                vectors = np.load(embeddings_path)
                if vectors.size:
                    self.dim = vectors.shape[1]
                    self._cpu_index = _BruteForceIndex(self.dim)
                    ids = np.arange(len(vectors), dtype=np.int64)
                    self._cpu_index.add_with_ids(vectors, ids)
                    self._next_id = len(vectors)
            return
        base = faiss.read_index(str(path))
        # Saved indices are already wrapped in an IndexIDMap; wrapping again
        # raises because the inner index is non-empty.
        idmap_types = tuple(
            t
            for t in (getattr(faiss, "IndexIDMap", None), getattr(faiss, "IndexIDMap2", None))
            if t is not None
        )
        if idmap_types and isinstance(base, idmap_types):
            self._cpu_index = base
        else:
            self._cpu_index = faiss.IndexIDMap(base)
        self._gpu_dirty = True
        self._gpu_index = None
        self._next_id = base.ntotal
        # Adopt the persisted vector dimension (callers may not know it yet,
        # e.g. when the embedder loads lazily on first use).
        self.dim = getattr(base, "d", self.dim)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_gpu(self) -> bool:
        return self._gpu_index is not None


# ---------------------------------------------------------------------------
# Brute-force fallback when faiss is unavailable
# ---------------------------------------------------------------------------


class _BruteForceIndex:
    """Simple in-memory brute-force index (for testing without faiss)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: dict[int, np.ndarray] = {}

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        for i, vid in enumerate(ids):
            self._vectors[int(vid)] = vectors[i].copy()

    def remove_ids(self, ids: np.ndarray) -> None:
        for vid in ids:
            self._vectors.pop(int(vid), None)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self._vectors:
            return (
                np.zeros((queries.shape[0], k), dtype=np.float32),
                np.full((queries.shape[0], k), -1, dtype=np.int64),
            )
        ids = np.array(list(self._vectors.keys()), dtype=np.int64)
        all_vecs = np.stack([self._vectors[int(v)] for v in ids], axis=0)
        scores = np.dot(all_vecs, queries.T).T  # (n_queries, n_vectors)
        # Top-k partial sort (never ask for more neighbours than exist)
        top_k = min(k, scores.shape[1])
        top_idx = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
        # Sort each row
        rows = []
        for i in range(scores.shape[0]):
            idx = top_idx[i]
            s = scores[i, idx]
            order = np.argsort(-s)
            rows.append((s[order], idx[order]))
        distances = np.stack([r[0] for r in rows], axis=0)
        indices = np.stack([r[1] for r in rows], axis=0)
        return distances, ids[indices]

    def reset(self) -> None:
        self._vectors.clear()

    @property
    def ntotal(self) -> int:
        return len(self._vectors)
