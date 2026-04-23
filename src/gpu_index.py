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

# faiss optional import handled gracefully
try:
    import faiss
    _HAS_FAISS = True
except Exception as exc:
    logger.warning("faiss not available (%s). Vector search will be brute-force.", exc)
    _HAS_FAISS = False
    faiss = None  # type: ignore[assignment]


class GpuIndex:
    """FAISS index manager with CPU source-of-truth + optional GPU mirror."""

    def __init__(self, dim: int, use_gpu: bool = True) -> None:
        self.dim = dim
        self._cpu_index: Any | None = None
        self._gpu_index: Any | None = None
        self._gpu_dirty = True
        self._gpu_available = False
        self._next_id = 0

        if _HAS_FAISS:
            # Build CPU IDMap over FlatIP (dot-product on L2-normalized vectors)
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
        except Exception:
            ngpu = 0
        if ngpu == 0:
            logger.info("No CUDA GPUs detected; using CPU FAISS.")
            return False
        try:
            self._gpu_res = faiss.StandardGpuResources()
            logger.info("CUDA GPU available for FAISS (%d device(s)).", ngpu)
            return True
        except Exception as exc:
            logger.warning("GPU init failed (%s); using CPU FAISS.", exc)
            return False

    def _sync_gpu(self) -> None:
        """Rebuild GPU mirror from CPU index if dirty."""
        if not self._gpu_available or not self._gpu_dirty or self._cpu_index is None:
            return
        try:
            self._gpu_index = faiss.index_cpu_to_gpu(self._gpu_res, 0, self._cpu_index)
            self._gpu_dirty = False
            logger.debug("GPU index synced (%d vectors).", self._cpu_index.ntotal)
        except Exception as exc:
            logger.warning("GPU sync failed (%s); falling back to CPU search.", exc)
            self._gpu_index = None

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

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search the index.

        Returns:
            distances: float32 array ``(n_queries, k)``.
            indices: int64 array ``(n_queries, k)``.
        """
        if self._cpu_index is None:
            return np.zeros((queries.shape[0], k), dtype=np.float32), np.full((queries.shape[0], k), -1, dtype=np.int64)

        # Prefer GPU if available and synced
        if self._gpu_available:
            self._sync_gpu()
        index = self._gpu_index if self._gpu_index is not None else self._cpu_index
        return index.search(queries, k)

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
            return
        base = faiss.read_index(str(path))
        self._cpu_index = faiss.IndexIDMap(base)
        self._gpu_dirty = True
        self._gpu_index = None
        self._next_id = base.ntotal

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
        # Top-k partial sort
        if k >= scores.shape[1]:
            top_k = k
        else:
            top_k = k
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
