"""Agent-facing tool interface for GPU-accelerated code embedding.

The tool never returns raw source code to the agent. It only returns buffer
handles, file paths, line numbers, and similarity scores.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.buffer_manager import BufferManager
from src.diff_engine import compute_diff, hash_lines
from src.embedder import Embedder
from src.flatten import flatten_embeddings
from src.metadata_store import load_metadata, save_metadata
from src.size_guard import check_size
from src.tokenizer import tokenize_file
from src.vulkan_context import VulkanContext

logger = logging.getLogger(__name__)


class CodeEmbeddingTool:
    """Embed a codebase into GPU/CPU buffers and expose search + cluster.

    Args:
        work_dir: Directory where buffer files and metadata are persisted.
        model_name: Sentence-transformers model name.
        device: torch device (``"cuda"``, ``"cpu"``, or ``None`` for auto).
        threshold_mb: Size-guard threshold in megabytes.
    """

    def __init__(
        self,
        work_dir: str | Path,
        model_name: str | None = None,
        device: str | None = None,
        threshold_mb: float = 500.0,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_mb = threshold_mb

        self._embedder = Embedder(model_name=model_name, device=device)
        self._embedding_dim = self._embedder.embedding_dim
        self._ctx = VulkanContext()
        self._buf_mgr = BufferManager(self._ctx)

        # Registry of embedded codebases: buffer_id -> metadata path
        self._registry_path = self.work_dir / "registry.json"
        self._registry: dict[str, dict[str, Any]] = (
            json.loads(self._registry_path.read_text(encoding="utf-8"))
            if self._registry_path.exists()
            else {}
        )

    # ------------------------------------------------------------------
    # Pre-flight size check
    # ------------------------------------------------------------------
    def check_codebase(
        self,
        path: str | Path,
        pattern: str = "*.py",
    ) -> dict[str, Any]:
        """Lightweight pre-flight size estimate without tokenizing.

        Args:
            path: Root directory or single file to scan.
            pattern: Glob pattern for files when *path* is a directory.

        Returns:
            Dict with ``status``, ``estimated_lines``, ``estimated_tokens``,
            ``estimated_mb``, ``threshold_mb``, and ``message``.
        """
        root = Path(path)
        if root.is_file():
            files = [root]
        else:
            files = sorted(root.rglob(pattern))

        if not files:
            return {
                "status": "warning",
                "message": f"No files matched pattern '{pattern}' in {root}",
            }

        total_lines = 0
        for f in files:
            try:
                with f.open("r", encoding="utf-8", errors="replace") as fh:
                    total_lines += sum(1 for _ in fh)
            except Exception as exc:
                logger.debug("Could not count lines in %s: %s", f, exc)

        # Rough heuristic: tokens per line ~ 8 on average
        estimated_tokens = total_lines * 8
        size_check = check_size(
            estimated_tokens, self._embedding_dim, self.threshold_mb
        )

        return {
            "status": size_check["status"],
            "estimated_lines": total_lines,
            "estimated_tokens": estimated_tokens,
            "estimated_mb": size_check["estimated_mb"],
            "threshold_mb": size_check["threshold_mb"],
            "message": (
                f"Estimated {total_lines} lines (~{estimated_tokens} tokens) "
                f"=> ~{size_check['estimated_mb']:.1f} MB"
            ),
        }

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def embed_codebase(
        self,
        path: str | Path,
        language_hint: str | None = None,
        pattern: str = "*.py",
    ) -> dict[str, Any]:
        """Ingest a codebase directory, embed it, and return a buffer handle.

        Args:
            path: Root directory or single file to ingest.
            language_hint: Optional language override (e.g. ``"python"``).
            pattern: Glob pattern for files when *path* is a directory.

        Returns:
            Dict with ``status``, ``buffer_id``, ``token_count``,
            ``size_bytes``, and ``message``.
        """
        root = Path(path)
        if root.is_file():
            files = [root]
        else:
            files = sorted(root.rglob(pattern))

        if not files:
            return {
                "status": "warning",
                "message": f"No files matched pattern '{pattern}' in {root}",
            }

        # Pre-flight size check (cheap line-count only)
        preflight = self.check_codebase(path, pattern)
        if preflight["status"] == "exceeds_threshold":
            return {
                "status": "warning",
                "message": (
                    f"Codebase too large for embedding ({preflight['estimated_mb']:.1f} MB "
                    f"exceeds threshold {preflight['threshold_mb']:.1f} MB). "
                    f"Suggest narrowing with a more specific glob or sub-directory."
                ),
                "suggested_max": f"{preflight['threshold_mb']:.0f} MB",
                "estimated_mb": preflight["estimated_mb"],
            }

        # Stage 1: tokenize
        all_lines: list[dict[str, Any]] = []
        file_index: list[dict[str, Any]] = []
        for f in files:
            try:
                lines = tokenize_file(f, language_hint=language_hint)
            except Exception as exc:
                logger.warning("Failed to tokenize %s: %s", f, exc)
                continue
            base_idx = len(all_lines)
            for rec in lines:
                all_lines.append(rec)
            file_index.append(
                {
                    "path": str(f.relative_to(root)),
                    "start_idx": base_idx,
                    "end_idx": len(all_lines),
                }
            )

        token_count = len(all_lines)
        if token_count == 0:
            return {
                "status": "warning",
                "message": "No tokens extracted from input files.",
            }

        # Stage 2: size guard
        size_check = check_size(token_count, self._embedding_dim, self.threshold_mb)
        if size_check["status"] == "exceeds_threshold":
            return {
                "status": "warning",
                "message": (
                    f"Codebase too large for embedding ({size_check['estimated_mb']:.1f} MB "
                    f"exceeds threshold {size_check['threshold_mb']:.1f} MB). "
                    f"Suggest narrowing with a more specific glob or sub-directory."
                ),
                "suggested_max": f"{size_check['threshold_mb']:.0f} MB",
                "estimated_mb": size_check["estimated_mb"],
            }

        # Stage 3: embed per-line text (use full line text for context)
        texts = [rec["text"] for rec in all_lines]
        embeddings = self._embedder.encode(texts)

        # Stage 4: flatten
        lines_data = []
        for rec, emb in zip(all_lines, embeddings):
            meta = dict(rec)
            meta["embedding"] = emb
            lines_data.append(meta)

        data_bytes, offsets_bytes, metadata_list = flatten_embeddings(
            lines_data, self._embedding_dim
        )

        # Stage 5: persist buffers and metadata
        buffer_id = str(uuid.uuid4())
        buffer_dir = self.work_dir / buffer_id
        buffer_dir.mkdir(parents=True, exist_ok=True)

        data_path = buffer_dir / "embeddings.bin"
        offsets_path = buffer_dir / "offsets.bin"
        meta_path = buffer_dir / "metadata.json"
        index_path = buffer_dir / "file_index.json"

        data_path.write_bytes(data_bytes)
        offsets_path.write_bytes(offsets_bytes)
        save_metadata(meta_path, metadata_list)
        index_path.write_text(
            json.dumps({"files": file_index}, indent=2), encoding="utf-8"
        )

        # Stage 6: register
        self._registry[buffer_id] = {
            "root": str(root),
            "buffer_dir": str(buffer_dir),
            "token_count": token_count,
            "embedding_dim": self._embedding_dim,
            "size_bytes": len(data_bytes),
        }
        self._save_registry()

        # Stage 7: optional GPU upload (skipped in CPU mode)
        if not self._ctx.cpu_mode:
            self._upload_to_gpu(buffer_id, data_bytes, offsets_bytes)

        return {
            "status": "ok",
            "buffer_id": buffer_id,
            "token_count": token_count,
            "size_bytes": len(data_bytes),
            "message": f"Embedded {token_count} lines from {len(files)} files.",
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def semantic_search(
        self,
        buffer_id: str,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Find the top-K lines most similar to *query*.

        Args:
            buffer_id: Handle returned by :meth:`embed_codebase`.
            query: Natural-language query string.
            top_k: Number of results to return.

        Returns:
            Dict with ``status`` and ``matches``. Each match contains
            ``file``, ``line``, ``score`` (never raw source text).
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        # Load embeddings from disk
        data_path = Path(info["buffer_dir"]) / "embeddings.bin"
        meta_path = Path(info["buffer_dir"]) / "metadata.json"
        index_path = Path(info["buffer_dir"]) / "file_index.json"

        embeddings = np.fromfile(data_path, dtype=np.float32).reshape(
            -1, info["embedding_dim"]
        )
        metadata = load_metadata(meta_path)
        file_index = json.loads(index_path.read_text(encoding="utf-8"))

        # Embed query
        q_emb = self._embedder.encode([query])[0]

        # Compute scores
        scores = self._ctx.similarity_search(embeddings, q_emb)

        # Top-K partial sort on CPU (fast enough for most codebases)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        matches = []
        for idx in top_indices:
            meta = metadata[int(idx)]
            file_path = self._resolve_file(int(idx), file_index["files"])
            matches.append(
                {
                    "file": file_path,
                    "line": meta.get("line_num", 0),
                    "score": float(scores[idx]),
                }
            )

        return {"status": "ok", "matches": matches}

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    def cluster_code(
        self,
        buffer_id: str,
        threshold: float = 0.75,
    ) -> dict[str, Any]:
        """Group similar code regions into clusters.

        Args:
            buffer_id: Handle returned by :meth:`embed_codebase`.
            threshold: Cosine-similarity threshold for grouping
                (default 0.75).

        Returns:
            Dict with ``status`` and ``clusters``. Each cluster contains
            ``file``, ``start_line``, ``end_line``, ``size``,
            ``avg_score``.
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        data_path = Path(info["buffer_dir"]) / "embeddings.bin"
        meta_path = Path(info["buffer_dir"]) / "metadata.json"
        index_path = Path(info["buffer_dir"]) / "file_index.json"

        embeddings = np.fromfile(data_path, dtype=np.float32).reshape(
            -1, info["embedding_dim"]
        )
        metadata = load_metadata(meta_path)
        file_index = json.loads(index_path.read_text(encoding="utf-8"))

        clusters = self._ctx.cluster_regions(embeddings, threshold)

        results = []
        for c in clusters:
            start_idx = c["start_token"]
            end_idx = c["end_token"]
            start_meta = metadata[start_idx]
            end_meta = metadata[end_idx]
            file_path = self._resolve_file(start_idx, file_index["files"])
            results.append(
                {
                    "file": file_path,
                    "start_line": start_meta.get("line_num", 0),
                    "end_line": end_meta.get("line_num", 0),
                    "size": c["count"],
                    "avg_score": round(c["avg_score"], 4),
                }
            )

        return {"status": "ok", "clusters": results}

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------
    def update_codebase(
        self,
        buffer_id: str,
        file_path: str | Path,
        language_hint: str | None = None,
    ) -> dict[str, Any]:
        """Re-embed a single file and patch the existing buffer.

        Args:
            buffer_id: Existing buffer handle.
            file_path: Path to the file that changed.
            language_hint: Optional language override.

        Returns:
            Update status dict.
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        # Load previous metadata
        meta_path = Path(info["buffer_dir"]) / "metadata.json"
        prev_metadata = load_metadata(meta_path)

        # Re-tokenize
        new_lines = tokenize_file(file_path, language_hint=language_hint)
        new_texts = [rec["text"] for rec in new_lines]
        new_hashes = hash_lines(new_texts)

        # Find old lines for this file
        rel_path = str(Path(file_path).relative_to(info["root"]))
        old_hashes = [
            hashlib.sha256(
                rec.get("text", "").encode("utf-8", errors="replace")
            ).hexdigest()
            for rec in prev_metadata
            if rec.get("file", "") == rel_path
        ]

        changed = compute_diff(old_hashes, new_hashes)
        if not changed:
            return {"status": "ok", "message": "No changes detected.", "changed_lines": 0}

        # Re-embed changed lines only
        changed_texts = [new_texts[i] for i in changed]
        new_embeddings = self._embedder.encode(changed_texts)

        # TODO: patch embeddings.bin in place or rebuild if size changed
        # For simplicity, fall back to full rebuild of this file's region
        return {
            "status": "ok",
            "message": f"Detected {len(changed)} changed lines. Full rebuild recommended.",
            "changed_lines": len(changed),
        }

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------
    def list_buffers(self) -> dict[str, Any]:
        """Return all registered buffer handles (no raw code)."""
        return {
            "status": "ok",
            "buffers": [
                {"buffer_id": bid, **info} for bid, info in self._registry.items()
            ],
        }

    def delete_buffer(self, buffer_id: str) -> dict[str, Any]:
        """Remove a buffer and its on-disk files."""
        info = self._registry.pop(buffer_id, None)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}
        self._save_registry()
        import shutil

        shutil.rmtree(info["buffer_dir"], ignore_errors=True)
        return {"status": "ok", "message": f"Deleted buffer {buffer_id}"}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_buffer_info(self, buffer_id: str) -> dict[str, Any] | None:
        return self._registry.get(buffer_id)

    def _save_registry(self) -> None:
        with self._registry_path.open("w", encoding="utf-8") as fh:
            json.dump(self._registry, fh, indent=2)

    def _resolve_file(self, token_idx: int, file_index: list[dict[str, Any]]) -> str:
        for entry in file_index:
            if entry["start_idx"] <= token_idx < entry["end_idx"]:
                return entry["path"]
        return "unknown"

    def _upload_to_gpu(
        self,
        buffer_id: str,
        data_bytes: bytes,
        offsets_bytes: bytes,
    ) -> None:
        # Not yet fully implemented for Vulkan compute pipeline
        logger.info("GPU upload for %s skipped (compute pipeline WIP)", buffer_id)

    def close(self) -> None:
        """Release all GPU resources."""
        self._buf_mgr.destroy()
        self._ctx.destroy()

    def __enter__(self) -> CodeEmbeddingTool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
