"""Agent-facing tool interface for GPU-accelerated code embedding.

The tool never returns raw source code to the agent. It only returns buffer
handles, file paths, line numbers, and similarity scores.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
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
            Each buffer gets a ``.vkbuff/`` subdirectory.
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
    # Schema exposure (Phase 6.1)
    # ------------------------------------------------------------------
    @staticmethod
    def get_tool_schemas() -> list[dict[str, Any]]:
        """Return formal JSON schemas for all exposed tools."""
        from src.tool_schema import get_all_schemas

        return get_all_schemas()

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

        # Stage 5: persist buffers and metadata in .vkbuff/ directory
        buffer_id = str(uuid.uuid4())
        buffer_dir = self.work_dir / f"{buffer_id}.vkbuff"
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

        # Compute per-file content hashes for incremental reload / discard
        file_hashes: dict[str, str] = {}
        for f in files:
            rel = str(f.relative_to(root))
            file_hashes[rel] = hashlib.sha256(f.read_bytes()).hexdigest()

        # Build and persist source snapshot (authoritative in-buffer source text)
        source_snapshot: dict[str, list[str]] = {}
        for rec in all_lines:
            rel_file = rec.get("file", "")
            if not rel_file:
                # Fallback: derive from file_index
                for entry in file_index:
                    if entry["start_idx"] <= len(source_snapshot.get(rel_file, [])) < entry["end_idx"]:
                        rel_file = entry["path"]
                        break
            source_snapshot.setdefault(rel_file, []).append(rec["text"])

        snapshot_path = buffer_dir / "source_snapshot.json"
        snapshot_path.write_text(
            json.dumps(source_snapshot, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Stage 6: register
        self._registry[buffer_id] = {
            "root": str(root),
            "buffer_dir": str(buffer_dir),
            "token_count": token_count,
            "embedding_dim": self._embedding_dim,
            "size_bytes": len(data_bytes),
            "file_hashes": file_hashes,
            "pattern": pattern,
            "language_hint": language_hint,
            "dirty_files": {},
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
    # Reload without re-embedding (Phase 6.4)
    # ------------------------------------------------------------------
    def reload_codebase(
        self,
        buffer_id: str,
    ) -> dict[str, Any]:
        """Reload an existing buffer if file hashes match, avoiding re-embedding.

        Args:
            buffer_id: Existing buffer handle.

        Returns:
            Dict with ``status``, ``buffer_id``, ``token_count``, ``size_bytes``,
            and ``message``.
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        root = Path(info["root"])
        pattern = info.get("pattern", "*.py")
        if root.is_file():
            files = [root]
        else:
            files = sorted(root.rglob(pattern))

        old_hashes = info.get("file_hashes", {})
        mismatched: list[str] = []
        for f in files:
            rel = str(f.relative_to(root))
            current = hashlib.sha256(f.read_bytes()).hexdigest()
            if old_hashes.get(rel) != current:
                mismatched.append(rel)

        if not mismatched:
            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "token_count": info["token_count"],
                "size_bytes": info["size_bytes"],
                "message": "All file hashes match; reloaded without re-embedding.",
            }

        # Hash mismatch: fall back to full re-embed
        logger.info("Hash mismatch for %s in files %s; re-embedding.", buffer_id, mismatched)
        return self.embed_codebase(
            root,
            language_hint=info.get("language_hint"),
            pattern=pattern,
        )

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
        shutil.rmtree(info["buffer_dir"], ignore_errors=True)
        return {"status": "ok", "message": f"Deleted buffer {buffer_id}"}

    # ------------------------------------------------------------------
    # Read / Write / Commit (Agent editing workflow)
    # ------------------------------------------------------------------
    def read_code(
        self,
        buffer_id: str,
        file: str | None = None,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read raw source text from the buffer.

        .. note::
            This intentionally returns raw source text so that an agent can
            edit it. The original ``semantic_search`` / ``cluster_code``
            tools still never expose source text.

        Args:
            buffer_id: Handle returned by :meth:`embed_codebase`.
            file: Relative file path to read. If ``None``, all files are
                returned as a mapping.
            start_line: 1-based start line (inclusive).
            end_line: 1-based end line (exclusive). ``None`` means "to end".

        Returns:
            Dict with ``status`` and either ``lines`` or ``files``.
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing for buffer."}

        if file is not None:
            if file not in snapshot:
                return {"status": "error", "message": f"File not in buffer: {file}"}
            lines = snapshot[file]
            end = end_line if end_line is not None else len(lines) + 1
            selected = lines[start_line - 1 : end - 1]
            return {
                "status": "ok",
                "file": file,
                "start_line": start_line,
                "end_line": end,
                "lines": selected,
            }

        # Return all files
        result: dict[str, list[str]] = {}
        for fname, lines in snapshot.items():
            end = end_line if end_line is not None else len(lines) + 1
            result[fname] = lines[start_line - 1 : end - 1]
        return {"status": "ok", "files": result}

    def write_code(
        self,
        buffer_id: str,
        file: str,
        start_line: int,
        new_lines: list[str],
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Replace a range of lines in the buffer and re-embed the file.

        Args:
            buffer_id: Handle returned by :meth:`embed_codebase`.
            file: Relative file path to edit.
            start_line: 1-based start line (inclusive).
            new_lines: List of replacement line strings (no newlines).
            end_line: 1-based end line (exclusive). ``None`` means "to end
                of file".

        Returns:
            Dict with ``status``, ``changed_lines``, and ``file``.
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing for buffer."}

        if file not in snapshot:
            return {"status": "error", "message": f"File not in buffer: {file}"}

        old_lines = snapshot[file]
        end = end_line if end_line is not None else len(old_lines) + 1
        # Build new line list
        new_file_lines = old_lines[: start_line - 1] + new_lines + old_lines[end - 1 :]
        snapshot[file] = new_file_lines
        self._save_source_snapshot(buffer_id, snapshot)

        # Rebuild embeddings / metadata for this file
        rebuild_result = self._rebuild_file_region(
            buffer_id, file, new_file_lines, info.get("language_hint")
        )
        if rebuild_result.get("status") != "ok":
            return rebuild_result

        # Mark file dirty
        dirty = info.setdefault("dirty_files", {})
        dirty[file] = True
        self._save_registry()

        return {
            "status": "ok",
            "file": file,
            "changed_lines": len(new_lines),
            "replaced_lines": end - start_line,
            "total_lines": len(new_file_lines),
        }

    def diff(self, buffer_id: str) -> dict[str, Any]:
        """List files that differ from the original on-disk versions.

        Args:
            buffer_id: Handle returned by :meth:`embed_codebase`.

        Returns:
            Dict with ``status`` and ``changed_files``.
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        root = Path(info["root"])
        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}

        changed: list[dict[str, Any]] = []
        old_hashes = info.get("file_hashes", {})
        for rel_path, lines in snapshot.items():
            current_text = "\n".join(lines)
            current_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
            if old_hashes.get(rel_path) != current_hash:
                disk_path = root / rel_path
                disk_lines = disk_path.read_text(encoding="utf-8").splitlines() if disk_path.exists() else []
                changed.append(
                    {
                        "file": rel_path,
                        "buffer_lines": len(lines),
                        "disk_lines": len(disk_lines),
                        "dirty": info.get("dirty_files", {}).get(rel_path, False),
                    }
                )

        return {"status": "ok", "changed_files": changed}

    def discard(
        self,
        buffer_id: str,
        file: str | None = None,
    ) -> dict[str, Any]:
        """Revert buffer state for one or all files back to the on-disk original.

        Args:
            buffer_id: Handle returned by :meth:`embed_codebase`.
            file: Relative file path. If ``None``, all files are reverted.

        Returns:
            Dict with ``status`` and ``reverted_files``.
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        root = Path(info["root"])
        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}

        files_to_revert = [file] if file is not None else list(snapshot.keys())
        reverted: list[str] = []
        dirty = info.setdefault("dirty_files", {})

        for rel_path in files_to_revert:
            disk_path = root / rel_path
            if not disk_path.exists():
                continue
            with disk_path.open("r", encoding="utf-8", errors="replace") as fh:
                disk_lines = fh.read().splitlines()
            snapshot[rel_path] = disk_lines
            rebuild = self._rebuild_file_region(
                buffer_id, rel_path, disk_lines, info.get("language_hint")
            )
            if rebuild.get("status") == "ok":
                dirty.pop(rel_path, None)
                reverted.append(rel_path)

        self._save_source_snapshot(buffer_id, snapshot)
        self._save_registry()
        return {"status": "ok", "reverted_files": reverted}

    def commit(
        self,
        buffer_id: str,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Write dirty files from the buffer back to disk.

        Args:
            buffer_id: Handle returned by :meth:`embed_codebase`.
            dry_run: If ``True``, report what would be written without
                touching disk.

        Returns:
            Dict with ``status``, ``written_files``, and ``dry_run``.
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        root = Path(info["root"])
        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}

        dirty = info.get("dirty_files", {})
        if not dirty:
            return {"status": "ok", "written_files": [], "dry_run": dry_run}

        written: list[str] = []
        new_hashes: dict[str, str] = {}

        for rel_path in dirty:
            lines = snapshot.get(rel_path, [])
            disk_path = root / rel_path

            if not dry_run:
                # Safety: check hash hasn't changed on disk since embed
                if disk_path.exists():
                    disk_hash = hashlib.sha256(disk_path.read_bytes()).hexdigest()
                    old_hash = info.get("file_hashes", {}).get(rel_path)
                    if old_hash is not None and disk_hash != old_hash:
                        return {
                            "status": "error",
                            "message": (
                                f"File {rel_path} changed on disk since embedding "
                                f"(hash mismatch). Aborting commit to avoid overwriting."
                            ),
                        }
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                disk_path.write_text("\n".join(lines), encoding="utf-8")
                new_hashes[rel_path] = hashlib.sha256(
                    "\n".join(lines).encode("utf-8")
                ).hexdigest()

            written.append(rel_path)

        if not dry_run:
            info["file_hashes"].update(new_hashes)
            info["dirty_files"] = {}
            self._save_registry()

        return {"status": "ok", "written_files": written, "dry_run": dry_run}

    # ------------------------------------------------------------------
    # Buffer editing helpers
    # ------------------------------------------------------------------
    def _load_source_snapshot(self, buffer_id: str) -> dict[str, list[str]] | None:
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return None
        snapshot_path = Path(info["buffer_dir"]) / "source_snapshot.json"
        if not snapshot_path.exists():
            return None
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        # Ensure values are lists of strings
        return {k: list(v) for k, v in data.items()}

    def _save_source_snapshot(
        self, buffer_id: str, snapshot: dict[str, list[str]]
    ) -> None:
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return
        snapshot_path = Path(info["buffer_dir"]) / "source_snapshot.json"
        snapshot_path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _rebuild_file_region(
        self,
        buffer_id: str,
        rel_path: str,
        new_lines: list[str],
        language_hint: str | None,
    ) -> dict[str, Any]:
        """Re-tokenize and re-embed a single file, then splice into the buffer."""
        import tempfile

        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        buffer_dir = Path(info["buffer_dir"])
        meta_path = buffer_dir / "metadata.json"
        index_path = buffer_dir / "file_index.json"
        data_path = buffer_dir / "embeddings.bin"
        offsets_path = buffer_dir / "offsets.bin"

        # Load current structures
        metadata = load_metadata(meta_path)
        file_index = json.loads(index_path.read_text(encoding="utf-8"))
        old_embeddings = np.fromfile(data_path, dtype=np.float32).reshape(
            -1, info["embedding_dim"]
        )

        # Find file range
        file_entry: dict[str, Any] | None = None
        file_entry_idx = -1
        for idx, entry in enumerate(file_index["files"]):
            if entry["path"] == rel_path:
                file_entry = entry
                file_entry_idx = idx
                break

        if file_entry is None:
            return {"status": "error", "message": f"File not in index: {rel_path}"}

        old_start = file_entry["start_idx"]
        old_end = file_entry["end_idx"]
        old_count = old_end - old_start

        # Tokenize new file content via temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write("\n".join(new_lines))
            tmp_path = tmp.name

        try:
            new_line_recs = tokenize_file(tmp_path, language_hint=language_hint)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Ensure each record knows its file
        for rec in new_line_recs:
            rec["file"] = rel_path

        new_count = len(new_line_recs)
        new_texts = [rec["text"] for rec in new_line_recs]
        new_embeddings = self._embedder.encode(new_texts)

        # Build new metadata list
        new_metadata = metadata[:old_start] + [
            {k: v for k, v in rec.items() if k != "embedding"}
            for rec in new_line_recs
        ]
        # Each new_line_rec doesn't have "embedding" key since tokenize_file doesn't
        # produce it; but to be safe we strip it if present.
        new_metadata += metadata[old_end:]

        # Build new embeddings array
        new_emb_array = np.concatenate(
            [
                old_embeddings[:old_start],
                new_embeddings,
                old_embeddings[old_end:],
            ]
        )

        # Update file_index: adjust this file and all subsequent files
        delta = new_count - old_count
        file_index["files"][file_entry_idx]["end_idx"] = old_start + new_count
        for subsequent in file_index["files"][file_entry_idx + 1 :]:
            subsequent["start_idx"] += delta
            subsequent["end_idx"] += delta

        # Flatten and persist
        lines_data: list[dict[str, Any]] = []
        for rec, emb in zip(new_line_recs, new_embeddings):
            meta = dict(rec)
            meta["embedding"] = emb
            lines_data.append(meta)

        # Rebuild the full lines_data for flattening
        full_lines_data: list[dict[str, Any]] = []
        for i, meta in enumerate(new_metadata):
            if old_start <= i < old_start + new_count:
                # From newly embedded region
                rec = new_line_recs[i - old_start]
                full_rec = dict(rec)
                full_rec["embedding"] = new_embeddings[i - old_start]
                full_lines_data.append(full_rec)
            else:
                # From old embeddings
                full_rec = dict(meta)
                full_rec["embedding"] = new_emb_array[i]
                full_lines_data.append(full_rec)

        data_bytes, offsets_bytes, metadata_list = flatten_embeddings(
            full_lines_data, info["embedding_dim"]
        )

        data_path.write_bytes(data_bytes)
        offsets_path.write_bytes(offsets_bytes)
        save_metadata(meta_path, metadata_list)
        index_path.write_text(
            json.dumps(file_index, indent=2), encoding="utf-8"
        )

        return {"status": "ok", "rebuilt_lines": new_count}

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
