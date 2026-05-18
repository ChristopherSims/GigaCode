"""Buffer management layer for CodeEmbeddingTool.

Handles:
- Buffer registry (metadata for embedded codebases)
- Buffer lifecycle (embed, reload, delete)
- Snapshot management (external change detection, 3-way merge)
- File read/write/commit operations
- Audit logging
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from gigacode.audit_logger import AuditLogger

from gigacode.buffer_state import BufferState, BufferStateTransition
from gigacode.chunker import CodeChunk, chunk_file, chunk_text
from gigacode.constants import DEFAULT_THRESHOLD_MB, MAX_DIRTY_BEFORE_AUTO_REBUILD
from gigacode.json_logger import StructuredJsonLogger
from gigacode.size_guard import check_size
from gigacode.snapshot_manager import SnapshotManager
from gigacode.state_manager import StateManager

logger = logging.getLogger(__name__)
json_logger = StructuredJsonLogger("buffer_manager")


__all__ = [
    "BufferManager",
]


class BufferManager:
    """Manages buffer registry, persistence, and file I/O operations.

    Responsibilities:
    - Maintain registry of embedded codebases
    - Manage buffer lifecycle (create, reload, delete)
    - Handle file read/write/commit with 3-way merge
    - Snapshot management for change detection
    - Audit logging for compliance

    Args:
        work_dir: Directory where buffers and registry are persisted.
        state_manager: StateManager for crash recovery and transactions.
        embedding_dim: Dimension of embeddings (for size checks).
        threshold_mb: Size-guard threshold in megabytes.
    """

    def __init__(
        self,
        work_dir: Path,
        state_manager: StateManager,
        embedding_dim: int,
        threshold_mb: float = DEFAULT_THRESHOLD_MB,
        audit_logger: Optional["AuditLogger"] = None,
        user_id: str = "default",
    ) -> None:
        """Initialize BufferManager.

        Args:
            work_dir: Directory for buffer storage and registry
            state_manager: StateManager for crash recovery
            embedding_dim: Embedding dimension for size checks
            threshold_mb: Size guard threshold in MB
            audit_logger: Optional AuditLogger for structured operation logging
            user_id: User ID for audit logging context
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._state_manager = state_manager
        self._embedding_dim = embedding_dim
        self.threshold_mb = threshold_mb
        self._audit_logger = audit_logger
        self._user_id = user_id

        # Registry of embedded codebases: buffer_id -> metadata dict
        self._registry_path = self.work_dir / "registry.json"
        self._registry: dict[str, dict[str, Any]] = {}
        if self._registry_path.exists():
            try:
                self._registry = json.loads(self._registry_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                logger.warning("Corrupted registry.json — starting with empty registry")
                self._registry = {}

        # Snapshot managers: buffer_id -> SnapshotManager (one per buffer)
        self._snapshot_managers: dict[str, SnapshotManager] = {}

    # ------------------------------------------------------------------
    # Directory hash helpers (session persistence)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_dir_hash(path: str | Path, pattern: str = "*.py") -> str:
        """Recursively hash all files in a directory (or single file).

        Computes a stable SHA256 hash from sorted file paths combined with
        each file's mtime and size. This is used to detect if a codebase
        has changed since the last embed.

        Args:
            path: Directory or single file to hash.
            pattern: Glob pattern for recursive file matching when path is
                a directory. Ignored when path is a single file.

        Returns:
            A stable hex digest string.
        """
        root = Path(path)
        files = [root] if root.is_file() else sorted(root.rglob(pattern))
        hasher = hashlib.sha256()
        for f in files:
            if not f.is_file():
                continue
            stat = f.stat()
            entry = f"{f.as_posix()}::{stat.st_mtime}::{stat.st_size}"
            hasher.update(entry.encode("utf-8"))
        return hasher.hexdigest()

    def check_existing_buffer(
        self,
        path: str | Path,
        pattern: str = "*.py",
    ) -> dict[str, Any]:
        """Check if a buffer already exists for the given codebase.

        Computes the directory hash for ``path`` and scans the registry
        for a buffer whose ``source_hash`` matches.  If found and the
        hash matches, returns a resume response; otherwise returns a
        not-found response.

        Args:
            path: Directory or single file that was previously embedded.
            pattern: Glob pattern used when the buffer was created.

        Returns:
            ``{"status": "resumed", "buffer_id": ..., "num_chunks": ...}``
            if a matching buffer exists, or
            ``{"status": "not_found"}`` otherwise.
        """
        source_hash = self._compute_dir_hash(path, pattern)
        for buffer_id, info in self._registry.items():
            if info.get("source_hash") == source_hash:
                return {
                    "status": "resumed",
                    "buffer_id": buffer_id,
                    "num_chunks": info.get("chunk_count", 0),
                }
        return {"status": "not_found"}

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------
    def save_session(
        self,
        alias: str,
        buffer_ids: list[str],
    ) -> dict[str, Any]:
        """Save a named session to disk.

        Writes a JSON session file in ``work_dir / ".sessions"`` so that
        the set of buffers can be restored later via :meth:`load_session`.

        Args:
            alias: Human-readable name for the session.
            buffer_ids: List of buffer IDs to include in the session.

        Returns:
            ``{"status": "ok", "alias": ..., "session_path": ...}``
        """
        sessions_dir = self.work_dir / ".sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        session_path = sessions_dir / f"{alias}.json"
        payload = {
            "alias": alias,
            "buffer_ids": buffer_ids,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "bookmarks": {},
        }
        session_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {"status": "ok", "alias": alias, "session_path": str(session_path)}

    def load_session(self, alias: str) -> dict[str, Any]:
        """Load a previously saved session.

        Args:
            alias: Session name (the ``.json`` file under
                ``work_dir / ".sessions"`` without the extension).

        Returns:
            ``{"status": "ok", "buffer_ids": [...], "bookmarks": {}}``
            on success.  If the session file does not exist, returns
            ``{"status": "error", "message": "..."}``.  If any
            buffer IDs are no longer present in the registry, they are
            listed under the ``missing_buffer_ids`` key.
        """
        session_path = self.work_dir / ".sessions" / f"{alias}.json"
        if not session_path.exists():
            return {
                "status": "error",
                "message": f"Session '{alias}' not found at {session_path}",
            }
        try:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "status": "error",
                "message": f"Failed to read session '{alias}': {exc}",
            }
        buffer_ids = payload.get("buffer_ids", [])
        missing = [bid for bid in buffer_ids if bid not in self._registry]
        result: dict[str, Any] = {
            "status": "ok",
            "buffer_ids": buffer_ids,
            "bookmarks": payload.get("bookmarks", {}),
        }
        if missing:
            result["missing_buffer_ids"] = missing
        return result

    def list_sessions(self) -> dict[str, Any]:
        """List all saved sessions.

        Returns:
            ``{"sessions": [{"alias": ..., "saved_at": ..., "buffer_count": ...}, ...]}``
        """
        sessions_dir = self.work_dir / ".sessions"
        sessions: list[dict[str, Any]] = []
        if sessions_dir.exists():
            for fp in sorted(sessions_dir.glob("*.json")):
                try:
                    payload = json.loads(fp.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                sessions.append(
                    {
                        "alias": payload.get("alias", fp.stem),
                        "saved_at": payload.get("saved_at"),
                        "buffer_count": len(payload.get("buffer_ids", [])),
                    }
                )
        return {"sessions": sessions}

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------
    def _audit_log(
        self,
        operation: str,
        buffer_id: str | None = None,
        status: str = "ok",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an operation to the audit logger.

        Delegates to AuditLogger if available; otherwise skips (minimal footprint).

        Args:
            operation: Operation name (e.g., 'embed_codebase', 'write_code')
            buffer_id: Buffer ID associated with operation
            status: Status (e.g., 'ok', 'error', 'conflict')
            details: Additional operation-specific metadata
        """
        if not self._audit_logger:
            return

        try:
            # Map status to AuditStatus

            if status == "error":
                error_msg = details.get("error", "Unknown error") if details else "Unknown error"
                self._audit_logger.log_failure(
                    operation=operation,
                    user_id=self._user_id,
                    role="AGENT",  # BufferManager operations run as AGENT
                    error_message=error_msg,
                    buffer_id=buffer_id,
                    details=details or {},
                )
            else:
                self._audit_logger.log_success(
                    operation=operation,
                    user_id=self._user_id,
                    role="AGENT",
                    buffer_id=buffer_id,
                    details=details or {},
                )
        except (OSError, ValueError, AttributeError) as exc:
            json_logger.warning(
                operation="audit_log",
                status="error",
                message=f"Could not write audit log: {exc}",
            )

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------
    def _get_buffer_info(self, buffer_id: str) -> dict[str, Any] | None:
        """Get buffer metadata from registry."""
        return self._registry.get(buffer_id)

    def _get_snapshot_manager(self, buffer_id: str) -> SnapshotManager | None:
        """Get or load SnapshotManager for a buffer."""
        if buffer_id in self._snapshot_managers:
            return self._snapshot_managers[buffer_id]

        info = self._get_buffer_info(buffer_id)
        if not info:
            return None

        buffer_dir = Path(info["buffer_dir"])
        if not buffer_dir.exists():
            return None

        # Load snapshot manager from disk
        snapshot_mgr = SnapshotManager(buffer_dir)
        self._snapshot_managers[buffer_id] = snapshot_mgr
        return snapshot_mgr

    def _save_registry(self) -> None:
        """Atomically save registry to disk."""
        import os as _os
        import threading as _threading

        try:
            temp_path = (
                self._registry_path.parent
                / f".{self._registry_path.name}.tmp.{_os.getpid()}.{_threading.get_ident()}"
            )
            temp_path.write_text(
                json.dumps(self._registry, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temp_path.replace(self._registry_path)
        except (OSError, ValueError, TypeError) as exc:
            logger.error(f"Failed to save registry: {exc}")
            raise

    # ------------------------------------------------------------------
    # Buffer state machine management
    # ------------------------------------------------------------------
    def _get_buffer_state(self, buffer_id: str) -> BufferState:
        """Get current buffer state.

        Args:
            buffer_id: Buffer ID

        Returns:
            Current BufferState (defaults to READY if not set)
        """
        info = self._get_buffer_info(buffer_id)
        if not info:
            raise ValueError(f"Unknown buffer_id: {buffer_id}")

        state_str = info.get("state", BufferState.READY.value)
        return BufferState(state_str)

    def _set_buffer_state(self, buffer_id: str, new_state: BufferState) -> None:
        """Set buffer state with validation.

        Args:
            buffer_id: Buffer ID
            new_state: Desired new state

        Raises:
            ValueError: If state transition is invalid
        """
        info = self._get_buffer_info(buffer_id)
        if not info:
            raise ValueError(f"Unknown buffer_id: {buffer_id}")

        current_state = self._get_buffer_state(buffer_id)

        # Validate transition
        BufferStateTransition.validate_or_raise(current_state, new_state)

        # Update state and timestamp
        info["state"] = new_state.value
        info["state_changed_at"] = time.time()

        self._save_registry()

        # Log state change
        self._audit_log(
            operation="state_transition",
            buffer_id=buffer_id,
            status="ok",
            details={
                "from_state": str(current_state),
                "to_state": str(new_state),
            },
        )

    def _is_buffer_dirty(self, buffer_id: str) -> bool:
        """Check if buffer has dirty files.

        Args:
            buffer_id: Buffer ID

        Returns:
            True if buffer has dirty files
        """
        info = self._get_buffer_info(buffer_id)
        if not info:
            return False

        dirty_files = info.get("dirty_files", {})
        return len(dirty_files) > 0

    def _load_source_snapshot(self, buffer_id: str) -> dict[str, list[str]] | None:
        """Load source code snapshot from disk."""
        info = self._get_buffer_info(buffer_id)
        if not info:
            return None

        snapshot_path = Path(info["buffer_dir"]) / "source_snapshot.json"
        if not snapshot_path.exists():
            return None

        try:
            return json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.error(f"Failed to load source snapshot: {exc}")
            return None

    def _save_source_snapshot(
        self,
        buffer_id: str,
        snapshot: dict[str, list[str]],
    ) -> None:
        """Save source code snapshot to disk."""
        info = self._get_buffer_info(buffer_id)
        if not info:
            return

        snapshot_path = Path(info["buffer_dir"]) / "source_snapshot.json"
        try:
            snapshot_path.write_text(
                json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
            )
        except (OSError, ValueError, TypeError) as exc:
            logger.error(f"Failed to save source snapshot: {exc}")

    # ------------------------------------------------------------------
    # Buffer lifecycle
    # ------------------------------------------------------------------
    def embed_codebase(
        self,
        path: str | Path,
        language_hint: str | None = None,
        pattern: str = "*.py",
        sliding_window_size: int = 30,
    ) -> tuple[str, list[CodeChunk], list[Path]]:
        """Embed a codebase and return buffer_id, chunks, and file list.

        This method handles chunking and basic validation, but delegates
        actual index/embedding to IndexManager.

        Returns:
            (buffer_id, chunks, files) - Use to pass to IndexManager
        """
        root = Path(path)
        files = [root] if root.is_file() else sorted(root.rglob(pattern))

        if not files:
            raise ValueError(f"No files matched '{pattern}' in {root}")

        # Size-guard check
        preflight = self.check_codebase(path, pattern)
        if preflight["status"] == "exceeds_threshold":
            raise ValueError(
                f"Codebase too large ({preflight['estimated_mb']:.1f} MB "
                f"exceeds threshold {preflight['threshold_mb']:.1f} MB)"
            )

        # Chunk all files
        all_chunks: list[CodeChunk] = []
        file_chunks_map: dict[str, list[int]] = {}

        for f in files:
            try:
                chunks = chunk_file(
                    f, language_hint=language_hint, sliding_window_size=sliding_window_size
                )
            except (OSError, ValueError, TypeError) as exc:
                json_logger.warning(
                    operation="chunk_file",
                    message=f"Failed to chunk {f}: {exc}",
                )
                continue

            rel = str(f.relative_to(root))
            file_chunks_map[rel] = []
            for ch in chunks:
                ch.file = rel
                file_chunks_map[rel].append(len(all_chunks))
                all_chunks.append(ch)

        if not all_chunks:
            raise ValueError("No chunks extracted from input files.")

        # Size check
        token_count = len(all_chunks)
        size_check = check_size(token_count, self._embedding_dim, self.threshold_mb)
        if size_check["status"] == "exceeds_threshold":
            raise ValueError(
                f"Codebase too large ({size_check['estimated_mb']:.1f} MB "
                f"exceeds threshold {size_check['threshold_mb']:.1f} MB)"
            )

        # Create buffer directory and register
        buffer_id = str(uuid.uuid4())
        buffer_dir = self.work_dir / f"{buffer_id}.gcbuff"
        buffer_dir.mkdir(parents=True, exist_ok=True)

        # Create snapshot
        snapshot_mgr = SnapshotManager(buffer_dir)
        files_dict = {str(f.relative_to(root)): f for f in files}
        manifest = snapshot_mgr.create_snapshot(buffer_id, root, files_dict)

        # Create source snapshot
        source_snapshot: dict[str, list[str]] = {}
        for rel_path, file_path in files_dict.items():
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_snapshot[rel_path] = f.read().splitlines()

        snapshot_path = buffer_dir / "source_snapshot.json"
        snapshot_path.write_bytes(
            json.dumps(source_snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )

        # Get file hashes
        file_hashes: dict[str, str] = {
            rel_path: meta.hash for rel_path, meta in manifest.files.items()
        }

        # Compute source hash for change-detection
        source_hash = self._compute_dir_hash(root, pattern)

        # Register buffer
        self._registry[buffer_id] = {
            "root": str(root),
            "buffer_dir": str(buffer_dir),
            "chunk_count": token_count,
            "embedding_dim": self._embedding_dim,
            "size_bytes": token_count * self._embedding_dim * 4,
            "file_hashes": file_hashes,
            "pattern": pattern,
            "language_hint": language_hint,
            "sliding_window_size": sliding_window_size,
            "dirty_files": {},
            "state": BufferState.READY.value,
            "state_changed_at": time.time(),
            "source_hash": source_hash,
        }

        self._snapshot_managers[buffer_id] = snapshot_mgr
        self._save_registry()

        return buffer_id, all_chunks, files

    def check_codebase(
        self,
        path: str | Path,
        pattern: str = "*.py",
    ) -> dict[str, Any]:
        """Check if codebase is within size threshold."""
        root = Path(path)
        files = [root] if root.is_file() else sorted(root.rglob(pattern))

        if not files:
            return {
                "status": "ok",
                "file_count": 0,
                "estimated_mb": 0.0,
                "threshold_mb": self.threshold_mb,
            }

        # Rough estimate: average source file is ~10KB
        total_size_bytes = sum(f.stat().st_size for f in files if f.is_file())
        total_size_mb = total_size_bytes / 1024 / 1024

        return {
            "status": "exceeds_threshold" if total_size_mb > self.threshold_mb else "ok",
            "file_count": len(files),
            "estimated_mb": total_size_mb,
            "threshold_mb": self.threshold_mb,
        }

    def list_buffers(self) -> dict[str, Any]:
        """List all registered buffers."""
        return {
            "status": "ok",
            "buffers": [{"buffer_id": bid, **info} for bid, info in self._registry.items()],
        }

    def delete_buffer(self, buffer_id: str) -> dict[str, Any]:
        """Delete a buffer and its data."""
        info = self._registry.pop(buffer_id, None)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        self._save_registry()
        self._snapshot_managers.pop(buffer_id, None)
        cleanup_message = None
        try:
            shutil.rmtree(info["buffer_dir"])
        except FileNotFoundError:
            cleanup_message = "Buffer directory was already removed."
        except OSError as exc:
            logger.warning("Failed to remove buffer directory for %s: %s", buffer_id, exc)
            cleanup_message = f"Buffer directory cleanup failed: {type(exc).__name__}"

        self._audit_log(operation="delete_buffer", buffer_id=buffer_id, status="ok")

        if cleanup_message is not None:
            return {
                "status": "ok",
                "message": f"Deleted buffer {buffer_id}",
                "cleanup_message": cleanup_message,
            }

        return {"status": "ok", "message": f"Deleted buffer {buffer_id}"}

    # ------------------------------------------------------------------
    # Read/Write/Commit operations
    # ------------------------------------------------------------------
    def read_code(
        self,
        buffer_id: str,
        file: str | None = None,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read file contents from buffer."""
        t0 = time.perf_counter()
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        snapshot_mgr = self._get_snapshot_manager(buffer_id)
        if snapshot_mgr is None:
            return {"status": "error", "message": "Snapshot not available."}

        start_line = max(1, start_line)

        if file is not None:
            if file not in snapshot_mgr.manifest.files:
                return {"status": "error", "message": f"File not in buffer: {file}"}

            lines = snapshot_mgr.read_lines(file)
            if lines is None:
                return {"status": "error", "message": f"Failed to read file: {file}"}

            end = end_line if end_line is not None else len(lines) + 1
            end = max(start_line, end)
            selected = lines[start_line - 1 : end - 1]
            result = {
                "status": "ok",
                "file": file,
                "start_line": start_line,
                "end_line": end,
                "lines": selected,
            }

            self._audit_log(
                operation="read_code",
                buffer_id=buffer_id,
                status="ok",
                details={
                    "file": file,
                    "start_line": start_line,
                    "end_line": end,
                    "lines_count": len(selected),
                },
            )

            return result

        # Read all files
        result_dict: dict[str, list[str]] = {}
        for fname in snapshot_mgr.manifest.files.keys():
            lines = snapshot_mgr.read_lines(fname)
            if lines is None:
                json_logger.warning(
                    operation="read_code",
                    message=f"Failed to read file {fname}",
                )
                continue
            end = end_line if end_line is not None else len(lines) + 1
            end = max(start_line, end)
            result_dict[fname] = lines[start_line - 1 : end - 1]

        final_result = {"status": "ok", "files": result_dict}

        self._audit_log(
            operation="read_code",
            buffer_id=buffer_id,
            status="ok",
            details={
                "files_count": len(result_dict),
                "start_line": start_line,
                "end_line": end_line,
            },
        )

        return final_result

    def write_code(
        self,
        buffer_id: str,
        file: str,
        start_line: int,
        new_lines: list[str],
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Write code to buffer (in-memory, not yet committed)."""
        t0 = time.perf_counter()
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}
        if file not in snapshot:
            return {"status": "error", "message": f"File not in buffer: {file}"}

        # Check for conflicts
        snapshot_mgr = self._get_snapshot_manager(buffer_id)
        if snapshot_mgr is not None:
            current_buffer_lines = snapshot[file]
            diff_result = snapshot_mgr.compute_diff(file, current_buffer_lines)
            if diff_result.get("has_conflict", False):
                return {
                    "status": "conflict",
                    "file": file,
                    "message": (
                        f"Cannot write to {file}: both disk and buffer have been modified. "
                        "Use reload_codebase() to sync with disk."
                    ),
                    "disk_lines": len(diff_result.get("disk_lines") or []),
                    "buffer_lines": len(diff_result.get("buffer_lines") or []),
                }

        # Apply changes
        old_lines = snapshot[file]
        end = end_line if end_line is not None else len(old_lines) + 1
        sanitized_new_lines = [line.rstrip("\n\r") for line in new_lines]
        new_file_lines = old_lines[: start_line - 1] + sanitized_new_lines + old_lines[end:]
        snapshot[file] = new_file_lines
        self._save_source_snapshot(buffer_id, snapshot)

        dirty = info.setdefault("dirty_files", {})
        dirty[file] = True
        self._save_registry()

        result = {
            "status": "ok",
            "file": file,
            "changed_lines": len(sanitized_new_lines),
            "replaced_lines": end - start_line,
            "total_lines": len(new_file_lines),
        }

        self._audit_log(
            operation="write_code",
            buffer_id=buffer_id,
            status="ok",
            details={
                "file": file,
                "start_line": start_line,
                "end_line": end,
                "changed_lines": len(sanitized_new_lines),
            },
        )

        return result

    def commit(
        self,
        buffer_id: str,
        index_manager: Any,  # Avoid circular import
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Commit changes to disk with 3-way merge and crash recovery."""
        t0 = time.perf_counter()
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        root = Path(info["root"])
        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}

        dirty = info.get("dirty_files", {})
        if not dirty:
            return {
                "status": "ok",
                "written_files": [],
                "conflict_files": [],
                "dry_run": dry_run,
                "transaction_id": None,
            }

        # Begin transaction
        transaction_id = None
        if not dry_run:
            transaction_id = self._state_manager.start_transaction(
                operation="commit",
                buffer_id=buffer_id,
                file_path=",".join(dirty.keys()),
                start_line=0,
                end_line=None,
                new_lines=None,
            )

        try:
            # Rebuild embeddings for dirty files
            if not dry_run and index_manager:
                index_manager._rebuild_files(buffer_id, list(dirty.keys()))

            # Get snapshot manager
            snapshot_mgr = self._get_snapshot_manager(buffer_id)
            if snapshot_mgr is None:
                if transaction_id:
                    self._state_manager.rollback_transaction(transaction_id)
                return {"status": "error", "message": "Snapshot manager not available"}

            written: list[str] = []
            conflicts: list[dict[str, Any]] = []
            new_hashes: dict[str, str] = {}
            updated_files: dict[str, Path] = {}

            for rel_path in dirty:
                lines = snapshot.get(rel_path, [])
                disk_path = root / rel_path

                if dry_run:
                    diff_result = snapshot_mgr.compute_diff(rel_path, lines)
                    if diff_result.get("has_conflict"):
                        conflicts.append({"file": rel_path, "message": "3-way merge conflict"})
                    else:
                        written.append(rel_path)
                else:
                    merge_result = snapshot_mgr.write_file_with_merge(
                        rel_path, lines, allow_conflicts=False
                    )

                    if merge_result["status"] == "conflict":
                        diff_result = snapshot_mgr.compute_diff(rel_path, lines)
                        conflicts.append(
                            {
                                "file": rel_path,
                                "message": "3-way merge conflict",
                            }
                        )
                    elif merge_result["status"] == "ok":
                        written.append(rel_path)
                        updated_files[rel_path] = disk_path
                        new_hashes[rel_path] = hashlib.sha256(
                            "\n".join(lines).encode("utf-8")
                        ).hexdigest()
                    else:
                        if transaction_id:
                            self._state_manager.rollback_transaction(transaction_id)
                        return {"status": "error", "message": f"Failed to write {rel_path}"}

            if not dry_run:
                info["file_hashes"].update(new_hashes)
                for f in written:
                    info["dirty_files"].pop(f, None)
                self._save_registry()

                if updated_files:
                    snapshot_mgr.update_manifest_after_commit(updated_files)

                if transaction_id:
                    self._state_manager.commit_transaction(transaction_id)
                    self._state_manager.save_registry()

            status = "conflict" if conflicts else "ok"
            self._audit_log(
                operation="commit",
                buffer_id=buffer_id,
                status=status,
                details={
                    "dry_run": dry_run,
                    "written_files_count": len(written),
                    "conflict_files_count": len(conflicts),
                },
            )

            if index_manager:
                elapsed = time.perf_counter() - t0
                if index_manager._prometheus_exporter:
                    index_manager._prometheus_exporter.record_operation(
                        operation="commit",
                        duration_s=elapsed,
                        status=status,
                        chunk_count=len(written),
                    )

            return {
                "status": status,
                "written_files": written,
                "conflict_files": conflicts,
                "dry_run": dry_run,
                "transaction_id": transaction_id,
            }

        except (OSError, ValueError, RuntimeError) as e:
            if transaction_id:
                json_logger.error(
                    operation="commit",
                    buffer_id=buffer_id,
                    message=f"Commit failed: {e}; rolling back",
                )
                self._state_manager.rollback_transaction(transaction_id)
            raise

    def discard(self, buffer_id: str) -> dict[str, Any]:
        """Discard unsaved changes to buffer."""
        t0 = time.perf_counter()
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        # Reload source snapshot from disk
        buffer_dir = Path(info["buffer_dir"])
        snapshot_mgr = self._get_snapshot_manager(buffer_id)
        if snapshot_mgr is None:
            return {"status": "error", "message": "Snapshot not available"}

        # Reset dirty files
        dirty = info.get("dirty_files", {})
        discarded_files = list(dirty.keys())

        # Reload snapshot from disk
        for fname in discarded_files:
            lines = snapshot_mgr.read_lines(fname)
            if lines is not None:
                source_snapshot = self._load_source_snapshot(buffer_id)
                if source_snapshot:
                    source_snapshot[fname] = lines
                    self._save_source_snapshot(buffer_id, source_snapshot)

        info["dirty_files"] = {}
        self._save_registry()

        self._audit_log(
            operation="discard",
            buffer_id=buffer_id,
            status="ok",
            details={"discarded_files_count": len(discarded_files)},
        )

        elapsed = time.perf_counter() - t0
        return {
            "status": "ok",
            "discarded_files": discarded_files,
            "elapsed_s": elapsed,
        }

    def diff(
        self,
        buffer_id: str,
        file: str | None = None,
    ) -> dict[str, Any]:
        """Show diff between buffer and disk versions."""
        t0 = time.perf_counter()
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        snapshot_mgr = self._get_snapshot_manager(buffer_id)
        if snapshot_mgr is None:
            return {"status": "error", "message": "Snapshot not available"}

        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing"}

        diffs: dict[str, Any] = {}

        if file:
            if file not in snapshot:
                return {"status": "error", "message": f"File not in buffer: {file}"}

            diff_result = snapshot_mgr.compute_diff(file, snapshot[file])
            diffs[file] = {
                "has_conflict": diff_result.get("has_conflict", False),
                "added_lines": diff_result.get("added_lines", []),
                "removed_lines": diff_result.get("removed_lines", []),
                "disk_lines": diff_result.get("disk_lines", []),
                "buffer_lines": diff_result.get("buffer_lines", []),
            }
        else:
            for fname in snapshot:
                diff_result = snapshot_mgr.compute_diff(fname, snapshot[fname])
                diffs[fname] = {
                    "has_conflict": diff_result.get("has_conflict", False),
                    "added_lines": diff_result.get("added_lines", []),
                    "removed_lines": diff_result.get("removed_lines", []),
                }

        has_conflicts = any(d.get("has_conflict", False) for d in diffs.values())
        status = "conflict" if has_conflicts else "ok"

        self._audit_log(
            operation="diff",
            buffer_id=buffer_id,
            status=status,
        )

        elapsed = time.perf_counter() - t0
        return {
            "status": status,
            "diffs": diffs,
            "has_conflicts": has_conflicts,
            "elapsed_s": elapsed,
        }

    def reload_codebase(
        self,
        buffer_id: str,
        index_manager: Any = None,  # Avoid circular import
    ) -> dict[str, Any]:
        """Reload buffer from disk, detecting external changes.

        This performs a 3-way merge if both buffer and disk have changed:
        1. Compare disk version with last snapshot
        2. Compare buffer version with last snapshot
        3. Merge if no conflicts, otherwise report conflict
        """
        t0 = time.perf_counter()
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        snapshot_mgr = self._get_snapshot_manager(buffer_id)
        if snapshot_mgr is None:
            return {"status": "error", "message": "Snapshot not available"}

        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing"}

        # Check for external changes
        diffs: dict[str, Any] = {}
        merge_results: list[dict[str, Any]] = []
        dirty = info.get("dirty_files", {})

        for fname in snapshot_mgr.manifest.files.keys():
            diff_result = snapshot_mgr.compute_diff(fname, snapshot.get(fname, []))
            diffs[fname] = diff_result

            # Determine if disk changed vs snapshot (by line count or content)
            disk_lines = diff_result.get("disk_lines")
            snap_line_count = diff_result.get("snapshot_line_count")
            disk_changed = (
                disk_lines is not None
                and snap_line_count is not None
                and len(disk_lines) != snap_line_count
            )

            if disk_changed:
                if dirty.get(fname, False):
                    # Both disk AND buffer were modified → true 3-way conflict
                    merge_results.append(
                        {
                            "file": fname,
                            "status": "conflict",
                            "message": "3-way merge conflict between disk and buffer changes",
                        }
                    )
                else:
                    # Only disk changed, buffer clean → reload from disk
                    if disk_lines is not None:
                        snapshot[fname] = disk_lines
                        merge_results.append(
                            {
                                "file": fname,
                                "status": "merged",
                                "message": "Loaded disk version",
                            }
                        )

        # Save merged snapshot
        self._save_source_snapshot(buffer_id, snapshot)

        # Clear dirty flags for merged files
        dirty = info.get("dirty_files", {})
        for result in merge_results:
            if result["status"] == "merged":
                dirty.pop(result["file"], None)

        info["dirty_files"] = dirty
        self._save_registry()

        # Rebuild indices if needed
        if index_manager and merge_results:
            rebuilt_files = [r["file"] for r in merge_results if r["status"] == "merged"]
            if rebuilt_files:
                index_manager._rebuild_files(buffer_id, rebuilt_files)

        has_conflicts = any(r["status"] == "conflict" for r in merge_results)
        merged_count = sum(1 for r in merge_results if r["status"] == "merged")
        status = "conflict" if has_conflicts else "ok"
        if merged_count == 0 and not has_conflicts:
            message = "Hashes match; reloaded without re-embedding."
        elif merged_count > 0:
            message = f"Reloaded {merged_count} file(s) from disk."
        else:
            message = "Conflict detected; manual resolution required."

        self._audit_log(
            operation="reload_codebase",
            buffer_id=buffer_id,
            status=status,
            details={
                "merged_files": sum(1 for r in merge_results if r["status"] == "merged"),
                "conflict_files": sum(1 for r in merge_results if r["status"] == "conflict"),
            },
        )

        if index_manager:
            elapsed = time.perf_counter() - t0
            if index_manager._prometheus_exporter:
                index_manager._prometheus_exporter.record_operation(
                    operation="reload_codebase",
                    duration_s=elapsed,
                    status=status,
                )

        return {
            "status": status,
            "buffer_id": buffer_id,
            "message": message,
            "merge_results": merge_results,
            "elapsed_s": time.perf_counter() - t0,
        }

    def _rebuild_files(
        self,
        buffer_id: str,
        files: list[str],
        embeddings: list[Any] | None = None,
        chunks: list[CodeChunk] | None = None,
    ) -> dict[str, Any]:
        """Rebuild embeddings for specific files (called by IndexManager).

        This is an internal method called by IndexManager after file changes.
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        # Update hash for rebuilt files
        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error"}

        for fname in files:
            lines = snapshot.get(fname, [])
            content = "\n".join(lines)
            file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            info["file_hashes"][fname] = file_hash

        self._save_registry()

        return {
            "status": "ok",
            "rebuilt_files": files,
        }

    def _rebuild_dirty(
        self,
        buffer_id: str,
        index_manager: Any | None = None,
    ) -> dict[str, Any]:
        """Auto-rebuild dirty files if count exceeds threshold."""
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "ok", "rebuilt": False}

        dirty_files = info.get("dirty_files", {})
        if len(dirty_files) >= MAX_DIRTY_BEFORE_AUTO_REBUILD:
            if index_manager:
                index_manager._rebuild_files(buffer_id, list(dirty_files.keys()))
                dirty_files.clear()
                self._save_registry()
                return {
                    "status": "ok",
                    "rebuilt": True,
                    "rebuilt_files_count": len(dirty_files),
                }

        return {"status": "ok", "rebuilt": False}

    def embed_file_with_streaming(
        self,
        file_path: str | Path,
        language_hint: str | None = None,
        streaming_threshold_mb: int = 50,
    ) -> list[CodeChunk]:
        """Chunk a file using streaming for large files.

        Automatically detects large files and uses streaming to avoid OOM.
        For small files, uses standard chunking.

        Args:
            file_path: Path to file to chunk
            language_hint: Programming language hint for chunking
            streaming_threshold_mb: Size threshold for streaming (default 50MB)

        Returns:
            List of CodeChunk objects
        """
        from gigacode.streaming_support import StreamingChunker, supports_streaming

        file_path = Path(file_path)
        file_size = file_path.stat().st_size if file_path.exists() else 0

        # For small files, use standard chunking
        if not supports_streaming(file_size, threshold_mb=streaming_threshold_mb):
            try:
                return chunk_file(file_path, language_hint=language_hint)
            except (OSError, ValueError, TypeError) as e:
                json_logger.warning(
                    operation="embed_file_with_streaming",
                    message=f"Failed to chunk {file_path}: {e}",
                )
                return []

        # For large files, use streaming
        json_logger.info(
            operation="embed_file_with_streaming",
            message=f"Using streaming for {file_size / 1024 / 1024:.1f}MB file",
            file=str(file_path),
        )

        chunker = StreamingChunker(
            max_chunk_bytes=1024 * 1024,  # 1MB chunks
            language=language_hint or "python",
        )

        all_chunks: list[CodeChunk] = []

        def process_chunk(content: str, start_line: int, end_line: int) -> None:
            """Process a file chunk using standard chunking."""
            try:
                chunks = chunk_text(
                    content,
                    file_path=str(file_path),
                    language_hint=language_hint,
                    line_offset=start_line - 1,
                )
                for chunk in chunks:
                    all_chunks.append(chunk)
            except (OSError, ValueError, TypeError) as e:
                json_logger.warning(
                    operation="embed_file_with_streaming",
                    message=f"Failed to chunk text block {start_line}-{end_line}: {e}",
                )

        try:
            chunker.stream_chunks(str(file_path), process_chunk)
        except (OSError, ValueError, RuntimeError) as e:
            json_logger.error(
                operation="embed_file_with_streaming",
                message=f"Streaming failed for {file_path}: {e}",
            )
            # Fall back to standard chunking
            try:
                return chunk_file(file_path, language_hint=language_hint)
            except (OSError, ValueError, TypeError):
                return []

        return all_chunks
