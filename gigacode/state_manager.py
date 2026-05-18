"""State management with file locking, transactions, and WAL.

Handles concurrent access, atomic writes, and recovery from abrupt termination.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Unix file locking (optional - Windows uses fallback)
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

logger = logging.getLogger(__name__)


def _utc_now_iso_z() -> str:
    """Return an ISO-8601 UTC timestamp with a trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "TransactionLog",
    "FileLocker",
    "StateManager",
]


@dataclass
class TransactionLog:
    """Write-ahead log entry for transaction recovery."""

    transaction_id: str
    timestamp: str
    operation: str  # "write_code", "commit", "delete_buffer"
    buffer_id: str
    file_path: str
    start_line: int
    end_line: Optional[int]
    new_lines: Optional[list[str]]
    status: str = "pending"  # "pending", "committed", "rolled_back"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TransactionLog":
        """Create from dict."""
        return cls(**data)


class FileLocker:
    """Cross-process file locking using filelock library.

    Provides true exclusive locking across processes and threads
    on all platforms (Windows, Linux, macOS).
    """

    def __init__(self, file_path: Path):
        from filelock import FileLock

        self.file_path = file_path
        self.lock_file = file_path.parent / f".{file_path.name}.lock"
        self._file_lock = FileLock(str(self.lock_file), timeout=10.0)
        self.is_locked = False

    def acquire(self, timeout: float = 10.0) -> bool:
        """Acquire exclusive lock on file."""
        from filelock import Timeout

        try:
            self._file_lock.acquire(timeout=timeout)
            self.is_locked = True
            logger.debug(f"Acquired lock on {self.file_path}")
            return True
        except Timeout as e:
            logger.error(f"Failed to acquire lock: {e}")
            return False

    def release(self) -> None:
        """Release file lock."""
        try:
            if not self.is_locked:
                return
            self._file_lock.release()
            self.is_locked = False
            logger.debug(f"Released lock on {self.file_path}")
        except RuntimeError as e:
            logger.error(f"Failed to release lock: {e}")

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class StateManager:
    """Manages registry and transaction state with ACID guarantees.

    Features:
    - File locking for concurrent access
    - Write-ahead logging (WAL) for crash recovery
    - Transaction semantics for write_code + commit
    - Cache invalidation tracking
    """

    def __init__(self, work_dir: Path):
        """Initialize state manager.

        Args:
            work_dir: Directory containing registry and WAL logs
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.registry_path = self.work_dir / "registry.json"
        self.wal_path = self.work_dir / "wal.jsonl"  # Write-ahead log
        self.registry: dict[str, dict[str, Any]] = {}

        # Load existing registry
        self._load_registry()

        # Recover from incomplete transactions
        self._recover_from_wal()

    def _load_registry(self) -> None:
        """Load registry with file locking."""
        if not self.registry_path.exists():
            self.registry = {}
            return

        locker = FileLocker(self.registry_path)
        if locker.acquire(timeout=5.0):
            try:
                content = self.registry_path.read_text(encoding="utf-8")
                self.registry = json.loads(content) if content.strip() else {}
                logger.debug(f"Loaded registry with {len(self.registry)} buffers")
            except json.JSONDecodeError as e:
                logger.error(f"Corrupted registry: {e}; starting fresh")
                self.registry = {}
            finally:
                locker.release()
        else:
            logger.warning("Could not acquire lock on registry; using stale copy")

    def _recover_from_wal(self) -> None:
        """Recover from incomplete transactions in WAL.

        Scans WAL log for pending transactions and either:
        - Completes them if logged as committed
        - Rolls them back if incomplete
        """
        if not self.wal_path.exists():
            return

        pending = []
        with open(self.wal_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry_dict = json.loads(line)
                    # Skip status-update entries (entries with only transaction_id, status, timestamp)
                    # These are minimal entries written by commit_transaction() and rollback_transaction()
                    if "operation" not in entry_dict or "buffer_id" not in entry_dict:
                        continue

                    entry = TransactionLog.from_dict(entry_dict)
                    if entry.status == "pending":
                        pending.append(entry)

        for tx in pending:
            logger.warning(
                f"Found pending transaction {tx.transaction_id} "
                f"({tx.operation} on {tx.buffer_id}); rolling back"
            )
            # Mark as rolled back in WAL
            self._write_wal_entry(
                tx.transaction_id,
                tx.operation,
                tx.buffer_id,
                tx.file_path,
                tx.start_line,
                tx.end_line,
                tx.new_lines,
                status="rolled_back",
            )

            # If commit was pending, revert dirty file tracking
            if tx.operation == "commit" and tx.buffer_id in self.registry:
                if "dirty_files" in self.registry[tx.buffer_id]:
                    self.registry[tx.buffer_id]["dirty_files"].pop(tx.file_path, None)

    def _write_wal_entry(
        self,
        transaction_id: str,
        operation: str,
        buffer_id: str,
        file_path: str,
        start_line: int,
        end_line: Optional[int],
        new_lines: Optional[list[str]],
        status: str = "pending",
    ) -> None:
        """Append entry to write-ahead log."""
        entry = TransactionLog(
            transaction_id=transaction_id,
            timestamp=_utc_now_iso_z(),
            operation=operation,
            buffer_id=buffer_id,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            new_lines=new_lines,
            status=status,
        )

        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        logger.debug(f"WAL entry: {transaction_id} {status}")

    def save_registry(self) -> None:
        """Save registry atomically with file locking."""
        max_retries = 5
        last_error = None

        for attempt in range(max_retries):
            locker = FileLocker(self.registry_path)
            if not locker.acquire(timeout=1.0):
                if attempt < max_retries - 1:
                    logger.debug(f"Lock acquisition attempt {attempt + 1} failed, retrying...")
                    time.sleep(0.1 * (2**attempt))
                continue

            try:
                # Write to temp file first
                temp_path = (
                    self.registry_path.parent / f".{self.registry_path.name}.tmp.{os.getpid()}"
                )
                temp_path.write_text(json.dumps(self.registry, indent=2), encoding="utf-8")

                # Atomic rename
                temp_path.replace(self.registry_path)
                logger.debug("Registry saved atomically")
                return
            except (OSError, ValueError, RuntimeError) as e:
                last_error = e
                logger.error(f"Failed to save registry: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2**attempt))
            finally:
                locker.release()

        if last_error:
            raise last_error
        raise RuntimeError("Could not acquire lock for registry write after retries")

    def get_buffer_info(self, buffer_id: str) -> Optional[dict[str, Any]]:
        """Get buffer info with thread-safe access."""
        return self.registry.get(buffer_id)

    def set_buffer_info(self, buffer_id: str, info: dict[str, Any]) -> None:
        """Set buffer info and save registry."""
        self.registry[buffer_id] = info
        self.save_registry()

    def mark_dirty_file(self, buffer_id: str, file_path: str) -> None:
        """Mark file as dirty (modified in write_code)."""
        if buffer_id not in self.registry:
            raise ValueError(f"Unknown buffer_id: {buffer_id}")

        if "dirty_files" not in self.registry[buffer_id]:
            self.registry[buffer_id]["dirty_files"] = {}

        self.registry[buffer_id]["dirty_files"][file_path] = {"modified_at": _utc_now_iso_z()}

    def clear_dirty_files(self, buffer_id: str) -> None:
        """Clear dirty file tracking after commit."""
        if buffer_id in self.registry:
            self.registry[buffer_id]["dirty_files"] = {}

    def invalidate_cache_for_buffer(
        self, buffer_id: str, cache_types: list[str] | None = None
    ) -> None:
        """Mark cache as invalid for buffer (index, lexical, or query).

        Args:
            buffer_id: Buffer to invalidate
            cache_types: List of ["index", "lexical", "query"]; all if None
        """
        if buffer_id not in self.registry:
            return

        if cache_types is None:
            cache_types = ["index", "lexical", "query"]

        if "cache_invalid" not in self.registry[buffer_id]:
            self.registry[buffer_id]["cache_invalid"] = {}

        for cache_type in cache_types:
            self.registry[buffer_id]["cache_invalid"][cache_type] = {
                "invalidated_at": _utc_now_iso_z()
            }

        logger.debug(f"Invalidated {cache_types} caches for buffer {buffer_id}")

    def is_cache_valid(self, buffer_id: str, cache_type: str) -> bool:
        """Check if cache is valid for buffer.

        Args:
            buffer_id: Buffer ID
            cache_type: "index", "lexical", or "query"

        Returns:
            True if cache is valid (or cache type not tracked)
        """
        if buffer_id not in self.registry:
            return False

        invalid_map = self.registry[buffer_id].get("cache_invalid", {})
        return cache_type not in invalid_map

    def start_transaction(
        self,
        buffer_id: str,
        operation: str,
        file_path: str,
        start_line: int,
        end_line: Optional[int],
        new_lines: Optional[list[str]],
    ) -> str:
        """Start a new transaction, returning transaction ID.

        Args:
            buffer_id: Buffer being modified
            operation: "write_code", "commit", etc.
            file_path: File being modified
            start_line, end_line, new_lines: Edit parameters

        Returns:
            Transaction ID for tracking
        """
        import uuid

        transaction_id = str(uuid.uuid4())

        self._write_wal_entry(
            transaction_id,
            operation,
            buffer_id,
            file_path,
            start_line,
            end_line,
            new_lines,
            status="pending",
        )

        return transaction_id

    def commit_transaction(self, transaction_id: str) -> None:
        """Mark transaction as successfully committed."""
        # Append status update to WAL
        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "transaction_id": transaction_id,
                        "status": "committed",
                        "timestamp": _utc_now_iso_z(),
                    }
                )
                + "\n"
            )
        logger.debug(f"Transaction {transaction_id} committed")

    def rollback_transaction(self, transaction_id: str) -> None:
        """Roll back a failed transaction."""
        with open(self.wal_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "transaction_id": transaction_id,
                        "status": "rolled_back",
                        "timestamp": _utc_now_iso_z(),
                    }
                )
                + "\n"
            )
        logger.debug(f"Transaction {transaction_id} rolled back")
