"""
Audit Trail (Event Logging) System.

This module keeps a detailed record of everything that happens - who did what,
when they did it, and whether it succeeded or failed. Think of it like a
security camera that records all activity for compliance and debugging.

Key Concepts:
    - AuditStatus: Whether an action succeeded, failed, or was denied
    - AuditLogEntry: A single record of one action
"""
from __future__ import annotations

# Additional documentation:
#     - AuditLogger: The system that records all actions to a log file
#
# Why This Matters:
#     - Compliance: Prove who accessed what and when (for security audits)
#     - Debugging: Understand what happened when something breaks
#     - Security: Detect suspicious activity patterns
#     - Accountability: Track which user did which action
#
# Example:
#     >>> logger = AuditLogger()
#     >>> logger.log_success(
#     ...     operation="write_code",
#     ...     user_id="alice",
#     ...     role="analyst",
#     ...     buffer_id="buf-123",
#     ...     details={"lines_written": 50},
#     ... )
#     >>> # Action is recorded to audit.jsonl file
#
# The audit log is stored as JSONL (JSON Lines) format - each line is one
# complete JSON record, making it easy to parse and analyze.
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "AuditStatus",
    "AuditLogEntry",
    "AuditLogger",
]


class AuditStatus(Enum):
    """
    > Outcome Status of an Operation.
    >
    > This tells you what happened with an operation:
    > - SUCCESS: The operation completed successfully
    > - FAILURE: The operation tried but something went wrong
    > - DENIED: The permission system blocked the operation before it ran
    >
    > Attributes:
    >     SUCCESS: Operation worked (e.g., code was written successfully)
    >     FAILURE: Operation failed (e.g., network error, file not found)
    >     DENIED: Operation was blocked (e.g., no permission)
    """

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"  # Permission denied


@dataclass
class AuditLogEntry:
    """
    > A Single Record of One Operation.
    >
    > This is what gets written to the audit log. It's like a detailed
    > receipt that shows: who, what, when, and how it went.
    >
    > Attributes:
    >     timestamp: When the action happened (seconds since 1970)
    >     operation: What action was performed (e.g., "write_code", "delete_buffer")
    >     buffer_id: Which code buffer was involved (if applicable)
    >     user_id: Who performed the action
    >     role: What role the user had
    >     status: Did it succeed, fail, or get denied?
    >     details: Extra info about what happened
    >     duration_ms: How long the operation took (milliseconds)
    >     error_message: If it failed, why?
    """

    timestamp: float
    operation: str
    buffer_id: Optional[str]
    user_id: str
    role: str
    status: str  # AuditStatus value
    details: Dict[str, Any]
    duration_ms: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> "AuditLogEntry":
        """Create from dict."""
        return AuditLogEntry(**data)


class AuditLogger:
    """
    > The Audit Logger - Record Keeper.
    >
    > This class is responsible for recording all operations to a permanent log.
    > Every action gets saved with a timestamp, user info, and outcome. It's
    > like a ledger that shows the complete history of what happened.
    >
    > The log is stored in JSONL format (JSON Lines) - one JSON object per line.
    > This makes it easy to read, parse, and analyze.
    >
    > Example:
    >     >>> logger = AuditLogger()
    >     >>> logger.log_success(
    >     ...     operation="search",
    >     ...     user_id="alice",
    >     ...     role="analyst",
    >     ...     details={"query": "def process_data"},
    >     ... )
    >     >>> # Query is recorded to the audit log
    >
    > File Location:
    >     Default: ~/.gigacode/audit.jsonl
    >     Each line is one complete audit entry as JSON
    """

    def __init__(self, log_file: Optional[Path] = None):
        """
        > Initialize the Audit Logger.
        >
        > This sets up the audit logging system and creates the log file
        > if it doesn't exist.
        >
        > Args:
        >     log_file: Where to save the audit log. If None, uses
        >               ~/.gigacode/audit.jsonl (default location)
        >
        > Example:
        >     >>> # Use default location
        >     >>> logger = AuditLogger()
        >     >>> # Use custom location
        >     >>> logger = AuditLogger("/var/log/gigacode/audit.jsonl")
        """
        if log_file is None:
            log_file = Path.home() / ".gigacode" / "audit.jsonl"

        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.touch(exist_ok=True)

    def close(self) -> None:
        """Close the audit logger (no-op; file handles are opened per-write)."""
        pass

    def log_operation(
        self,
        operation: str,
        user_id: str,
        role: str,
        status: AuditStatus,
        buffer_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        duration_ms: int = 0,
        error_message: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log an operation.

        Args:
            operation: Operation name
            user_id: User performing operation
            role: User's role
            status: Operation status
            buffer_id: Related buffer (optional)
            details: Operation-specific details
            duration_ms: How long operation took
            error_message: Error message if failed

        Returns:
            AuditLogEntry that was logged
        """
        if details is None:
            details = {}

        entry = AuditLogEntry(
            timestamp=time.time(),
            operation=operation,
            buffer_id=buffer_id,
            user_id=user_id,
            role=role,
            status=status.value,
            details=details,
            duration_ms=duration_ms,
            error_message=error_message,
        )

        # Append to log file (JSONL format)
        with self.log_file.open("a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

        return entry

    def log_success(
        self,
        operation: str,
        user_id: str,
        role: str,
        buffer_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        duration_ms: int = 0,
    ) -> AuditLogEntry:
        """Log successful operation."""
        return self.log_operation(
            operation=operation,
            user_id=user_id,
            role=role,
            status=AuditStatus.SUCCESS,
            buffer_id=buffer_id,
            details=details,
            duration_ms=duration_ms,
        )

    def log_failure(
        self,
        operation: str,
        user_id: str,
        role: str,
        error_message: str,
        buffer_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        duration_ms: int = 0,
    ) -> AuditLogEntry:
        """Log failed operation."""
        return self.log_operation(
            operation=operation,
            user_id=user_id,
            role=role,
            status=AuditStatus.FAILURE,
            buffer_id=buffer_id,
            details=details,
            duration_ms=duration_ms,
            error_message=error_message,
        )

    def log_denied(
        self,
        operation: str,
        user_id: str,
        role: str,
        reason: str,
        buffer_id: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log permission denied."""
        return self.log_operation(
            operation=operation,
            user_id=user_id,
            role=role,
            status=AuditStatus.DENIED,
            buffer_id=buffer_id,
            error_message=reason,
        )

    def query_logs(
        self,
        user_id: Optional[str] = None,
        operation: Optional[str] = None,
        buffer_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AuditLogEntry]:
        """Query audit logs.

        Args:
            user_id: Filter by user
            operation: Filter by operation type
            buffer_id: Filter by buffer
            status: Filter by status
            limit: Max number of results

        Returns:
            Matching audit entries
        """
        if not self.log_file.exists():
            return []

        results = []

        with self.log_file.open("r") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    entry_dict = json.loads(line)
                    entry = AuditLogEntry.from_dict(entry_dict)

                    # Apply filters
                    if user_id and entry.user_id != user_id:
                        continue
                    if operation and entry.operation != operation:
                        continue
                    if buffer_id and entry.buffer_id != buffer_id:
                        continue
                    if status and entry.status != status:
                        continue

                    results.append(entry)
                except (json.JSONDecodeError, TypeError):
                    continue

        # Return most recent first
        results.reverse()

        if limit:
            results = results[:limit]

        return results

    def get_user_activity(self, user_id: str, limit: int = 100) -> List[AuditLogEntry]:
        """Get recent activity for user."""
        return self.query_logs(user_id=user_id, limit=limit)

    def get_buffer_history(self, buffer_id: str, limit: int = 100) -> List[AuditLogEntry]:
        """Get operation history for buffer."""
        return self.query_logs(buffer_id=buffer_id, limit=limit)

    def query(
        self,
        since: str | None = None,
        operations: list[str] | None = None,
        buffer_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit log entries.

        Args:
            since: ISO timestamp string (e.g., "2026-05-01T00:00:00").
            operations: Filter by operation types (e.g., ["write_code", "commit"]).
            buffer_id: Filter by buffer ID.
            limit: Maximum entries to return.

        Returns:
            List of audit log entries as dicts (newest first).
        """
        if not self.log_file.exists():
            return []

        # Parse since timestamp if provided
        since_timestamp: float | None = None
        if since is not None:
            try:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
                since_timestamp = since_dt.timestamp()
            except ValueError:
                since_timestamp = None

        results: list[dict[str, Any]] = []

        with self.log_file.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter by since (compare timestamp)
                if since_timestamp is not None:
                    entry_ts = entry.get("timestamp")
                    if entry_ts is None or entry_ts < since_timestamp:
                        continue

                # Filter by operations (check if operation in list)
                if operations is not None:
                    op = entry.get("operation")
                    if op not in operations:
                        continue

                # Filter by buffer_id
                if buffer_id is not None:
                    entry_buffer_id = entry.get("buffer_id")
                    if entry_buffer_id != buffer_id:
                        continue

                results.append(entry)

        # Return most recent limit entries (newest first)
        results.reverse()
        return results[:limit]

    def stats(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        if not self.log_file.exists():
            return {"total_entries": 0, "log_file": str(self.log_file)}

        entries = self.query_logs()

        if not entries:
            return {"total_entries": 0, "log_file": str(self.log_file)}

        # Count by status
        status_counts = {}
        for entry in entries:
            status_counts[entry.status] = status_counts.get(entry.status, 0) + 1

        # Count by operation
        operation_counts = {}
        for entry in entries:
            operation_counts[entry.operation] = operation_counts.get(entry.operation, 0) + 1

        # Count by user
        user_counts = {}
        for entry in entries:
            user_counts[entry.user_id] = user_counts.get(entry.user_id, 0) + 1

        return {
            "total_entries": len(entries),
            "status_counts": status_counts,
            "operation_counts": operation_counts,
            "user_counts": user_counts,
            "log_file": str(self.log_file),
        }
