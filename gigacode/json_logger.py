"""Structured JSON logging for GigaCode.

Provides JSON-formatted logging with consistent structure across all operations.
Each log entry includes: timestamp, level, operation, buffer_id, elapsed_time, and details.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

__all__ = ["StructuredJsonLogger", "json_logger"]


@dataclass
class LogEntry:
    """Structured log entry with JSON serialization."""

    timestamp: float  # Unix epoch time
    level: str  # DEBUG, INFO, WARNING, ERROR
    operation: str  # e.g., 'semantic_search', 'embed_codebase', 'commit'
    buffer_id: str | None = None
    elapsed_s: float | None = None
    status: str | None = None  # 'ok', 'error', 'conflict'
    details: dict[str, Any] | None = None
    message: str | None = None  # Human-readable message

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "timestamp": self.timestamp,
            "level": self.level,
            "operation": self.operation,
        }

        if self.buffer_id is not None:
            data["buffer_id"] = self.buffer_id

        if self.elapsed_s is not None:
            data["elapsed_s"] = round(self.elapsed_s, 4)

        if self.status is not None:
            data["status"] = self.status

        if self.message is not None:
            data["message"] = self.message

        if self.details is not None:
            data["details"] = self.details

        return json.dumps(data, separators=(",", ":"), default=str)


class StructuredJsonLogger:
    """Structured JSON logger that outputs consistent log entries.

    Usage:
        logger = StructuredJsonLogger('embed')
        logger.info('semantic_search', buffer_id='buf123', elapsed_s=0.45,
                    details={'query': 'search term', 'top_k': 10, 'matches': 5})
    """

    def __init__(self, module_name: str):
        """Initialize logger for a specific module.

        Args:
            module_name: Module or component name for identification
        """
        self.module_name = module_name
        self._logger = logging.getLogger(f"gigacode.{module_name}")

    def _log(
        self,
        level: str,
        operation: str,
        buffer_id: str | None = None,
        elapsed_s: float | None = None,
        status: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a structured entry.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            operation: Operation name
            buffer_id: Buffer/resource ID (optional)
            elapsed_s: Elapsed time in seconds (optional)
            status: Operation status: 'ok', 'error', 'conflict' (optional)
            message: Human-readable message (optional)
            details: Additional structured details (optional)
        """
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            operation=operation,
            buffer_id=buffer_id,
            elapsed_s=elapsed_s,
            status=status,
            message=message,
            details=details,
        )

        json_str = entry.to_json()
        log_level = getattr(logging, level.upper(), logging.INFO)
        self._logger.log(log_level, json_str)

    def debug(
        self,
        operation: str,
        buffer_id: str | None = None,
        elapsed_s: float | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a DEBUG-level structured entry."""
        self._log("DEBUG", operation, buffer_id, elapsed_s, None, message, details)

    def info(
        self,
        operation: str,
        buffer_id: str | None = None,
        elapsed_s: float | None = None,
        status: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an INFO-level structured entry."""
        self._log("INFO", operation, buffer_id, elapsed_s, status, message, details)

    def warning(
        self,
        operation: str,
        buffer_id: str | None = None,
        elapsed_s: float | None = None,
        status: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a WARNING-level structured entry."""
        self._log("WARNING", operation, buffer_id, elapsed_s, status, message, details)

    def error(
        self,
        operation: str,
        buffer_id: str | None = None,
        elapsed_s: float | None = None,
        status: str | None = "error",
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an ERROR-level structured entry."""
        self._log("ERROR", operation, buffer_id, elapsed_s, status, message, details)


# Global singleton logger for GigaCode
json_logger = StructuredJsonLogger("core")


def configure_json_logging(level: int = logging.INFO) -> None:
    """Configure JSON logging for all GigaCode loggers.

    Should be called once at application startup.

    Args:
        level: Logging level (default INFO)
    """
    # Get root GigaCode logger
    gigacode_logger = logging.getLogger("gigacode")
    gigacode_logger.setLevel(level)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # JSON formatter - just pass through JSON strings
    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            # Message is already JSON
            return record.getMessage()

    formatter = JsonFormatter()
    handler.setFormatter(formatter)

    # Remove existing handlers and add new one
    gigacode_logger.handlers.clear()
    gigacode_logger.addHandler(handler)

    # Prevent propagation to root logger
    gigacode_logger.propagate = False
