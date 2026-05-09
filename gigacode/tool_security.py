"""Security layer for CodeEmbeddingTool operations.

Encapsulates RBAC, audit logging, rate limiting, and input validation
for consistent enforcement across all public API methods.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from gigacode.access_control import AccessControl, Role
from gigacode.audit_logger import AuditLogger
from gigacode.exceptions import InvalidPathError, QueryLimitExceeded, RateLimitExceeded
from gigacode.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


__all__ = [
    "ToolSecurityLayer",
]


# Default rate limits per operation type
DEFAULT_RATE_LIMITS: Dict[str, int] = {
    "embed_codebase": 10,
    "semantic_search": 120,
    "hybrid_search": 120,
    "cluster_code": 30,
    "find_duplicates": 20,
    "read_code": 600,
    "write_code": 120,
    "commit": 30,
    "diff": 120,
    "discard": 60,
    "reload_codebase": 10,
    "delete_buffer": 10,
    "check_codebase": 60,
    "list_buffers": 60,
}


class ToolSecurityLayer:
    """Unified security interface for access control, audit logging, and rate limiting."""

    def __init__(
        self,
        work_dir: Any,
        current_user_id: str = "default",
        rate_limits: Optional[Dict[str, int]] = None,
    ):
        """Initialize security layer with all three components.

        Args:
            work_dir: Work directory for audit log storage
            current_user_id: Default user ID (usually "default" for local dev)
            rate_limits: Optional dict mapping operation names to per-minute limits
        """
        self._access_control = AccessControl()
        self._audit_logger = AuditLogger(Path(work_dir) / "audit.jsonl")
        self._rate_limiter = RateLimiter()
        self._current_user_id = current_user_id
        self._rate_limits = rate_limits or DEFAULT_RATE_LIMITS

        # Register default user with AGENT role (full operational access for AI agents)
        self._access_control.register_user("default", Role.AGENT)

    def get_current_user(self):
        """Get current user object for role/permission checks."""
        return self._access_control.get_user(self._current_user_id)

    def get_current_user_role_name(self) -> str:
        """Get current user's role name as string."""
        return self.get_current_user().role.name

    def check_rate_limit(
        self, operation: str, buffer_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Check if operation is within rate limit.

        Args:
            operation: Operation type to check
            buffer_id: Optional buffer ID for buffer-level limits

        Returns:
            Tuple of (allowed, error_message). error_message is None if allowed.
        """
        role = self.get_current_user().role
        limit_per_minute = self._rate_limits.get(operation, 60)

        return self._rate_limiter.check_all_limits(
            user_id=self._current_user_id,
            role=role,
            buffer_id=buffer_id,
            operation_type=operation,
        )

    def enforce_rate_limit(
        self,
        operation: str,
        buffer_id: Optional[str] = None,
    ) -> None:
        """Enforce rate limit by raising RateLimitExceeded if exceeded.

        Args:
            operation: Operation type to check
            buffer_id: Optional buffer ID for buffer-level limits

        Raises:
            RateLimitExceeded: If rate limit is exceeded.
        """
        allowed, reason = self.check_rate_limit(operation, buffer_id)
        if not allowed:
            self.log_failure(operation, reason or "Rate limit exceeded", buffer_id)
            raise RateLimitExceeded(reason or "Rate limit exceeded")

    def validate_query(
        self,
        query: str,
        max_length: int = 10_000,
    ) -> None:
        """Validate search query input.

        Args:
            query: Query string to validate
            max_length: Maximum allowed query length

        Raises:
            QueryLimitExceeded: If query is empty or too long.
        """
        if not query or not query.strip():
            raise QueryLimitExceeded("query must be a non-empty string")
        if len(query) > max_length:
            raise QueryLimitExceeded(
                f"query exceeds maximum length of {max_length} characters " f"(got {len(query)}"
            )

    def validate_buffer_id(self, buffer_id: str) -> None:
        """Validate buffer ID format.

        Args:
            buffer_id: Buffer ID to validate

        Raises:
            InvalidPathError: If buffer ID is empty or invalid.
        """
        if not buffer_id or not buffer_id.strip():
            raise InvalidPathError("buffer_id must be a non-empty string")
        try:
            uuid.UUID(buffer_id)
        except ValueError:
            pass  # Allow non-UUID format for legacy buffer IDs

    def validate_top_k(self, top_k: int, max_k: int = 1000) -> None:
        """Validate top_k parameter.

        Args:
            top_k: Number of results requested
            max_k: Maximum allowed top_k

        Raises:
            QueryLimitExceeded: If top_k is out of range.
        """
        if not isinstance(top_k, int) or top_k < 1 or top_k > max_k:
            raise QueryLimitExceeded(
                f"top_k must be an integer between 1 and {max_k} (got {top_k})"
            )

    def log_success(
        self,
        operation: str,
        buffer_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log successful operation to audit log.

        Args:
            operation: Operation name (e.g., 'semantic_search')
            buffer_id: Buffer ID if applicable
            details: Additional details to log
        """
        audit_details = details or {}
        if buffer_id:
            audit_details["buffer_id"] = buffer_id

        self._audit_logger.log_success(
            operation=operation,
            user_id=self._current_user_id,
            role=self.get_current_user_role_name(),
            buffer_id=buffer_id,
            details=audit_details,
        )

    def log_failure(
        self,
        operation: str,
        message: str,
        buffer_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log failed operation to audit log.

        Args:
            operation: Operation name
            message: Failure reason
            buffer_id: Buffer ID if applicable
            details: Additional details to log
        """
        audit_details = details or {}
        if buffer_id:
            audit_details["buffer_id"] = buffer_id

        self._audit_logger.log_failure(
            operation=operation,
            user_id=self._current_user_id,
            role=self.get_current_user_role_name(),
            error_message=message,
            buffer_id=buffer_id,
            details=audit_details,
        )

    def close(self) -> None:
        """Close audit logger and clean up resources."""
        self._audit_logger.close()
