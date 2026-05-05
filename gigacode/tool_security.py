"""Security layer for CodeEmbeddingTool operations.

Encapsulates RBAC, audit logging, and rate limiting for consistent enforcement.
"""

from typing import Any, Dict, Optional
import logging

from gigacode.access_control import AccessControl, Role
from gigacode.audit_logger import AuditLogger
from gigacode.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class ToolSecurityLayer:
    """Unified security interface for access control, audit logging, and rate limiting."""
    
    def __init__(self, work_dir: Any, current_user_id: str = "default"):
        """Initialize security layer with all three components.
        
        Args:
            work_dir: Work directory for audit log storage
            current_user_id: Default user ID (usually "default" for local dev)
        """
        self._access_control = AccessControl()
        self._audit_logger = AuditLogger(work_dir / "audit.jsonl")
        self._rate_limiter = RateLimiter()
        self._current_user_id = current_user_id
        
        # Register default user with AGENT role (full operational access for AI agents)
        self._access_control.register_user("default", Role.AGENT)
    
    def get_current_user(self):
        """Get current user object for role/permission checks."""
        return self._access_control.get_user(self._current_user_id)
    
    def get_current_user_role_name(self) -> str:
        """Get current user's role name as string."""
        return self.get_current_user().role.name
    
    def check_rate_limit(self, operation: str) -> bool:
        """Check if operation is within rate limit.
        
        Args:
            operation: Operation type to check
            
        Returns:
            True if operation is allowed, False if rate limited
        """
        return self._rate_limiter.check_rate_limit(self._current_user_id, operation)
    
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
            message=message,
            buffer_id=buffer_id,
            details=audit_details,
        )
    
    def close(self) -> None:
        """Close audit logger and clean up resources."""
        self._audit_logger.close()
