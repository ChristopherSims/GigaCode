"""Tests for Access Control, Audit Logging, and Rate Limiting."""

# CRITICAL: Initialize sklearn FIRST before any gigacode imports
import types

try:
    import sklearn

    if getattr(sklearn, "__spec__", None) is None:
        sklearn.__spec__ = types.ModuleSpec("sklearn", getattr(sklearn, "__file__", None))
except Exception:
    pass


import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from gigacode.access_control import (
    AccessControl,
    Permission,
    Role,
    User,
)
from gigacode.audit_logger import (
    AuditLogger,
)
from gigacode.rate_limiter import (
    RateLimiter,
    TokenBucket,
)

# ============================================================================
# RBAC Tests
# ============================================================================


class TestRoleBasedAccessControl:
    """Test RBAC system."""

    def test_admin_has_all_permissions(self):
        """Test ADMIN role has all permissions."""
        user = User("admin1", Role.ADMIN)

        assert user.has_permission(Permission.CREATE_BUFFER)
        assert user.has_permission(Permission.WRITE_CODE)
        assert user.has_permission(Permission.DELETE_BUFFER)
        assert user.has_permission(Permission.VIEW_AUDIT_LOG)

    def test_analyst_has_create_and_own_edit(self):
        """Test ANALYST role can create and edit own buffers."""
        user = User("analyst1", Role.ANALYST)

        assert user.has_permission(Permission.CREATE_BUFFER)
        assert user.has_permission(Permission.WRITE_CODE)
        assert user.has_permission(Permission.SEARCH)
        assert not user.has_permission(Permission.VIEW_AUDIT_LOG)

    def test_reader_has_read_only(self):
        """Test READER role is read-only."""
        user = User("reader1", Role.READER)

        assert user.has_permission(Permission.READ_CODE)
        assert user.has_permission(Permission.SEARCH)
        assert not user.has_permission(Permission.WRITE_CODE)
        assert not user.has_permission(Permission.DELETE_BUFFER)

    def test_guest_has_limited_access(self):
        """Test GUEST role has limited access."""
        user = User("guest1", Role.GUEST)

        assert user.has_permission(Permission.SEARCH)
        assert user.has_permission(Permission.READ_CODE)
        assert not user.has_permission(Permission.WRITE_CODE)
        assert not user.has_permission(Permission.CREATE_BUFFER)

    def test_analyst_can_access_own_buffer(self):
        """Test ANALYST can access own buffer."""
        user = User("alice", Role.ANALYST)

        assert user.can_access_buffer("buf-123", "alice")

    def test_analyst_cannot_access_other_buffer(self):
        """Test ANALYST cannot access other's buffer."""
        user = User("alice", Role.ANALYST)

        assert not user.can_access_buffer("buf-456", "bob")

    def test_admin_can_access_any_buffer(self):
        """Test ADMIN can access any buffer."""
        user = User("admin", Role.ADMIN)

        assert user.can_access_buffer("buf-123", "alice")
        assert user.can_access_buffer("buf-456", "bob")

    def test_reader_can_access_any_buffer(self):
        """Test READER can access any buffer."""
        user = User("reader", Role.READER)

        assert user.can_access_buffer("buf-123", "alice")
        assert user.can_access_buffer("buf-456", "bob")


class TestAccessControl:
    """Test AccessControl enforcement."""

    def test_register_user(self):
        """Test user registration."""
        ac = AccessControl()

        user = ac.register_user("test_user", Role.ANALYST)
        assert user.user_id == "test_user"
        assert user.role == Role.ANALYST

    def test_get_or_create_user(self):
        """Test get_user creates if not exists."""
        ac = AccessControl()

        user = ac.get_user("new_user")
        assert user.user_id == "new_user"
        assert user.role == Role.ANALYST  # Default

    def test_check_permission_allowed(self):
        """Test permission check when allowed."""
        ac = AccessControl()
        ac.register_user("analyst1", Role.ANALYST)

        allowed, reason = ac.check_permission("analyst1", Permission.CREATE_BUFFER)
        assert allowed is True
        assert reason == ""

    def test_check_permission_denied(self):
        """Test permission check when denied."""
        ac = AccessControl()
        ac.register_user("guest1", Role.GUEST)

        allowed, reason = ac.check_permission("guest1", Permission.WRITE_CODE)
        assert allowed is False
        assert "not authorized" in reason.lower()

    def test_check_operation_by_name(self):
        """Test checking operation by string name."""
        ac = AccessControl()
        ac.register_user("reader1", Role.READER)

        allowed, reason = ac.check_operation("reader1", "search")
        assert allowed is True

    def test_ownership_check_analyst(self):
        """Test ownership validation for ANALYST."""
        ac = AccessControl()
        ac.register_user("alice", Role.ANALYST)

        # Own buffer
        allowed, _ = ac.check_operation("alice", "write_code", buffer_owner="alice")
        assert allowed is True

        # Other's buffer
        allowed, reason = ac.check_operation("alice", "write_code", buffer_owner="bob")
        assert allowed is False


# ============================================================================
# Audit Logging Tests
# ============================================================================


class TestAuditLogging:
    """Test audit trail system."""

    def test_log_success(self):
        """Test logging successful operation."""
        with TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir) / "audit.jsonl")

            entry = logger.log_success(
                operation="write_code",
                user_id="alice",
                role="analyst",
                buffer_id="buf-123",
                details={"lines": 42},
                duration_ms=150,
            )

            assert entry.operation == "write_code"
            assert entry.user_id == "alice"
            assert entry.status == "success"
            assert entry.buffer_id == "buf-123"

    def test_log_failure(self):
        """Test logging failed operation."""
        with TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir) / "audit.jsonl")

            entry = logger.log_failure(
                operation="commit",
                user_id="alice",
                role="analyst",
                error_message="Conflict detected",
                buffer_id="buf-123",
            )

            assert entry.status == "failure"
            assert entry.error_message == "Conflict detected"

    def test_log_denied(self):
        """Test logging permission denied."""
        with TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir) / "audit.jsonl")

            entry = logger.log_denied(
                operation="delete_buffer",
                user_id="alice",
                role="reader",
                reason="Reader cannot delete buffers",
                buffer_id="buf-123",
            )

            assert entry.status == "denied"
            assert entry.error_message == "Reader cannot delete buffers"

    def test_query_by_user(self):
        """Test querying logs by user."""
        with TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir) / "audit.jsonl")

            logger.log_success("write_code", "alice", "analyst")
            logger.log_success("commit", "bob", "analyst")
            logger.log_success("search", "alice", "analyst")

            alice_logs = logger.query_logs(user_id="alice")
            assert len(alice_logs) == 2
            assert all(e.user_id == "alice" for e in alice_logs)

    def test_query_by_operation(self):
        """Test querying logs by operation."""
        with TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir) / "audit.jsonl")

            logger.log_success("write_code", "alice", "analyst")
            logger.log_success("commit", "alice", "analyst")
            logger.log_success("write_code", "bob", "analyst")

            write_logs = logger.query_logs(operation="write_code")
            assert len(write_logs) == 2

    def test_query_by_buffer(self):
        """Test querying logs by buffer."""
        with TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir) / "audit.jsonl")

            logger.log_success("write_code", "alice", "analyst", "buf-1")
            logger.log_success("write_code", "alice", "analyst", "buf-2")
            logger.log_success("write_code", "bob", "analyst", "buf-1")

            buf1_logs = logger.query_logs(buffer_id="buf-1")
            assert len(buf1_logs) == 2

    def test_query_by_status(self):
        """Test querying logs by status."""
        with TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir) / "audit.jsonl")

            logger.log_success("write_code", "alice", "analyst")
            logger.log_failure("commit", "alice", "analyst", "Conflict")
            logger.log_denied("delete_buffer", "bob", "reader", "Not authorized")

            success_logs = logger.query_logs(status="success")
            assert len(success_logs) == 1
            assert success_logs[0].operation == "write_code"

    def test_get_user_activity(self):
        """Test getting user activity."""
        with TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir) / "audit.jsonl")

            for i in range(5):
                logger.log_success(f"op{i}", "alice", "analyst")

            activity = logger.get_user_activity("alice")
            assert len(activity) == 5

    def test_get_buffer_history(self):
        """Test getting buffer history."""
        with TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir) / "audit.jsonl")

            for i in range(3):
                logger.log_success("write_code", f"user{i}", "analyst", "buf-1")

            history = logger.get_buffer_history("buf-1")
            assert len(history) == 3

    def test_audit_stats(self):
        """Test audit log statistics."""
        with TemporaryDirectory() as tmpdir:
            logger = AuditLogger(Path(tmpdir) / "audit.jsonl")

            logger.log_success("write_code", "alice", "analyst")
            logger.log_failure("commit", "alice", "analyst", "Error")
            logger.log_success("write_code", "bob", "analyst")

            stats = logger.stats()
            assert stats["total_entries"] == 3
            assert stats["status_counts"]["success"] == 2
            assert stats["status_counts"]["failure"] == 1
            assert "user_counts" in stats


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestTokenBucket:
    """Test token bucket implementation."""

    def test_bucket_initialization(self):
        """Test bucket creation."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.capacity == 10
        assert bucket.refill_rate == 1.0

    def test_consume_success(self):
        """Test successful token consumption."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.tokens = 5  # Start with 5 tokens

        assert bucket.consume(3) is True
        assert math.isclose(bucket.tokens, 2.0, abs_tol=0.01)

    def test_consume_fail(self):
        """Test failed consumption (insufficient tokens)."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.tokens = 2

        assert bucket.consume(5) is False
        assert math.isclose(bucket.tokens, 2, abs_tol=0.01)  # Unchanged

    def test_refill_over_time(self):
        """Test token refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.tokens = 0

        # Wait for 2 seconds, should gain 2 tokens
        time.sleep(0.1)
        bucket._refill()

        # Refill should have added some tokens (at least 1)
        assert bucket.tokens > 0

    def test_capacity_limit(self):
        """Test tokens capped at capacity."""
        bucket = TokenBucket(capacity=5, refill_rate=100.0)  # Fast refill
        bucket.tokens = 2

        # Wait for refill (should want to add many tokens)
        time.sleep(0.05)
        bucket._refill()

        # Should not exceed capacity
        assert bucket.tokens <= 5


class TestRateLimiter:
    """Test rate limiter system."""

    def test_user_limit_by_role(self):
        """Test user limits vary by role."""
        limiter = RateLimiter()

        admin_bucket = limiter.get_user_limit("admin", Role.ADMIN)
        guest_bucket = limiter.get_user_limit("guest", Role.GUEST)

        # ADMIN should have higher limit
        assert admin_bucket.capacity > guest_bucket.capacity

    def test_check_user_limit_success(self):
        """Test successful user limit check."""
        limiter = RateLimiter()

        allowed, _ = limiter.check_user_limit("user1", Role.ANALYST)
        assert allowed is True

    def test_check_buffer_limit_success(self):
        """Test successful buffer limit check."""
        limiter = RateLimiter()

        allowed, _ = limiter.check_buffer_limit("buf-1", limit_per_minute=60)
        assert allowed is True

    def test_check_all_limits_success(self):
        """Test all limits check when allowed."""
        limiter = RateLimiter()

        allowed, reason = limiter.check_all_limits(
            "user1",
            Role.ANALYST,
            buffer_id="buf-1",
        )

        assert allowed is True
        assert reason is None

    def test_rate_limit_exhaustion(self):
        """Test rate limit exhaustion."""
        limiter = RateLimiter()

        # Create bucket with tiny capacity
        bucket = limiter.get_user_limit("user1", Role.ANALYST, "operations")
        bucket.capacity = 1
        bucket.refill_rate = 0.0  # No refill
        bucket.tokens = 0  # No tokens

        # Should fail
        allowed, reason = limiter.check_user_limit("user1", Role.ANALYST)
        assert allowed is False
        assert "rate limit" in reason.lower()


# ============================================================================
# Integration Tests
# ============================================================================


class TestAccessControlIntegration:
    """Integration tests for access control, audit, and rate limiting."""

    def test_permission_and_audit_workflow(self):
        """Test permission check and audit logging together."""
        with TemporaryDirectory() as tmpdir:
            ac = AccessControl()
            logger = AuditLogger(Path(tmpdir) / "audit.jsonl")

            # User tries to delete buffer they don't own
            allowed, reason = ac.check_operation("alice", "delete_buffer", "bob")
            assert allowed is False

            # Log the denied operation
            logger.log_denied("delete_buffer", "alice", "analyst", reason, "buf-123")

            # Check it was logged
            denied_logs = logger.query_logs(status="denied")
            assert len(denied_logs) == 1
            assert denied_logs[0].user_id == "alice"

    def test_rate_limit_and_audit_workflow(self):
        """Test rate limiting with audit logging."""
        with TemporaryDirectory() as tmpdir:
            limiter = RateLimiter()
            logger = AuditLogger(Path(tmpdir) / "audit.jsonl")

            # Exhaust a user's rate limit
            bucket = limiter.get_user_limit("user1", Role.GUEST, "operations")
            bucket.capacity = 1
            bucket.refill_rate = 0
            bucket.tokens = 0

            # Try operation
            allowed, reason = limiter.check_all_limits("user1", Role.GUEST)

            # Log the attempt
            if allowed:
                logger.log_success("search", "user1", "guest")
            else:
                logger.log_failure("search", "user1", "guest", reason)

            # Verify logged
            entries = logger.query_logs(user_id="user1")
            assert len(entries) == 1
            assert entries[0].status == "failure"
