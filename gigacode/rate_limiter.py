"""
Rate Limiting System (Token Bucket).

This module prevents users from making too many requests too quickly. It's like
a water meter that limits how much water you can use per minute. Different user
roles get different limits.

Key Concepts:
    - Token: Permission to do one operation
    - Bucket: A container that holds tokens
    - Refill: As time passes, tokens are added back
    - Consume: Using an operation costs one token
    - Rate Limit: Max operations per minute

How It Works:
    1. User gets a bucket full of tokens (e.g., 60 tokens = 60 operations/minute)
    2. Each operation costs 1 token
    3. Tokens are added back over time (refill)
    4. If they run out of tokens, they have to wait until more are added

Example:
    >>> limiter = RateLimiter()
    >>> allowed, msg = limiter.check_user_limit(
    ...     user_id="alice",
    ...     role=Role.ANALYST,
    ...     operation_type="operations"
    ... )
    >>> if allowed:
    ...     # Do the operation
    ... else:
    ...     # Tell user to wait (msg has how long)

Different Roles Get Different Limits:
    - ADMIN: 300 operations/minute (very generous)
    - ANALYST: 60 operations/minute (normal use)
    - READER: 30 operations/minute (limited)
    - GUEST: 10 operations/minute (very limited)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum

from gigacode.access_control import Role


class RateLimitError(Exception):
    """Exception thrown when rate limit is exceeded.
    
    This exception is raised when a user has run out of tokens and
    is trying to do too many operations too quickly.
    
    Attributes:
        retry_after_seconds: How many seconds to wait before trying again
    
    Example:
        >>> try:
        ...     check_limit(user_id)
        ... except RateLimitError as e:
        ...     print(f"Wait {e.retry_after_seconds} seconds")
    """
    
    def __init__(self, retry_after_seconds: float):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Rate limit exceeded. Retry after {retry_after_seconds:.1f}s")


@dataclass
class TokenBucket:
    """A Container of Tokens for Rate Limiting.
    
    This is like a water bucket that slowly fills up. Each operation costs
    one token. As time passes, the bucket is refilled with new tokens.
    When the bucket is empty, you have to wait for new tokens to be added.
    
    Attributes:
        capacity: Maximum tokens the bucket can hold (e.g., 60)
        refill_rate: Tokens added per second (e.g., 1 token/sec = 60/minute)
        tokens: Current tokens in bucket (starts full)
        last_refill: When we last added tokens
    
    Example:
        >>> bucket = TokenBucket(capacity=60, refill_rate=1.0)
        >>> # Try to use 1 token
        >>> bucket.consume(1)
        True  # Success, 1 token used
        >>> # Check how long until next token
        >>> bucket.time_to_next_token()
        0.5  # Wait 0.5 seconds for next token
    """
    
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(default_factory=lambda: 0.0)
    last_refill: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize the bucket with full capacity."""
        self.tokens = float(self.capacity)
    
    def _refill(self):
        """Add tokens to the bucket based on elapsed time.
        
        This is called automatically before every operation to calculate
        how many new tokens should be added since the last refill.
        
        How It Works:
            1. Calculate seconds elapsed since last refill
            2. Add (elapsed_seconds * refill_rate) tokens
            3. Cap at maximum capacity so bucket doesn't overflow
            4. Record the refill time for next calculation
        """
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to use tokens from the bucket.
        
        This checks if there are enough tokens, and if so, removes them.
        This is called when the user wants to do an operation.
        
        Args:
            tokens: How many tokens to use (default: 1 for one operation)
        
        Returns:
            bool: True if successful (tokens were available and removed),
                  False if not enough tokens (operation blocked)
        
        Example:
            >>> bucket = TokenBucket(capacity=10, refill_rate=1.0)
            >>> bucket.consume()  # Use 1 token
            True
            >>> bucket.consume(20)  # Try to use 20 tokens
            False  # Not enough tokens
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def time_to_next_token(self) -> float:
        """Calculate how long to wait for the next available token.
        
        This tells you how many seconds to wait before another token
        will be available. Useful for telling users when to retry.
        
        Returns:
            float: Seconds until next token is available. 0.0 if tokens
                   are already available.
        
        Example:
            >>> bucket = TokenBucket(capacity=60, refill_rate=1.0)
            >>> bucket.consume(60)  # Use all tokens
            >>> bucket.time_to_next_token()
            1.0  # Wait 1 second for next token
        """
        self._refill()
        
        if self.tokens >= 1:
            return 0.0
        
        tokens_needed = 1 - self.tokens
        return tokens_needed / self.refill_rate if self.refill_rate > 0 else 0.0


# Default limits by role (operations per minute)
DEFAULT_LIMITS = {
    Role.ADMIN: {
        "operations_per_minute": 300,
        "queries_per_minute": 600,
    },
    Role.ANALYST: {
        "operations_per_minute": 60,
        "queries_per_minute": 120,
    },
    Role.READER: {
        "operations_per_minute": 30,
        "queries_per_minute": 60,
    },
    Role.GUEST: {
        "operations_per_minute": 10,
        "queries_per_minute": 20,
    },
}




class RateLimiter:
    """Rate Limiter Manager - The Gatekeeper of Operations.
    
    This class manages all the token buckets for all users and all buffers.
    It's responsible for checking if someone can do an operation before they
    do it. Different users get different rate limits based on their role.
    
    Rate limits prevent someone from:
        - Running too many queries at once (avoids crashes)
        - Overloading the system with requests
        - Denying service to other users
    
    Example:
        >>> limiter = RateLimiter()
        >>> allowed, msg = limiter.check_user_limit(
        ...     user_id="alice",
        ...     role=Role.ANALYST
        ... )
        >>> if not allowed:
        ...     print(msg)  # "Wait X seconds before trying again"
    
    How Limits Work:
        ADMIN gets 300 ops/minute (very fast, can do lots)
        ANALYST gets 60 ops/minute (normal, for regular work)
        READER gets 30 ops/minute (slower, read-only)
        GUEST gets 10 ops/minute (very slow, limited access)
    """
    
    def __init__(self):
        """Initialize the Rate Limiter.
        
        This sets up empty buckets for users and buffers. Buckets are
        created on-demand when a user first tries an operation.
        """
        self.user_buckets: Dict[str, Dict[str, TokenBucket]] = {}
        self.buffer_buckets: Dict[str, Dict[str, TokenBucket]] = {}
    
    def get_user_limit(self, user_id: str, role: Role, limit_type: str = "operations") -> TokenBucket:
        """Get or Create a Token Bucket for a User.
        
        This finds the user's bucket (or creates it if new). Different
        types of operations can have different limits (general ops vs queries).
        
        Args:
            user_id: Who the limit is for (e.g., "alice@example.com")
            role: User's role, determines their limits
            limit_type: What kind of operation: "operations" or "queries"
        
        Returns:
            TokenBucket: The user's bucket for this operation type
        
        Example:
            >>> limiter = RateLimiter()
            >>> bucket = limiter.get_user_limit("alice", Role.ANALYST)
            >>> # alice now has 60 operations/minute limit
        """
        if user_id not in self.user_buckets:
            self.user_buckets[user_id] = {}
        
        key = f"{limit_type}_per_minute"
        
        if limit_type not in self.user_buckets[user_id]:
            limit = DEFAULT_LIMITS.get(role, {}).get(key, 60)
            # Convert per-minute to per-second
            refill_rate = limit / 60.0
            self.user_buckets[user_id][limit_type] = TokenBucket(
                capacity=limit,
                refill_rate=refill_rate,
            )
        
        return self.user_buckets[user_id][limit_type]
    
    def get_buffer_limit(
        self,
        buffer_id: str,
        limit_type: str = "operations",
        limit_per_minute: int = 60,
    ) -> TokenBucket:
        """Get or create rate limit bucket for buffer.
        
        Args:
            buffer_id: Buffer identifier
            limit_type: "operations", "queries", or "writes"
            limit_per_minute: Max operations per minute
            
        Returns:
            TokenBucket for this buffer
        """
        if buffer_id not in self.buffer_buckets:
            self.buffer_buckets[buffer_id] = {}
        
        if limit_type not in self.buffer_buckets[buffer_id]:
            refill_rate = limit_per_minute / 60.0
            self.buffer_buckets[buffer_id][limit_type] = TokenBucket(
                capacity=limit_per_minute,
                refill_rate=refill_rate,
            )
        
        return self.buffer_buckets[buffer_id][limit_type]
    
    def check_user_limit(
        self,
        user_id: str,
        role: Role,
        operation_type: str = "operations",
        tokens: int = 1,
    ) -> tuple[bool, Optional[str]]:
        """Check if User Has Enough Tokens for an Operation.
        
        This is the main method you call before allowing a user to do something.
        It checks if they have tokens available, and if so, removes them.
        
        Args:
            user_id: Who is trying to do the operation
            role: Their role (determines their limits)
            operation_type: What kind of operation ("operations" or "queries")
            tokens: How many tokens it costs (default 1)
        
        Returns:
            tuple: (allowed, error_message)
                   - allowed: True if operation is allowed, False if blocked
                   - error_message: If blocked, tells user when to retry
        
        Example:
            >>> allowed, msg = limiter.check_user_limit(
            ...     user_id="alice",
            ...     role=Role.ANALYST,
            ... )
            >>> if not allowed:
            ...     return {"error": msg}  # Tell user to wait
        """
        bucket = self.get_user_limit(user_id, role, operation_type)
        
        if bucket.consume(tokens):
            return True, None
        
        retry_after = bucket.time_to_next_token()
        return False, f"Rate limit exceeded. Retry after {retry_after:.1f}s"
    
    def check_buffer_limit(
        self,
        buffer_id: str,
        operation_type: str = "operations",
        limit_per_minute: int = 60,
        tokens: int = 1,
    ) -> tuple[bool, Optional[str]]:
        """Check buffer rate limit.
        
        Args:
            buffer_id: Buffer identifier
            operation_type: Type of operation
            limit_per_minute: Max per minute
            tokens: Number of tokens to consume
            
        Returns:
            (allowed, error_message) tuple
        """
        bucket = self.get_buffer_limit(buffer_id, operation_type, limit_per_minute)
        
        if bucket.consume(tokens):
            return True, None
        
        retry_after = bucket.time_to_next_token()
        return False, f"Buffer rate limit exceeded. Retry after {retry_after:.1f}s"
    
    def check_all_limits(
        self,
        user_id: str,
        role: Role,
        buffer_id: Optional[str] = None,
        operation_type: str = "operations",
    ) -> tuple[bool, Optional[str]]:
        """Check all applicable rate limits.
        
        Args:
            user_id: User identifier
            role: User's role
            buffer_id: Buffer (optional)
            operation_type: Type of operation
            
        Returns:
            (allowed, error_message) tuple
        """
        # Check user limit
        allowed, reason = self.check_user_limit(user_id, role, operation_type)
        if not allowed:
            return False, reason
        
        # Check buffer limit if specified
        if buffer_id:
            allowed, reason = self.check_buffer_limit(
                buffer_id,
                operation_type,
                limit_per_minute=60,
            )
            if not allowed:
                return False, reason
        
        return True, None
