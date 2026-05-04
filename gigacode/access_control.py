"""
Role-Based Access Control (RBAC) System.

This module implements a simple permission system for controlling who can do what.
Think of it like a library system: different users (ADMIN, ANALYST, READER, GUEST)
have different permissions for different actions.

Key Concepts:
    - Role: What type of user you are (determines your permissions)
    - Permission: What action you want to do (create, write, delete, etc.)
    - AccessControl: The gatekeeper that checks if you're allowed to do something
    - User: A person with a specific role

How It Works:
    1. Each user gets a Role (ADMIN, ANALYST, READER, or GUEST)
    2. Each Role has a set of Permissions it can do
    3. AccessControl checks permissions before allowing operations
    4. Some permissions are restricted per-buffer (e.g., ANALYST can only edit own)

Example:
    >>> ac = AccessControl()
    >>> user = ac.register_user("alice", Role.ANALYST)
    >>> allowed, reason = ac.check_permission(
    ...     user_id="alice",
    ...     permission=Permission.WRITE_CODE,
    ...     buffer_owner="alice"
    ... )
    >>> # allowed = True (ANALYST can write their own buffer)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Set


class Role(Enum):
    """
    > User Role Categories.
    >
    > Each role represents a different level of access. Think of it like job titles:
    > ADMIN is the manager, AGENT/ANALYST do the work, READER watches, GUEST has minimal access.
    >
    > Attributes:
    >     ADMIN: Can do everything - create, edit, delete, and view audit logs
    >     AGENT: AI agent role with full operational access (same as ANALYST)
    >     ANALYST: Can create and edit their own buffers, search, and read code
    >     READER: Can only search and read code (no writing or creating)
    >     GUEST: Can search and read code with strict rate limits
    """
    
    ADMIN = "admin"        # Full access
    AGENT = "agent"        # AI agent - can create/edit own buffers (same as ANALYST)
    ANALYST = "analyst"    # Can create/edit own buffers
    READER = "reader"      # Read-only access
    GUEST = "guest"        # Limited read with rate limiting


class Permission(Enum):
    """
    > Individual Permissions (Actions You Can Do).
    >
    > Each permission represents one specific action. Before you can do any action,
    > the system checks if your role allows it.
    >
    > Common Permissions:
    >     CREATE_BUFFER: Make a new code buffer
    >     WRITE_CODE: Modify code in a buffer
    >     SEARCH: Look for code using search
    >     DELETE_BUFFER: Remove a buffer
    >     COMMIT: Save changes to a buffer
    >
    > Admin Permissions:
    >     RELOAD_CODEBASE: Re-read all code files from disk
    >     VIEW_AUDIT_LOG: See who did what and when
    """
    
    CREATE_BUFFER = "create_buffer"
    DELETE_BUFFER = "delete_buffer"
    WRITE_CODE = "write_code"
    COMMIT = "commit"
    DISCARD = "discard"
    RELOAD_CODEBASE = "reload_codebase"
    SEARCH = "search"
    READ_CODE = "read_code"
    EMBED_CODEBASE = "embed_codebase"
    VIEW_AUDIT_LOG = "view_audit_log"


# Permission matrix: role -> set of allowed permissions
PERMISSION_MATRIX: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        Permission.CREATE_BUFFER,
        Permission.DELETE_BUFFER,
        Permission.WRITE_CODE,
        Permission.COMMIT,
        Permission.DISCARD,
        Permission.RELOAD_CODEBASE,
        Permission.SEARCH,
        Permission.READ_CODE,
        Permission.EMBED_CODEBASE,
        Permission.VIEW_AUDIT_LOG,
    },
    Role.AGENT: {
        Permission.CREATE_BUFFER,
        Permission.DELETE_BUFFER,  # Own buffers only
        Permission.WRITE_CODE,     # Own buffers only
        Permission.COMMIT,         # Own buffers only
        Permission.DISCARD,        # Own buffers only
        Permission.RELOAD_CODEBASE,  # Own buffers only
        Permission.SEARCH,
        Permission.READ_CODE,
        Permission.EMBED_CODEBASE,  # Own buffers only
    },
    Role.ANALYST: {
        Permission.CREATE_BUFFER,
        Permission.DELETE_BUFFER,  # Own buffers only
        Permission.WRITE_CODE,     # Own buffers only
        Permission.COMMIT,         # Own buffers only
        Permission.DISCARD,        # Own buffers only
        Permission.RELOAD_CODEBASE,  # Own buffers only
        Permission.SEARCH,
        Permission.READ_CODE,
        Permission.EMBED_CODEBASE,  # Own buffers only
    },
    Role.READER: {
        Permission.SEARCH,
        Permission.READ_CODE,
    },
    Role.GUEST: {
        Permission.SEARCH,  # Rate-limited
        Permission.READ_CODE,  # Basic metadata only
    },
}


@dataclass
class User:
    """
    > Represents a User with Their Role and Permissions.
    >
    > A User is a person who has a specific role. The role determines what
    > permissions they have. Think of it like a badge that lets them do certain things.
    >
    > Attributes:
    >     user_id (str): Unique identifier for the user (e.g., "alice@example.com")
    >     role (Role): What type of user they are (ADMIN, ANALYST, READER, GUEST)
    >     buffer_owner (str): Who owns the buffer (used to check if they can edit it)
    >
    > Example:
    >     >>> analyst = User("alice", Role.ANALYST)
    >     >>> analyst.has_permission(Permission.SEARCH)
    >     True  # ANALYST can search
    >     >>> analyst.has_permission(Permission.DELETE_BUFFER)
    >     True  # ANALYST can delete (their own buffers)
    """
    
    user_id: str
    role: Role = Role.ANALYST  # Default to ANALYST
    buffer_owner: Optional[str] = None  # For "own buffers only" checks
    
    def has_permission(self, permission: Permission) -> bool:
        """
        > Check if this user can do a specific action.
        >
        > This looks up the user's role in the permission matrix and checks if
        > that role is allowed to do the requested action.
        >
        > Args:
        >     permission: The action to check (Permission enum value)
        >
        > Returns:
        >     bool: True if allowed, False if not allowed
        >
        > Example:
        >     >>> user = User("alice", Role.ANALYST)
        >     >>> user.has_permission(Permission.WRITE_CODE)
        >     True
        >     >>> user.has_permission(Permission.VIEW_AUDIT_LOG)
        >     False  # Only ADMIN can view audit logs
        """
        return permission in PERMISSION_MATRIX.get(self.role, set())
    
    def can_access_buffer(self, buffer_id: str, buffer_owner: str) -> bool:
        """
        > Check if this user can access a specific buffer.
        >
        > This enforces the "own buffers only" rule for ANALYST users:
        > - ADMIN and READER can access any buffer
        > - ANALYST can only access their own buffers
        > - GUEST can access any buffer (but rate-limited)
        >
        > Args:
        >     buffer_id: Which buffer to check access for
        >     buffer_owner: Who owns the buffer
        >
        > Returns:
        >     bool: True if they can access, False otherwise
        >
        > Example:
        >     >>> alice = User("alice", Role.ANALYST)
        >     >>> alice.can_access_buffer("buf-123", "alice")
        >     True  # Alice can access her own buffer
        >     >>> alice.can_access_buffer("buf-456", "bob")
        >     False  # Alice cannot access Bob's buffer
        """
        # ADMINs can access all buffers
        if self.role == Role.ADMIN:
            return True
        
        # ANALYSTS can access their own buffers
        if self.role == Role.ANALYST:
            return self.user_id == buffer_owner
        
        # READERS and GUESTS can read any buffer
        if self.role in (Role.READER, Role.GUEST):
            return True
        
        return False
    
    def to_dict(self) -> Dict:
        """Serialize user."""
        return {
            "user_id": self.user_id,
            "role": self.role.value,
            "buffer_owner": self.buffer_owner,
        }



class AccessControl:
    """
    > Access Control Manager - The Gatekeeper.
    >
    > This class is responsible for enforcing all permission checks. It maintains
    > a registry of users and their roles, and answers the question: "Is user X
    > allowed to do action Y?"
    >
    > It's like a security guard who checks credentials and permissions before
    > letting anyone do anything important.
    >
    > Example:
    >     >>> ac = AccessControl()
    >     >>> ac.register_user("alice", Role.ANALYST)
    >     >>> allowed, reason = ac.check_permission(
    >     ...     user_id="alice",
    >     ...     permission=Permission.WRITE_CODE,
    >     ...     buffer_owner="alice"
    >     ... )
    >     >>> if allowed:
    >     ...     # Do the write operation
    >     ... else:
    >     ...     # Deny and show reason to user
    """
    
    def __init__(self):
        """
        > Initialize the Access Control system.
        >
        > This sets up the permission system and creates a default user
        > for local development.
        """
        self.users: Dict[str, User] = {}
        self._register_default_user()
    
    def _register_default_user(self):
        """Register default user for local development."""
        self.register_user("default", Role.ANALYST)
    
    def register_user(self, user_id: str, role: Role = Role.ANALYST) -> User:
        """Register a user."""
        user = User(user_id=user_id, role=role)
        self.users[user_id] = user
        return user
    
    def get_user(self, user_id: str) -> User:
        """Get or create user."""
        if user_id not in self.users:
            return self.register_user(user_id)
        return self.users[user_id]
    
    def check_permission(
        self,
        user_id: str,
        permission: Permission,
        buffer_owner: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Check if user has permission.
        
        Args:
            user_id: User identifier
            permission: Permission to check
            buffer_owner: Buffer owner (for "own buffers only" checks)
            
        Returns:
            (allowed, reason) tuple
        """
        user = self.get_user(user_id)
        
        # Check basic permission
        if not user.has_permission(permission):
            return False, f"User role '{user.role.value}' not authorized for {permission.value}"
        
        # Check buffer ownership for "own buffers only" permissions
        ownership_required = {
            Permission.DELETE_BUFFER,
            Permission.WRITE_CODE,
            Permission.COMMIT,
            Permission.DISCARD,
            Permission.RELOAD_CODEBASE,
            Permission.EMBED_CODEBASE,
        }
        
        if permission in ownership_required:
            if buffer_owner and user.role == Role.ANALYST:
                if not user.can_access_buffer("dummy", buffer_owner):
                    return False, f"User '{user_id}' not authorized to modify buffer owned by '{buffer_owner}'"
        
        return True, ""
    
    def check_operation(
        self,
        user_id: str,
        operation: str,
        buffer_owner: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Check if operation is allowed.
        
        Args:
            user_id: User identifier
            operation: Operation name (maps to Permission)
            buffer_owner: Buffer owner (for ownership checks)
            
        Returns:
            (allowed, reason) tuple
        """
        # Map operation string to Permission
        try:
            permission = Permission[operation.upper()]
        except KeyError:
            return False, f"Unknown operation: {operation}"
        
        return self.check_permission(user_id, permission, buffer_owner)
