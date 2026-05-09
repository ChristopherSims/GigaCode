"""Path validation utilities to prevent traversal attacks.

Provides safe path resolution and boundary checking for codebase operations.
"""

from pathlib import Path
from typing import List, Union

__all__ = [
    "validate_buffer_path",
    "validate_buffer_paths",
    "is_valid_buffer_path",
]


def validate_buffer_path(user_path: Union[str, Path], allowed_root: Union[str, Path]) -> Path:
    """Resolve and verify path is under allowed_root.

    Prevents path traversal attacks by ensuring the resolved path is
    a child of or equal to the allowed root directory.

    Args:
        user_path: User-provided path (may be relative, absolute, or contain ..)
        allowed_root: Root directory to constrain path within

    Returns:
        Resolved Path object guaranteed to be under allowed_root

    Raises:
        ValueError: If path escapes allowed_root or other validation fails

    Examples:
        >>> root = Path("/code")
        >>> validate_buffer_path("src/main.py", root)
        Path('/code/src/main.py')

        >>> validate_buffer_path("../etc/passwd", root)  # Raises ValueError
        ValueError: Path ... escapes allowed root ...

        >>> validate_buffer_path("/etc/passwd", root)  # Raises ValueError
        ValueError: Path ... escapes allowed root ...
    """
    user_path = Path(user_path)
    allowed_root = Path(allowed_root)

    # Resolve both paths to absolute canonical form
    try:
        resolved = user_path.resolve()
        allowed_root_resolved = allowed_root.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Cannot resolve path: {e}") from e

    # Ensure resolved path is under allowed_root
    try:
        # This will raise ValueError if resolved is not relative to allowed_root_resolved
        resolved.relative_to(allowed_root_resolved)
    except ValueError as _e:
        raise ValueError(
            f"Path {user_path} (resolved to {resolved}) escapes allowed root {allowed_root}"
        ) from _e

    return resolved


def validate_buffer_paths(
    paths: List[Union[str, Path]], allowed_root: Union[str, Path]
) -> List[Path]:
    """Validate a list of paths are all under allowed_root.

    Args:
        paths: List of user-provided paths
        allowed_root: Root directory to constrain paths within

    Returns:
        List of resolved Path objects, all guaranteed under allowed_root

    Raises:
        ValueError: If any path escapes allowed_root
    """
    return [validate_buffer_path(p, allowed_root) for p in paths]


def is_valid_buffer_path(user_path: Union[str, Path], allowed_root: Union[str, Path]) -> bool:
    """Check if path is valid without raising exceptions.

    Args:
        user_path: User-provided path to check
        allowed_root: Root directory to constrain path within

    Returns:
        True if path is valid, False otherwise
    """
    try:
        validate_buffer_path(user_path, allowed_root)
        return True
    except ValueError:
        return False
