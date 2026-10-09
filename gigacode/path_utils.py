"""Path validation utilities to prevent traversal attacks.

Provides safe path resolution and boundary checking for codebase operations.
"""

import os
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import List, Union

__all__ = [
    "validate_buffer_path",
    "validate_buffer_paths",
    "is_valid_buffer_path",
    "resolve_source_path",
    "SourcePathError",
]


class SourcePathError(ValueError):
    """An actionable source-path failure, without exposing paths outside the root."""

    def __init__(self, message: str, code: str, candidates: list[str] | None = None):
        super().__init__(message)
        self.code = code
        self.candidates = candidates or []


def resolve_source_path(
    user_path: str, allowed_root: Union[str, Path], source_keys: Iterable[str],
) -> str:
    """Accept safe path spellings and return the existing snapshot key.

    Responses can use POSIX separators while persisted snapshots retain their
    platform-native keys. Root-prefixed relative paths are accepted only when
    the ordinary root-relative path does not identify an existing source file.
    """
    if not isinstance(user_path, str) or not user_path.strip() or "\0" in user_path:
        raise SourcePathError("file must be a non-empty source path.", "invalid_path")
    if re.search(r"</?arg_(?:key|value)>", user_path, re.IGNORECASE):
        raise SourcePathError(
            "file contains tool-call XML delimiters. Resend the edit as a native JSON object "
            "with file as a plain path string; do not put argument names or XML wrappers inside values.",
            "invalid_arguments",
        )
    raw = user_path.strip().replace("\\", "/")
    if ".." in PurePosixPath(raw).parts:
        raise SourcePathError("Parent traversal is not allowed.", "path_outside_root")
    windows = PureWindowsPath(raw)
    if os.name != "nt" and windows.drive:
        raise SourcePathError("File must be inside the configured project root.", "path_outside_root")
    if os.name == "nt" and windows.drive and not windows.is_absolute():
        raise SourcePathError("Drive-relative paths are not allowed.", "invalid_path")
    root = Path(allowed_root).resolve()
    try:
        resolved = validate_buffer_path(raw, root)
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise SourcePathError("File must be inside the configured project root.", "path_outside_root") from exc
    def fold(value: str) -> str:
        return value.casefold() if os.name == "nt" else value
    lookup: dict[str, list[str]] = {}
    for key in source_keys:
        canonical = PurePosixPath(key.replace("\\", "/")).as_posix()
        lookup.setdefault(fold(canonical), []).append(key)
    alternatives = [relative]
    # An agent's cwd can be the parent of a project root named "project".
    parts = PurePosixPath(raw).parts
    if not Path(raw).is_absolute() and parts and fold(parts[0]) == fold(root.name):
        alternatives.append(PurePosixPath(*parts[1:]).as_posix())
    for candidate in alternatives:
        keys = lookup.get(fold(candidate), [])
        if len(keys) > 1:
            raise SourcePathError("Ambiguous source path; use an exact relative path.", "ambiguous_file")
        if keys:
            # Validate the chosen key as well, including root-prefix stripping
            # and symlinks. Never turn an outside path into an inside basename.
            try:
                validate_buffer_path(keys[0].replace("\\", "/"), root)
            except ValueError as exc:
                raise SourcePathError("File resolves outside the project root.", "path_outside_root") from exc
            return keys[0]
    basename = PurePosixPath(relative).name
    candidates = sorted(
        key.replace("\\", "/") for keys in lookup.values() for key in keys
        if fold(PurePosixPath(key.replace("\\", "/")).name) == fold(basename)
    )
    if len(parts) == 1 and len(candidates) == 1:
        # Safe, unique basename lookup is useful for file-only queries.
        key = lookup[fold(candidates[0])][0]
        try:
            validate_buffer_path(key.replace("\\", "/"), root)
        except ValueError as exc:
            raise SourcePathError("File resolves outside the project root.", "path_outside_root") from exc
        return key
    code = "ambiguous_file" if len(candidates) > 1 else "file_not_found"
    raise SourcePathError(
        "File not in buffer; use a returned root-relative file path.",
        code, candidates[:5],
    )


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

    # If path is relative, resolve it relative to allowed_root
    if not user_path.is_absolute():
        combined = (allowed_root / user_path).resolve()
    else:
        combined = user_path.resolve()

    allowed_root_resolved = allowed_root.resolve()

    # Ensure resolved path is under allowed_root
    try:
        # This will raise ValueError if combined is not relative to allowed_root_resolved
        combined.relative_to(allowed_root_resolved)
    except ValueError as _e:
        raise ValueError(
            f"Path {user_path} (resolved to {combined}) escapes allowed root {allowed_root}"
        ) from _e

    return combined


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
