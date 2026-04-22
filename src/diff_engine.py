"""Line-level diff engine based on content hashing."""

from __future__ import annotations

import hashlib
from typing import Sequence


def hash_lines(lines_texts: Sequence[str]) -> list[str]:
    """Hash each line with SHA-256 (truncated to 64 hex chars).

    Args:
        lines_texts: Sequence of line strings.

    Returns:
        List of hexadecimal digest strings, one per input line.
    """
    hashes: list[str] = []
    for text in lines_texts:
        # Use UTF-8 encoding, replacing invalid chars
        data = text.encode("utf-8", errors="replace")
        digest = hashlib.sha256(data).hexdigest()
        hashes.append(digest)
    return hashes


def compute_diff(
    old_hashes: Sequence[str],
    new_hashes: Sequence[str],
) -> list[int]:
    """Compare current line hashes to previous and return changed indices.

    The comparison is purely line-index based (no move detection).
    If the file grew, all new lines are considered changed.
    If the file shrank, the removed lines are not reported.

    Args:
        old_hashes: Hashes from the previous version.
        new_hashes: Hashes from the current version.

    Returns:
        Sorted list of 0-based line indices that differ.
    """
    changed: list[int] = []
    old_len = len(old_hashes)
    for i, h in enumerate(new_hashes):
        if i >= old_len or old_hashes[i] != h:
            changed.append(i)
    return changed
