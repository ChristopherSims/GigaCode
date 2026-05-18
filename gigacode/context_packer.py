"""LLM context packing — greedily pack high-relevance chunks into a token budget.

Token counting is approximate (1 token ≈ 4 chars for English/code) since we
do not require tiktoken as a dependency.  If ``tiktoken`` is installed it is
used for more accurate counts.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from gigacode.constants import CHARS_PER_TOKEN

logger = logging.getLogger(__name__)


__all__ = [
    "pack_context",
    "pack_context_smart",
    "strip_boilerplate_text",
    "deduplicate_chunks",
]


def _approx_token_count(text: str) -> int:
    """Fast approximate token count."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def _exact_token_count(text: str) -> int:
    """Try tiktoken; fall back to approximation."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except (ImportError, ModuleNotFoundError):
        return _approx_token_count(text)


# License header patterns
_LICENSE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)^\s*#.*(?:license|copyright|spdx)"),
    re.compile(r"(?i)^\s*\/\/.*(?:license|copyright|spdx)"),
    re.compile(r"(?i)^\s*\*\s.*(?:license|copyright|spdx)"),
    re.compile(r"(?i)^\s*\s*\*\s.*(?:license|copyright|spdx)"),
    re.compile(r"(?i)^\s*#\s*this file is part of"),
    re.compile(r"(?i)^\s*#\s*all rights reserved"),
]

# Import block patterns by language
_IMPORT_BLOCK_PATTERNS: dict[str, re.Pattern] = {
    "python": re.compile(r"^(?:from\s+\S+\s+import|import\s+\S+|#\s*\-\*\-).*", re.MULTILINE),
    "javascript": re.compile(r"^(?:import\s+.*?from\s+['\"]|require\s*\().*", re.MULTILINE),
    "typescript": re.compile(r"^(?:import\s+.*?from\s+['\"]|require\s*\().*", re.MULTILINE),
    "rust": re.compile(r"^(?:use\s+\S+|extern\s+crate\s+\S+).*", re.MULTILINE),
    "go": re.compile(r"^(?:import\s*\(|package\s+\S+).*", re.MULTILINE),
    "java": re.compile(r"^(?:import\s+\S+|package\s+\S+).*", re.MULTILINE),
}

# Test file patterns
_TEST_FILE_RE: list[re.Pattern] = [
    re.compile(r"^test_.*\.py$"),
    re.compile(r".*_test\.py$"),
    re.compile(r".*\.(test|spec)\.(js|ts|jsx|tsx)$"),
    re.compile(r".*_test\.rs$"),
    re.compile(r".*_test\.go$"),
    re.compile(r".*Test\.java$"),
]


def _is_test_file(file_path: str) -> bool:
    """Check if a file path looks like a test file."""
    from pathlib import Path

    name = Path(file_path).name
    return any(p.match(name) for p in _TEST_FILE_RE)


def _is_boilerplate_line(line: str) -> bool:
    """Check if a line is boilerplate (license, encoding, shebang, __all__)."""
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#!/"):
        return True
    if "coding:" in stripped or "-*-" in stripped:
        return True
    if "__all__" in stripped:
        return True
    for pattern in _LICENSE_PATTERNS:
        if pattern.match(stripped):
            return True
    return False


def _detect_language(file_path: str) -> str:
    """Detect programming language from file extension."""
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".h": "c",
    }
    from pathlib import Path

    ext = Path(file_path).suffix.lower()
    return ext_map.get(ext, "python")


def strip_boilerplate_text(
    text: str,
    language: str = "python",
    strip_docstrings: bool = False,
) -> str:
    """Remove boilerplate lines from code text.

    Removes:
    - License/copyright headers
    - Shebang lines
    - Encoding declarations
    - __all__ definitions
    - Import blocks (optional, via language-specific patterns)
    - Empty lines at start/end

    Args:
        text: Source code text.
        language: Programming language for import stripping.
        strip_docstrings: Also remove triple-quoted docstrings.

    Returns:
        Cleaned text.
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    in_docstring = False
    docstring_quote = ""

    for line in lines:
        stripped = line.strip()

        # Skip boilerplate lines
        if _is_boilerplate_line(line):
            continue

        # Handle docstring stripping
        if strip_docstrings:
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if not (stripped.endswith('"""') and len(stripped) > 3) and not (
                        stripped.endswith("'''") and len(stripped) > 3
                    ):
                        in_docstring = True
                        docstring_quote = '"""' if '"""' in stripped else "'''"
                    continue
            else:
                if docstring_quote in stripped:
                    in_docstring = False
                continue

        cleaned.append(line)

    # Strip leading/trailing empty lines
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()

    return "\n".join(cleaned)


def _chunk_signature(text: str, chunk_type: str, name: str | None) -> str:
    """Extract just the signature/definition line + docstring from a chunk."""
    lines = text.splitlines()
    if not lines:
        return text

    # First line is usually the definition
    signature_lines = [lines[0]]

    # Include docstring if present (next 1-5 lines)
    if len(lines) > 1:
        second = lines[1].strip()
        if second.startswith('"""') or second.startswith("'''"):
            docstring_lines = []
            in_doc = True
            for line in lines[1:]:
                docstring_lines.append(line)
                if line.strip().endswith('"""') or line.strip().endswith("'''"):
                    break
            signature_lines.extend(docstring_lines)

    return "\n".join(signature_lines)


def deduplicate_chunks(
    chunks: list[Any],
    scores: list[float],
    threshold: float = 0.95,
) -> tuple[list[Any], list[float], dict[str, int]]:
    """Remove near-duplicate chunks based on text hash similarity.

    Args:
        chunks: List of chunks.
        scores: Parallel scores.
        threshold: Min hash similarity (0.0-1.0) to consider duplicate.

    Returns:
        (filtered_chunks, filtered_scores, stats) tuple.
    """
    if len(chunks) != len(scores):
        return chunks, scores, {"error": "mismatched lengths"}

    kept: list[Any] = []
    kept_scores: list[float] = []
    seen_hashes: set[str] = set()
    duplicates = 0

    for ch, score in zip(chunks, scores, strict=False):
        # Normalize text: remove whitespace, lowercase
        normalized = re.sub(r"\s+", "", ch.text.lower())
        text_hash = hashlib.sha256(normalized.encode()).hexdigest()

        if text_hash in seen_hashes:
            duplicates += 1
            continue

        seen_hashes.add(text_hash)
        kept.append(ch)
        kept_scores.append(score)

    stats = {
        "original": len(chunks),
        "kept": len(kept),
        "duplicates_removed": duplicates,
    }
    return kept, kept_scores, stats


def pack_context_smart(
    chunks: list[Any],
    scores: list[float],
    query: str = "",
    max_tokens: int = 8192,
    use_tiktoken: bool = False,
    # Smart packing options
    deduplicate: bool = True,
    strip_boilerplate: bool = True,
    strip_docstrings: str = "auto",
    exclude_types: list[str] | None = None,
    exclude_test_files: bool = False,
    min_lines: int = 3,
    max_lines: int = 200,
    granularity: str = "smart",  # "signatures" | "bodies" | "smart"
) -> dict[str, Any]:
    """Smart context packing with token-saving optimizations.

    Applies multiple filters to reduce token usage while maintaining relevance:
    1. Deduplication: remove copy-pasted chunks
    2. Boilerplate stripping: remove imports, licenses, __all__
    3. Docstring stripping: remove verbose docstrings (unless query asks for docs)
    4. Type filtering: skip orphans, sliding windows, tests
    5. Size filtering: skip tiny or huge chunks
    6. Granularity: use signatures-only for low-relevance chunks

    Args:
        chunks: List of CodeChunk-like objects.
        scores: Parallel relevance scores.
        query: Original query (for auto strip_docstrings detection).
        max_tokens: Target token budget.
        use_tiktoken: Use exact token counting.
        deduplicate: Remove near-duplicate chunks.
        strip_boilerplate: Remove license/import headers.
        strip_docstrings: "auto" | "always" | "never". Auto detects "doc" in query.
        exclude_types: Chunk types to skip (e.g., ["orphan", "sliding"]).
        exclude_test_files: Skip test files.
        min_lines: Minimum chunk line count.
        max_lines: Maximum chunk line count.
        granularity: "signatures" (defs only), "bodies" (full), "smart" (signatures for low-relevance).

    Returns:
        Dict with packed chunks, token stats, and filter report.
    """
    if len(chunks) != len(scores):
        return {
            "status": "error",
            "message": f"chunks ({len(chunks)}) and scores ({len(scores)}) length mismatch",
        }

    token_fn = _exact_token_count if use_tiktoken else _approx_token_count
    exclude_types = exclude_types or []

    # Determine docstring stripping
    if strip_docstrings == "auto":
        strip_docs = "doc" not in query.lower() and "document" not in query.lower()
    else:
        strip_docs = strip_docstrings == "always"

    filter_stats = {
        "original": len(chunks),
        "orphans": 0,
        "boilerplate": 0,
        "duplicates": 0,
        "test_files": 0,
        "too_small": 0,
        "too_large": 0,
        "stripped_tokens": 0,
    }

    # --- Step 1: Filter by type, size, test files ---
    filtered_chunks: list[Any] = []
    filtered_scores: list[float] = []

    for ch, score in zip(chunks, scores, strict=False):
        # Exclude by type
        if ch.type in exclude_types:
            filter_stats["orphans"] += 1
            continue

        # Exclude test files
        if exclude_test_files and _is_test_file(ch.file):
            filter_stats["test_files"] += 1
            continue

        # Size filter
        line_count = ch.end_line - ch.start_line + 1
        if line_count < min_lines:
            filter_stats["too_small"] += 1
            continue
        if line_count > max_lines:
            filter_stats["too_large"] += 1
            continue

        filtered_chunks.append(ch)
        filtered_scores.append(score)

    # --- Step 2: Deduplicate ---
    if deduplicate:
        filtered_chunks, filtered_scores, dedup_stats = deduplicate_chunks(
            filtered_chunks, filtered_scores
        )
        filter_stats["duplicates"] = dedup_stats.get("duplicates_removed", 0)

    # --- Step 3: Sort by score and pack ---
    indexed = sorted(enumerate(filtered_scores), key=lambda x: x[1], reverse=True)

    packed: list[dict[str, Any]] = []
    total_tokens = 0

    for rank, (idx, score) in enumerate(indexed):
        ch = filtered_chunks[idx]

        # Determine granularity for this chunk
        if granularity == "signatures":
            text = _chunk_signature(ch.text, ch.type, ch.name)
        elif granularity == "smart":
            # Top 3 chunks get full text, rest get signatures
            if rank < 3:
                text = ch.text
            else:
                text = _chunk_signature(ch.text, ch.type, ch.name)
        else:
            text = ch.text

        # Strip boilerplate
        if strip_boilerplate:
            language = _detect_language(ch.file)
            original_len = len(text)
            text = strip_boilerplate_text(text, language, strip_docs)
            filter_stats["stripped_tokens"] += (original_len - len(text)) // 4

        cost = token_fn(text)

        if total_tokens + cost > max_tokens:
            break

        packed.append(
            {
                "file": ch.file,
                "start_line": ch.start_line,
                "end_line": ch.end_line,
                "name": ch.name,
                "type": ch.type,
                "score": round(score, 4),
                "tokens": cost,
                "granularity": (
                    granularity
                    if rank < 3
                    else "signature"
                    if granularity == "smart"
                    else granularity
                ),
                "text_preview": text[:120] + "..." if len(text) > 120 else text,
            }
        )
        total_tokens += cost

    # Re-sort by file order for coherent reading
    packed.sort(key=lambda x: (x["file"], x["start_line"]))

    naive_tokens = sum(token_fn(ch.text) for ch in chunks)
    saved_tokens = naive_tokens - total_tokens

    return {
        "status": "ok",
        "packed_chunks": packed,
        "total_tokens": total_tokens,
        "remaining_tokens": max(0, max_tokens - total_tokens),
        "count": len(packed),
        "naive_token_estimate": naive_tokens,
        "saved_tokens": saved_tokens,
        "savings_percent": round((saved_tokens / max(naive_tokens, 1)) * 100, 1),
        "filter_stats": filter_stats,
    }


def pack_context(
    chunks: list[Any],
    scores: list[float],
    max_tokens: int = 8192,
    use_tiktoken: bool = False,
) -> dict[str, Any]:
    """Pack chunks by relevance using best-fit strategy until *max_tokens* is reached.

    Uses a deterministic best-fit algorithm:
    1. Sort chunks by relevance score (descending)
    2. Accumulate highest-scored chunks that fit within budget
    3. Stop when adding the next chunk would exceed budget

    This ensures:
    - Predictable behavior (no skipping then accepting)
    - Optimal token usage for the set of chunks considered
    - Consistent results across runs

    Args:
        chunks: List of CodeChunk-like objects (must have ``.text``, ``.file``, ``.start_line``, ``.end_line``, ``.name``).
        scores: Parallel list of relevance scores (higher = better).
        max_tokens: Target token budget.
        use_tiktoken: If True, use tiktoken for exact counts (slower).

    Returns:
        Dict with ``status``, ``packed_chunks``, ``total_tokens``, and ``remaining_tokens``.
    """
    return pack_context_smart(
        chunks=chunks,
        scores=scores,
        max_tokens=max_tokens,
        use_tiktoken=use_tiktoken,
        # Default pack_context behavior: no smart optimizations
        deduplicate=False,
        strip_boilerplate=False,
        strip_docstrings="never",
        exclude_types=[],
        exclude_test_files=False,
        min_lines=1,
        max_lines=999999,
        granularity="bodies",
    )
