"""Skeleton views: compressed source lines for agent context.

Drops docstrings and comment-only lines and collapses blank-line runs while
preserving original line numbers, so the caller can keep addressing lines and
place anchors for edits.
"""

from __future__ import annotations

import ast

__all__ = [
    "skeletonize",
]

_COMMENT_TOKENS: dict[str, set[str]] = {
    "#": {".py", ".pyw", ".sh", ".bash", ".rb", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".gitignore"},
    "//": {
        ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".go", ".rs", ".c", ".h",
        ".cpp", ".hpp", ".cc", ".cxx", ".java", ".kt", ".kts", ".swift", ".cs",
        ".php", ".dart", ".scala", ".sql", ".zig", ".v",
    },
}

_PYTHON_SUFFIXES = {".py", ".pyw"}


def _docstring_ranges(source: str) -> set[int]:
    """Return 1-based line numbers covered by module/class/function docstrings."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return set()

    spans: list[tuple[int, int]] = []

    def _collect(node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            spans.append((first.lineno, first.end_lineno or first.lineno))

    _collect(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _collect(node)

    dropped: set[int] = set()
    for start, end in spans:
        dropped.update(range(start, end + 1))
    return dropped


def _comment_prefixes(file_name: str) -> str | None:
    lowered = str(file_name).lower()
    for token, suffixes in _COMMENT_TOKENS.items():
        if any(lowered.endswith(suffix) for suffix in suffixes):
            return token
    return None


def _is_blank(line: str) -> bool:
    return not line.strip()


def _is_comment(line: str, prefix: str | None) -> bool:
    return bool(prefix) and line.lstrip().startswith(prefix)


def skeletonize(lines: list[str], file_name: str) -> tuple[list[str], list[int]]:
    """Return kept lines with their original 1-based line numbers.

    Rules: whole-file template strings are parsed only for Python (docstring
    removal via AST; fallback is comment + blank handling on parse failure);
    comment-only lines are dropped for languages with a known comment token;
    runs of blank lines are collapsed to a single blank line.
    """
    lowered = str(file_name).lower()
    is_python = any(lowered.endswith(suffix) for suffix in _PYTHON_SUFFIXES)
    prefix = _comment_prefixes(file_name)

    dropped: set[int] = set()
    if is_python:
        dropped = _docstring_ranges("\n".join(lines))

    kept: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        lineno = i + 1
        if lineno in dropped:
            continue
        if _is_comment(line, prefix):
            continue
        if _is_blank(line) and kept and _is_blank(kept[-1][1]):
            continue
        kept.append((lineno, line))

    while kept and _is_blank(kept[-1][1]):
        kept.pop()

    out_lines = [line for _, line in kept]
    out_numbers = [lineno for lineno, _ in kept]
    return out_lines, out_numbers
