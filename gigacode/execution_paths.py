"""Execution path tracing for code symbols.

Traces all execution paths through a function/method using AST branch
detection. Returns each path as a chain of calls with branch counts.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ExecutionPath",
    "trace_execution_paths",
]


@dataclass
class ExecutionPath:
    """A single execution path through a function."""

    path: list[str]  # ["handle_request → validate_input → check_auth"]
    branches: int  # Number of branch points along this path
    calls: list[str]  # Individual function calls in order
    conditions: list[str]  # Condition expressions at branch points

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_branches_and_calls(source: str) -> tuple[int, list[str], list[str]]:
    """Extract branch count, calls, and conditions from Python source.

    Returns:
        (branch_count, calls_list, conditions_list)
    """
    branches = 0
    calls: list[str] = []
    conditions: list[str] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fallback: regex-based extraction
        return _extract_branches_regex(source)

    for node in ast.walk(tree):
        # Count branches
        if isinstance(node, (ast.If, ast.AsyncIf)):
            branches += 1
            conditions.append(_node_to_str(node.test))
        elif isinstance(node, ast.For):
            branches += 1
            conditions.append(f"for {_node_to_str(node.target)}")
        elif isinstance(node, ast.AsyncFor):
            branches += 1
            conditions.append(f"async for {_node_to_str(node.target)}")
        elif isinstance(node, ast.While):
            branches += 1
            conditions.append(_node_to_str(node.test))
        elif isinstance(node, ast.ExceptHandler):
            branches += 1
            conditions.append(f"except {node.type and _node_to_str(node.type)}")
        elif isinstance(node, ast.BoolOp):
            # and/or create implicit branches
            for value in node.values:
                branches += 1
                conditions.append(_node_to_str(value))

        # Extract calls
        if isinstance(node, ast.Call):
            call_name = _get_call_name(node)
            if call_name and call_name not in calls:
                calls.append(call_name)

    return branches, calls, conditions


def _extract_branches_regex(source: str) -> tuple[int, list[str], list[str]]:
    """Fallback regex-based extraction for non-Python or unparseable code."""
    branches = 0
    calls: list[str] = []
    conditions: list[str] = []

    # Branch patterns
    branch_patterns = [
        (r"\bif\b\s+(.+?):", "if"),
        (r"\belif\b\s+(.+?):", "elif"),
        (r"\bfor\b\s+\w+\b", "for"),
        (r"\bwhile\b\s+(.+?):", "while"),
        (r"\bexcept\b\s*(\w+)?", "except"),
        (r"\band\b", "and"),
        (r"\bor\b", "or"),
    ]
    for pattern, kind in branch_patterns:
        for match in re.finditer(pattern, source):
            branches += 1
            conditions.append(match.group(1) if match.lastindex else kind)

    # Call patterns
    call_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(")
    keywords = {
        "if",
        "while",
        "for",
        "with",
        "assert",
        "raise",
        "return",
        "print",
        "len",
        "range",
        "isinstance",
        "type",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "bool",
        "super",
    }
    for match in call_pattern.finditer(source):
        name = match.group(1)
        if name not in keywords and name not in calls:
            calls.append(name)

    return branches, calls, conditions


def _node_to_str(node: ast.AST) -> str:
    """Convert an AST node to a readable string."""
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return str(getattr(node, "id", getattr(node, "attr", "?")))


def _get_call_name(node: ast.Call) -> str | None:
    """Extract the function name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        parts = []
        current = node.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        return ".".join(parts)
    return None


def trace_execution_paths(
    chunks: list[Any],
    symbol: str,
    max_depth: int = 3,
) -> list[ExecutionPath]:
    """Trace all execution paths through a symbol.

    Step 1: Extract branches and calls from the symbol's chunk.
    Step 2: Follow each call to its definition and repeat up to max_depth.

    Args:
        chunks: List of CodeChunk objects.
        symbol: Symbol name to trace paths for.
        max_depth: Maximum depth to follow calls (default: 3).

    Returns:
        List of ExecutionPath objects, each with path chain, branch count,
        and individual calls.
    """
    # Find the chunk for this symbol
    target_chunk = None
    for chunk in chunks:
        if hasattr(chunk, "name") and chunk.name == symbol and chunk.text:
            target_chunk = chunk
            break

    if target_chunk is None:
        # Try partial match
        for chunk in chunks:
            if hasattr(chunk, "name") and symbol in (chunk.name or "") and chunk.text:
                target_chunk = chunk
                break

    if target_chunk is None or not target_chunk.text:
        return []

    # Build definition lookup
    definitions: dict[str, Any] = {}
    for chunk in chunks:
        if hasattr(chunk, "name") and chunk.name:
            definitions[chunk.name] = chunk

    # Trace paths using DFS
    paths: list[ExecutionPath] = []
    visited: set[str] = set()

    def _trace(current_sym: str, chain: list[str], depth: int) -> None:
        if depth > max_depth:
            return
        if current_sym in visited:
            chain.append(f"{current_sym} (cycle)")
            paths.append(
                ExecutionPath(
                    path=[" → ".join(chain)],
                    branches=0,
                    calls=list(chain),
                    conditions=[],
                )
            )
            chain.pop()
            return

        visited.add(current_sym)
        chunk = definitions.get(current_sym)
        if chunk is None or not chunk.text:
            chain.append(current_sym)
            paths.append(
                ExecutionPath(
                    path=[" → ".join(chain)],
                    branches=0,
                    calls=list(chain),
                    conditions=[],
                )
            )
            visited.discard(current_sym)
            chain.pop()
            return

        branches, calls, conditions = _extract_branches_and_calls(chunk.text)

        if not calls or depth == max_depth:
            chain.append(current_sym)
            paths.append(
                ExecutionPath(
                    path=[" → ".join(chain)],
                    branches=branches,
                    calls=list(chain),
                    conditions=conditions,
                )
            )
            visited.discard(current_sym)
            chain.pop()
            return

        # Follow each call
        for call_sym in calls:
            chain.append(current_sym)
            _trace(call_sym, chain, depth + 1)
            chain.pop()

        visited.discard(current_sym)

    # Start tracing
    root_branches, root_calls, root_conditions = _extract_branches_and_calls(target_chunk.text)

    if root_calls:
        for call_sym in root_calls:
            _trace(call_sym, [symbol], 1)
    else:
        # No calls — single path
        paths.append(
            ExecutionPath(
                path=[symbol],
                branches=root_branches,
                calls=[symbol],
                conditions=root_conditions,
            )
        )

    # Deduplicate and limit
    seen_paths: set[str] = set()
    unique_paths: list[ExecutionPath] = []
    for p in paths:
        key = " → ".join(p.calls)
        if key not in seen_paths:
            seen_paths.add(key)
            unique_paths.append(p)

    return unique_paths[:20]
