"""Type-aware search for finding functions and classes by type signatures.

Provides:
- Search by type signature (e.g., "Callable[[str, int], bool]")
- Find implementations of an interface/protocol
- Extract type annotations from chunks
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "TypeSignature",
    "TypeMatch",
    "TypeSearcher",
    "search_by_type",
    "find_implementations",
]


@dataclass
class TypeSignature:
    """Extracted type signature from a code chunk."""

    name: str
    file: str
    line: int
    parameters: list[dict[str, str]]  # [{"name": "x", "type": "str"}, ...]
    return_type: str | None
    is_async: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TypeMatch:
    """A match from type-aware search."""

    file: str
    start_line: int
    end_line: int
    name: str
    type_signature: TypeSignature
    match_score: float  # 0.0-1.0
    match_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Language-specific type extraction patterns
_TYPE_PATTERNS: dict[str, re.Pattern] = {
    "python": re.compile(
        r"(?:async\s+)?def\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*self\s*\)?\s*->\s*(?P<return_type>[a-zA-Z_][a-zA-Z0-9_\[\]|,\s]*)"
    ),
}


def _extract_python_types(text: str) -> list[TypeSignature]:
    """Extract function type signatures from Python code."""
    signatures: list[TypeSignature] = []

    # Find function definitions with type annotations
    # Pattern: def name(param: type, ...) -> return_type:
    func_pattern = re.compile(
        r"(?:async\s+)?def\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(?P<params>.*?)\s*\)\s*(?:->\s*(?P<return_type>[a-zA-Z_][a-zA-Z0-9_\[\]|,\s\.]+))?\s*:",
        re.DOTALL,
    )

    for match in func_pattern.finditer(text):
        name = match.group("name")
        params_str = match.group("params") or ""
        return_type = match.group("return_type")
        is_async = bool(re.match(r"\s*async\s", text[match.start():match.end()]))

        # Parse parameters
        parameters: list[dict[str, str]] = []
        if params_str.strip() and params_str.strip() != "self" and not params_str.strip().startswith("self,"):
            # Split by comma, respecting nested brackets
            param_parts = _split_params(params_str)
            for part in param_parts:
                part = part.strip()
                if not part or part == "self" or part == "cls":
                    continue
                # Parse "name: type = default" or "name: type" or "name"
                param_match = re.match(
                    r"(?P<param_name>[a-zA-Z_][a-zA-Z0-9_]*)\s*(?::\s*(?P<param_type>[^=]+))?(?:\s*=\s*.+)?",
                    part,
                )
                if param_match:
                    parameters.append({
                        "name": param_match.group("param_name"),
                        "type": (param_match.group("param_type") or "").strip(),
                    })

        # Estimate line number (approximate from text position)
        line = text[:match.start()].count("\n") + 1

        signatures.append(TypeSignature(
            name=name,
            file="",
            line=line,
            parameters=parameters,
            return_type=return_type.strip() if return_type else None,
            is_async=is_async,
        ))

    return signatures


def _split_params(params_str: str) -> list[str]:
    """Split parameter string by comma, respecting nested brackets."""
    parts: list[str] = []
    current = ""
    depth = 0
    for char in params_str:
        if char in "([{":
            depth += 1
            current += char
        elif char in ")]}]":
            depth -= 1
            current += char
        elif char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current)
    return parts


def _match_type_pattern(
    signature: TypeSignature,
    type_pattern: str,
) -> tuple[float, str]:
    """Match a type signature against a pattern.

    Returns:
        (score, reason) tuple.
    """
    score = 0.0
    reasons: list[str] = []

    pattern_lower = type_pattern.lower()

    # Check return type
    if signature.return_type:
        ret_lower = signature.return_type.lower()
        if pattern_lower in ret_lower or ret_lower in pattern_lower:
            score += 0.3
            reasons.append(f"Return type '{signature.return_type}' matches pattern")

    # Check parameter types
    param_types = [p.get("type", "").lower() for p in signature.parameters if p.get("type")]
    for pt in param_types:
        if pattern_lower in pt or pt in pattern_lower:
            score += 0.2
            reasons.append(f"Parameter type matches pattern")
            break

    # Check parameter count heuristics for Callable patterns
    callable_match = re.search(r"callable\s*\[\s*\[(.*?)\]\s*,\s*(.*?)\s*\]", pattern_lower)
    if callable_match:
        expected_params = [p.strip() for p in callable_match.group(1).split(",") if p.strip()]
        expected_return = callable_match.group(2).strip()

        # Compare parameter count
        if len(signature.parameters) == len(expected_params):
            score += 0.3
            reasons.append("Parameter count matches")

        # Compare parameter types
        for i, (ep, sp) in enumerate(zip(expected_params, [p.get("type", "").lower() for p in signature.parameters])):
            if ep == sp or ep in sp or sp in ep:
                score += 0.1

        # Compare return type
        if signature.return_type and expected_return in signature.return_type.lower():
            score += 0.2
            reasons.append("Return type matches Callable pattern")

    if not reasons:
        return 0.0, "No type match found"

    return min(score, 1.0), "; ".join(reasons)


def _extract_class_hierarchy(text: str) -> dict[str, list[str]]:
    """Extract class inheritance from Python code.

    Returns:
        Dict mapping class name to list of parent classes.
    """
    hierarchy: dict[str, list[str]] = {}
    # Pattern: class Name(Parent1, Parent2):
    class_pattern = re.compile(
        r"class\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\((?P<bases>.*?)\))?\s*:",
    )
    for match in class_pattern.finditer(text):
        name = match.group("name")
        bases_str = match.group("bases") or ""
        bases = [b.strip() for b in bases_str.split(",") if b.strip()]
        hierarchy[name] = bases
    return hierarchy


class TypeSearcher:
    """Search codebase by type signatures and interfaces."""

    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self._type_index: dict[str, list[TypeSignature]] = {}
        self._class_hierarchy: dict[str, list[str]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Build type signature index from chunks."""
        for chunk in self.chunks:
            if not chunk.text:
                continue

            # Detect language from file extension
            language = "python"  # Default
            if hasattr(chunk, "file"):
                if chunk.file.endswith(".py"):
                    language = "python"
                elif chunk.file.endswith(".ts"):
                    language = "typescript"
                elif chunk.file.endswith(".js"):
                    language = "javascript"
                elif chunk.file.endswith(".rs"):
                    language = "rust"
                elif chunk.file.endswith(".go"):
                    language = "go"
                elif chunk.file.endswith(".java"):
                    language = "java"

            if language == "python":
                sigs = _extract_python_types(chunk.text)
                for sig in sigs:
                    sig.file = chunk.file
                    sig.line = chunk.start_line + sig.line - 1
                    self._type_index.setdefault(sig.name, []).append(sig)

                # Build class hierarchy
                hierarchy = _extract_class_hierarchy(chunk.text)
                for cls, bases in hierarchy.items():
                    self._class_hierarchy[cls] = bases

    def search_by_type(self, type_pattern: str, top_k: int = 10) -> list[TypeMatch]:
        """Find functions/classes matching a type signature pattern.

        Args:
            type_pattern: Type pattern to search for (e.g., "Callable[[str], bool]").
            top_k: Maximum results.

        Returns:
            List of TypeMatch with scores and explanations.
        """
        matches: list[tuple[TypeMatch, float]] = []

        for name, signatures in self._type_index.items():
            for sig in signatures:
                score, reason = _match_type_pattern(sig, type_pattern)
                if score > 0:
                    # Find the chunk for this signature
                    chunk = None
                    for c in self.chunks:
                        if c.file == sig.file and c.start_line <= sig.line <= c.end_line:
                            chunk = c
                            break

                    match = TypeMatch(
                        file=sig.file,
                        start_line=chunk.start_line if chunk else sig.line,
                        end_line=chunk.end_line if chunk else sig.line,
                        name=sig.name,
                        type_signature=sig,
                        match_score=round(score, 3),
                        match_reason=reason,
                    )
                    matches.append((match, score))

        # Sort by score desc
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches[:top_k]]

    def find_implementations(self, interface_name: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Find classes that implement a given interface/protocol.

        Args:
            interface_name: Name of the interface/protocol (e.g., "DataStore").
            top_k: Maximum results.

        Returns:
            List of class info dicts.
        """
        results: list[dict[str, Any]] = []
        seen: set[str] = set()

        for class_name, bases in self._class_hierarchy.items():
            # Direct inheritance
            if interface_name in bases:
                key = f"{class_name}:direct"
                if key not in seen:
                    seen.add(key)
                    results.append({
                        "class_name": class_name,
                        "inheritance_type": "direct",
                        "bases": bases,
                        "confidence": "high",
                    })

            # Check if any base class inherits from the interface (transitive)
            for base in bases:
                base_bases = self._class_hierarchy.get(base, [])
                if interface_name in base_bases:
                    key = f"{class_name}:transitive"
                    if key not in seen:
                        seen.add(key)
                        results.append({
                            "class_name": class_name,
                            "inheritance_type": "transitive",
                            "bases": bases,
                            "via": base,
                            "confidence": "medium",
                        })

        # Also search for duck-typing: classes with methods matching interface
        # (This would require knowing interface method signatures — future enhancement)

        return results[:top_k]


def search_by_type(chunks: list[Any], type_pattern: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Convenience function: search by type signature."""
    searcher = TypeSearcher(chunks)
    matches = searcher.search_by_type(type_pattern, top_k)
    return [m.to_dict() for m in matches]


def find_implementations(chunks: list[Any], interface_name: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Convenience function: find interface implementations."""
    searcher = TypeSearcher(chunks)
    return searcher.find_implementations(interface_name, top_k)
