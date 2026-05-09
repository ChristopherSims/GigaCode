"""Symbol-level operations: definitions, references, and symbol search.

Builds an index of symbols (functions, classes, methods, variables) extracted
from chunks, enabling exact symbol search, jump-to-definition, and find-references.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "SymbolIndex",
    "SymbolEntry",
    "SearchSymbolsResult",
    "search_symbols",
    "get_symbol_definition",
    "get_symbol_references",
]


@dataclass
class SymbolEntry:
    """A single symbol definition."""

    name: str
    file: str
    start_line: int
    end_line: int
    type: str  # "function", "class", "method", "variable", "trait", "interface"
    chunk_id: int | None = None
    parent: str | None = None  # class name for methods

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchSymbolsResult:
    """Result from symbol search."""

    query: str
    exact_matches: list[SymbolEntry]
    prefix_matches: list[SymbolEntry]
    fuzzy_matches: list[SymbolEntry]
    total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "exact_matches": [m.to_dict() for m in self.exact_matches],
            "prefix_matches": [m.to_dict() for m in self.prefix_matches],
            "fuzzy_matches": [m.to_dict() for m in self.fuzzy_matches],
            "total": self.total,
        }


@dataclass
class ReferenceResult:
    """A single reference to a symbol."""

    file: str
    line: int
    context: str  # line of code containing the reference
    confidence: str  # "high" | "medium" | "low"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SymbolIndex:
    """Index of symbols extracted from code chunks.

    Provides:
    - Exact and fuzzy symbol search by name
    - Jump-to-definition
    - Find-references (approximate via text search)
    """

    def __init__(self, chunks: list[Any]) -> None:
        """Build symbol index from chunks.

        Args:
            chunks: List of CodeChunk-like objects with .name, .file, .start_line,
                    .end_line, .type, .id, .text
        """
        self.chunks = chunks
        self._definitions: dict[str, list[SymbolEntry]] = {}
        self._file_symbols: dict[str, list[SymbolEntry]] = {}
        self._all_names: set[str] = set()

        for chunk in chunks:
            if chunk.name:
                # Determine symbol type from chunk type
                sym_type = self._chunk_type_to_symbol_type(chunk.type)
                entry = SymbolEntry(
                    name=chunk.name,
                    file=chunk.file,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    type=sym_type,
                    chunk_id=chunk.id if hasattr(chunk, "id") else None,
                )
                self._definitions.setdefault(chunk.name, []).append(entry)
                self._file_symbols.setdefault(chunk.file, []).append(entry)
                self._all_names.add(chunk.name)

            # Also index symbols_defined if present
            for sym_name in chunk.symbols_defined or []:
                if sym_name == chunk.name:
                    continue  # Already indexed above
                entry = SymbolEntry(
                    name=sym_name,
                    file=chunk.file,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    type="function",  # default
                    chunk_id=chunk.id if hasattr(chunk, "id") else None,
                )
                self._definitions.setdefault(sym_name, []).append(entry)
                self._file_symbols.setdefault(chunk.file, []).append(entry)
                self._all_names.add(sym_name)

    @staticmethod
    def _chunk_type_to_symbol_type(chunk_type: str) -> str:
        """Map chunk type to symbol type."""
        mapping = {
            "function": "function",
            "nested_function": "function",
            "method": "method",
            "class": "class",
            "trait": "trait",
            "interface": "interface",
            "lambda": "function",
            "macro": "macro",
            "module": "module",
        }
        return mapping.get(chunk_type, "variable")

    def search(self, query: str) -> SearchSymbolsResult:
        """Search for symbols by name.

        Returns exact matches, prefix matches, and fuzzy matches.
        """
        query_lower = query.lower()
        exact: list[SymbolEntry] = []
        prefix: list[SymbolEntry] = []
        fuzzy: list[SymbolEntry] = []

        for name in self._all_names:
            name_lower = name.lower()
            entries = self._definitions.get(name, [])

            if name_lower == query_lower:
                exact.extend(entries)
            elif name_lower.startswith(query_lower):
                prefix.extend(entries)
            elif query_lower in name_lower or _fuzzy_match(query_lower, name_lower):
                fuzzy.extend(entries)

        # Remove duplicates that appear in both exact and prefix
        seen = {id(e) for e in exact}
        prefix = [e for e in prefix if id(e) not in seen]
        seen.update(id(e) for e in prefix)
        fuzzy = [e for e in fuzzy if id(e) not in seen]

        return SearchSymbolsResult(
            query=query,
            exact_matches=exact,
            prefix_matches=prefix,
            fuzzy_matches=fuzzy,
            total=len(exact) + len(prefix) + len(fuzzy),
        )

    def get_definition(self, symbol: str) -> list[SymbolEntry]:
        """Get definition location(s) for a symbol.

        Args:
            symbol: Symbol name (may be qualified: "ClassName.method_name")

        Returns:
            List of SymbolEntry definitions.
        """
        # Handle qualified names
        parts = symbol.split(".")
        if len(parts) == 1:
            return self._definitions.get(symbol, [])

        # Qualified: try full match first, then method-only
        full_match = self._definitions.get(symbol, [])
        if full_match:
            return full_match

        # Try just the method name
        method_name = parts[-1]
        return self._definitions.get(method_name, [])

    def get_references(self, symbol: str, top_k: int = 50) -> list[ReferenceResult]:
        """Find references to a symbol across all chunks.

        Uses a combination of:
        1. Exact word-boundary text search (high confidence)
        2. Symbol-call matching from symbols_called metadata (medium confidence)

        Args:
            symbol: Symbol name to search for.
            top_k: Maximum number of references to return.

        Returns:
            List of ReferenceResult with context and confidence.
        """
        results: list[ReferenceResult] = []
        seen_locations: set[tuple[str, int]] = set()

        # Pattern for word-boundary match
        pattern = re.compile(rf"\b{re.escape(symbol)}\b")

        for chunk in self.chunks:
            if not chunk.text:
                continue

            # Skip the definition itself
            if chunk.name == symbol or symbol in (chunk.symbols_defined or []):
                continue

            # Method 1: symbols_called metadata (if available)
            calls = chunk.symbols_called or []
            if symbol in calls:
                # Find the line containing the call
                for i, line in enumerate(chunk.text.splitlines(), start=chunk.start_line):
                    if symbol in line:
                        loc = (chunk.file, i)
                        if loc not in seen_locations:
                            seen_locations.add(loc)
                            results.append(
                                ReferenceResult(
                                    file=chunk.file,
                                    line=i,
                                    context=line.strip(),
                                    confidence="high",
                                )
                            )
                        break

            # Method 2: text search with word boundary
            for match in pattern.finditer(chunk.text):
                # Estimate line number from match position
                line_offset = chunk.text[: match.start()].count("\n")
                line_num = chunk.start_line + line_offset
                loc = (chunk.file, line_num)
                if loc not in seen_locations:
                    seen_locations.add(loc)
                    # Get context line
                    lines = chunk.text.splitlines()
                    if line_offset < len(lines):
                        context = lines[line_offset].strip()
                    else:
                        context = chunk.text[match.start() : match.end()]
                    results.append(
                        ReferenceResult(
                            file=chunk.file,
                            line=line_num,
                            context=context,
                            confidence="medium",
                        )
                    )

        # Sort by confidence (high first) then by file/line
        results.sort(key=lambda r: (0 if r.confidence == "high" else 1, r.file, r.line))
        return results[:top_k]

    def get_file_symbols(self, file: str) -> list[SymbolEntry]:
        """Get all symbols defined in a specific file."""
        return self._file_symbols.get(file, [])


def _fuzzy_match(query: str, target: str, max_distance: int = 2) -> bool:
    """Simple fuzzy match using Levenshtein distance threshold."""
    # For short queries, use a smaller threshold
    if len(query) <= 3:
        return query in target

    distance = _levenshtein(query, target)
    return distance <= max_distance


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def search_symbols(chunks: list[Any], query: str) -> dict[str, Any]:
    """Convenience function: search symbols and return as dict."""
    index = SymbolIndex(chunks)
    result = index.search(query)
    return result.to_dict()


def get_symbol_definition(chunks: list[Any], symbol: str) -> list[dict[str, Any]]:
    """Convenience function: get definition and return as list of dicts."""
    index = SymbolIndex(chunks)
    entries = index.get_definition(symbol)
    return [e.to_dict() for e in entries]


def get_symbol_references(chunks: list[Any], symbol: str, top_k: int = 50) -> list[dict[str, Any]]:
    """Convenience function: get references and return as list of dicts."""
    index = SymbolIndex(chunks)
    refs = index.get_references(symbol, top_k)
    return [r.to_dict() for r in refs]
