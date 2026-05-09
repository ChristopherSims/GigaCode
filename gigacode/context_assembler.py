"""Cross-file context assembly for AI agents.

Provides smart context assembly: given a chunk, find its callers, tests,
interfaces, imports, and semantic neighborhood.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gigacode.chunker import CodeChunk

logger = logging.getLogger(__name__)


__all__ = [
    "RelatedContext",
    "ContextAssembler",
    "assemble_related_context",
]


# Test file name patterns by language
_TEST_FILE_PATTERNS: dict[str, list[re.Pattern]] = {
    "python": [re.compile(r"^test_.*\.py$"), re.compile(r".*_test\.py$")],
    "javascript": [re.compile(r".*\.(test|spec)\.(js|ts|jsx|tsx)$")],
    "typescript": [re.compile(r".*\.(test|spec)\.(js|ts|jsx|tsx)$")],
    "rust": [re.compile(r".*_test\.rs$")],
    "go": [re.compile(r".*_test\.go$")],
    "java": [re.compile(r".*Test\.java$")],
    "cpp": [re.compile(r".*_test\.(cpp|cc|cxx)$")],
}

# Call statement patterns (very approximate, language-specific)
_CALL_PATTERNS: dict[str, re.Pattern] = {
    "python": re.compile(r"(?P<callee>[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\("),
    "javascript": re.compile(
        r"(?P<callee>[a-zA-Z_$][a-zA-Z0-9_$]*(?:\.[a-zA-Z_$][a-zA-Z0-9_$]*)*)\s*\("
    ),
    "typescript": re.compile(
        r"(?P<callee>[a-zA-Z_$][a-zA-Z0-9_$]*(?:\.[a-zA-Z_$][a-zA-Z0-9_$]*)*)\s*\("
    ),
    "java": re.compile(r"(?P<callee>[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\("),
    "rust": re.compile(r"(?P<callee>[a-zA-Z_][a-zA-Z0-9_]*(?:::[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\("),
    "go": re.compile(r"(?P<callee>[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\("),
}

# Import patterns
_IMPORT_PATTERNS: dict[str, re.Pattern] = {
    "python": re.compile(
        r"^(?:from\s+(?P<module>[a-zA-Z_][a-zA-Z0-9_.]*)\s+import\s+(?P<names>.*)|import\s+(?P<module_only>[a-zA-Z_][a-zA-Z0-9_.]*(?:\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*)?))"
    ),
    "javascript": re.compile(
        r"(?:import\s+.*?\s+from\s+['\"](?P<module>[^'\"]+)['\"]|require\s*\(\s*['\"](?P<module_req>[^'\"]+)['\"]\s*\))"
    ),
    "typescript": re.compile(
        r"(?:import\s+.*?\s+from\s+['\"](?P<module>[^'\"]+)['\"]|require\s*\(\s*['\"](?P<module_req>[^'\"]+)['\"]\s*\))"
    ),
}


def _is_test_file(filename: str, language: str = "python") -> bool:
    """Check if a filename looks like a test file."""
    basename = Path(filename).name
    patterns = _TEST_FILE_PATTERNS.get(language, [])
    return any(p.match(basename) for p in patterns)


def _extract_calls(text: str, language: str = "python") -> list[str]:
    """Extract function/method call names from source text."""
    pattern = _CALL_PATTERNS.get(language)
    if not pattern:
        return []
    calls: set[str] = set()
    for match in pattern.finditer(text):
        callee = match.group("callee")
        if callee and not callee.startswith("("):
            # Filter out keywords and builtins
            if callee not in ("if", "while", "for", "switch", "catch", "assert"):
                calls.add(callee)
    return sorted(calls)


def _extract_imports(text: str, language: str = "python") -> list[str]:
    """Extract import/module names from source text."""
    pattern = _IMPORT_PATTERNS.get(language)
    if not pattern:
        return []
    imports: set[str] = set()
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            module = (
                match.group("module") or match.group("module_only") or match.group("module_req")
            )
            if module:
                imports.add(module)
    return sorted(imports)


def _build_symbol_index(chunks: list[CodeChunk]) -> dict[str, list[CodeChunk]]:
    """Build an index of symbol names to defining chunks."""
    index: dict[str, list[CodeChunk]] = {}
    for chunk in chunks:
        for sym in chunk.symbols_defined or []:
            index.setdefault(sym, []).append(chunk)
    return index


def _find_callers(
    chunks: list[CodeChunk],
    target_name: str,
    embeddings: np.ndarray | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Find chunks that call the target symbol."""
    candidates: list[tuple[CodeChunk, int]] = []
    for chunk in chunks:
        calls = chunk.symbols_called or []
        if target_name in calls:
            candidates.append((chunk, 100))  # Exact call match = high score
        elif embeddings is not None:
            # Semantic fallback: chunks with similar embeddings
            # (handled by caller if no exact call matches)
            pass

    # Sort by score desc
    candidates.sort(key=lambda x: x[1], reverse=True)
    results = []
    for chunk, score in candidates[:top_k]:
        results.append(
            {
                "file": chunk.file,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "type": chunk.type,
                "name": chunk.name,
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "score": score / 100.0,
                "match_type": "exact_call",
            }
        )
    return results


def _find_tests(
    chunks: list[CodeChunk],
    target_file: str,
    target_name: str | None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Find test chunks related to a target file/symbol."""
    # Strategy 1: test files that reference the target file name or symbol
    results = []
    target_basename = Path(target_file).stem
    for chunk in chunks:
        if not _is_test_file(chunk.file):
            continue
        score = 0
        text = chunk.text
        # File name reference
        if target_basename in text:
            score += 50
        # Symbol name reference
        if target_name and target_name in text:
            score += 40
        # Import reference
        imports = chunk.imports or []
        if any(target_basename in imp for imp in imports):
            score += 30
        if score > 0:
            results.append((chunk, score))

    results.sort(key=lambda x: x[1], reverse=True)
    out = []
    for chunk, score in results[:top_k]:
        out.append(
            {
                "file": chunk.file,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "type": chunk.type,
                "name": chunk.name,
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "score": min(score / 100.0, 1.0),
                "match_type": "test_reference",
            }
        )
    return out


def _find_interfaces(
    chunks: list[CodeChunk],
    target_name: str | None,
    target_type: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Find interface definitions (classes, protocols, traits) for a symbol."""
    if not target_name or target_type not in ("function", "method"):
        return []
    # Heuristic: find class chunks that define the same method name
    results = []
    for chunk in chunks:
        if chunk.type in ("class", "trait", "interface"):
            # Check if this class has a method with the same name
            # (requires symbols_defined analysis)
            symbols = chunk.symbols_defined or []
            if target_name in symbols:
                results.append((chunk, 90))
            elif target_name in chunk.text:
                results.append((chunk, 30))

    results.sort(key=lambda x: x[1], reverse=True)
    out = []
    for chunk, score in results[:top_k]:
        out.append(
            {
                "file": chunk.file,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "type": chunk.type,
                "name": chunk.name,
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "score": score / 100.0,
                "match_type": "interface",
            }
        )
    return out


def _semantic_neighborhood(
    chunks: list[CodeChunk],
    target_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    top_k: int = 10,
    exclude_file: str | None = None,
) -> list[dict[str, Any]]:
    """Find chunks semantically similar to the target, excluding same file."""
    if chunk_embeddings.size == 0:
        return []
    # Dot product on L2-normalized vectors
    scores = np.dot(chunk_embeddings, target_embedding)
    # Get top_k indices
    indices = np.argsort(scores)[::-1]
    results = []
    for idx in indices:
        chunk = chunks[idx]
        if exclude_file and chunk.file == exclude_file:
            continue
        score = float(scores[idx])
        if score < 0.3:  # Minimum relevance threshold
            continue
        results.append(
            {
                "file": chunk.file,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "type": chunk.type,
                "name": chunk.name,
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "score": score,
                "match_type": "semantic",
            }
        )
        if len(results) >= top_k:
            break
    return results


@dataclass
class RelatedContext:
    """Assembled context for a code chunk."""

    file: str
    start_line: int
    end_line: int
    name: str | None
    callers: list[dict[str, Any]]
    tests: list[dict[str, Any]]
    interfaces: list[dict[str, Any]]
    imports: list[str]
    semantic_neighbors: list[dict[str, Any]]
    total_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextAssembler:
    """Assemble cross-file context for a given code location."""

    def __init__(
        self,
        chunks: list[CodeChunk],
        embeddings: np.ndarray | None = None,
        language: str = "python",
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.language = language
        self._symbol_index = _build_symbol_index(chunks)

    def assemble(
        self,
        file: str,
        start_line: int,
        end_line: int | None = None,
        include: list[str] | None = None,
        max_tokens: int = 8192,
    ) -> RelatedContext:
        """Assemble related context for a code location.

        Args:
            file: Source file path.
            start_line: Start line (1-based).
            end_line: End line (1-based). If None, uses start_line.
            include: Context types to include:
                ["callers", "tests", "interfaces", "imports", "semantic"]
                Default: all.
            max_tokens: Token budget for assembled context.

        Returns:
            RelatedContext dataclass with assembled pieces.
        """
        include = include or ["callers", "tests", "interfaces", "imports", "semantic"]
        end_line = end_line or start_line

        # Find the target chunk
        target_chunk: CodeChunk | None = None
        target_idx: int = -1
        for i, chunk in enumerate(self.chunks):
            if chunk.file == file and chunk.start_line <= start_line <= chunk.end_line:
                target_chunk = chunk
                target_idx = i
                break

        if target_chunk is None:
            # Create a synthetic chunk for the query range
            target_chunk = CodeChunk(
                id=-1,
                file=file,
                start_line=start_line,
                end_line=end_line,
                type="query",
                name=None,
                text="",
            )

        target_name = target_chunk.name
        target_embedding = (
            self.embeddings[target_idx] if self.embeddings is not None and target_idx >= 0 else None
        )

        # --- Gather raw candidates ---
        callers: list[dict[str, Any]] = []
        tests: list[dict[str, Any]] = []
        interfaces: list[dict[str, Any]] = []
        imports: list[str] = target_chunk.imports or []
        semantic_neighbors: list[dict[str, Any]] = []

        if "callers" in include and target_name:
            callers = _find_callers(self.chunks, target_name, embeddings=self.embeddings, top_k=10)

        if "tests" in include:
            tests = _find_tests(self.chunks, file, target_name, top_k=10)

        if "interfaces" in include and target_name:
            interfaces = _find_interfaces(self.chunks, target_name, target_chunk.type, top_k=5)

        if "semantic" in include and target_embedding is not None:
            semantic_neighbors = _semantic_neighborhood(
                self.chunks,
                target_embedding,
                self.embeddings,
                top_k=10,
                exclude_file=file,
            )

        # --- Token-budgeted assembly ---
        # Priority order: target chunk → callers → tests → interfaces → semantic neighbors
        # Imports are metadata-only (cheap) and always included.
        budget = max_tokens
        used = 0

        def _tokens(text: str | None) -> int:
            return (len(text or "") // 4) + 1  # rough +1 for metadata overhead

        # Reserve target chunk (truncated if needed)
        target_text = target_chunk.text or ""
        target_tok = _tokens(target_text)
        if target_tok > budget:
            # Truncate target to fit budget
            max_chars = budget * 4
            target_text = target_text[:max_chars] + "\n... [truncated]"
            target_tok = budget
        used += target_tok

        def _fit(items: list[dict[str, Any]], reserve: int = 0) -> list[dict[str, Any]]:
            """Return prefix of items that fits within remaining budget."""
            nonlocal used
            remaining = budget - used - reserve
            fitted: list[dict[str, Any]] = []
            for it in items:
                cost = _tokens(it.get("text", ""))
                if cost > remaining:
                    break
                fitted.append(it)
                remaining -= cost
                used += cost
            return fitted

        # Fit callers (high priority)
        callers = _fit(callers, reserve=0)
        # Fit tests (medium-high priority)
        tests = _fit(tests, reserve=0)
        # Fit interfaces (medium priority)
        interfaces = _fit(interfaces, reserve=0)
        # Fit semantic neighbors (lowest priority)
        semantic_neighbors = _fit(semantic_neighbors, reserve=0)

        total_tokens = used + sum(_tokens(imp) for imp in imports)

        return RelatedContext(
            file=file,
            start_line=start_line,
            end_line=end_line,
            name=target_name,
            callers=callers,
            tests=tests,
            interfaces=interfaces,
            imports=imports,
            semantic_neighbors=semantic_neighbors,
            total_tokens=total_tokens,
        )


def assemble_related_context(
    chunks: list[CodeChunk],
    file: str,
    start_line: int,
    end_line: int | None = None,
    embeddings: np.ndarray | None = None,
    language: str = "python",
    include: list[str] | None = None,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    """Convenience function: assemble related context and return as dict."""
    assembler = ContextAssembler(chunks, embeddings=embeddings, language=language)
    ctx = assembler.assemble(
        file=file,
        start_line=start_line,
        end_line=end_line,
        include=include,
        max_tokens=max_tokens,
    )
    return ctx.to_dict()
