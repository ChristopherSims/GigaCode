"""Call and dependency graph analysis."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import json

logger = logging.getLogger(__name__)

__all__ = ["DependencyGraph", "CallChainResult", "build_dependency_graph"]

@dataclass
class CallChainResult:
    path_found: bool
    chain: list[dict[str, Any]]
    depth: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class DependencyGraph:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self._callers: dict[str, list[dict[str, Any]]] = {}   # symbol -> callers
        self._callees: dict[str, list[dict[str, Any]]] = {}   # symbol -> callees
        self._dependencies: dict[str, list[str]] = {}          # file -> imported modules
        self._reverse_deps: dict[str, list[str]] = {}          # file -> files importing it
        self._build_graph()

    def _build_graph(self) -> None:
        """Build call graph from chunk metadata."""
        # Map symbols to their defining chunks
        symbol_to_chunk: dict[str, Any] = {}
        for ch in self.chunks:
            for sym in (ch.symbols_defined or []):
                symbol_to_chunk[sym] = ch

        # Build caller/callee relationships
        for ch in self.chunks:
            for callee in (ch.symbols_called or []):
                caller_entry = {
                    "file": ch.file,
                    "line": ch.start_line,
                    "symbol": ch.name,
                }
                self._callers.setdefault(callee, []).append(caller_entry)
                if callee in symbol_to_chunk:
                    callee_chunk = symbol_to_chunk[callee]
                    callee_entry = {
                        "file": callee_chunk.file,
                        "line": callee_chunk.start_line,
                        "symbol": callee,
                    }
                    self._callees.setdefault(ch.name or ch.file, []).append(callee_entry)

        # Build file dependencies from imports
        for ch in self.chunks:
            for imp in (ch.imports or []):
                self._dependencies.setdefault(ch.file, []).append(imp)

        # Build reverse dependencies (which files import which modules)
        for ch in self.chunks:
            for imp in (ch.imports or []):
                # imp might be like "fastapi.security" or "models.user"
                # Find files that define symbols from this module
                for other_ch in self.chunks:
                    if other_ch.file == ch.file:
                        continue
                    # Heuristic: if import path appears in file path
                    if imp.replace(".", "/") in other_ch.file or imp.split(".")[-1] in Path(other_ch.file).stem:
                        self._reverse_deps.setdefault(other_ch.file, []).append(ch.file)

    def trace_call_chain(self, from_symbol: str, to_symbol: str, max_depth: int = 10) -> CallChainResult:
        """Find call chain from from_symbol to to_symbol using BFS."""
        visited: set[str] = set()
        queue: list[tuple[str, list[dict[str, Any]]]] = [(from_symbol, [{"symbol": from_symbol, "file": "", "line": 0}])]

        while queue:
            current, chain = queue.pop(0)
            if current == to_symbol:
                return CallChainResult(path_found=True, chain=chain[1:], depth=len(chain) - 1)
            if current in visited or len(chain) >= max_depth:
                continue
            visited.add(current)

            for callee in self._callees.get(current, []):
                if callee["symbol"] not in visited:
                    new_chain = chain + [callee]
                    queue.append((callee["symbol"], new_chain))

        return CallChainResult(path_found=False, chain=[], depth=0)

    def get_callers(self, symbol: str) -> list[dict[str, Any]]:
        """Get all callers of a symbol."""
        return self._callers.get(symbol, [])

    def get_dependencies(self, file: str, direction: str = "outgoing") -> list[str]:
        """Get dependencies for a file."""
        if direction == "outgoing":
            return list(set(self._dependencies.get(file, [])))
        elif direction == "incoming":
            return list(set(self._reverse_deps.get(file, [])))
        else:
            return list(set(
                self._dependencies.get(file, []) + self._reverse_deps.get(file, [])
            ))

    def find_cycles(self) -> list[list[str]]:
        """Find circular import/reference cycles."""
        # Build simplified file graph
        graph: dict[str, set[str]] = {}
        for ch in self.chunks:
            file = ch.file
            graph.setdefault(file, set())
            for imp in (ch.imports or []):
                for other_ch in self.chunks:
                    if other_ch.file != file and (imp.replace(".", "/") in other_ch.file or imp.split(".")[-1] in Path(other_ch.file).stem):
                        graph[file].add(other_ch.file)

        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if cycle not in cycles:
                        cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles

    def export_graph(self, format: str = "json") -> dict[str, Any]:
        """Export graph in specified format."""
        if format == "json":
            return {
                "nodes": list(set(ch.file for ch in self.chunks)),
                "edges": [
                    {"source": ch.file, "target": imp, "type": "import"}
                    for ch in self.chunks for imp in (ch.imports or [])
                ],
                "calls": [
                    {"caller": ch.name or ch.file, "callee": callee, "file": ch.file}
                    for ch in self.chunks for callee in (ch.symbols_called or [])
                ],
            }
        else:
            return {"status": "error", "message": f"Format '{format}' not supported"}

def build_dependency_graph(chunks: list[Any]) -> DependencyGraph:
    return DependencyGraph(chunks)
