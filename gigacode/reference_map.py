"""Incremental reference map with lazy on-demand construction.

Three-step strategy:
1. Lazy/On-Demand: Build direct caller/callee neighborhood on first query, cache it.
2. Incremental Update on File Changes: Invalidate affected subgraph on file changes.
3. Background Fill: Asynchronously expand the call graph after serving first query.

Unlike DependencyGraph (eager, rebuilt wholesale), this module caches per-symbol
neighborhoods and only invalidates what changed.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ReferenceNeighborhood",
    "ReferenceMap",
]


@dataclass
class ReferenceNeighborhood:
    """Caller/callee neighborhood for a single symbol."""

    symbol: str
    file: str
    line: int
    callers: list[dict[str, Any]]  # [{file, line, symbol, context}]
    callees: list[dict[str, Any]]  # [{file, line, symbol, context}]
    direction: str = "both"  # "both", "calls", "called_by"
    depth: int = 1  # How many levels deep we've explored
    complete: bool = False  # Whether background fill has finished

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class ReferenceMap:
    """Incremental call graph with lazy on-demand construction.

    Step 1 (Lazy): On first query for a symbol, build only its direct
    caller/callee neighborhood from chunks. Cache results.

    Step 2 (Incremental): On file changes, invalidate only the
    subgraph for symbols defined in or referenced by changed files.

    Step 3 (Background): After serving the initial query, mark the
    neighborhood as expandable. A caller can trigger background fill
    to expand to unvisited symbols.
    """

    def __init__(self, chunks: list[Any], max_cache: int = 1000) -> None:
        self._chunks = chunks
        self._max_cache = max_cache

        # Per-symbol cached neighborhoods (LRU)
        self._cache: OrderedDict[str, ReferenceNeighborhood] = OrderedDict()

        # Reverse index: file -> set of symbol names defined in that file
        self._file_to_symbols: dict[str, set[str]] = defaultdict(set)

        # Reverse index: file -> set of symbol names that reference (call) something in that file
        self._file_to_callers: dict[str, set[str]] = defaultdict(set)

        # Symbol definitions: symbol -> (file, start_line, end_line, chunk)
        self._definitions: dict[str, list[tuple[str, int, int, Any]]] = defaultdict(list)

        # Pre-build definition index (cheap, no full graph)
        for chunk in chunks:
            if not hasattr(chunk, "file"):
                continue
            name = getattr(chunk, "name", None)
            if name:
                self._definitions[name].append(
                    (chunk.file, chunk.start_line, chunk.end_line, chunk)
                )
                self._file_to_symbols[chunk.file].add(name)

            # Track symbols_defined
            for sym in getattr(chunk, "symbols_defined", []) or []:
                if sym != name:
                    self._definitions[sym].append(
                        (chunk.file, chunk.start_line, chunk.end_line, chunk)
                    )
                    self._file_to_symbols[chunk.file].add(sym)

    def get_references(
        self,
        symbol: str,
        direction: str = "both",
        top_k: int = 50,
    ) -> dict[str, Any]:
        """Get caller/callee references for a symbol.

        Step 1: Lazy on-demand. Builds neighborhood on first query, caches it.

        Args:
            symbol: Symbol name to find references for.
            direction: "both", "calls" (callees), or "called_by" (callers).
            top_k: Maximum references per direction.

        Returns:
            Dict with symbol, file, line, callers, callees, direction, cached.
        """
        cache_key = f"{symbol}:{direction}"

        # Check cache
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            result = self._cache[cache_key]
            result_dict = result.to_dict()
            result_dict["cached"] = True
            return {"status": "ok", **result_dict}

        # Step 1: Build neighborhood on demand
        neighborhood = self._build_neighborhood(symbol, direction, top_k)

        if neighborhood is None:
            return {
                "status": "ok",
                "symbol": symbol,
                "callers": [],
                "callees": [],
                "message": f"Symbol '{symbol}' not found in codebase",
            }

        # Cache it
        self._cache[cache_key] = neighborhood
        # Evict LRU if over capacity
        while len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)

        result_dict = neighborhood.to_dict()
        result_dict["cached"] = False
        return {"status": "ok", **result_dict}

    def invalidate_file(self, file: str) -> int:
        """Step 2: Invalidate all cached entries related to a changed file.

        Removes neighborhoods for symbols defined in the file and
        symbols that call into the file.

        Returns the number of entries evicted.
        """
        keys_to_evict: set[str] = set()

        # Evict symbols defined in this file
        for sym in self._file_to_symbols.get(file, set()):
            for direction in ("both", "calls", "called_by"):
                keys_to_evict.add(f"{sym}:{direction}")

        # Evict symbols that reference things in this file
        for sym in self._file_to_callers.get(file, set()):
            for direction in ("both", "calls", "called_by"):
                keys_to_evict.add(f"{sym}:{direction}")

        count = 0
        for key in keys_to_evict:
            if key in self._cache:
                del self._cache[key]
                count += 1

        # Also update the definition index for this file
        self._file_to_symbols.pop(file, None)
        self._file_to_callers.pop(file, None)

        # Rebuild definitions for chunks in this file
        for chunk in self._chunks:
            if not hasattr(chunk, "file") or chunk.file != file:
                continue
            name = getattr(chunk, "name", None)
            if name:
                self._file_to_symbols[file].add(name)

        if count > 0:
            logger.debug(f"Invalidated {count} reference map entries for file: {file}")
        return count

    def expand_neighborhood(
        self,
        symbol: str,
        max_depth: int = 3,
        direction: str = "both",
    ) -> dict[str, Any]:
        """Step 3: Background fill - expand neighborhood to deeper levels.

        Explores callers of callers and callees of callees up to max_depth.
        Returns the expanded neighborhood.

        Args:
            symbol: Symbol name to expand.
            max_depth: Maximum depth to explore (default: 3).
            direction: "both", "calls", or "called_by".

        Returns:
            Dict with expanded callers/callees and the depth reached.
        """
        cache_key = f"{symbol}:{direction}"

        # Ensure base neighborhood exists
        if cache_key not in self._cache:
            self.get_references(symbol, direction=direction)

        base = self._cache.get(cache_key)
        if base is None:
            return {"status": "ok", "symbol": symbol, "callers": [], "callees": [], "depth": 0}

        # Expand callers
        all_callers: list[dict[str, Any]] = list(base.callers)
        all_callees: list[dict[str, Any]] = list(base.callees)

        visited: set[str] = {symbol}
        current_callers = [c["symbol"] for c in base.callers if c.get("symbol")]
        current_callees = [c["symbol"] for c in base.callees if c.get("symbol")]

        for depth in range(2, max_depth + 1):
            next_callers: list[str] = []
            next_callees: list[str] = []

            if direction in ("both", "called_by"):
                for caller_sym in current_callers:
                    if caller_sym in visited:
                        continue
                    visited.add(caller_sym)
                    sub = self._build_neighborhood(caller_sym, "called_by", top_k=10)
                    if sub:
                        for ref in sub.callers:
                            if ref.get("symbol") not in visited:
                                all_callers.append(
                                    {
                                        **ref,
                                        "via": caller_sym,
                                        "depth": depth,
                                    }
                                )
                                next_callers.append(ref["symbol"])

            if direction in ("both", "calls"):
                for callee_sym in current_callees:
                    if callee_sym in visited:
                        continue
                    visited.add(callee_sym)
                    sub = self._build_neighborhood(callee_sym, "calls", top_k=10)
                    if sub:
                        for ref in sub.callees:
                            if ref.get("symbol") not in visited:
                                all_callees.append(
                                    {
                                        **ref,
                                        "via": callee_sym,
                                        "depth": depth,
                                    }
                                )
                                next_callees.append(ref["symbol"])

            current_callers = next_callers
            current_callees = next_callees

            if not current_callers and not current_callees:
                break

        # Update cache with expanded neighborhood
        base.callers = all_callers
        base.callees = all_callees
        base.depth = max_depth
        base.complete = True

        return {"status": "ok", **base.to_dict()}

    def _build_neighborhood(
        self,
        symbol: str,
        direction: str,
        top_k: int,
    ) -> Optional[ReferenceNeighborhood]:
        """Build the direct caller/callee neighborhood for a symbol."""
        # Find definition
        defs = self._definitions.get(symbol)
        if not defs:
            return None

        def_file, def_line, def_end, def_chunk = defs[0]

        callers: list[dict[str, Any]] = []
        callees: list[dict[str, Any]] = []

        if direction in ("both", "called_by"):
            callers = self._find_callers(symbol, top_k)

        if direction in ("both", "calls"):
            callees = self._find_callees(symbol, def_chunk, top_k)

        return ReferenceNeighborhood(
            symbol=symbol,
            file=def_file,
            line=def_line,
            callers=callers,
            callees=callees,
            direction=direction,
            depth=1,
            complete=False,
        )

    def _find_callers(self, symbol: str, top_k: int) -> list[dict[str, Any]]:
        """Find all chunks that call this symbol."""
        callers: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()

        for chunk in self._chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue

            # High confidence: symbols_called metadata
            symbols_called = getattr(chunk, "symbols_called", None) or []
            if symbol in symbols_called and getattr(chunk, "name", None) != symbol:
                # Find exact line
                line, context = self._find_call_line(chunk, symbol)
                key = (chunk.file, line)
                if key not in seen:
                    seen.add(key)
                    callers.append(
                        {
                            "file": chunk.file,
                            "line": line,
                            "symbol": getattr(chunk, "name", ""),
                            "context": context,
                            "confidence": "high",
                        }
                    )

            # Medium confidence: regex text search (only if we haven't hit top_k)
            if len(callers) >= top_k:
                break

        # Fallback: text search for any remaining
        if len(callers) < top_k:
            pattern = re.compile(rf"\b{re.escape(symbol)}\b")
            for chunk in self._chunks:
                if not hasattr(chunk, "text") or not chunk.text:
                    continue
                if getattr(chunk, "name", None) == symbol:
                    continue
                for match in pattern.finditer(chunk.text):
                    line = chunk.start_line + chunk.text[: match.start()].count("\n")
                    key = (chunk.file, line)
                    if key not in seen:
                        seen.add(key)
                        callers.append(
                            {
                                "file": chunk.file,
                                "line": line,
                                "symbol": getattr(chunk, "name", ""),
                                "context": match.group(0),
                                "confidence": "medium",
                            }
                        )
                        if len(callers) >= top_k:
                            break
                if len(callers) >= top_k:
                    break

        # Track file -> caller relationships for invalidation
        for c in callers:
            if c.get("file"):
                self._file_to_callers[c["file"]].add(symbol)

        return callers[:top_k]

    def _find_callees(self, symbol: str, chunk: Any, top_k: int) -> list[dict[str, Any]]:
        """Find all symbols called by this symbol's chunk."""
        callees: list[dict[str, Any]] = []

        symbols_called = getattr(chunk, "symbols_called", None) or []
        for called_sym in symbols_called:
            # Find where the called symbol is defined
            defs = self._definitions.get(called_sym)
            if defs:
                def_file, def_line, _, _ = defs[0]
                # Find the call line in the caller chunk
                line, context = self._find_call_line(chunk, called_sym)
                callees.append(
                    {
                        "file": def_file,
                        "line": def_line,
                        "symbol": called_sym,
                        "context": context,
                        "confidence": "high",
                    }
                )
            else:
                # Symbol not defined in this codebase (external/built-in)
                line, context = self._find_call_line(chunk, called_sym)
                callees.append(
                    {
                        "file": getattr(chunk, "file", ""),
                        "line": line,
                        "symbol": called_sym,
                        "context": context,
                        "confidence": "medium",
                    }
                )

        return callees[:top_k]

    @staticmethod
    def _find_call_line(chunk: Any, symbol: str) -> tuple[int, str]:
        """Find the line number and context where a symbol is called in a chunk."""
        if not hasattr(chunk, "text") or not chunk.text:
            return getattr(chunk, "start_line", 0), ""

        pattern = re.compile(rf"\b{re.escape(symbol)}\b")
        match = pattern.search(chunk.text)
        if match:
            line = chunk.start_line + chunk.text[: match.start()].count("\n")
            # Get the full line of code for context
            lines = chunk.text.split("\n")
            line_offset = chunk.text[: match.start()].count("\n")
            context = lines[line_offset].strip() if line_offset < len(lines) else match.group(0)
            return line, context
        return getattr(chunk, "start_line", 0), ""

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "cached_symbols": len(self._cache),
            "max_cache": self._max_cache,
            "files_tracked": len(self._file_to_symbols),
            "symbols_indexed": len(self._definitions),
        }

    def clear(self) -> None:
        """Clear all cached neighborhoods."""
        self._cache.clear()
