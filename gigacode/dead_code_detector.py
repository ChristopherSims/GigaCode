"""Dead code and unused symbol detection."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["DeadCodeDetector", "DeadSymbol"]

@dataclass
class DeadSymbol:
    symbol: str
    file: str
    line: int
    type: str
    confidence: str  # "high", "medium", "low"
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class DeadCodeDetector:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self._all_defined: dict[str, list[Any]] = {}
        self._all_called: set[str] = set()
        self._build_index()

    def _build_index(self) -> None:
        for ch in self.chunks:
            for sym in (ch.symbols_defined or []):
                self._all_defined.setdefault(sym, []).append(ch)
            for sym in (ch.symbols_called or []):
                self._all_called.add(sym)

    def find_dead_code(self, min_confidence: str = "medium") -> list[DeadSymbol]:
        confidence_levels = {"high": 3, "medium": 2, "low": 1}
        min_level = confidence_levels.get(min_confidence, 2)

        results: list[DeadSymbol] = []

        for symbol, chunks_list in self._all_defined.items():
            for ch in chunks_list:
                # Check if symbol is called anywhere
                call_count = sum(1 for other_ch in self.chunks if symbol in (other_ch.symbols_called or []))

                if call_count == 0:
                    # High confidence: never called
                    results.append(DeadSymbol(
                        symbol=symbol,
                        file=ch.file,
                        line=ch.start_line,
                        type=ch.type,
                        confidence="high",
                        reason="Defined but never called anywhere in the codebase.",
                    ))
                elif call_count <= 2:
                    # Low confidence: rarely called
                    results.append(DeadSymbol(
                        symbol=symbol,
                        file=ch.file,
                        line=ch.start_line,
                        type=ch.type,
                        confidence="low",
                        reason=f"Only called {call_count} times. May be dead code or test-only.",
                    ))

        # Filter by minimum confidence
        return [r for r in results if confidence_levels.get(r.confidence, 0) >= min_level]

    def find_unused_imports(self) -> list[dict[str, Any]]:
        """Find imports that are not referenced in the same file."""
        results: list[dict[str, Any]] = []

        for ch in self.chunks:
            for imp in (ch.imports or []):
                # Check if import is used (symbol called matches import)
                module_name = imp.split(".")[-1]
                used = False
                for other_ch in self.chunks:
                    if other_ch.file != ch.file:
                        continue
                    for call in (other_ch.symbols_called or []):
                        if call.startswith(module_name) or call == module_name:
                            used = True
                            break
                if not used:
                    results.append({
                        "file": ch.file,
                        "import": imp,
                        "line": ch.start_line,
                        "confidence": "medium",
                        "reason": f"Import '{imp}' may be unused in this file.",
                    })

        return results

def find_dead_code(chunks: list[Any], min_confidence: str = "medium") -> list[dict[str, Any]]:
    detector = DeadCodeDetector(chunks)
    return [s.to_dict() for s in detector.find_dead_code(min_confidence)]
