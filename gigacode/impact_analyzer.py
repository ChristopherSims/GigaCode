"""Dependency risk analysis on edit — Phase 4.

Before commit, analyze how many upstream callers and dependent files
are affected by dirty changes.  Computes a risk score from call-chain
depth, file count, and test-coverage presence.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gigacode.chunker import CodeChunk
from gigacode.dependency_graph import DependencyGraph

logger = logging.getLogger(__name__)

__all__ = [
    "ImpactLevel",
    "SymbolImpact",
    "FileImpact",
    "ImpactAnalysis",
    "ImpactAnalyzer",
    "analyze_impact",
]

# Files that are considered "critical" — high blast-radius
_CRITICAL_FILE_PATTERNS = [
    "main.py",
    "app.py",
    "api.py",
    "server.py",
    "__init__.py",
    "config.py",
    "settings.py",
    "middleware.py",
    "models.py",
    "schema.py",
    "urls.py",
    "router.py",
    "views.py",
    "controllers.py",
]


def _is_critical_file(path: str) -> bool:
    name = Path(path).name.lower()
    return name in _CRITICAL_FILE_PATTERNS or "config" in name or "main" in name


def _is_test_file(path: str) -> bool:
    name = Path(path).name.lower()
    return name.startswith("test_") or name.endswith("_test.py") or ".test." in name


class ImpactLevel:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class SymbolImpact:
    """Impact assessment for a single modified symbol."""

    symbol: str
    modified_file: str
    callers: list[dict[str, Any]] = field(default_factory=list)
    max_call_depth: int = 0
    unique_caller_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FileImpact:
    """Impact assessment for a single dirty file."""

    file: str
    modified_symbols: list[str] = field(default_factory=list)
    incoming_deps: list[str] = field(default_factory=list)
    incoming_dep_count: int = 0
    is_critical: bool = False
    has_tests: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImpactAnalysis:
    """Complete pre-commit impact analysis."""

    status: str  # "ok" | "error"
    risk_level: str
    risk_score: float  # 0.0 – 1.0
    dirty_files: list[str] = field(default_factory=list)
    modified_symbols: list[str] = field(default_factory=list)
    symbol_impacts: list[SymbolImpact] = field(default_factory=list)
    file_impacts: list[FileImpact] = field(default_factory=list)
    total_impacted_files: int = 0
    max_call_depth: int = 0
    test_coverage_present: bool = False
    recommendations: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImpactAnalyzer:
    """Analyze dependency blast-radius of pending edits."""

    def __init__(
        self,
        chunks: list[CodeChunk],
        dependency_graph: DependencyGraph | None = None,
    ) -> None:
        self.chunks = chunks
        self.graph = dependency_graph or DependencyGraph(chunks)

        # Index test chunks by file for quick lookup
        self._test_chunks = [ch for ch in chunks if _is_test_file(ch.file)]

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        dirty_files: list[str],
        modified_symbols: list[str] | None = None,
        max_depth: int = 6,
    ) -> ImpactAnalysis:
        """Analyze impact of dirty files + modified symbols.

        Args:
            dirty_files: List of relative file paths with uncommitted changes.
            modified_symbols: Optional list of symbol names that were changed.
            max_depth: Max call-chain depth to traverse upward.

        Returns:
            ImpactAnalysis with risk scoring and recommendations.
        """
        modified_symbols = modified_symbols or []

        # --- Symbol-level impact ---
        symbol_impacts: list[SymbolImpact] = []
        all_caller_entries: list[dict[str, Any]] = []

        for sym in modified_symbols:
            # Find which file defines this symbol
            def_file = self._find_defining_file(sym)
            callers = self._gather_callers_recursive(sym, max_depth=max_depth)
            unique_files = sorted({c["file"] for c in callers})
            max_depth_for_sym = max(
                (c.get("depth", 0) for c in callers),
                default=0,
            )

            si = SymbolImpact(
                symbol=sym,
                modified_file=def_file or "unknown",
                callers=callers,
                max_call_depth=max_depth_for_sym,
                unique_caller_files=unique_files,
            )
            symbol_impacts.append(si)
            all_caller_entries.extend(callers)

        # --- File-level impact ---
        file_impacts: list[FileImpact] = []
        for df in dirty_files:
            incoming = self.graph.get_dependencies(df, direction="incoming")
            # Also look at callers of symbols defined in this file
            file_symbols = self._symbols_defined_in_file(df)
            file_callers: list[str] = []
            for fsym in file_symbols:
                c = self.graph.get_callers(fsym)
                file_callers.extend([entry["file"] for entry in c])

            all_deps = sorted(set(incoming + file_callers))
            has_tests = any(
                t.file in all_deps or _file_references_symbol(t, file_symbols)
                for t in self._test_chunks
            )

            fi = FileImpact(
                file=df,
                modified_symbols=file_symbols,
                incoming_deps=all_deps,
                incoming_dep_count=len(all_deps),
                is_critical=_is_critical_file(df),
                has_tests=has_tests,
            )
            file_impacts.append(fi)

        # --- Aggregate metrics ---
        all_impacted_files = sorted(
            {c["file"] for c in all_caller_entries}
            | {dep for fi in file_impacts for dep in fi.incoming_deps}
        )

        max_call_depth = max(
            (si.max_call_depth for si in symbol_impacts),
            default=0,
        )

        test_coverage_present = any(fi.has_tests for fi in file_impacts)

        risk_score = self._compute_risk_score(
            dirty_files=dirty_files,
            impacted_files=all_impacted_files,
            max_call_depth=max_call_depth,
            symbol_impacts=symbol_impacts,
            file_impacts=file_impacts,
            test_coverage_present=test_coverage_present,
        )

        risk_level = self._score_to_level(risk_score)

        recommendations = self._make_recommendations(
            risk_level,
            risk_score,
            max_call_depth,
            all_impacted_files,
            test_coverage_present,
            file_impacts,
        )

        return ImpactAnalysis(
            status="ok",
            risk_level=risk_level,
            risk_score=risk_score,
            dirty_files=dirty_files,
            modified_symbols=modified_symbols,
            symbol_impacts=symbol_impacts,
            file_impacts=file_impacts,
            total_impacted_files=len(all_impacted_files),
            max_call_depth=max_call_depth,
            test_coverage_present=test_coverage_present,
            recommendations=recommendations,
            message=f"Risk: {risk_level} ({risk_score:.2f}) — {len(all_impacted_files)} impacted file(s)",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_defining_file(self, symbol: str) -> str | None:
        for ch in self.chunks:
            if ch.name == symbol or symbol in (ch.symbols_defined or []):
                return ch.file
        return None

    def _symbols_defined_in_file(self, file: str) -> list[str]:
        syms: set[str] = set()
        for ch in self.chunks:
            if ch.file == file:
                if ch.name:
                    syms.add(ch.name)
                syms.update(ch.symbols_defined or [])
        return sorted(syms)

    def _gather_callers_recursive(
        self,
        symbol: str,
        max_depth: int = 6,
    ) -> list[dict[str, Any]]:
        """Gather all callers of `symbol` up to `max_depth` levels.

        Returns list of dicts with keys: file, line, symbol, depth.
        """
        results: list[dict[str, Any]] = []
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(symbol, 0)]

        while queue:
            current_sym, depth = queue.pop(0)
            if current_sym in visited or depth >= max_depth:
                continue
            visited.add(current_sym)

            for caller in self.graph.get_callers(current_sym):
                caller_sym = caller.get("symbol")
                entry = {
                    "file": caller.get("file", ""),
                    "line": caller.get("line", 0),
                    "symbol": caller_sym,
                    "depth": depth + 1,
                }
                # Deduplicate by (file, line, symbol)
                key = (entry["file"], entry["line"], entry["symbol"])
                if not any((r["file"], r["line"], r["symbol"]) == key for r in results):
                    results.append(entry)

                if caller_sym and caller_sym not in visited:
                    queue.append((caller_sym, depth + 1))

        return results

    @staticmethod
    def _compute_risk_score(
        dirty_files: list[str],
        impacted_files: list[str],
        max_call_depth: int,
        symbol_impacts: list[SymbolImpact],
        file_impacts: list[FileImpact],
        test_coverage_present: bool,
    ) -> float:
        """Compute a 0.0 – 1.0 risk score.

        Factors:
          - impacted file count (0–0.3)
          - max call depth (0–0.25)
          - critical files touched (0–0.25)
          - absence of test coverage (0–0.2)
        """
        score = 0.0

        # Impacted files factor
        n_impacted = len(impacted_files)
        if n_impacted == 0:
            score += 0.0
        elif n_impacted <= 3:
            score += 0.05
        elif n_impacted <= 8:
            score += 0.15
        else:
            score += 0.30

        # Call depth factor
        if max_call_depth >= 5:
            score += 0.25
        elif max_call_depth >= 3:
            score += 0.15
        elif max_call_depth >= 1:
            score += 0.05

        # Critical files factor
        critical_count = sum(1 for fi in file_impacts if fi.is_critical)
        if critical_count > 0:
            score += min(critical_count * 0.10, 0.25)

        # Test coverage penalty
        if not test_coverage_present:
            score += 0.20

        return min(score, 1.0)

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score < 0.25:
            return ImpactLevel.LOW
        elif score < 0.60:
            return ImpactLevel.MEDIUM
        return ImpactLevel.HIGH

    @staticmethod
    def _make_recommendations(
        risk_level: str,
        risk_score: float,
        max_call_depth: int,
        impacted_files: list[str],
        test_coverage_present: bool,
        file_impacts: list[FileImpact],
    ) -> list[str]:
        recs: list[str] = []

        if risk_level == ImpactLevel.HIGH:
            recs.append(
                f"HIGH risk ({risk_score:.2f}): {len(impacted_files)} files impacted. "
                "Consider breaking this into smaller commits or running full test suite first."
            )
        elif risk_level == ImpactLevel.MEDIUM:
            recs.append(
                f"MEDIUM risk ({risk_score:.2f}): {len(impacted_files)} files may be affected. "
                "Review impacted callers before committing."
            )

        if max_call_depth >= 4:
            recs.append(
                f"Deep call chain detected (depth {max_call_depth}). "
                "Changes may propagate through multiple layers."
            )

        critical = [fi.file for fi in file_impacts if fi.is_critical]
        if critical:
            recs.append(
                f"Critical file(s) modified: {', '.join(critical)}. "
                "These have high blast-radius — double-check interfaces."
            )

        if not test_coverage_present:
            recs.append(
                "No test coverage detected for impacted code. "
                "Consider adding tests before committing."
            )

        if not recs:
            recs.append("Low risk — safe to commit.")

        return recs


def _file_references_symbol(test_chunk: CodeChunk, symbols: list[str]) -> bool:
    text = test_chunk.text
    return any(sym in text for sym in symbols)


def analyze_impact(
    chunks: list[CodeChunk],
    dirty_files: list[str],
    modified_symbols: list[str] | None = None,
    max_depth: int = 6,
) -> dict[str, Any]:
    """Convenience function: analyze impact and return as dict."""
    analyzer = ImpactAnalyzer(chunks)
    result = analyzer.analyze(dirty_files, modified_symbols, max_depth)
    return result.to_dict()
