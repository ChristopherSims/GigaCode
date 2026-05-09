"""Automated test-after-edit feedback loop for AI agents.

Maps dirty files / modified symbols to impacted test files and runs them,
returning pass/fail with traces and token counts.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gigacode.chunker import CodeChunk

logger = logging.getLogger(__name__)

__all__ = [
    "TestResult",
    "TestRunSummary",
    "TestRunner",
    "run_impacted_tests",
]


# Test file name patterns by language (same as context_assembler.py)
_TEST_FILE_PATTERNS: dict[str, list[re.Pattern]] = {
    "python": [re.compile(r"^test_.*\.py$"), re.compile(r".*_test\.py$")],
    "javascript": [re.compile(r".*\.(test|spec)\.(js|ts|jsx|tsx)$")],
    "typescript": [re.compile(r".*\.(test|spec)\.(js|ts|jsx|tsx)$")],
    "rust": [re.compile(r".*_test\.rs$")],
    "go": [re.compile(r".*_test\.go$")],
    "java": [re.compile(r".*Test\.java$")],
    "cpp": [re.compile(r".*_test\.(cpp|cc|cxx)$")],
}


def _is_test_file(filename: str, language: str = "python") -> bool:
    """Check if a filename looks like a test file."""
    basename = Path(filename).name
    patterns = _TEST_FILE_PATTERNS.get(language, [])
    return any(p.match(basename) for p in patterns)


@dataclass
class TestResult:
    """A single test case result."""

    name: str
    file: str
    line: int
    outcome: str  # "passed" | "failed" | "skipped" | "error"
    duration_ms: float
    traceback: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file,
            "line": self.line,
            "outcome": self.outcome,
            "duration_ms": self.duration_ms,
            "traceback": self.traceback,
            "message": self.message,
        }


@dataclass
class TestRunSummary:
    """Summary of a test run."""

    status: str  # "ok" | "error" | "no_tests" | "skipped"
    passed: int
    failed: int
    skipped: int
    total: int
    duration_sec: float
    test_file_count: int
    tests: list[TestResult] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    token_estimate: int = 0
    impacted_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "total": self.total,
            "duration_sec": self.duration_sec,
            "test_file_count": self.test_file_count,
            "tests": [t.to_dict() for t in self.tests],
            "stdout": self.stdout,
            "stderr": self.stderr,
            "token_estimate": self.token_estimate,
            "impacted_files": self.impacted_files,
        }


class TestRunner:
    """Find and run tests impacted by code changes."""

    def __init__(
        self,
        chunks: list[CodeChunk],
        root_dir: str | Path,
        language: str = "python",
    ) -> None:
        self.chunks = chunks
        self.root_dir = Path(root_dir)
        self.language = language
        self._test_chunks = [ch for ch in chunks if _is_test_file(ch.file, language)]

    # ------------------------------------------------------------------
    # Impact detection
    # ------------------------------------------------------------------

    def find_impacted_tests(
        self,
        modified_files: list[str],
        modified_symbols: list[str] | None = None,
        top_k: int = 20,
    ) -> list[CodeChunk]:
        """Find test chunks likely impacted by recent changes.

        Scoring:
          +50  test file references a modified file basename
          +40  test file references a modified symbol name
          +20  test imports from a modified file
          +10  semantic overlap (test chunk text contains symbol)
        """
        modified_symbols = modified_symbols or []
        scored: list[tuple[CodeChunk, int]] = []

        for tchunk in self._test_chunks:
            score = 0
            text = tchunk.text

            # File name references
            for mf in modified_files:
                basename = Path(mf).stem
                if basename in text:
                    score += 50
                if mf in text:
                    score += 50

            # Symbol name references
            for sym in modified_symbols:
                if sym in text:
                    score += 40

            # Import references
            imports = tchunk.imports or []
            for mf in modified_files:
                basename = Path(mf).stem
                if any(basename in imp for imp in imports):
                    score += 20

            # Symbols_called overlap
            calls = tchunk.symbols_called or []
            for sym in modified_symbols:
                if sym in calls:
                    score += 30

            if score > 0:
                scored.append((tchunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        seen_files: set[str] = set()
        unique: list[CodeChunk] = []
        for ch, _ in scored:
            if ch.file not in seen_files:
                seen_files.add(ch.file)
                unique.append(ch)
            if len(unique) >= top_k:
                break
        return unique

    def extract_symbols_from_diff(
        self,
        old_lines: list[str],
        new_lines: list[str],
    ) -> list[str]:
        """Extract likely modified symbol names from a line diff.

        Heuristic: look for added function/class/method definitions.
        """
        symbols: set[str] = set()
        # Unified diff style: added lines in new_lines not in old_lines
        old_set = set(old_lines)
        for line in new_lines:
            if line in old_set:
                continue
            stripped = line.strip()
            # Python
            m = re.match(r"^(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", stripped)
            if m:
                symbols.add(m.group(1))
            m = re.match(r"^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]", stripped)
            if m:
                symbols.add(m.group(1))
            # JavaScript / TypeScript
            m = re.match(r"^(?:async\s+)?(?:function\s+)?([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(", stripped)
            if m:
                symbols.add(m.group(1))
            m = re.match(
                r"^const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:async\s+)?\(?.*\)?\s*=>", stripped
            )
            if m:
                symbols.add(m.group(1))
            # Java
            m = re.match(
                r"^(?:public|private|protected|static|\s)*\s*(?:<[^>]+>\s+)?[\w<>\[\]]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
                stripped,
            )
            if m:
                symbols.add(m.group(1))
        return sorted(symbols)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_tests(
        self,
        test_paths: list[str],
        timeout: int = 120,
        extra_args: list[str] | None = None,
    ) -> TestRunSummary:
        """Run pytest on specific test files and return parsed results.

        Args:
            test_paths: Relative paths to test files.
            timeout: Max seconds to wait.
            extra_args: Additional pytest arguments.

        Returns:
            TestRunSummary with parsed results.
        """
        if not test_paths:
            return TestRunSummary(
                status="no_tests",
                passed=0,
                failed=0,
                skipped=0,
                total=0,
                duration_sec=0.0,
                test_file_count=0,
                token_estimate=50,
                impacted_files=[],
            )

        # Build pytest command with JSON report if available
        cmd = [
            "python",
            "-m",
            "pytest",
            *test_paths,
            "-v",
            "--tb=short",
            "-q",
            "--no-header",
        ]
        if extra_args:
            cmd.extend(extra_args)

        # Try JSON report (pytest-json-report plugin)
        json_output: dict[str, Any] | None = None
        json_path = self.root_dir / ".pytest_test_runner_output.json"
        json_cmd = cmd + [
            "--json-report",
            "--json-report-file",
            str(json_path),
        ]

        t0 = __import__("time").perf_counter()
        try:
            proc = subprocess.run(
                json_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.root_dir,
            )
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_output = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass
                finally:
                    try:
                        json_path.unlink()
                    except OSError:
                        pass
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            # pytest not available — fall back to plain run
            logger.warning(f"JSON report pytest failed ({e}), falling back to plain.")
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.root_dir,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e2:
                return TestRunSummary(
                    status="skipped",
                    passed=0,
                    failed=0,
                    skipped=0,
                    total=0,
                    duration_sec=0.0,
                    test_file_count=len(test_paths),
                    stdout="",
                    stderr=f"Could not run pytest: {e2}",
                    token_estimate=100,
                    impacted_files=test_paths,
                )

        duration = __import__("time").perf_counter() - t0

        # Parse results
        tests: list[TestResult] = []
        passed = failed = skipped = 0

        if json_output:
            for test in json_output.get("tests", []):
                outcome = test.get("outcome", "unknown")
                tr = TestResult(
                    name=test.get("nodeid", "unknown"),
                    file=test.get("location", ["", 0])[0],
                    line=test.get("location", ["", 0])[1],
                    outcome=outcome,
                    duration_ms=test.get("duration", 0.0) * 1000,
                    traceback=(
                        "\n".join(test.get("call", {}).get("longrepr", "").splitlines())
                        if isinstance(test.get("call", {}).get("longrepr"), str)
                        else ""
                    ),
                    message=(
                        test.get("call", {}).get("crash", {}).get("message", "")
                        if test.get("call", {}).get("crash")
                        else ""
                    ),
                )
                tests.append(tr)
                if outcome == "passed":
                    passed += 1
                elif outcome == "failed":
                    failed += 1
                elif outcome == "skipped":
                    skipped += 1

            summary = json_output.get("summary", {})
            total = summary.get("total", len(tests))
        else:
            # Plain-text parse
            total, passed, failed, skipped = self._parse_plain_output(
                proc.stdout + proc.stderr, tests
            )

        token_estimate = len(proc.stdout + proc.stderr) // 4

        return TestRunSummary(
            status="ok",
            passed=passed,
            failed=failed,
            skipped=skipped,
            total=total or len(tests),
            duration_sec=duration,
            test_file_count=len(test_paths),
            tests=tests,
            stdout=proc.stdout[:4000],
            stderr=proc.stderr[:2000],
            token_estimate=token_estimate,
            impacted_files=test_paths,
        )

    @staticmethod
    def _parse_plain_output(
        output: str,
        tests_out: list[TestResult],
    ) -> tuple[int, int, int, int]:
        """Parse pytest plain-text output for counts and failures.

        Returns (total, passed, failed, skipped).
        """
        total = passed = failed = skipped = 0
        # Summary line: "3 passed, 1 failed, 2 skipped in 0.05s"
        for line in output.splitlines():
            lower = line.lower()
            if "passed" in lower or "failed" in lower or "skipped" in lower or "error" in lower:
                # Try to extract counts
                for token in lower.split(","):
                    token = token.strip()
                    for keyword, dest in [
                        ("passed", "passed"),
                        ("failed", "failed"),
                        ("skipped", "skipped"),
                        ("error", "failed"),
                    ]:
                        if keyword in token:
                            try:
                                count = int(token.split()[0])
                                if dest == "passed":
                                    passed += count
                                elif dest == "failed":
                                    failed += count
                                elif dest == "skipped":
                                    skipped += count
                            except ValueError:
                                pass
                # Total
                try:
                    parts = lower.split("in")
                    if len(parts) >= 2 and "s" in parts[-1]:
                        total = passed + failed + skipped
                except Exception:
                    pass

        # Extract failures for tests_out
        current_failure: dict[str, Any] = {}
        in_traceback = False
        for line in output.splitlines():
            if line.startswith("FAILED ") or line.startswith("ERROR "):
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[1]
                    file_part = ""
                    line_num = 0
                    # Try to find file:line in traceback below
                    current_failure = {
                        "name": name,
                        "file": "",
                        "line": 0,
                        "traceback": [],
                    }
                    in_traceback = True
            elif in_traceback:
                if line.strip().startswith("File "):
                    # File "path/to/file.py", line 42
                    m = re.search(r'File "([^"]+)", line (\d+)', line)
                    if m:
                        current_failure["file"] = m.group(1)
                        current_failure["line"] = int(m.group(2))
                elif line.strip().startswith("_ ") or line.strip() == "":
                    # End of traceback section
                    if current_failure:
                        tests_out.append(
                            TestResult(
                                name=current_failure.get("name", ""),
                                file=current_failure.get("file", ""),
                                line=current_failure.get("line", 0),
                                outcome="failed",
                                duration_ms=0.0,
                                traceback="\n".join(current_failure.get("traceback", [])),
                            )
                        )
                    current_failure = {}
                    in_traceback = False
                else:
                    current_failure.setdefault("traceback", []).append(line)

        return total, passed, failed, skipped

    # ------------------------------------------------------------------
    # High-level: impacted + run
    # ------------------------------------------------------------------

    def run_impacted(
        self,
        modified_files: list[str],
        modified_symbols: list[str] | None = None,
        top_k: int = 20,
        timeout: int = 120,
    ) -> TestRunSummary:
        """Find impacted tests and run them.

        Returns:
            TestRunSummary with impacted file list + results.
        """
        impacted = self.find_impacted_tests(modified_files, modified_symbols, top_k)
        if not impacted:
            return TestRunSummary(
                status="no_tests",
                passed=0,
                failed=0,
                skipped=0,
                total=0,
                duration_sec=0.0,
                test_file_count=0,
                token_estimate=50,
                impacted_files=[],
            )

        test_paths = [ch.file for ch in impacted]
        return self.run_tests(test_paths, timeout=timeout)


def run_impacted_tests(
    chunks: list[CodeChunk],
    root_dir: str | Path,
    modified_files: list[str],
    modified_symbols: list[str] | None = None,
    language: str = "python",
    top_k: int = 20,
    timeout: int = 120,
) -> dict[str, Any]:
    """Convenience function: find and run impacted tests, return dict."""
    runner = TestRunner(chunks, root_dir, language)
    summary = runner.run_impacted(modified_files, modified_symbols, top_k, timeout)
    return summary.to_dict()
