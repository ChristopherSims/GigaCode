"""Execution sandbox for running arbitrary code snippets against a codebase.

Provides a restricted environment for AI agents to:
- Run Python/JS snippets with stdout/stderr capture
- Inject the buffer root into sys.path for codebase imports
- Enforce timeouts and AST-based security whitelisting
"""

from __future__ import annotations

import ast
import hashlib
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "ExecutionResult",
    "SandboxExecutor",
    "execute_in_context",
]


# ---------------------------------------------------------------------------
# Security policy
# ---------------------------------------------------------------------------

# Banned built-in / import names (exact match)
_BANNED_IMPORTS: set[str] = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "ctypes",
    "multiprocessing",
    "threading",
    "concurrent",
    "pickle",
    "marshal",
    "compileall",
    "py_compile",
    "importlib",
    "pkgutil",
    "site",
    "builtins",
    "__builtin__",
}

# Banned AST node types (all removed in Python 3; kept as a hook for future bans)
_BANNED_AST_NODES: tuple[type[ast.AST], ...] = ()

# Banned attribute chains (e.g., os.system, subprocess.run)
_BANNED_ATTRIBUTE_PREFIXES: tuple[str, ...] = (
    "os.",
    "sys.",
    "subprocess.",
    "shutil.",
    "pathlib.",
    "socket.",
    "urllib.",
    "builtins.__import__",
    "__import__",
)

# Allowed built-in functions (subset)
_ALLOWED_BUILTINS: set[str] = {
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "chr",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hasattr",
    "hash",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    "vars",
    "help",
    "input",  # reads from stdin — acceptable in sandbox context
}


class SecurityError(ValueError):
    """Raised when submitted code violates the sandbox security policy."""

    pass


# ---------------------------------------------------------------------------
# AST scanner
# ---------------------------------------------------------------------------


class _SecurityScanner(ast.NodeVisitor):
    """Walks an AST and collects security violations."""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            mod = alias.name.split(".")[0]
            if mod in _BANNED_IMPORTS:
                self.violations.append(f"Banned import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module:
            mod = node.module.split(".")[0]
            if mod in _BANNED_IMPORTS:
                self.violations.append(f"Banned import from: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # Detect open(), eval(), exec(), compile(), __import__()
        if isinstance(node.func, ast.Name):
            if node.func.id in ("open", "eval", "exec", "compile", "__import__"):
                self.violations.append(f"Banned function call: {node.func.id}()")
        elif isinstance(node.func, ast.Attribute):
            chain = _attr_chain(node.func)
            for banned in _BANNED_ATTRIBUTE_PREFIXES:
                if chain.startswith(banned):
                    self.violations.append(f"Banned attribute call: {chain}")
                    break
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        # Detect dunder attribute access like __class__, __bases__, etc.
        if isinstance(node.attr, str) and node.attr.startswith("__") and node.attr.endswith("__"):
            # Allow a few safe dunders
            if node.attr not in ("__name__", "__file__", "__doc__", "__class__"):
                self.violations.append(f"Suspicious dunder access: {node.attr}")
        self.generic_visit(node)


def _attr_chain(node: ast.Attribute | ast.Name) -> str:
    """Return the dotted attribute chain, e.g. 'os.system'."""
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    parts.reverse()
    return ".".join(parts)


def _scan_security(code: str) -> list[str]:
    """Parse code and return list of security violations."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    scanner = _SecurityScanner()
    scanner.visit(tree)
    return scanner.violations


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Result of a sandbox execution."""

    status: str  # "ok" | "error" | "timeout" | "security_violation"
    returncode: int
    stdout: str
    stderr: str
    execution_time_sec: float
    language: str
    violations: list[str] = None  # type: ignore[assignment]
    truncated: bool = False

    def __post_init__(self) -> None:
        if self.violations is None:
            self.violations = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time_sec": self.execution_time_sec,
            "language": self.language,
            "violations": self.violations,
            "truncated": self.truncated,
        }


# ---------------------------------------------------------------------------
# Sandbox executor
# ---------------------------------------------------------------------------


class SandboxExecutor:
    """Execute code snippets in a restricted subprocess sandbox."""

    def __init__(
        self,
        root_dir: Path,
        language: str = "python",
        max_output_bytes: int = 64_000,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.language = language
        self.max_output_bytes = max_output_bytes

    # ------------------------------------------------------------------
    # Python execution
    # ------------------------------------------------------------------

    def _run_python(
        self,
        code: str,
        timeout: int = 30,
    ) -> ExecutionResult:
        """Run Python code in a temporary file with restricted builtins."""
        # Security scan
        violations = _scan_security(code)
        if violations:
            return ExecutionResult(
                status="security_violation",
                returncode=-1,
                stdout="",
                stderr="; ".join(violations),
                execution_time_sec=0.0,
                language="python",
                violations=violations,
            )

        # Build wrapper script that injects root into sys.path and restricts builtins
        wrapper = self._build_python_wrapper(code)

        # Write to temp file inside root_dir (so relative imports work)
        script_path = self.root_dir / ".sandbox_exec.py"
        try:
            script_path.write_text(wrapper, encoding="utf-8")
        except OSError as e:
            return ExecutionResult(
                status="error",
                returncode=-1,
                stdout="",
                stderr=f"Failed to write sandbox script: {e}",
                execution_time_sec=0.0,
                language="python",
            )

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                timeout=timeout,
                cwd=self.root_dir,
            )
            elapsed = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            _safe_unlink(script_path)
            return ExecutionResult(
                status="timeout",
                returncode=-1,
                stdout="",
                stderr=f"Execution exceeded {timeout} seconds",
                execution_time_sec=elapsed,
                language="python",
            )
        except FileNotFoundError:
            _safe_unlink(script_path)
            return ExecutionResult(
                status="error",
                returncode=-1,
                stdout="",
                stderr="Python interpreter not found",
                execution_time_sec=0.0,
                language="python",
            )

        _safe_unlink(script_path)

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        truncated = False
        total_len = len(stdout) + len(stderr)
        if total_len > self.max_output_bytes:
            # Truncate proportionally
            keep = self.max_output_bytes // 2
            stdout = stdout[:keep] + "\n... [truncated]"
            stderr = stderr[:keep] + "\n... [truncated]"
            truncated = True

        return ExecutionResult(
            status="ok" if proc.returncode == 0 else "error",
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            execution_time_sec=elapsed,
            language="python",
            truncated=truncated,
        )

    def _build_python_wrapper(self, user_code: str) -> str:
        """Build a wrapper script that injects sys.path and restricts builtins."""
        # Hash user code to avoid cache issues
        code_hash = hashlib.sha256(user_code.encode("utf-8")).hexdigest()[:8]

        wrapper_lines = [
            "# Auto-generated sandbox wrapper",
            f"# hash={code_hash}",
            "import sys",
            f"sys.path.insert(0, {repr(str(self.root_dir))})",
            "",
            "# Restrict builtins",
            "import builtins",
            "_original_builtins = builtins.__dict__.copy()",
            "",
        ]

        # Rebuild builtins dict with only allowed names
        wrapper_lines.append("_allowed = {")
        for name in sorted(_ALLOWED_BUILTINS):
            wrapper_lines.append(f"    {repr(name)}: _original_builtins.get({repr(name)}),")
        wrapper_lines.append("}")
        wrapper_lines.append("builtins.__dict__.clear()")
        wrapper_lines.append("builtins.__dict__.update(_allowed)")
        wrapper_lines.append("")

        # Ban file open attempts at runtime (last-resort guard)
        wrapper_lines.extend(
            [
                "def _banned(*args, **kwargs):",
                "    raise RuntimeError('file I/O is disabled in the sandbox')",
                "builtins.__dict__['open'] = _banned",
                "",
            ]
        )

        # User code
        wrapper_lines.append("# --- user code ---")
        wrapper_lines.append(user_code)
        wrapper_lines.append("# --- end user code ---")

        return "\n".join(wrapper_lines)

    # ------------------------------------------------------------------
    # JavaScript execution (best-effort via node if available)
    # ------------------------------------------------------------------

    def _run_javascript(
        self,
        code: str,
        timeout: int = 30,
    ) -> ExecutionResult:
        """Run JavaScript via Node.js if available."""
        # Simple text-based security scan
        lower = code.lower()
        violations: list[str] = []
        banned_js = [
            "require('fs')",
            'require("fs")',
            "require('child_process')",
            'require("child_process")',
            "require('os')",
            'require("os")',
            "require('net')",
            'require("net")',
            "eval(",
            "new function",
            "document.",
            "window.",
            "fetch(",
            "xmlhttprequest",
        ]
        for b in banned_js:
            if b in code:
                violations.append(f"Banned JS pattern: {b}")

        if violations:
            return ExecutionResult(
                status="security_violation",
                returncode=-1,
                stdout="",
                stderr="; ".join(violations),
                execution_time_sec=0.0,
                language="javascript",
                violations=violations,
            )

        script_path = self.root_dir / ".sandbox_exec.js"
        try:
            script_path.write_text(code, encoding="utf-8")
        except OSError as e:
            return ExecutionResult(
                status="error",
                returncode=-1,
                stdout="",
                stderr=f"Failed to write sandbox script: {e}",
                execution_time_sec=0.0,
                language="javascript",
            )

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                ["node", str(script_path)],
                capture_output=True,
                timeout=timeout,
                cwd=self.root_dir,
            )
            elapsed = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            _safe_unlink(script_path)
            return ExecutionResult(
                status="timeout",
                returncode=-1,
                stdout="",
                stderr=f"Execution exceeded {timeout} seconds",
                execution_time_sec=elapsed,
                language="javascript",
            )
        except FileNotFoundError:
            _safe_unlink(script_path)
            return ExecutionResult(
                status="error",
                returncode=-1,
                stdout="",
                stderr="Node.js not found",
                execution_time_sec=0.0,
                language="javascript",
            )

        _safe_unlink(script_path)

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        truncated = False
        if len(stdout) + len(stderr) > self.max_output_bytes:
            keep = self.max_output_bytes // 2
            stdout = stdout[:keep] + "\n... [truncated]"
            stderr = stderr[:keep] + "\n... [truncated]"
            truncated = True

        return ExecutionResult(
            status="ok" if proc.returncode == 0 else "error",
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            execution_time_sec=elapsed,
            language="javascript",
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        code: str,
        language: str | None = None,
        timeout: int = 30,
    ) -> ExecutionResult:
        """Execute code with security restrictions and output capture.

        Args:
            code: Source code string.
            language: "python" or "javascript". Defaults to executor language.
            timeout: Max seconds.

        Returns:
            ExecutionResult.
        """
        lang = (language or self.language).lower()
        if lang == "python":
            return self._run_python(code, timeout)
        elif lang in ("javascript", "js", "node"):
            return self._run_javascript(code, timeout)
        else:
            return ExecutionResult(
                status="error",
                returncode=-1,
                stdout="",
                stderr=f"Unsupported sandbox language: {lang}",
                execution_time_sec=0.0,
                language=lang,
            )


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def execute_in_context(
    root_dir: Path,
    code: str,
    language: str = "python",
    timeout: int = 30,
) -> dict[str, Any]:
    """Convenience function: execute code in sandbox and return dict."""
    executor = SandboxExecutor(root_dir, language)
    result = executor.execute(code, timeout=timeout)
    return result.to_dict()
