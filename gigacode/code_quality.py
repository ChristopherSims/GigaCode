"""Code quality tools: auto-formatting (Black/Ruff) and auto-linting (Ruff).

Layer 1 primitives: auto_format, auto_lint (independent, full-featured).
Layer 2 convenience wrappers: auto_polish (delegates to Layer 1).

All tools operate on entire directories by default (files=None), matching
native `black .` and `ruff check .` behavior.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "auto_format",
    "auto_lint",
    "auto_polish",
]


def _run_command(
    args: list[str],
    cwd: str | Path,
    timeout: int = 120,
) -> tuple[int, str, str]:
    """Run a subprocess command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", f"Command not found: {args[0]}"


def auto_format(
    work_dir: str | Path,
    files: Optional[list[str]] = None,
    formatter: str = "black",
    line_length: int = 88,
    skip_magic_trailing_comma: bool = False,
    dry_run: bool = True,
    exclude_patterns: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Format code using Black or ruff format.

    Args:
        work_dir: Directory to format (or root if files specified).
        files: Specific files to format. If None, format entire directory.
        formatter: "black" or "ruff.format".
        line_length: Maximum line length (default: 88).
        skip_magic_trailing_comma: Skip Black's magic trailing comma (default: False).
        dry_run: Preview only, don't modify files (default: True).
        exclude_patterns: Glob patterns to exclude.

    Returns:
        Dict with status, total_files, formatted_files, already_formatted,
        changes list, and summary.
    """
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        return {"status": "error", "message": f"Directory not found: {work_dir}"}

    targets = [str(Path(work_dir) / f) for f in files] if files else ["."]

    if formatter == "black":
        return _format_black(
            work_dir=work_dir,
            targets=targets,
            line_length=line_length,
            skip_magic_trailing_comma=skip_magic_trailing_comma,
            dry_run=dry_run,
            exclude_patterns=exclude_patterns,
        )
    elif formatter in ("ruff.format", "ruff"):
        return _format_ruff(
            work_dir=work_dir,
            targets=targets,
            line_length=line_length,
            dry_run=dry_run,
            exclude_patterns=exclude_patterns,
        )
    else:
        return {"status": "error", "message": f"Unknown formatter: {formatter}"}


def _format_black(
    work_dir: Path,
    targets: list[str],
    line_length: int,
    skip_magic_trailing_comma: bool,
    dry_run: bool,
    exclude_patterns: Optional[list[str]],
) -> dict[str, Any]:
    """Format with Black."""
    args = ["black", "--line-length", str(line_length)]

    if skip_magic_trailing_comma:
        args.append("--skip-magic-trailing-comma")

    if dry_run:
        args.append("--diff")

    if exclude_patterns:
        for pattern in exclude_patterns:
            args.extend(["--extend-exclude", pattern])

    args.extend(targets)

    returncode, stdout, stderr = _run_command(args, cwd=work_dir)

    if returncode == -1:
        return {"status": "error", "message": stderr}

    # Parse Black output
    changes: list[dict[str, Any]] = []
    formatted_count = 0
    already_formatted = 0

    if dry_run and stdout.strip():
        # In diff mode, Black shows diffs for files that need formatting
        # Split by file diff headers
        diff_sections = stdout.split("diff --git")
        for section in diff_sections:
            if not section.strip():
                continue
            # Extract file name from diff header
            lines = section.strip().split("\n")
            file_name = ""
            for line in lines:
                if line.startswith("--- a/"):
                    file_name = line[6:]
                    break
            if file_name:
                added = sum(
                    1
                    for diff_line in lines
                    if diff_line.startswith("+") and not diff_line.startswith("+++")
                )
                removed = sum(
                    1
                    for diff_line in lines
                    if diff_line.startswith("-") and not diff_line.startswith("---")
                )
                changes.append(
                    {
                        "file": file_name,
                        "added_lines": added,
                        "removed_lines": removed,
                        "diff": "diff --git" + section,
                    }
                )
                formatted_count += 1

    if not dry_run:
        # Black in apply mode: parse "reformatted" and "left unchanged"
        for line in (stdout + stderr).split("\n"):
            if "reformatted" in line:
                formatted_count = 1  # Summary line
            if "left unchanged" in line:
                already_formatted = 1

    summary = (
        f"Formatted {formatted_count} files" if formatted_count else "All files already formatted"
    )

    return {
        "status": "ok",
        "formatter": "black",
        "formatted_files": formatted_count,
        "already_formatted": already_formatted,
        "changes": changes if dry_run else [],
        "summary": summary,
    }


def _format_ruff(
    work_dir: Path,
    targets: list[str],
    line_length: int,
    dry_run: bool,
    exclude_patterns: Optional[list[str]],
) -> dict[str, Any]:
    """Format with ruff format."""
    args = ["ruff", "format", "--line-length", str(line_length)]

    if dry_run:
        args.append("--diff")

    if exclude_patterns:
        for pattern in exclude_patterns:
            args.extend(["--exclude", pattern])

    args.extend(targets)

    returncode, stdout, stderr = _run_command(args, cwd=work_dir)

    if returncode == -1:
        return {"status": "error", "message": stderr}

    changes: list[dict[str, Any]] = []
    formatted_count = 0

    if dry_run and stdout.strip():
        diff_sections = stdout.split("diff --git")
        for section in diff_sections:
            if not section.strip():
                continue
            lines = section.strip().split("\n")
            file_name = ""
            for line in lines:
                if line.startswith("--- a/"):
                    file_name = line[6:]
                    break
            if file_name:
                added = sum(
                    1
                    for diff_line in lines
                    if diff_line.startswith("+") and not diff_line.startswith("+++")
                )
                removed = sum(
                    1
                    for diff_line in lines
                    if diff_line.startswith("-") and not diff_line.startswith("---")
                )
                changes.append(
                    {
                        "file": file_name,
                        "added_lines": added,
                        "removed_lines": removed,
                        "diff": "diff --git" + section,
                    }
                )
                formatted_count += 1

    # ruff format exit code 0 = no changes, 1 = changes made (in non-diff mode)
    if not dry_run and returncode == 1:
        formatted_count = 1  # Summary

    summary = (
        f"Formatted {formatted_count} files" if formatted_count else "All files already formatted"
    )

    return {
        "status": "ok",
        "formatter": "ruff.format",
        "formatted_files": formatted_count,
        "already_formatted": 0,
        "changes": changes if dry_run else [],
        "summary": summary,
    }


def auto_lint(
    work_dir: str | Path,
    files: Optional[list[str]] = None,
    linter: str = "ruff",
    select: Optional[list[str]] = None,
    ignore: Optional[list[str]] = None,
    auto_fix: bool = False,
    dry_run: bool = True,
    exclude_patterns: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Lint code using Ruff.

    Args:
        work_dir: Directory to lint (or root if files specified).
        files: Specific files to lint. If None, lint entire directory.
        linter: Linter to use (currently only "ruff" supported).
        select: Rule categories to check (e.g., ["E", "F", "W"]).
        ignore: Rule codes to ignore (e.g., ["E501"]).
        auto_fix: Auto-fix fixable issues (default: False).
        dry_run: Preview only (default: True). Implies auto_fix=False.
        exclude_patterns: Glob patterns to exclude.

    Returns:
        Dict with status, files_with_issues, total_issues, issues list,
        fixed_count, unfixed_count, and by_rule breakdown.
    """
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        return {"status": "error", "message": f"Directory not found: {work_dir}"}

    targets = [str(Path(work_dir) / f) for f in files] if files else ["."]

    if linter != "ruff":
        return {
            "status": "error",
            "message": f"Unsupported linter: {linter}. Only 'ruff' is supported.",
        }

    args = ["ruff", "check"]

    if select:
        args.extend(["--select", ",".join(select)])
    if ignore:
        args.extend(["--ignore", ",".join(ignore)])
    if auto_fix and not dry_run:
        args.append("--fix")
    if exclude_patterns:
        for pattern in exclude_patterns:
            args.extend(["--exclude", pattern])

    args.extend(targets)

    returncode, stdout, stderr = _run_command(args, cwd=work_dir)

    if returncode == -1:
        return {"status": "error", "message": stderr}

    # Parse Ruff output
    issues: list[dict[str, Any]] = []
    by_rule: dict[str, dict[str, Any]] = {}

    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        # Ruff format: file:line:col: CODE message
        parts = line.split(":", 3)
        if len(parts) >= 4:
            file_path = parts[0].strip()
            try:
                line_no = int(parts[1].strip())
            except ValueError:
                continue
            col = parts[2].strip()
            rest = parts[3].strip()
            # Split code from message
            code_msg = rest.split(None, 1)
            code = code_msg[0] if code_msg else ""
            message = code_msg[1] if len(code_msg) > 1 else ""

            issue = {
                "file": file_path,
                "line": line_no,
                "code": code,
                "message": message,
                "fixed": False,
            }
            issues.append(issue)

            # Track by rule
            if code not in by_rule:
                by_rule[code] = {"count": 0, "files": []}
            by_rule[code]["count"] += 1
            if file_path not in by_rule[code]["files"]:
                by_rule[code]["files"].append(file_path)

    files_with_issues = len({issue["file"] for issue in issues})

    # Count fixed vs unfixed
    fixed_count = 0
    unfixed_count = len(issues)

    return {
        "status": "ok",
        "linter": "ruff",
        "files_with_issues": files_with_issues,
        "total_issues": len(issues),
        "issues": issues,
        "fixed_count": fixed_count,
        "unfixed_count": unfixed_count,
        "by_rule": by_rule,
        "auto_fixed_code_available": auto_fix and not dry_run,
    }


def auto_polish(
    work_dir: str | Path,
    files: Optional[list[str]] = None,
    format_with: str = "black",
    lint_with: str = "ruff",
    auto_fix_lints: bool = True,
    line_length: int = 88,
    ruff_select: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Convenience wrapper: format then lint in one call.

    Delegates to auto_format() then auto_lint() (Layer 2 wrapper over Layer 1 primitives).
    Format runs first so lint checks the formatted code.

    Args:
        work_dir: Directory to polish.
        files: Specific files. If None, polish entire directory.
        format_with: Formatter to use ("black" or "ruff.format").
        lint_with: Linter to use (currently only "ruff").
        auto_fix_lints: Auto-fix fixable lint issues (default: True).
        line_length: Maximum line length (default: 88).
        ruff_select: Ruff rule categories (e.g., ["E", "F", "W"]).
        exclude_patterns: Glob patterns to exclude.
        dry_run: Preview only (default: True).

    Returns:
        Dict with formatting and linting sub-results, plus ready_to_commit flag.
    """
    # Layer 2: delegate to Layer 1 primitives
    format_result = auto_format(
        work_dir=work_dir,
        files=files,
        formatter=format_with,
        line_length=line_length,
        dry_run=dry_run,
        exclude_patterns=exclude_patterns,
    )

    lint_result = auto_lint(
        work_dir=work_dir,
        files=files,
        linter=lint_with,
        select=ruff_select,
        auto_fix=auto_fix_lints and not dry_run,
        dry_run=dry_run,
        exclude_patterns=exclude_patterns,
    )

    # Determine commit readiness
    has_formatting_changes = format_result.get("formatted_files", 0) > 0
    has_lint_issues = lint_result.get("unfixed_count", 0) > 0

    ready_to_commit = not has_lint_issues

    summary_parts = []
    if has_formatting_changes:
        summary_parts.append(f"Formatted {format_result['formatted_files']} files")
    else:
        summary_parts.append("No formatting changes needed")
    if has_lint_issues:
        summary_parts.append(f"{lint_result['unfixed_count']} lint issues need manual fix")
    else:
        summary_parts.append("No lint issues")

    return {
        "status": "ok",
        "formatting": format_result,
        "linting": lint_result,
        "ready_to_commit": ready_to_commit,
        "summary": ", ".join(summary_parts),
    }
