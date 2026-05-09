"""Higher-level refactoring operations for AI agents.

Provides AST-aware transformations that are safer and more ergonomic
than line-by-line edits:
- Symbol renaming across files
- Import add/remove
- Unified diff application
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


__all__ = [
    "RefactorChange",
    "RefactorResult",
    "RefactorEngine",
    "rename_symbol",
    "add_import_statement",
    "apply_unified_diff",
    "edit_symbol",
    "add_parameter",
    "extract_method",
]


# Language-aware import syntax templates
_IMPORT_TEMPLATES: dict[str, dict[str, str]] = {
    "python": {
        "module": "import {module}",
        "symbols": "from {module} import {symbols}",
        "alias": "import {module} as {alias}",
    },
    "javascript": {
        "module": "import * as {alias} from '{module}';",
        "symbols": "import {{ {symbols} }} from '{module}';",
        "default": "import {alias} from '{module}';",
    },
    "typescript": {
        "module": "import * as {alias} from '{module}';",
        "symbols": "import {{ {symbols} }} from '{module}';",
        "default": "import {alias} from '{module}';",
    },
    "rust": {
        "module": "use {module};",
        "symbols": "use {module}::{{{symbols}}};",
    },
    "go": {
        "module": 'import "{module}"',
        "alias": '{alias} "{module}"',
    },
    "java": {
        "symbols": "import {module}.{symbols};",
        "module": "import {module}.*;",
    },
}


# Regex patterns for finding import sections by language
_IMPORT_SECTION_RE: dict[str, re.Pattern] = {
    "python": re.compile(r"^(from\s+\S+\s+import\s+.*|import\s+.*)$", re.MULTILINE),
    "javascript": re.compile(r"^import\s+.*\s+from\s+['\"].*['\"];?$", re.MULTILINE),
    "typescript": re.compile(r"^import\s+.*\s+from\s+['\"].*['\"];?$", re.MULTILINE),
    "rust": re.compile(r"^use\s+.*;", re.MULTILINE),
    "go": re.compile(r"^import\s+.*", re.MULTILINE),
    "java": re.compile(r"^import\s+.*;", re.MULTILINE),
}


@dataclass
class RefactorChange:
    """A single change in a refactoring operation."""

    file: str
    start_line: int
    end_line: int
    old_text: str
    new_text: str
    change_type: str  # "rename", "import_add", "import_remove", "diff"

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "change_type": self.change_type,
        }


@dataclass
class RefactorResult:
    """Result of a refactoring operation."""

    status: str  # "ok" | "error" | "preview"
    changed_files: list[str]
    changes: list[RefactorChange]
    message: str
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "changed_files": self.changed_files,
            "changes": [c.to_dict() for c in self.changes],
            "message": self.message,
            "dry_run": self.dry_run,
        }


class RefactorEngine:
    """Engine for higher-level refactoring operations."""

    def __init__(self, chunks: list[Any]) -> None:
        """Initialize with codebase chunks.

        Args:
            chunks: List of CodeChunk-like objects with .file, .text, .name, etc.
        """
        self.chunks = chunks
        self._file_chunks: dict[str, list[Any]] = {}
        for ch in chunks:
            self._file_chunks.setdefault(ch.file, []).append(ch)

    def rename_symbol(
        self,
        old_name: str,
        new_name: str,
        scope: str = "buffer",
        dry_run: bool = True,
    ) -> RefactorResult:
        """Rename a symbol across chunks.

        Args:
            old_name: Current symbol name.
            new_name: Desired symbol name.
            scope: "buffer" (all files), "file" (single file only — not yet supported).
            dry_run: If True, compute changes but do not modify chunks.

        Returns:
            RefactorResult with list of changes.
        """
        changes: list[RefactorChange] = []
        changed_files: set[str] = set()

        # Simple word-boundary replacement
        word_re = re.compile(rf"\b{re.escape(old_name)}\b")

        for chunk in self.chunks:
            if old_name == chunk.name:
                # This chunk defines the symbol
                new_text = word_re.sub(new_name, chunk.text)
                if new_text != chunk.text:
                    changes.append(
                        RefactorChange(
                            file=chunk.file,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            old_text=chunk.text,
                            new_text=new_text,
                            change_type="rename",
                        )
                    )
                    changed_files.add(chunk.file)
                    if not dry_run:
                        chunk.text = new_text
                        chunk.name = new_name
            else:
                # Check if this chunk references the symbol
                new_text = word_re.sub(new_name, chunk.text)
                if new_text != chunk.text:
                    changes.append(
                        RefactorChange(
                            file=chunk.file,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            old_text=chunk.text,
                            new_text=new_text,
                            change_type="rename",
                        )
                    )
                    changed_files.add(chunk.file)
                    if not dry_run:
                        chunk.text = new_text

        if not changes:
            return RefactorResult(
                status="warning",
                changed_files=[],
                changes=[],
                message=f"Symbol '{old_name}' not found in any chunk.",
                dry_run=dry_run,
            )

        return RefactorResult(
            status="ok",
            changed_files=sorted(changed_files),
            changes=changes,
            message=f"Renamed '{old_name}' -> '{new_name}' in {len(changed_files)} file(s), {len(changes)} chunk(s).",
            dry_run=dry_run,
        )

    def add_import(
        self,
        file: str,
        module: str,
        symbols: list[str] | None = None,
        language: str = "python",
        dry_run: bool = True,
    ) -> RefactorResult:
        """Add an import statement to a file.

        Args:
            file: Target file path.
            module: Module to import (e.g., "fastapi.security").
            symbols: Specific symbols to import. If None, imports entire module.
            language: Programming language for syntax selection.
            dry_run: If True, compute change but do not modify.

        Returns:
            RefactorResult with the import addition.
        """
        file_chunks = self._file_chunks.get(file, [])
        if not file_chunks:
            return RefactorResult(
                status="error",
                changed_files=[],
                changes=[],
                message=f"File '{file}' not found in buffer.",
                dry_run=dry_run,
            )

        # Sort chunks by line number
        file_chunks.sort(key=lambda c: c.start_line)

        # Find import section (first chunk with imports, or first chunk if none)
        import_chunk = file_chunks[0]
        for ch in file_chunks:
            if ch.imports:
                import_chunk = ch
                break

        # Build import statement
        templates = _IMPORT_TEMPLATES.get(language, _IMPORT_TEMPLATES["python"])
        if symbols:
            import_line = templates.get("symbols", "from {module} import {symbols}").format(
                module=module,
                symbols=", ".join(symbols),
            )
        else:
            import_line = templates.get("module", "import {module}").format(module=module)

        # Insert import at the beginning of the import chunk
        old_text = import_chunk.text
        lines = old_text.splitlines()
        # Find a good insertion point (after existing imports, or at top)
        insert_idx = 0
        import_re = _IMPORT_SECTION_RE.get(language)
        if import_re:
            for i, line in enumerate(lines):
                if import_re.match(line.strip()):
                    insert_idx = i + 1

        new_lines = lines[:insert_idx] + [import_line] + lines[insert_idx:]
        new_text = "\n".join(new_lines)

        change = RefactorChange(
            file=file,
            start_line=import_chunk.start_line,
            end_line=import_chunk.start_line,
            old_text=old_text,
            new_text=new_text,
            change_type="import_add",
        )

        if not dry_run:
            import_chunk.text = new_text
            if import_chunk.imports is None:
                import_chunk.imports = []
            import_chunk.imports.append(module)

        return RefactorResult(
            status="ok",
            changed_files=[file],
            changes=[change],
            message=f"Added import: {import_line}",
            dry_run=dry_run,
        )

    def remove_import(
        self,
        file: str,
        module: str,
        language: str = "python",
        dry_run: bool = True,
    ) -> RefactorResult:
        """Remove an import statement from a file.

        Args:
            file: Target file path.
            module: Module to remove.
            language: Programming language.
            dry_run: If True, compute change but do not modify.

        Returns:
            RefactorResult with the import removal.
        """
        file_chunks = self._file_chunks.get(file, [])
        if not file_chunks:
            return RefactorResult(
                status="error",
                changed_files=[],
                changes=[],
                message=f"File '{file}' not found in buffer.",
                dry_run=dry_run,
            )

        # Find chunk containing the import
        for chunk in file_chunks:
            # Simple line-by-line removal
            lines = chunk.text.splitlines()
            removed_lines: list[str] = []
            removed = False
            for line in lines:
                stripped = line.strip()
                # Match "import module" or "from module import ..."
                if (
                    stripped.startswith(f"import {module}")
                    or stripped.startswith(f"from {module} import")
                    or stripped.startswith(f"from {module}")
                ):
                    removed = True
                    continue
                removed_lines.append(line)

            if removed:
                new_text = "\n".join(removed_lines)
                change = RefactorChange(
                    file=file,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    old_text=chunk.text,
                    new_text=new_text,
                    change_type="import_remove",
                )
                if not dry_run:
                    chunk.text = new_text
                    if chunk.imports and module in chunk.imports:
                        chunk.imports.remove(module)
                return RefactorResult(
                    status="ok",
                    changed_files=[file],
                    changes=[change],
                    message=f"Removed import of '{module}' from {file}.",
                    dry_run=dry_run,
                )

        return RefactorResult(
            status="warning",
            changed_files=[],
            changes=[],
            message=f"Import of '{module}' not found in {file}.",
            dry_run=dry_run,
        )


def rename_symbol(
    chunks: list[Any],
    old_name: str,
    new_name: str,
    scope: str = "buffer",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Convenience function: rename symbol and return as dict."""
    engine = RefactorEngine(chunks)
    result = engine.rename_symbol(old_name, new_name, scope, dry_run)
    return result.to_dict()


def add_import_statement(
    chunks: list[Any],
    file: str,
    module: str,
    symbols: list[str] | None = None,
    language: str = "python",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Convenience function: add import and return as dict."""
    engine = RefactorEngine(chunks)
    result = engine.add_import(file, module, symbols, language, dry_run)
    return result.to_dict()


class DiffHunk:
    """A single hunk from a unified diff."""

    def __init__(
        self,
        old_start: int,
        old_count: int,
        new_start: int,
        new_count: int,
        lines: list[str],
    ) -> None:
        self.old_start = old_start
        self.old_count = old_count
        self.new_start = new_start
        self.new_count = new_count
        self.lines = lines

    def apply(self, old_lines: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Apply this hunk to old_lines.

        Returns:
            Tuple of (new_lines, context_before, context_after) for preview.

        Raises:
            ValueError: If hunk cannot be applied cleanly.
        """
        # Find the context lines in old_lines
        # old_start is 1-based, convert to 0-based index
        start_idx = self.old_start - 1

        # Validate we have enough lines
        if start_idx < 0 or start_idx > len(old_lines):
            raise ValueError(
                f"Hunk start line {self.old_start} out of range (file has {len(old_lines)} lines)"
            )

        # Build the new lines by processing hunk lines
        new_lines: list[str] = old_lines[:start_idx]
        removed: list[str] = []
        added: list[str] = []

        old_idx = start_idx
        for line in self.lines:
            if line.startswith(" "):
                # Context line - must match
                expected = line[1:]
                if old_idx >= len(old_lines) or old_lines[old_idx] != expected:
                    # Try fuzzy match: check nearby lines
                    found = False
                    for offset in range(-3, 4):
                        check_idx = old_idx + offset
                        if 0 <= check_idx < len(old_lines) and old_lines[check_idx] == expected:
                            # Adjust: add intervening lines to new_lines
                            if offset > 0:
                                new_lines.extend(old_lines[len(new_lines) : check_idx])
                            elif offset < 0:
                                # Backtrack
                                backtrack = -offset
                                new_lines = new_lines[:-backtrack]
                            old_idx = check_idx
                            found = True
                            break
                    if not found:
                        raise ValueError(
                            f"Context line mismatch at line {old_idx + 1}: "
                            f"expected '{expected[:50]}...', got '{old_lines[old_idx][:50]}...' "
                            if old_idx < len(old_lines)
                            else "(end of file)"
                        )
                new_lines.append(expected)
                old_idx += 1
            elif line.startswith("-"):
                # Removed line
                expected = line[1:]
                if old_idx >= len(old_lines) or old_lines[old_idx] != expected:
                    raise ValueError(
                        f"Removed line mismatch at line {old_idx + 1}: "
                        f"expected '{expected[:50]}...'"
                    )
                removed.append(expected)
                old_idx += 1
            elif line.startswith("+"):
                # Added line
                added.append(line[1:])
                new_lines.append(line[1:])
            elif line.startswith("\\"):
                # "\ No newline at end of file" - ignore
                pass

        # Append remaining lines after the hunk
        new_lines.extend(old_lines[old_idx:])

        return new_lines, removed, added


def _parse_unified_diff(diff_text: str) -> dict[str, list[DiffHunk]]:
    """Parse a unified diff into hunks grouped by file.

    Returns:
        Dict mapping file path to list of DiffHunk objects.
    """
    files: dict[str, list[DiffHunk]] = {}
    current_file: str | None = None
    current_hunk: DiffHunk | None = None
    hunk_lines: list[str] = []

    for line in diff_text.splitlines():
        if line.startswith("--- "):
            # Old file path
            old_path = line[4:].split("\t")[0]
            if old_path.startswith("a/"):
                old_path = old_path[2:]
            continue

        if line.startswith("+++ "):
            # New file path
            new_path = line[4:].split("\t")[0]
            if new_path.startswith("b/"):
                new_path = new_path[2:]
            current_file = new_path
            files.setdefault(current_file, [])
            continue

        if line.startswith("@@"):
            # Save previous hunk
            if current_file and current_hunk and hunk_lines:
                current_hunk.lines = hunk_lines
                files[current_file].append(current_hunk)
                hunk_lines = []

            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            match = re.match(
                r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@",
                line,
            )
            if match:
                old_start = int(match.group(1))
                old_count = int(match.group(2)) if match.group(2) else 1
                new_start = int(match.group(3))
                new_count = int(match.group(4)) if match.group(4) else 1
                current_hunk = DiffHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=[],
                )
            continue

        # Hunk line
        if (
            line.startswith(" ")
            or line.startswith("+")
            or line.startswith("-")
            or line.startswith("\\")
        ):
            hunk_lines.append(line)
        elif line.strip() == "" and current_hunk:
            # Empty lines in hunk context
            hunk_lines.append(" " + line)

    # Save final hunk
    if current_file and current_hunk and hunk_lines:
        current_hunk.lines = hunk_lines
        files[current_file].append(current_hunk)

    return files


class PatchApplier:
    """Apply unified diffs to buffer snapshots."""

    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self._file_chunks: dict[str, list[Any]] = {}
        for ch in chunks:
            self._file_chunks.setdefault(ch.file, []).append(ch)

    def apply_patch(
        self,
        diff_text: str,
        dry_run: bool = True,
    ) -> RefactorResult:
        """Apply a unified diff to the buffer.

        Args:
            diff_text: Unified diff string.
            dry_run: If True, compute changes but do not modify chunks.

        Returns:
            RefactorResult with applied changes.
        """
        try:
            file_hunks = _parse_unified_diff(diff_text)
        except (ValueError, TypeError) as e:
            return RefactorResult(
                status="error",
                changed_files=[],
                changes=[],
                message=f"Failed to parse diff: {e}",
                dry_run=dry_run,
            )

        if not file_hunks:
            return RefactorResult(
                status="warning",
                changed_files=[],
                changes=[],
                message="No hunks found in diff text.",
                dry_run=dry_run,
            )

        all_changes: list[RefactorChange] = []
        changed_files: set[str] = set()

        for file_path, hunks in file_hunks.items():
            # Get current file content from chunks
            file_chunks = self._file_chunks.get(file_path, [])
            if not file_chunks:
                return RefactorResult(
                    status="error",
                    changed_files=[],
                    changes=[],
                    message=f"File '{file_path}' not found in buffer.",
                    dry_run=dry_run,
                )

            # Sort chunks by line number and concatenate text
            file_chunks.sort(key=lambda c: c.start_line)
            old_lines: list[str] = []
            for ch in file_chunks:
                ch_lines = ch.text.splitlines()
                old_lines.extend(ch_lines)

            # Apply hunks in order
            new_lines = old_lines[:]
            for hunk in hunks:
                try:
                    new_lines, removed, added = hunk.apply(new_lines)
                except ValueError as e:
                    return RefactorResult(
                        status="error",
                        changed_files=[],
                        changes=[],
                        message=f"Failed to apply hunk for {file_path}: {e}",
                        dry_run=dry_run,
                    )

            # Only proceed if there are actual changes
            if new_lines != old_lines:
                new_text = "\n".join(new_lines)
                all_changes.append(
                    RefactorChange(
                        file=file_path,
                        start_line=1,
                        end_line=len(old_lines),
                        old_text="\n".join(old_lines),
                        new_text=new_text,
                        change_type="diff",
                    )
                )
                changed_files.add(file_path)

                if not dry_run:
                    # Update chunks: replace all chunks for this file with one big chunk
                    # (simplification: re-chunking happens on commit)
                    for ch in file_chunks:
                        ch.text = ""
                    if file_chunks:
                        file_chunks[0].text = new_text
                        file_chunks[0].start_line = 1
                        file_chunks[0].end_line = len(new_lines)
                        # Remove other chunks
                        for ch in file_chunks[1:]:
                            if ch in self.chunks:
                                self.chunks.remove(ch)

        if not changed_files:
            return RefactorResult(
                status="ok",
                changed_files=[],
                changes=[],
                message="Patch applied cleanly but made no changes.",
                dry_run=dry_run,
            )

        return RefactorResult(
            status="ok",
            changed_files=sorted(changed_files),
            changes=all_changes,
            message=f"Applied patch to {len(changed_files)} file(s) with {len(all_changes)} change(s).",
            dry_run=dry_run,
        )


def apply_unified_diff(
    chunks: list[Any],
    diff_text: str,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Apply a unified diff to chunks and return as dict."""
    applier = PatchApplier(chunks)
    result = applier.apply_patch(diff_text, dry_run)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Phase 2: AST-Aware Edit Operations
# ---------------------------------------------------------------------------


def _find_python_function_range(
    lines: list[str], symbol_name: str
) -> tuple[int, int, list[str]] | None:
    """Find (body_start_1based, body_end_exclusive_1based, decorator_lines).

    Returns None if symbol not found or is a one-liner.
    """
    # Pattern: optional decorators followed by def symbol_name(...):
    sig_re = re.compile(rf"^(?P<indent>\s*)(?P<async>async\s+)?def\s+{re.escape(symbol_name)}\s*\(")
    for i, line in enumerate(lines):
        m = sig_re.match(line)
        if not m:
            continue
        indent = m.group("indent")
        base_indent_len = len(indent)

        # Collect decorators immediately above
        decorators: list[str] = []
        d = i - 1
        while d >= 0 and lines[d].strip().startswith("@"):
            decorators.insert(0, lines[d])
            d -= 1

        # Find where body starts (line after signature, or one-liner)
        # Check if it's a one-liner
        if line.rstrip().endswith(":"):
            # Multi-line body
            body_start = i + 2  # 1-based index of first body line
            for j in range(i + 1, len(lines)):
                stripped = lines[j].rstrip()
                if not stripped:
                    continue
                line_indent_len = len(lines[j]) - len(lines[j].lstrip())
                if line_indent_len <= base_indent_len and not stripped.startswith("#"):
                    body_end = j  # exclusive, 1-based would be j+1, but we use 0-based here
                    return (body_start, body_end + 1, decorators)
            # Ran to EOF
            return (body_start, len(lines) + 1, decorators)
        else:
            # One-liner: e.g., "def foo(): return 1" — no separate body
            return None
    return None


def _replace_body(
    lines: list[str],
    body_start_1based: int,
    body_end_exclusive_1based: int,
    new_body_lines: list[str],
) -> list[str]:
    """Replace lines [body_start_1based-1 : body_end_exclusive_1based-1] with new_body_lines."""
    start_idx = body_start_1based - 1
    end_idx = body_end_exclusive_1based - 1
    return lines[:start_idx] + new_body_lines + lines[end_idx:]


def _add_parameter_to_signature(
    lines: list[str],
    symbol_name: str,
    param: str,
    default: str | None,
) -> list[str] | None:
    """Add a parameter to a function signature. Returns new lines or None if not found."""
    sig_re = re.compile(
        rf"^(?P<pre>\s*(?P<async>async\s+)?def\s+{re.escape(symbol_name)}\s*\()(?P<params>.*)(?P<post>\)\s*:.*)$"
    )
    for i, line in enumerate(lines):
        m = sig_re.match(line)
        if m:
            pre = m.group("pre")
            params = m.group("params").strip()
            post = m.group("post")
            new_param = f"{param}={default}" if default else param
            if params:
                new_params = f"{params}, {new_param}"
            else:
                new_params = new_param
            lines[i] = f"{pre}{new_params}{post}"
            return lines

    # Try multi-line signature: find the closing ):
    start_re = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(symbol_name)}\s*\(")
    for i, line in enumerate(lines):
        if start_re.match(line):
            # Scan forward for closing ) followed by :
            for j in range(i, len(lines)):
                stripped = lines[j].strip()
                if re.search(r"\)\s*:", stripped):
                    # Insert parameter before the closing )
                    # Find last ) before :
                    match = re.search(r"\)(\s*:\s*)$", stripped)
                    if match:
                        colon_part = match.group(1)
                        line_before = stripped[: match.start()]
                        new_param = f", {param}={default}" if default else f", {param}"
                        lines[j] = (
                            lines[j].rstrip()[: -len(stripped) + match.start()]
                            + new_param
                            + colon_part
                        )
                    else:
                        match = re.search(r"\)(\s*:\s*)", stripped)
                        if match:
                            colon_part = match.group(1)
                            lines[j] = (
                                stripped[: match.start()] + f", {param}={default}" + colon_part
                                if default
                                else stripped[: match.start()] + f", {param}" + colon_part
                            )
                    return lines
            break
    return None


class SymbolEditor:
    """AST-aware symbol editing utilities.

    Works on lists of file lines (not chunks) so that line numbers are stable.
    """

    @staticmethod
    def edit_symbol_body(
        lines: list[str],
        symbol_name: str,
        new_body_lines: list[str],
        language: str = "python",
    ) -> tuple[list[str], int, int] | None:
        """Replace the body of a symbol while preserving its signature.

        Returns:
            Tuple of (new_lines, body_start_1based, body_end_exclusive_1based)
            or None if symbol not found.
        """
        if language != "python":
            # Fallback: simple regex for other languages — best-effort
            return SymbolEditor._edit_symbol_body_fallback(lines, symbol_name, new_body_lines)

        rng = _find_python_function_range(lines, symbol_name)
        if rng is None:
            return None
        body_start, body_end, _ = rng
        new_lines = _replace_body(lines, body_start, body_end, new_body_lines)
        return (new_lines, body_start, body_end)

    @staticmethod
    def _edit_symbol_body_fallback(
        lines: list[str],
        symbol_name: str,
        new_body_lines: list[str],
    ) -> tuple[list[str], int, int] | None:
        """Best-effort body replacement for non-Python languages."""
        # Very naive: find "function name(...)" or "name(...)" then replace
        # everything until next line with same or less indentation
        sig_re = re.compile(
            rf"^(?P<indent>\s*)(?:function\s+)?{re.escape(symbol_name)}\s*[\(:]|\b{re.escape(symbol_name)}\s*[=:]\s*(?:function|=>|{{)"
        )
        for i, line in enumerate(lines):
            if sig_re.search(line):
                indent = len(line) - len(line.lstrip())
                body_start = i + 2
                for j in range(i + 1, len(lines)):
                    stripped = lines[j].rstrip()
                    if not stripped:
                        continue
                    line_indent = len(lines[j]) - len(lines[j].lstrip())
                    if (
                        line_indent <= indent
                        and not stripped.startswith("//")
                        and not stripped.startswith("/*")
                    ):
                        body_end = j + 1
                        new_lines = _replace_body(lines, body_start, body_end, new_body_lines)
                        return (new_lines, body_start, body_end)
                body_end = len(lines) + 1
                new_lines = _replace_body(lines, body_start, body_end, new_body_lines)
                return (new_lines, body_start, body_end)
        return None

    @staticmethod
    def add_parameter(
        lines: list[str],
        symbol_name: str,
        param: str,
        default: str | None = None,
        language: str = "python",
    ) -> tuple[list[str], int] | None:
        """Add a parameter to a function signature.

        Returns:
            Tuple of (new_lines, modified_line_1based) or None if not found.
        """
        if language != "python":
            return None  # Not yet supported for other languages
        result = _add_parameter_to_signature(lines, symbol_name, param, default)
        if result is None:
            return None
        # Find the modified line
        for i, (old, new) in enumerate(zip(lines, result, strict=False)):
            if old != new:
                return (result, i + 1)
        return (result, 1)

    @staticmethod
    def extract_method(
        lines: list[str],
        start_line_1based: int,
        end_line_1based: int,
        new_name: str,
        language: str = "python",
    ) -> tuple[list[str], int, int] | None:
        """Extract a range of lines into a new method.

        Returns:
            Tuple of (new_lines, inserted_at_line_1based, call_line_1based)
            or None on failure.
        """
        if language != "python":
            return None

        start_idx = start_line_1based - 1
        end_idx = end_line_1based - 1
        if start_idx < 0 or end_idx > len(lines) or start_idx >= end_idx:
            return None

        extracted = lines[start_idx:end_idx]
        if not extracted:
            return None

        # Determine indentation of extracted block
        indents = [len(line) - len(line.lstrip()) for line in extracted if line.strip()]
        if not indents:
            return None
        min_indent = min(indents)

        # Dedent extracted lines
        dedented = []
        for line in extracted:
            if line.strip():
                dedented.append(line[min_indent:])
            else:
                dedented.append(line)

        # Find the function/class that contains the extracted block
        # to determine where to insert the new method
        container_indent = 0
        for i in range(start_idx - 1, -1, -1):
            stripped = lines[i].strip()
            if (
                stripped.startswith("def ")
                or stripped.startswith("class ")
                or stripped.startswith("async def ")
            ):
                container_indent = len(lines[i]) - len(lines[i].lstrip())
                break

        # Build new function with same indentation as container
        prefix = " " * container_indent
        new_func_lines = [
            f"{prefix}def {new_name}():",
        ]
        for line in dedented:
            new_func_lines.append(prefix + "    " + line)
        new_func_lines.append("")  # blank line after function

        # Replace extracted block with a call to the new function
        call_line = f"{prefix}    {new_name}()"
        new_lines = lines[:start_idx] + [call_line] + lines[end_idx:]

        # Insert new function before the container function
        insert_at = start_line_1based  # 1-based line where extracted block was
        # Actually, we should insert before the container. Let's insert right before the start_line
        # but at container indentation level.
        # Better: insert right before the block we extracted (same position, but at container indent)
        new_lines = new_lines[: insert_at - 1] + new_func_lines + new_lines[insert_at - 1 :]

        return (new_lines, insert_at, start_line_1based + len(new_func_lines))


def edit_symbol(
    chunks: list[Any],
    symbol_name: str,
    new_body: str,
    language: str = "python",
    dry_run: bool = True,
) -> RefactorResult:
    """Replace the body of a symbol while preserving its signature.

    Returns a RefactorResult with a single RefactorChange.
    """
    # Find the chunk defining the symbol
    target_chunk = None
    for chunk in chunks:
        if chunk.name == symbol_name or symbol_name in (chunk.symbols_defined or []):
            target_chunk = chunk
            break

    if target_chunk is None:
        return RefactorResult(
            status="error",
            changed_files=[],
            changes=[],
            message=f"Symbol '{symbol_name}' not found in chunks.",
            dry_run=dry_run,
        )

    lines = target_chunk.text.splitlines()
    result = SymbolEditor.edit_symbol_body(lines, symbol_name, new_body.splitlines(), language)
    if result is None:
        return RefactorResult(
            status="error",
            changed_files=[],
            changes=[],
            message=f"Could not parse body of '{symbol_name}'. Is it a one-liner or unsupported language?",
            dry_run=dry_run,
        )

    new_lines, body_start, body_end = result
    new_text = "\n".join(new_lines)
    change = RefactorChange(
        file=target_chunk.file,
        start_line=body_start,
        end_line=body_end,
        old_text="\n".join(lines),
        new_text=new_text,
        change_type="edit_symbol",
    )

    if not dry_run:
        target_chunk.text = new_text

    return RefactorResult(
        status="ok",
        changed_files=[target_chunk.file],
        changes=[change],
        message=f"Edited body of '{symbol_name}' (lines {body_start}-{body_end}).",
        dry_run=dry_run,
    )


def add_parameter(
    chunks: list[Any],
    symbol_name: str,
    param: str,
    default: str | None = None,
    language: str = "python",
    dry_run: bool = True,
) -> RefactorResult:
    """Add a parameter to a function signature.

    Returns a RefactorResult with a single RefactorChange.
    """
    target_chunk = None
    for chunk in chunks:
        if chunk.name == symbol_name or symbol_name in (chunk.symbols_defined or []):
            target_chunk = chunk
            break

    if target_chunk is None:
        return RefactorResult(
            status="error",
            changed_files=[],
            changes=[],
            message=f"Symbol '{symbol_name}' not found in chunks.",
            dry_run=dry_run,
        )

    lines = target_chunk.text.splitlines()
    result = SymbolEditor.add_parameter(lines, symbol_name, param, default, language)
    if result is None:
        return RefactorResult(
            status="error",
            changed_files=[],
            changes=[],
            message=f"Could not find signature of '{symbol_name}'.",
            dry_run=dry_run,
        )

    new_lines, modified_line = result
    new_text = "\n".join(new_lines)
    change = RefactorChange(
        file=target_chunk.file,
        start_line=modified_line,
        end_line=modified_line,
        old_text="\n".join(lines),
        new_text=new_text,
        change_type="add_parameter",
    )

    if not dry_run:
        target_chunk.text = new_text

    return RefactorResult(
        status="ok",
        changed_files=[target_chunk.file],
        changes=[change],
        message=f"Added parameter '{param}' to '{symbol_name}' at line {modified_line}.",
        dry_run=dry_run,
    )


def extract_method(
    file_lines: list[str],
    start_line_1based: int,
    end_line_1based: int,
    new_name: str,
    language: str = "python",
    dry_run: bool = True,
) -> RefactorResult:
    """Extract a range of lines into a new method.

    Works on full file lines (not chunks) to preserve line numbering.
    """
    result = SymbolEditor.extract_method(
        file_lines, start_line_1based, end_line_1based, new_name, language
    )
    if result is None:
        return RefactorResult(
            status="error",
            changed_files=[],
            changes=[],
            message="Extract method failed: invalid range or unsupported language.",
            dry_run=dry_run,
        )

    new_lines, insert_at, call_line = result
    # Two changes: insertion of new function + replacement of extracted block
    changes = [
        RefactorChange(
            file="<input>",
            start_line=start_line_1based,
            end_line=end_line_1based,
            old_text="\n".join(file_lines),
            new_text="\n".join(new_lines),
            change_type="extract_method",
        ),
    ]

    return RefactorResult(
        status="ok",
        changed_files=["<input>"],
        changes=changes,
        message=f"Extracted lines {start_line_1based}-{end_line_1based} into '{new_name}' (inserted at line {insert_at}, call at line {call_line}).",
        dry_run=dry_run,
    )
