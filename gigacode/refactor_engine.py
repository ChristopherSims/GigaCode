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
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


__all__ = [
    "RefactorChange",
    "RefactorResult",
    "RefactorEngine",
    "rename_symbol",
    "add_import_statement",
    "apply_unified_diff",
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
                    changes.append(RefactorChange(
                        file=chunk.file,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        old_text=chunk.text,
                        new_text=new_text,
                        change_type="rename",
                    ))
                    changed_files.add(chunk.file)
                    if not dry_run:
                        chunk.text = new_text
                        chunk.name = new_name
            else:
                # Check if this chunk references the symbol
                new_text = word_re.sub(new_name, chunk.text)
                if new_text != chunk.text:
                    changes.append(RefactorChange(
                        file=chunk.file,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        old_text=chunk.text,
                        new_text=new_text,
                        change_type="rename",
                    ))
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
                                new_lines.extend(old_lines[len(new_lines):check_idx])
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
        if line.startswith(" ") or line.startswith("+") or line.startswith("-") or line.startswith("\\"):
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
                all_changes.append(RefactorChange(
                    file=file_path,
                    start_line=1,
                    end_line=len(old_lines),
                    old_text="\n".join(old_lines),
                    new_text=new_text,
                    change_type="diff",
                ))
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
