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


def apply_unified_diff(
    chunks: list[Any],
    diff_text: str,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Apply a unified diff to chunks.

    This is a simplified implementation that parses the diff and applies
    line-by-line changes. For production use, consider using `patch` library.

    Args:
        chunks: List of chunks to modify.
        diff_text: Unified diff string (like `git diff` output).
        dry_run: If True, compute changes but do not modify.

    Returns:
        RefactorResult as dict.
    """
    # TODO: Implement proper unified diff parsing
    # For now, return a placeholder
    return RefactorResult(
        status="warning",
        changed_files=[],
        changes=[],
        message="Unified diff application is not yet fully implemented. Use write_code for now.",
        dry_run=dry_run,
    ).to_dict()
