"""AST-based code chunking.

Chunks source code at function/class/method boundaries using tree-sitter.
Falls back to sliding-window chunking for unsupported languages or parse failures.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Node types that define logical boundaries (language-agnostic fallback)
_DEFINITION_TYPES = {
    "function_definition",
    "function_item",          # Rust
    "function_declaration",   # C/Go/JS/TS
    "class_definition",
    "class_item",             # Rust
    "class_declaration",      # JS/TS/Java
    "method_definition",
    "method_item",            # Rust
    "method_declaration",     # Java/C#/Go
    "struct_item",            # Rust
    "impl_item",              # Rust
    "interface_definition",   # TypeScript / Java
    "interface_declaration",  # Java
    "enum_item",              # Rust
    "enum_declaration",       # Java/TS
    "trait_item",             # Rust
    "constructor_declaration",# Java/C#
    "arrow_function",         # JS/TS
    "async_function_definition", # Python
    "module",                 # Elixir
}

# Language-specific definition node types (takes precedence over global set)
_DEFINITION_TYPES_BY_LANGUAGE: dict[str, set[str]] = {
    "python": {"function_definition", "async_function_definition", "class_definition"},
    "javascript": {"function_declaration", "method_definition", "arrow_function", "class_declaration"},
    "typescript": {"function_declaration", "method_definition", "arrow_function", "class_declaration", "interface_declaration", "enum_declaration"},
    "java": {"method_declaration", "constructor_declaration", "class_declaration", "interface_declaration", "enum_declaration"},
    "c": {"function_definition"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item", "class_item"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "ruby": {"method", "singleton_method", "class", "module"},
    "php": {"function_definition", "method_declaration", "class_declaration"},
    "csharp": {"method_declaration", "constructor_declaration", "class_declaration", "interface_declaration", "enum_declaration"},
    "swift": {"function_declaration", "class_declaration", "struct_declaration", "enum_declaration"},
    "kotlin": {"function_declaration", "class_declaration", "object_declaration"},
    "scala": {"function_definition", "class_definition", "trait_definition"},
    "lua": {"function_declaration"},
    "elixir": {"function", "module"},
    "bash": {"function_definition"},
}


@dataclass(slots=True)
class CodeChunk:
    """A semantic chunk of source code."""

    id: int
    file: str
    start_line: int          # 1-based, inclusive
    end_line: int            # 1-based, inclusive
    type: str                # "function", "class", "method", "orphan", "sliding"
    name: str | None
    text: str
    embedding: list[float] | None = None

    def dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# Tree-sitter helpers
# ---------------------------------------------------------------------------

_grammar_cache: dict[str, Any] = {}


def _get_grammar(language_hint: str | None) -> Any | None:
    """Import and cache a tree-sitter Language object."""
    if not language_hint:
        return None
    hint = language_hint.lower()
    if hint in _grammar_cache:
        return _grammar_cache[hint]

    # Map common aliases to package names
    aliases: dict[str, str] = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "tsx": "typescript",
        "rs": "rust",
        "cpp": "cpp",
        "c++": "cpp",
        "c": "c",
        "go": "go",
        "java": "java",
        "rb": "ruby",
    }
    lang_name = aliases.get(hint, hint)
    pkg = f"tree_sitter_{lang_name}"

    try:
        mod = __import__(pkg)
        lang = getattr(mod, "language", None)
        if lang is None:
            return None
        from tree_sitter import Language
        grammar = Language(lang() if callable(lang) else lang)
        _grammar_cache[hint] = grammar
        return grammar
    except ModuleNotFoundError:
        logger.debug(
            "Grammar package %s not installed. Install with: pip install %s",
            pkg,
            pkg.replace("_", "-")
        )
        return None
    except Exception as exc:
        logger.debug("Could not load grammar %s: %s", pkg, exc)
        return None


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _extract_name(node, source_bytes: bytes) -> str | None:
    """Try to find an identifier child and return its text."""
    for child in node.children:
        if child.type in ("identifier", "name"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return None


def _chunk_with_tree_sitter(text: str, language_hint: str | None, file_hint: str = "") -> list[CodeChunk] | None:
    """Parse text with tree-sitter and return definition-based chunks.

    Returns None if parsing fails or no grammar is available.
    """
    grammar = _get_grammar(language_hint)
    if grammar is None:
        return None

    try:
        from tree_sitter import Parser
        parser = Parser(grammar)
    except Exception as exc:
        logger.debug("Parser init failed: %s", exc)
        return None

    source_bytes = text.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    lines = text.splitlines()
    if not lines:
        return []

    # Determine which node types to treat as boundaries for this language
    normalized_hint = (language_hint or "").lower()
    definition_types = _DEFINITION_TYPES_BY_LANGUAGE.get(normalized_hint, _DEFINITION_TYPES)

    # Collect definition nodes with their line ranges
    defs: list[tuple[int, int, str, str | None]] = []
    for node in _walk(root):
        if node.type in definition_types:
            start = node.start_point[0] + 1  # 0-index -> 1-index
            end = node.end_point[0] + 1
            name = _extract_name(node, source_bytes)
            if "class" in node.type or node.type in ("struct_item", "struct_specifier", "object_declaration"):
                chunk_type = "class"
            elif node.type in ("method_declaration", "method_definition", "method_item"):
                chunk_type = "method"
            elif "function" in node.type or node.type in ("arrow_function", "function_item", "function_declaration"):
                chunk_type = "function"
            elif node.type == "module":
                chunk_type = "module"
            else:
                chunk_type = "function"
            defs.append((start, end, chunk_type, name))

    if not defs:
        return None

    defs.sort(key=lambda x: x[0])

    # Build chunks: definitions + orphan gaps between them
    chunks: list[CodeChunk] = []
    next_id = 0
    prev_end = 0

    for start, end, ctype, name in defs:
        if start > prev_end + 1:
            # Orphan chunk
            orphan_lines = lines[prev_end:start - 1]
            orphan_text = "\n".join(orphan_lines)
            if orphan_text.strip():
                chunks.append(CodeChunk(
                    id=next_id,
                    file=file_hint,
                    start_line=prev_end + 1,
                    end_line=start - 1,
                    type="orphan",
                    name=None,
                    text=orphan_text,
                ))
                next_id += 1
        chunk_lines = lines[start - 1:end]
        chunk_text = "\n".join(chunk_lines)
        chunks.append(CodeChunk(
            id=next_id,
            file=file_hint,
            start_line=start,
            end_line=end,
            type=ctype,
            name=name,
            text=chunk_text,
        ))
        next_id += 1
        prev_end = end

    # Trailing orphan
    if prev_end < len(lines):
        orphan_lines = lines[prev_end:]
        orphan_text = "\n".join(orphan_lines)
        if orphan_text.strip():
            chunks.append(CodeChunk(
                id=next_id,
                file=file_hint,
                start_line=prev_end + 1,
                end_line=len(lines),
                type="orphan",
                name=None,
                text=orphan_text,
            ))

    return chunks


def _chunk_sliding_window(text: str, file_hint: str = "", window: int = 30, overlap: int = 5) -> list[CodeChunk]:
    """Fallback chunker: sliding windows of *window* lines with *overlap*."""
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[CodeChunk] = []
    step = window - overlap
    idx = 0
    cid = 0
    for start in range(0, len(lines), step):
        end = min(start + window, len(lines))
        chunk_lines = lines[start:end]
        chunk_text = "\n".join(chunk_lines)
        chunks.append(CodeChunk(
            id=cid,
            file=file_hint,
            start_line=start + 1,
            end_line=end,
            type="sliding",
            name=None,
            text=chunk_text,
        ))
        cid += 1
        if end == len(lines):
            break
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(text: str, language_hint: str | None = None, filename_hint: str = "") -> list[CodeChunk]:
    """Split source text into semantic chunks.

    Tries tree-sitter AST chunking first, falls back to sliding windows.
    """
    chunks = _chunk_with_tree_sitter(text, language_hint, filename_hint)
    if chunks is not None:
        logger.debug("tree-sitter chunking produced %d chunks for %s", len(chunks), filename_hint)
        return chunks
    logger.debug("Falling back to sliding-window chunking for %s", filename_hint)
    return _chunk_sliding_window(text, filename_hint)


def chunk_file(path: str | Path, language_hint: str | None = None) -> list[CodeChunk]:
    """Chunk a file on disk."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    if language_hint is None:
        from src.language_detect import detect_language
        language_hint = detect_language(p)
    return chunk_text(text, language_hint=language_hint, filename_hint=str(p))
