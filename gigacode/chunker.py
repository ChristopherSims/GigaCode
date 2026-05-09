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


__all__ = [
    "CodeChunk",
    "chunk_text",
    "chunk_file",
]

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
    "lambda",                 # Python
    "lambda_expression",      # Java
    "proc",                   # Ruby (lambda/proc)
    "preprocessing_directive", # C/C++ macros
}

# Language-specific definition node types (takes precedence over global set)
_DEFINITION_TYPES_BY_LANGUAGE: dict[str, set[str]] = {
    "python": {"function_definition", "async_function_definition", "class_definition", "lambda"},
    "javascript": {"function_declaration", "method_definition", "arrow_function", "class_declaration"},
    "typescript": {"function_declaration", "method_definition", "arrow_function", "class_declaration", "interface_declaration", "enum_declaration"},
    "java": {"method_declaration", "constructor_declaration", "class_declaration", "interface_declaration", "enum_declaration", "lambda_expression"},
    "c": {"function_definition", "preprocessing_directive"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier", "preprocessing_directive"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item", "class_item"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "ruby": {"method", "singleton_method", "class", "module", "proc"},
    "php": {"function_definition", "method_declaration", "class_declaration"},
    "csharp": {"method_declaration", "constructor_declaration", "class_declaration", "interface_declaration", "enum_declaration"},
    "swift": {"function_declaration", "class_declaration", "struct_declaration", "enum_declaration"},
    "kotlin": {"function_declaration", "class_declaration", "object_declaration"},
    "scala": {"function_definition", "class_definition", "trait_definition"},
    "lua": {"function_declaration"},
    "elixir": {"function", "module"},
    "bash": {"function_definition"},
}


@dataclass
class CodeChunk:
    """A semantic chunk of source code."""

    id: int
    file: str
    start_line: int          # 1-based, inclusive
    end_line: int            # 1-based, inclusive
    type: str                # "function", "class", "method", "lambda", "macro", "nested_function", "orphan", "sliding"
    name: str | None
    text: str
    embedding: list[float] | None = None
    # Symbol metadata for context assembly and cross-reference
    symbols_defined: list[str] | None = None   # e.g., ["UserRepository", "validate_email"]
    symbols_called: list[str] | None = None    # e.g., ["hash_password", "db.query"]
    imports: list[str] | None = None           # e.g., ["fastapi.security", "jwt"]

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
    except (ImportError, AttributeError, TypeError) as exc:
        logger.debug("Could not load grammar %s: %s", pkg, exc)
        return None


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _extract_name(node, source_bytes: bytes) -> str | None:
    """Try to find an identifier child and return its text.
    
    Handles various node types: identifier, name, function_name, etc.
    For macros, extracts the macro name after #define.
    """
    # Handle preprocessing directives (macros) - extract name after #define
    if node.type == "preprocessing_directive":
        text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        # Extract macro name: #define MACRO_NAME ...
        import re
        match = re.match(r'#\s*define\s+(\w+)', text)
        if match:
            return match.group(1)
        return None
    
    # For regular definitions, find identifier/name child
    for child in node.children:
        if child.type in ("identifier", "name", "function_name", "property_identifier"):
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    
    return None


def _is_nested_definition(node) -> bool:
    """Check if a function/lambda is nested inside another function/lambda/class.
    
    Returns True if the node's parent is a function, lambda, class, or method.
    """
    parent = node.parent
    while parent is not None:
        # Check if parent is a definition that would contain this node
        if parent.type in (
            "function_definition", "async_function_definition",
            "function_declaration", "method_declaration", "method_definition",
            "lambda", "lambda_expression", "arrow_function",
            "class_definition", "class_declaration", "class_specifier",
            "function_item",  # Rust
        ):
            return True
        parent = parent.parent
    return False


def _chunk_with_tree_sitter(text: str, language_hint: str | None, file_hint: str = "") -> list[CodeChunk] | None:
    """Parse text with tree-sitter and return definition-based chunks.

    Returns None if parsing fails or no grammar is available.
    """
    grammar = _get_grammar(language_hint)
    if grammar is None:
        logger.debug(
            "No tree-sitter grammar available for language_hint=%r; will fall back to sliding window.",
            language_hint
        )
        return None

    try:
        from tree_sitter import Parser
        parser = Parser(grammar)
    except (ImportError, ModuleNotFoundError, TypeError) as exc:
        logger.debug(
            "Parser initialization failed for language_hint=%r (file=%s): %s; will fall back to sliding window.",
            language_hint, file_hint, exc
        )
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
            
            # Classify chunk type
            if node.type == "preprocessing_directive":
                # C/C++ macro
                chunk_type = "macro"
            elif node.type in ("class_definition", "class_declaration", "class_item", "class_specifier", "object_declaration"):
                chunk_type = "class"
            elif node.type in ("method_declaration", "method_definition", "method_item"):
                chunk_type = "method"
            elif node.type in ("lambda", "lambda_expression"):
                chunk_type = "lambda"
            elif node.type == "proc":
                # Ruby lambda/proc
                chunk_type = "lambda"
            elif "function" in node.type or node.type in ("arrow_function", "function_item", "function_declaration"):
                # Check if nested
                if _is_nested_definition(node):
                    chunk_type = "nested_function"
                else:
                    chunk_type = "function"
            elif node.type == "module":
                chunk_type = "module"
            else:
                # Default fallback
                if "class" in node.type or node.type in ("struct_item", "struct_specifier", "object_declaration"):
                    chunk_type = "class"
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

    # Determine language for import/call extraction
    normalized_hint = (language_hint or "").lower()
    from gigacode.context_assembler import _extract_calls, _extract_imports

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
                    symbols_defined=[],
                    symbols_called=_extract_calls(orphan_text, normalized_hint),
                    imports=_extract_imports(orphan_text, normalized_hint),
                ))
                next_id += 1
        chunk_lines = lines[start - 1:end]
        chunk_text = "\n".join(chunk_lines)
        # Extract symbols: the defined symbol + any calls inside
        symbols_defined = [name] if name else []
        symbols_called = _extract_calls(chunk_text, normalized_hint)
        # Remove self-calls from symbols_called
        if name and name in symbols_called:
            symbols_called = [c for c in symbols_called if c != name]
        imports = _extract_imports(chunk_text, normalized_hint)
        chunks.append(CodeChunk(
            id=next_id,
            file=file_hint,
            start_line=start,
            end_line=end,
            type=ctype,
            name=name,
            text=chunk_text,
            symbols_defined=symbols_defined,
            symbols_called=symbols_called,
            imports=imports,
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
                symbols_defined=[],
                symbols_called=_extract_calls(orphan_text, normalized_hint),
                imports=_extract_imports(orphan_text, normalized_hint),
            ))

    return chunks


def _chunk_sliding_window(
    text: str,
    file_hint: str = "",
    window: int = 30,
    overlap: int = 5,
    language_hint: str | None = None,
) -> list[CodeChunk]:
    """Fallback chunker: sliding windows of *window* lines with *overlap*."""
    lines = text.splitlines()
    if not lines:
        return []

    normalized_hint = (language_hint or "").lower()
    from gigacode.context_assembler import _extract_calls, _extract_imports

    chunks: list[CodeChunk] = []
    step = window - overlap
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
            symbols_defined=[],
            symbols_called=_extract_calls(chunk_text, normalized_hint),
            imports=_extract_imports(chunk_text, normalized_hint),
        ))
        cid += 1
        if end == len(lines):
            break
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    language_hint: str | None = None,
    filename_hint: str = "",
    sliding_window_size: int = 30
) -> list[CodeChunk]:
    """Split source text into semantic chunks.

    Tries tree-sitter AST chunking first, falls back to sliding windows.

    **Supported chunk types:**
    - `function` — Top-level functions
    - `nested_function` — Functions defined inside other functions
    - `class` — Class definitions
    - `method` — Class methods
    - `lambda` — Lambda expressions / anonymous functions (Python, Java, Ruby, etc.)
    - `macro` — Preprocessor macros (C/C++)
    - `module` — Module definitions (Elixir)
    - `orphan` — Code between definitions (semantically insignificant)
    - `sliding` — Fallback sliding windows when AST parsing fails

    **Warning:** Falling back to sliding windows means:
    - Chunks are fixed-size windows with no semantic awareness
    - Function/class boundaries are ignored
    - Search quality may be degraded for that file
    - Consider improving language detection or using an explicit language_hint

    Args:
        text: Source code text to chunk.
        language_hint: Language name (python, javascript, rust, etc.) for AST parsing.
                       If None, will be auto-detected.
        filename_hint: File name for logging/debugging.
        sliding_window_size: Number of lines per window in fallback mode (default 30).
                             Increase for fewer but larger chunks, decrease for more granular chunking.

    Returns:
        List of CodeChunk objects with type field set to one of the supported chunk types above.
    """
    chunks = _chunk_with_tree_sitter(text, language_hint, filename_hint)
    if chunks is not None:
        logger.debug("tree-sitter chunking produced %d chunks for %s", len(chunks), filename_hint)
        return chunks
    
    logger.warning(
        "AST chunking failed for %s (language_hint=%r); falling back to %d-line sliding windows. "
        "This reduces search quality — consider specifying language_hint or using AST-supported languages. "
        "Adjust sliding_window_size parameter if chunk granularity is not suitable.",
        filename_hint or "(unknown)", language_hint, sliding_window_size
    )
    return _chunk_sliding_window(text, filename_hint, window=sliding_window_size)


def chunk_file(
    path: str | Path,
    language_hint: str | None = None,
    sliding_window_size: int = 30
) -> list[CodeChunk]:
    """Chunk a file on disk.
    
    Args:
        path: Path to source file.
        language_hint: Language for AST parsing. Auto-detected if None.
        sliding_window_size: Lines per window in fallback mode (default 30).
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    if language_hint is None:
        from gigacode.language_detect import detect_language
        language_hint = detect_language(p)
    chunks = chunk_text(text, language_hint=language_hint, filename_hint=str(p), sliding_window_size=sliding_window_size)
    # Store full-file imports at chunk level for sliding window chunks
    # (tree-sitter chunks already have per-chunk imports)
    from gigacode.context_assembler import _extract_imports
    full_file_imports = _extract_imports(text, (language_hint or "").lower())
    for chunk in chunks:
        if not chunk.imports:
            chunk.imports = full_file_imports
    return chunks
