"""Code tokenizer with tree-sitter, tiktoken, or regex fallback."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Regex-based fallback tokenization
_WORD_RE = re.compile(r"[A-Za-z_]\w*")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_PUNCT_RE = re.compile(r"[+\-*/%=<>!&|^~:;.,?{}\[\]()@#`\\]")
_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"|' r"'(?:[^'\\]|\\.)*'")
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/|\#[^\n]*", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


def _regex_tokenize(text: str) -> list[str]:
    """Simple regex-based tokenizer fallback.

    Tries to capture words, numbers, strings, comments, and punctuation
    in a single left-to-right pass over *text*.
    """
    tokens: list[str] = []
    pos = 0
    length = len(text)
    while pos < length:
        match = None
        # Try longer / more-specific patterns first
        for pattern in (_COMMENT_RE, _STRING_RE, _WORD_RE, _NUMBER_RE, _PUNCT_RE):
            match = pattern.match(text, pos)
            if match:
                token = match.group(0)
                if token:
                    tokens.append(token)
                pos = match.end()
                break
        if not match:
            # Skip unrecognized / whitespace
            ws = _WHITESPACE_RE.match(text, pos)
            if ws:
                pos = ws.end()
            else:
                tokens.append(text[pos])
                pos += 1
    return tokens


def _tiktoken_tokenize(text: str) -> list[str]:
    """Tokenize with tiktoken (cl100k_base)."""
    import tiktoken  # type: ignore[import-untyped]

    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text, disallowed_special=())
    # Decode each token individually to keep a 1:1 mapping
    return [enc.decode([tid]) for tid in token_ids]


def _tree_sitter_tokenize(text: str, language_hint: str | None) -> list[tuple[int, list[str]]]:
    """Tokenize source code with tree-sitter, returning per-line tokens."""
    from tree_sitter import Language, Parser  # type: ignore[import-untyped]

    # Build parser from a language hint.  We try a few common tree-sitter
    # language package naming conventions.
    parser = Parser()
    lang = _resolve_tree_sitter_language(language_hint)
    parser.set_language(lang)
    tree = parser.parse(bytes(text, "utf8"))
    root = tree.root_node

    line_tokens: dict[int, list[str]] = {}
    for node in _walk(root):
        if node.type == "comment":
            continue
        if node.child_count == 0 or node.type in (
            "identifier",
            "string",
            "number",
            "operator",
            "punctuation",
        ):
            line_num = node.start_point[0] + 1  # 0-indexed -> 1-indexed
            token_text = text[node.start_byte : node.end_byte]
            if token_text.strip():
                line_tokens.setdefault(line_num, []).append(token_text)
    # Ensure every line has an entry even if empty
    max_line = max(line_tokens.keys()) if line_tokens else 0
    result: list[tuple[int, list[str]]] = []
    for ln in range(1, max_line + 1):
        result.append((ln, line_tokens.get(ln, [])))
    return result


def _walk(node):
    """Yield every node in the tree (pre-order)."""
    yield node
    for child in node.children:
        yield from _walk(child)


def _resolve_tree_sitter_language(language_hint: str | None):
    """Attempt to import and return a tree-sitter Language object."""
    # Try a handful of common community package names
    candidates: list[str] = []
    if language_hint:
        hint = language_hint.lower()
        candidates.append(f"tree_sitter_{hint}")
        # Common aliases
        if hint == "cpp":
            candidates.append("tree_sitter_cpp")
        elif hint == "c++":
            candidates.append("tree_sitter_cpp")
        elif hint == "js":
            candidates.append("tree_sitter_javascript")
        elif hint == "ts":
            candidates.append("tree_sitter_typescript")
        elif hint == "py":
            candidates.append("tree_sitter_python")
        elif hint == "rs":
            candidates.append("tree_sitter_rust")
        elif hint == "go":
            candidates.append("tree_sitter_go")
        elif hint == "java":
            candidates.append("tree_sitter_java")
        elif hint == "rb":
            candidates.append("tree_sitter_ruby")

    # Fallback: try a generic set
    for pkg in [
        "tree_sitter_python",
        "tree_sitter_javascript",
        "tree_sitter_cpp",
        "tree_sitter_rust",
    ]:
        if pkg not in candidates:
            candidates.append(pkg)

    for pkg in candidates:
        try:
            mod = __import__(pkg)
            lang = getattr(mod, "language", None)
            if lang is None:
                continue
            if callable(lang):
                return Language(lang())
            return Language(lang)
        except Exception:
            continue
    raise RuntimeError(
        f"Could not resolve tree-sitter language for hint={language_hint!r}. "
        "Install a language package such as tree-sitter-python."
    )


def tokenize_string(
    text: str,
    language_hint: str | None = None,
    filename_hint: str = "<string>",
) -> list[dict[str, Any]]:
    """Tokenize source text into per-line token records.

    This is the core implementation used by :func:`tokenize_file`. It avoids
    disk I/O so callers that already have the text in memory (e.g. the agent
    edit buffer) don't need to write a temporary file.

    Args:
        text: Raw source code.
        language_hint: Optional language identifier (e.g. ``"python"``).
        filename_hint: Used only for debug logging.

    Returns:
        A list of dicts, one per line, with keys:
        ``line_num`` (int), ``tokens`` (list[str]), ``text`` (str).
    """
    lines = text.splitlines()

    # --- Strategy 1: tree-sitter ---
    try:
        ts_result = _tree_sitter_tokenize(text, language_hint)
        output: list[dict[str, Any]] = []
        for ln, tokens in ts_result:
            output.append(
                {
                    "line_num": ln,
                    "tokens": tokens,
                    "text": lines[ln - 1] if ln <= len(lines) else "",
                }
            )
        logger.debug("Used tree-sitter tokenizer for %s", filename_hint)
        return output
    except Exception as exc:
        logger.debug("tree-sitter tokenizer failed (%s), falling back", exc)

    # --- Strategy 2: tiktoken ---
    try:
        all_tokens = _tiktoken_tokenize(text)
        output = []
        token_idx = 0
        char_pos = 0
        for i, line in enumerate(lines, start=1):
            line_tokens: list[str] = []
            line_start = char_pos
            line_end = char_pos + len(line)
            while token_idx < len(all_tokens):
                tok = all_tokens[token_idx]
                tok_start = text.find(tok, char_pos)
                if tok_start == -1:
                    token_idx += 1
                    continue
                if tok_start >= line_end:
                    break
                line_tokens.append(tok)
                char_pos = tok_start + len(tok)
                token_idx += 1
            output.append({"line_num": i, "tokens": line_tokens, "text": line})
            char_pos = line_end + 1
        logger.debug("Used tiktoken tokenizer for %s", filename_hint)
        return output
    except Exception as exc:
        logger.debug("tiktoken tokenizer failed (%s), falling back to regex", exc)

    # --- Strategy 3: regex fallback ---
    output = []
    for i, line in enumerate(lines, start=1):
        tokens = _regex_tokenize(line)
        output.append({"line_num": i, "tokens": tokens, "text": line})
    logger.debug("Used regex fallback tokenizer for %s", filename_hint)
    return output


def tokenize_file(path: str | Path, language_hint: str | None = None) -> list[dict[str, Any]]:
    """Tokenize a source file into per-line token records.

    Thin wrapper around :func:`tokenize_string` that reads the file from disk.

    Args:
        path: Path to the source file.
        language_hint: Optional language identifier.

    Returns:
        A list of dicts, one per line, with keys:
        ``line_num`` (int), ``tokens`` (list[str]), ``text`` (str).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    raw_text = p.read_text(encoding="utf-8", errors="replace")
    return tokenize_string(raw_text, language_hint=language_hint, filename_hint=str(p))
