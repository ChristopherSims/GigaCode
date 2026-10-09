"""Source-aware retrieval and edit guards, independent of embedding models."""

from __future__ import annotations

import ast
import re
from pathlib import PurePosixPath
from typing import Any

DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def text_replacement(
    lines: list[str], old_text: str, new_text: str,
) -> tuple[int, int, list[str]]:
    """Turn one unique literal replacement into a minimal line edit.

    Line endings are canonicalized, but indentation and all other whitespace
    must match. No fuzzy replacement, guessing, or execution is involved.
    """
    if not isinstance(old_text, str) or not old_text:
        raise ValueError("old_text must be non-empty exact source text.")
    if not isinstance(new_text, str):
        raise ValueError("new_text must be a string (empty is allowed for deletion).")
    old_text = old_text.replace("\r\n", "\n")
    new_text = new_text.replace("\r\n", "\n")
    if "\r" in old_text or "\r" in new_text:
        raise ValueError("Use LF or CRLF line endings, not standalone carriage returns.")
    text = "\n".join(lines) + ("\n" if lines else "")
    first = text.find(old_text)
    if first < 0:
        raise ValueError("old_text does not match current source. Use exact text from code_find.")
    if text.find(old_text, first + 1) >= 0:
        raise ValueError("old_text matches more than once. Include distinguishing surrounding lines.")
    candidate = (text[:first] + new_text + text[first + len(old_text):]).splitlines()
    prefix = 0
    while prefix < min(len(lines), len(candidate)) and lines[prefix] == candidate[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < min(len(lines), len(candidate)) - prefix
        and lines[len(lines) - suffix - 1] == candidate[len(candidate) - suffix - 1]
    ):
        suffix += 1
    start, end = prefix + 1, len(lines) - suffix
    replacement = candidate[prefix:len(candidate) - suffix if suffix else len(candidate)]
    if end < start:
        if prefix:
            start = end = prefix
            replacement = [lines[prefix - 1], *replacement]
        elif lines:
            start = end = 1
            replacement = [*replacement, lines[0]]
        else:
            raise ValueError("Cannot anchor an empty file; use write_code for file creation.")
    return start, end, replacement


def definition_spans(lines: list[str], file: str) -> list[tuple[int, int]]:
    """Parse a file once and collect complete Python definition bounds."""
    if not file.endswith(".py"):
        return []
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return []
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, DEFINITIONS):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            end = node.end_lineno or node.lineno
            spans.append((start, end))
    return spans


def definition_range(lines: list[str], file: str, line: int) -> tuple[int, int] | None:
    """Return the smallest complete Python definition containing a hit."""
    candidates = [span for span in definition_spans(lines, file) if span[0] <= line <= span[1]]
    return min(candidates, key=lambda span: span[1] - span[0]) if candidates else None


def exact_source_matches(
    snapshot: dict[str, list[str]],
    query: str,
    top_k: int,
) -> dict[str, Any] | None:
    """Route identifiers and file names without semantic nearest-neighbor guesses.

    None means natural-language retrieval should handle this query. An empty
    result means an explicit identifier/path was requested but does not exist.
    """
    words = query.strip().strip("`").split()
    extensions = {PurePosixPath(f.replace("\\", "/")).suffix for f in snapshot}
    extensions.discard("")
    file_tokens = [
        w for w in words
        if PurePosixPath(w.replace("\\", "/")).suffix in extensions
        or "/" in w or "\\" in w
    ]
    files = [
        f
        for f in snapshot
        if not file_tokens
        or any(
            f.replace("\\", "/") == token.replace("\\", "/")
            or PurePosixPath(f.replace("\\", "/")).name == token
            for token in file_tokens
        )
    ]
    names = [w for w in words if w not in file_tokens and w not in {"def", "class", "async"}]
    explicit = (
        bool(file_tokens)
        or (len(names) == 1 and bool(re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", names[0])))
        or (words and words[0] in {"def", "class", "async"})
    )
    identifiers = [
        w
        for w in names
        if re.fullmatch(r"[A-Za-z_]\w*", w) and ("_" in w or re.search(r"[a-z][A-Z]", w))
    ]
    declared = names[0] if names and words[0] in {"def", "class", "async"} else None
    needle = declared or (identifiers[0] if identifiers else (names[0] if len(names) == 1 else None))
    matches = []
    if needle:
        pattern = re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)")
        for file in files:
            lines = snapshot[file]
            spans = None
            for number, text in enumerate(lines, 1):
                if pattern.search(text):
                    if spans is None:
                        spans = definition_spans(lines, file)
                    candidates = [span for span in spans if span[0] <= number <= span[1]]
                    span = min(candidates, key=lambda s: s[1] - s[0]) if candidates else None
                    start, end = span or (number, number)
                    if any(m["file"] == file and m["start_line"] == start for m in matches):
                        continue
                    matches.append(
                        {
                            "file": file,
                            "start_line": start,
                            "end_line": end,
                            "name": needle,
                            "score": 1.0,
                            "confidence": "exact",
                            "text": "\n".join(lines[start - 1 : end]),
                            "definition_match": any(
                                re.match(
                                    rf"\s*(?:async\s+)?(?:def|class)\s+{re.escape(needle)}\b", row
                                )
                                for row in lines[start - 1 : end]
                            ),
                        }
                    )
        # Exact matches win even in a mixed natural-language query.
        if matches:
            matches.sort(key=lambda m: not m["definition_match"])
            return {"status": "ok", "mode": "exact", "matches": matches[:top_k]}
    if file_tokens and len(names) > 1 and not identifiers:
        return None
    if explicit:
        if file_tokens and not names:
            matches = [
                {
                    "file": f,
                    "start_line": 1,
                    "end_line": min(len(snapshot[f]), 80),
                    "score": 1.0,
                    "confidence": "exact",
                }
                for f in files
            ]
        return {
            "status": "ok",
            "mode": "exact",
            "matches": matches[:top_k],
            "hint": "No exact match; use a behavioral description or an existing file/symbol."
            if not matches
            else "Exact source match.",
        }
    return None


def validate_replacement(
    lines: list[str],
    file: str,
    start: int,
    end: int,
    new_lines: list[str],
    preserve_definitions: bool = False,
) -> str | None:
    """Validate the complete candidate before changing buffer or disk state."""
    if not isinstance(new_lines, list) or any(
        not isinstance(line, str) or "\n" in line or "\r" in line for line in new_lines
    ):
        return "new_lines must be a list of individual source lines."
    if not 1 <= start <= end <= len(lines):
        return "Invalid replacement range."
    if not file.endswith(".py"):
        return None
    candidate = lines[: start - 1] + new_lines + lines[end:]
    try:
        after = ast.parse("\n".join(candidate), filename=file)
        compile(after, file, "exec", dont_inherit=True)
    except SyntaxError as exc:
        return f"Edit rejected before mutation: {exc.msg} at line {exc.lineno}."
    if preserve_definitions:
        try:
            before = ast.parse("\n".join(lines), filename=file)
        except SyntaxError:
            return "Cannot safely resume an edit of an invalid Python file; use explicit anchors."

        def count(tree):
            return sum(isinstance(n, DEFINITIONS) for n in ast.walk(tree))

        if count(after) < count(before):
            return (
                "Edit would remove a definition. Replace the complete editable range, "
                "including its def/class declaration, or supply explicit anchors."
            )
    return None


def scoped_source_matches(
    snapshot: dict[str, list[str]],
    query: str,
    top_k: int,
) -> dict[str, Any]:
    """Rank definitions in an explicitly selected file without a global search."""
    from gigacode.lexical_index import LexicalIndex

    index = LexicalIndex()
    entries = []
    for file, lines in snapshot.items():
        spans = definition_spans(lines, file) or [(1, len(lines))]
        for start, end in spans:
            text = "\n".join(lines[start - 1 : end])
            entry = {
                "file": file,
                "start_line": start,
                "end_line": end,
                "text": text,
                "confidence": "candidate",
            }
            index.add(len(entries), text)
            entries.append(entry)
    matches = [
        {**entries[hit["doc_id"]], "score": hit["score"]} for hit in index.search(query, top_k)
    ]
    return {
        "status": "ok",
        "mode": "scoped_lexical",
        "matches": matches,
        "selection_hint": "Inspect the selected definition before editing.",
    }
