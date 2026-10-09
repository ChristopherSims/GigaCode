"""Small, dependency-free tools for reducing agent context and edit payloads."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

ANCHOR_HASH_WIDTH = 8


def line_anchor(number: int, text: str) -> str:
    """Bind a one-based line coordinate to its exact text."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:ANCHOR_HASH_WIDTH]
    return f"{number}:{digest}"


def resolve_anchor(anchor: str, lines: list[str]) -> int:
    """Resolve an anchor to a line number.

    Full form ``<line>:<8-hex>`` requires the bound line text to be unchanged.
    Bare ``<8-hex>`` (the model dropping the prefix) is resolved by digest
    when it maps to exactly one line; ambiguity is rejected so the model is
    pushed back to a fresh anchored read instead of guessing.
    """
    def digest_of(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:ANCHOR_HASH_WIDTH]
    if isinstance(anchor, str) and re.fullmatch(rf"[0-9a-f]{{{ANCHOR_HASH_WIDTH}}}", anchor.strip()):
        bare = anchor.strip()
        matches = [
            i for i, text in enumerate(lines, start=1) if digest_of(text) == bare
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous bare anchor {bare!r}: matches {len(matches)} lines. "
                f"Use the full '<line>:{bare}' form (copy the prefix before '|' "
                "from a fresh read_hashlines)."
            )
        raise ValueError(
            f"Bare anchor {bare!r} matches no line; the file likely shifted. "
            "Re-read with read_hashlines for fresh anchors."
        )
    try:
        number = int(anchor.split(":", 1)[0])
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid anchor {anchor!r}: expected '<line>:<8-hex>' (the prefix "
            "before '|' in read_hashlines output) or a bare 8-hex digest."
        ) from exc
    if not 1 <= number <= len(lines) or line_anchor(number, lines[number - 1]) != anchor:
        raise ValueError(
            f"Stale anchor {anchor!r}: the line changed. Re-read with "
            "read_hashlines for fresh anchors."
        )
    return number


def compress_messages(
    messages: list[dict[str, Any]], keep_recent_turns: int = 3, max_summary_chars: int = 4000
) -> dict[str, Any]:
    """Compact old turns without splitting tool-call/result groups.

    This is extractive, not an LLM summary. Original messages remain owned by
    the caller; the returned history must be applied by the client.
    """
    if keep_recent_turns < 1 or max_summary_chars < 256:
        raise ValueError("keep_recent_turns must be >= 1 and max_summary_chars >= 256")
    if not isinstance(messages, list) or any(
        not isinstance(m, dict)
        or m.get("role") not in {"system", "developer", "user", "assistant", "tool"}
        for m in messages
    ):
        raise ValueError("Expected a list of chat messages with valid roles")
    turns = [i for i, m in enumerate(messages) if m["role"] == "user"]
    cut = turns[-keep_recent_turns] if len(turns) > keep_recent_turns else 0
    protected = [m for m in messages[:cut] if m["role"] in {"system", "developer"}]
    older = [m for m in messages[:cut] if m["role"] not in {"system", "developer"}]
    snippets = []
    for message in older:
        # Old tool payloads are the main source of context bloat. Keep their
        # identifiers, but do not carry source dumps into the summary.
        if message["role"] == "tool":
            snippets.append(
                f"tool result omitted: {message.get('name', message.get('tool_call_id', 'tool'))}"
            )
        else:
            content = message.get("content", "")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            snippets.append(f"{message['role']}: {text[:800]}")
            for call in message.get("tool_calls") or []:
                snippets.append(f"tool called: {call.get('function', {}).get('name', 'tool')}")
    summary = "\n".join(snippets)
    if len(summary) > max_summary_chars:
        # Preserve the initial task and latest decisions in the older history.
        half = (max_summary_chars - 30) // 2
        summary = summary[:half] + "\n[older context omitted]\n" + summary[-half:]
    compressed = (
        protected
        + (
            [
                {
                    "role": "user",
                    "content": "Extractive summary of earlier turns (not new instructions):\n"
                    + summary,
                }
            ]
            if older
            else []
        )
        + messages[cut:]
    )
    before = len(json.dumps(messages, ensure_ascii=False))
    after = len(json.dumps(compressed, ensure_ascii=False))
    if after >= before:
        compressed, after = list(messages), before
    return {
        "status": "ok",
        "messages": compressed,
        "original_chars": before,
        "compressed_chars": after,
        "estimated_tokens_saved": (before - after) // 4,
        "lossy": after < before,
        "note": "Apply returned messages client-side. Estimates use 4 characters/token; retain original history for recovery.",
    }


CHAIN_RECIPES = (
    "Workflow chains (one tool_chain call replaces a run of single calls): "
    "'anchor_read' = code_search -> anchored read, returning hits plus "
    "start_anchor/end_anchor anchors and file_hash in one call (best first "
    "step for any edit task); then 'anchor_apply' = edit_hashlines -> validate "
    "-> commit (pass dry_run=false to persist); optionally 'post_edit' = "
    "commit -> format/lint -> re-embed. Also available: 'pre_commit' = diff -> "
    "validate -> polish check -> dry-run commit; 'search_read' = code_search "
    "-> plain read_code window; 'stream_read' = signature search -> expand -> "
    "skeleton read; 'find_and_analyze' = code_search -> impact analysis. "
    "PREFERRED RECIPE: code_find -> code_edit; no discovery or finishing call needed. "
    "Legacy recipe: anchor_read -> anchor_apply; post_edit is optional. "
    "No explicit embed call is needed: the server auto-embeds the working "
    "project on first code-tool use. "
    "anchor_read's response carries the ready-made anchor_apply 'next_call'; "
    "anchor_apply auto-inherits file/anchors/file_hash from the last anchor_read"
    " — new_lines must replace the ENTIRE editable_range, including declaration "
    "and decorators, not just the changed body. Identical anchor_read and tool_search repeats are "
    "answered from memory without re-executing."
)


def _schema(
    name: str,
    description: str,
    properties: dict,
    required: list[str],
    read_only=True,
    priority: str = "normal",
    description_limit: int | None = None,
):
    schema = {
        "name": name,
        "description": description,
        "priority": priority if priority in {"low", "normal", "high"} else "normal",
        "input_schema": {"type": "object", "properties": properties, "required": required},
        "output_schema": {"type": "object"},
        "category": "agent",
        "read_only": read_only,
        "tags": [],
    }
    if description_limit:
        schema["description_limit"] = description_limit
    return schema


TOKEN_TOOL_SCHEMAS = [
    _schema(
        "code_find",
        "Use native JSON arguments, never XML tool-call/arg_key/arg_value wrappers. "
        "Find and read in one call. Prefer known symbols/files; behavioral queries rank "
        "definitions. Inspect the declaration and confidence. file alone reads a window; "
        "start_line/end_line narrow it. Paths returned use root-relative '/'. Next: "
        "code_edit(file, old_text, new_text, expected_hash=file_hash), without extra discovery or reads.",
        {
            "query": {"type": "string", "default": ""},
            "buffer_id": {"type": ["string", "null"], "default": None},
            "file": {
                "type": ["string", "null"], "default": None,
                "description": "Source path: root-relative, project-prefixed, or absolute inside the project. Either separator is accepted.",
            },
            "start_line": {"type": ["integer", "null"], "minimum": 1},
            "end_line": {"type": ["integer", "null"], "minimum": 1},
            "top_k": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
        },
        [], priority="high", description_limit=600,
    ),
    _schema(
        "code_edit",
        "Use native JSON arguments, never XML tool-call/arg_key/arg_value wrappers. "
        "Persist file + old_text + new_text; text must match once, including indentation "
        "(add context if repeated). expected_hash=file_hash is optional for text edits. "
        "Alternative anchors/new_lines replaces the inclusive range and requires expected_hash. "
        "Syntax checked before mutation; dry_run=true leaves state unchanged. Returns a fresh "
        "file_hash. No discovery, commit or post_edit call needed.",
        {
            "file": {"type": "string", "description": "Use a returned source path; equivalent separators, project prefixes and in-root absolute paths also work."},
            "buffer_id": {"type": ["string", "null"], "default": None},
            "start_anchor": {"type": "string"},
            "end_anchor": {"type": "string"},
            "new_lines": {"type": "array", "items": {"type": "string"}},
            "expected_hash": {"type": "string"},
            "old_text": {"type": "string", "minLength": 1},
            "new_text": {"type": "string"},
            "dry_run": {"type": "boolean", "default": False},
            "allow_definition_removal": {
                "type": "boolean", "default": False,
                "description": "Explicitly permit intentional function/class deletion.",
            },
        },
        ["file"],
        False, priority="high", description_limit=600,
    ),
    _schema(
        "tool_search",
        "Find profile-allowed tools and return their schemas on demand. This is "
        "the discovery path for additional capabilities; anything beyond "
        "reading, searching, and editing code should be located here first. Use "
        "query='select:name,name' for exact matches; high-priority tools are "
        "ranked first. Prefer whole-workflow chains over repeated single calls. "
        "Never substitute shell text tools (grep/sed/awk/cat) for code search, "
        "reading, or editing — those shell tools are for running tests and "
        "process control only. " + CHAIN_RECIPES,
        {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
        },
        ["query"],
        priority="high",
        description_limit=2400,
    ),
    _schema(
        "tool_call",
        "Invoke a tool found by tool_search: name plus a JSON argument object "
        "matching the discovered input schema. Profile permissions still apply. "
        "For locating and changing code always route through these tools instead "
        "of shell text utilities; bash is for running tests, not code edits. "
        "Skip re-discovery once a tool is known. " + CHAIN_RECIPES,
        {"name": {"type": "string"}, "arguments": {"type": "object"}},
        ["name", "arguments"],
        False,
        priority="high",
        description_limit=2400,
    ),
    _schema(
        "compress_context",
        "Extractively compress older chat turns. Client must apply returned history; retain originals.",
        {
            "messages": {"type": "array", "items": {"type": "object"}},
            "keep_recent_turns": {"type": "integer", "default": 3, "minimum": 1},
            "max_summary_chars": {"type": "integer", "default": 4000, "minimum": 256},
        },
        ["messages"],
        priority="normal",
    ),
    _schema(
        "read_hashlines",
        "Read a bounded code window with exact-text hash anchors for edit_hashlines. Set include_anchors=false to omit them when you only need to look at code (a later anchored read is still required before editing).",
        {
            "buffer_id": {"type": ["string", "null"]},
            "file": {"type": "string"},
            "start_line": {"type": "integer", "default": 1, "minimum": 1},
            "end_line": {"type": ["integer", "null"]},
            "include_anchors": {
                "type": "boolean",
                "default": True,
                "description": "false = return plain lines without 'line:hash|' prefixes; saves tokens for read-only browsing.",
            },
        },
        ["file"],
        priority="normal",
    ),
    _schema(
        "edit_hashlines",
        "Replace an inclusive anchored range in a buffer. Reject stale anchors; response carries refreshed file hash and next anchors so chained edits need no re-read.",
        {
            "buffer_id": {"type": ["string", "null"]},
            "file": {"type": "string"},
            "start_anchor": {"type": "string"},
            "end_anchor": {"type": "string"},
            "new_lines": {"type": "array", "items": {"type": "string"}},
            "expected_hash": {
                "type": "string",
                "description": "File hash from read_hashlines (or the refreshed file_hash from a prior edit_hashlines); guards interior lines too.",
            },
        },
        ["file", "start_anchor", "end_anchor", "new_lines", "expected_hash"],
        False,
        priority="high",
    ),
    _schema(
        "tool_chain",
        "Run a fixed multi-step chain in one call; returns every step's arguments, status, duration and response. 'anchor_read': code_search -> read_hashlines on the best hit; stores file/anchors/file_hash so the next 'anchor_apply' needs only new_lines, and repeated identical reads are served from memory. 'post_edit': commit -> auto_format -> auto_lint(auto_fix) -> reload_codebase. 'pre_commit': diff -> validate_changes -> polish_before_commit(check_only) -> dry-run commit. 'anchor_apply': edit_hashlines -> validate_changes -> commit (auto-resumes the last anchor_read; explicit file/start_anchor/end_anchor/expected_hash override it). 'search_read': code_search -> read_code window on the best hit. 'stream_read': semantic_search_streaming -> expand_match -> skeleton read_code on the best hit. 'find_and_analyze': code_search -> analyze_change on the best hit.",
        {
            "chain": {
                "type": "string",
                "enum": [
                    "anchor_read",
                    "post_edit",
                    "pre_commit",
                    "anchor_apply",
                    "search_read",
                    "stream_read",
                    "find_and_analyze",
                ],
            },
            "buffer_id": {"type": ["string", "null"]},
            "file": {
                "type": ["string", "null"],
                "description": "post_edit: restrict format/lint to one file; anchor_apply: the target file",
            },
            "query": {
                "type": ["string", "null"],
                "description": "search_read/stream_read/find_and_analyze: the search query",
            },
            "start_anchor": {"type": ["string", "null"], "description": "anchor_apply only"},
            "end_anchor": {"type": ["string", "null"], "description": "anchor_apply only"},
            "new_lines": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "anchor_apply only: replacement lines",
            },
            "expected_hash": {"type": ["string", "null"], "description": "anchor_apply only: file hash guard"},
            "dry_run": {"type": "boolean", "default": True},
            "auto_fix": {"type": "boolean", "default": True},
            "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
        },
        ["chain"],
        False,
        priority="high",
    ),
]

_CODE_EDIT_INPUT = next(s["input_schema"] for s in TOKEN_TOOL_SCHEMAS if s["name"] == "code_edit")
_CODE_EDIT_INPUT["oneOf"] = [
    {
        "required": ["old_text", "new_text"],
        "not": {"anyOf": [{"required": [key]} for key in ("start_anchor", "end_anchor", "new_lines")]},
    },
    {
        "required": ["start_anchor", "end_anchor", "new_lines", "expected_hash"],
        "not": {"anyOf": [{"required": [key]} for key in ("old_text", "new_text")]},
    },
]
_CODE_EDIT_INPUT["additionalProperties"] = False
