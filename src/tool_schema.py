"""Formal JSON schemas for the GigaCode agent tool interface (Phase 6.1).

These schemas can be used with:
- OpenAI function calling
- Anthropic tool use
- MCP (Model Context Protocol) tool definitions
- Generic JSON-RPC or CLI wrappers

All schemas enforce the rule that raw source code is NEVER returned.
Only coordinates (file, line) and metadata (score, buffer_id) are exposed.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Shared schema fragments
# ---------------------------------------------------------------------------
def _buffer_id_param(required: bool = True) -> dict[str, Any]:
    d: dict[str, Any] = {
        "type": "string",
        "description": (
            "Opaque buffer handle returned by embed_codebase. "
            "The agent context should store only this UUID, never raw source text."
        ),
    }
    if required:
        d["enum"] = []  # placeholder; populated at runtime if desired
    return d


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
EMBED_CODEBASE_SCHEMA: dict[str, Any] = {
    "name": "embed_codebase",
    "description": (
        "Embed a directory or single file into a GPU/CPU buffer for semantic search "
        "and clustering. Returns a buffer handle; raw source code is never exposed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to a source directory or file.",
            },
            "language_hint": {
                "type": "string",
                "description": (
                    "Optional language override (e.g. 'python', 'javascript', 'rust', 'cpp'). "
                    "Auto-detected from file extension when omitted."
                ),
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern for files when path is a directory (default '*.py').",
                "default": "*.py",
            },
        },
        "required": ["path"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["ok", "warning", "error"],
                "description": "Result status.",
            },
            "buffer_id": {
                "type": "string",
                "description": "Opaque handle to the embedded buffer.",
            },
            "token_count": {
                "type": "integer",
                "description": "Number of embedded lines / tokens.",
            },
            "size_bytes": {
                "type": "integer",
                "description": "Size of the embeddings buffer in bytes.",
            },
            "message": {"type": "string"},
            "suggested_max": {
                "type": "string",
                "description": "Suggested maximum size if warning was returned.",
            },
            "estimated_mb": {
                "type": "number",
                "description": "Estimated megabytes if warning was returned.",
            },
        },
        "required": ["status"],
    },
}


SEMANTIC_SEARCH_SCHEMA: dict[str, Any] = {
    "name": "semantic_search",
    "description": (
        "Find the top-K code lines most similar to a natural-language query. "
        "Returns only file paths, line numbers, and scores — never raw source text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
            "query": {
                "type": "string",
                "description": "Natural-language query describing the code concept to find.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top matches to return (default 5).",
                "default": 5,
            },
        },
        "required": ["buffer_id", "query"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "score": {"type": "number"},
                    },
                    "required": ["file", "line", "score"],
                },
            },
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}


CLUSTER_CODE_SCHEMA: dict[str, Any] = {
    "name": "cluster_code",
    "description": (
        "Group similar code regions into semantic clusters. "
        "Returns only file paths, line ranges, and cluster metadata — never raw source text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
            "threshold": {
                "type": "number",
                "description": "Cosine-similarity threshold for grouping (default 0.75).",
                "default": 0.75,
            },
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "size": {"type": "integer"},
                        "avg_score": {"type": "number"},
                    },
                    "required": ["file", "start_line", "end_line", "size"],
                },
            },
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}


UPDATE_CODEBASE_SCHEMA: dict[str, Any] = {
    "name": "update_codebase",
    "description": (
        "Re-embed a single file and patch the existing buffer incrementally. "
        "Only changed lines are re-embedded when possible."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Existing buffer handle.",
            },
            "file_path": {
                "type": "string",
                "description": "Path to the file that changed.",
            },
            "language_hint": {
                "type": "string",
                "description": "Optional language override.",
            },
        },
        "required": ["buffer_id", "file_path"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "warning", "error"]},
            "message": {"type": "string"},
            "changed_lines": {"type": "integer"},
        },
        "required": ["status"],
    },
}


CHECK_CODEBASE_SCHEMA: dict[str, Any] = {
    "name": "check_codebase",
    "description": (
        "Lightweight pre-flight size estimate without embedding. "
        "Use this before embed_codebase to avoid large uploads."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory or file path to scan."},
            "pattern": {
                "type": "string",
                "description": "Glob pattern for files when path is a directory.",
                "default": "*.py",
            },
        },
        "required": ["path"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "warning", "exceeds_threshold"]},
            "estimated_lines": {"type": "integer"},
            "estimated_tokens": {"type": "integer"},
            "estimated_mb": {"type": "number"},
            "threshold_mb": {"type": "number"},
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}


LIST_BUFFERS_SCHEMA: dict[str, Any] = {
    "name": "list_buffers",
    "description": "List all embedded buffer handles with metadata (no raw code).",
    "input_schema": {
        "type": "object",
        "properties": {},
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "buffers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "buffer_id": {"type": "string"},
                        "root": {"type": "string"},
                        "token_count": {"type": "integer"},
                        "embedding_dim": {"type": "integer"},
                        "size_bytes": {"type": "integer"},
                    },
                },
            },
        },
        "required": ["status"],
    },
}


DELETE_BUFFER_SCHEMA: dict[str, Any] = {
    "name": "delete_buffer",
    "description": "Delete a buffer and free its on-disk resources.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle to delete.",
            },
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}

READ_CODE_SCHEMA: dict[str, Any] = {
    "name": "read_code",
    "description": (
        "Read raw source text from an embedded buffer. "
        "Unlike semantic_search, this returns actual code lines so the agent can edit them."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
            "file": {
                "type": "string",
                "description": "Relative file path. If omitted, all files are returned.",
            },
            "start_line": {
                "type": "integer",
                "description": "1-based start line (inclusive). Default 1.",
                "default": 1,
            },
            "end_line": {
                "type": "integer",
                "description": "1-based end line (exclusive). Null means to end of file.",
            },
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "file": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
            "lines": {
                "type": "array",
                "items": {"type": "string"},
            },
            "files": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}

WRITE_CODE_SCHEMA: dict[str, Any] = {
    "name": "write_code",
    "description": (
        "Replace a range of lines in a buffered file and re-embed the changed region. "
        "The file is marked dirty until commit is called."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
            "file": {
                "type": "string",
                "description": "Relative file path to edit.",
            },
            "start_line": {
                "type": "integer",
                "description": "1-based start line (inclusive).",
            },
            "new_lines": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Replacement line strings (no newlines).",
            },
            "end_line": {
                "type": "integer",
                "description": "1-based end line (exclusive). Null means to end of file.",
            },
        },
        "required": ["buffer_id", "file", "start_line", "new_lines"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "file": {"type": "string"},
            "changed_lines": {"type": "integer"},
            "replaced_lines": {"type": "integer"},
            "total_lines": {"type": "integer"},
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}

DIFF_SCHEMA: dict[str, Any] = {
    "name": "diff",
    "description": "List files that differ from the original on-disk versions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "changed_files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "buffer_lines": {"type": "integer"},
                        "disk_lines": {"type": "integer"},
                        "dirty": {"type": "boolean"},
                    },
                },
            },
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}

DISCARD_SCHEMA: dict[str, Any] = {
    "name": "discard",
    "description": "Revert one or all files in the buffer back to the on-disk originals.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
            "file": {
                "type": "string",
                "description": "Relative file path. If omitted, all files are reverted.",
            },
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "reverted_files": {
                "type": "array",
                "items": {"type": "string"},
            },
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}

COMMIT_SCHEMA: dict[str, Any] = {
    "name": "commit",
    "description": (
        "Write dirty files from the buffer back to disk, overwriting the originals. "
        "Use dry_run to preview changes first."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, report what would be written without touching disk.",
                "default": False,
            },
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "written_files": {
                "type": "array",
                "items": {"type": "string"},
            },
            "dry_run": {"type": "boolean"},
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}


# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------
ALL_SCHEMAS: list[dict[str, Any]] = [
    EMBED_CODEBASE_SCHEMA,
    SEMANTIC_SEARCH_SCHEMA,
    CLUSTER_CODE_SCHEMA,
    UPDATE_CODEBASE_SCHEMA,
    CHECK_CODEBASE_SCHEMA,
    LIST_BUFFERS_SCHEMA,
    DELETE_BUFFER_SCHEMA,
    READ_CODE_SCHEMA,
    WRITE_CODE_SCHEMA,
    DIFF_SCHEMA,
    DISCARD_SCHEMA,
    COMMIT_SCHEMA,
]


def get_schema(name: str) -> dict[str, Any] | None:
    """Return a single tool schema by name, or None if not found."""
    for schema in ALL_SCHEMAS:
        if schema.get("name") == name:
            return schema
    return None


def get_all_schemas() -> list[dict[str, Any]]:
    """Return all tool schemas."""
    return list(ALL_SCHEMAS)


def to_openai_functions() -> list[dict[str, Any]]:
    """Convert schemas to OpenAI function-calling format."""
    functions: list[dict[str, Any]] = []
    for schema in ALL_SCHEMAS:
        functions.append(
            {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema["description"],
                    "parameters": schema["input_schema"],
                },
            }
        )
    return functions


def to_mcp_tools() -> list[dict[str, Any]]:
    """Convert schemas to MCP tool format."""
    tools: list[dict[str, Any]] = []
    for schema in ALL_SCHEMAS:
        tools.append(
            {
                "name": schema["name"],
                "description": schema["description"],
                "inputSchema": schema["input_schema"],
            }
        )
    return tools
