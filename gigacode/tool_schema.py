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

__all__ = [
    "EMBED_CODEBASE_SCHEMA",
    "SEMANTIC_SEARCH_SCHEMA",
    "CLUSTER_CODE_SCHEMA",
    "SEARCH_FOR_SCHEMA",
    "SEARCH_SYMBOLS_SCHEMA",
    "RELOAD_CODEBASE_SCHEMA",
    "CHECK_CODEBASE_SCHEMA",
    "LIST_BUFFERS_SCHEMA",
    "DELETE_BUFFER_SCHEMA",
    "READ_CODE_SCHEMA",
    "LOOK_FOR_FILE_SCHEMA",
    "WRITE_CODE_SCHEMA",
    "DIFF_SCHEMA",
    "DISCARD_SCHEMA",
    "COMMIT_SCHEMA",
    "HYBRID_SEARCH_SCHEMA",
    "FIND_DUPLICATES_SCHEMA",
    "PACK_CONTEXT_SCHEMA",
    "INFER_TYPES_SCHEMA",
    "GET_SYMBOL_METADATA_SCHEMA",
    "ALL_SCHEMAS",
    "get_schema",
    "get_all_schemas",
    "to_openai_functions",
    "to_mcp_tools",
]


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
        "Find the top-K code blocks most similar to a natural-language query. "
        "Returns complete source code, file paths, line ranges, and relevance scores. "
        "Optionally includes inferred type hints (parameter types, return types, confidence scores)."
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
            "include_types": {
                "type": "boolean",
                "description": "Include inferred type hints in results (parameter types, return types). Default: false.",
                "default": False,
            },
            "type_inference_method": {
                "type": "string",
                "enum": ["llm", "ast"],
                "description": "Type inference method: 'llm' (accurate, ~50-300ms) or 'ast' (fast, ~1-5ms). Default: 'llm'.",
                "default": "llm",
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
                        "file": {"type": "string", "description": "Source file path"},
                        "start_line": {"type": "integer", "description": "First line of match"},
                        "end_line": {"type": "integer", "description": "Last line of match"},
                        "line": {"type": "integer", "description": "Alias for start_line (backward compatibility)"},
                        "score": {"type": "number", "description": "Relevance score (0.0-1.0)"},
                        "type": {"type": "string", "description": "Symbol type (function, class, etc.)"},
                        "name": {"type": "string", "description": "Symbol name if applicable"},
                        "text": {"type": "string", "description": "Complete source code for the match"},
                        "signature": {"type": "string", "description": "Function/class signature with type annotations (only when include_types=true)"},
                        "parameter_types": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                },
                            },
                            "description": "Inferred parameter types (only when include_types=true)",
                        },
                        "return_type": {"type": "string", "description": "Inferred return type (only when include_types=true)"},
                        "type_confidence": {"type": "number", "description": "Confidence score 0.0-1.0 (only for LLM inference)"},
                        "inference_method": {"type": "string", "enum": ["llm", "ast"], "description": "Inference method used"},
                    },
                    "required": ["file", "start_line", "end_line", "score"],
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

SEARCH_FOR_SCHEMA: dict[str, Any] = {
    "name": "search_for",
    "description": (
        "Literal substring search across the entire buffered codebase. "
        "Returns file paths, line numbers, and the matching line content."
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
                "description": "Substring to search for in every source line.",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "If true, match exact case. Default false.",
                "default": False,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matches to return (default 50).",
                "default": 50,
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
                        "content": {"type": "string"},
                    },
                    "required": ["file", "line", "content"],
                },
            },
            "total": {"type": "integer"},
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}

SEARCH_SYMBOLS_SCHEMA: dict[str, Any] = {
    "name": "search_symbols",
    "description": (
        "Find functions, classes, methods, and variables matching a query. "
        "Performs both name-based substring matching and semantic embedding search, "
        "then merges and deduplicates the results. Returns file paths, line ranges, "
        "symbol names, types, and scores — never full source text."
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
                "description": "Word or phrase describing the symbol to find (e.g. 'fetch_data', 'config loader').",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of symbol matches to return (default 10).",
                "default": 10,
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
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "type": {"type": "string"},
                        "name": {"type": ["string", "null"]},
                        "score": {"type": "number"},
                        "match_type": {
                            "type": "string",
                            "enum": ["name", "semantic"],
                        },
                    },
                    "required": ["file", "start_line", "end_line", "type", "name", "score"],
                },
            },
            "total": {"type": "integer"},
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}


RELOAD_CODEBASE_SCHEMA: dict[str, Any] = {
    "name": "reload_codebase",
    "description": (
        "Reload a codebase buffer from disk, re-embedding only if file hashes changed. "
        "Use this to refresh a buffer after external edits."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Existing buffer handle.",
            },
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "warning", "error"]},
            "buffer_id": {"type": "string"},
            "chunk_count": {"type": "integer"},
            "size_bytes": {"type": "integer"},
            "message": {"type": "string"},
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

LOOK_FOR_FILE_SCHEMA: dict[str, Any] = {
    "name": "look_for_file",
    "description": (
        "Find the location of a file within an embedded buffer. "
        "Tries exact match, then basename match, then partial substring match. "
        "Returns the relative file path and the absolute path on disk."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
            "file_name": {
                "type": "string",
                "description": (
                    "File name, relative path, or path fragment to look for. "
                    "Examples: 'gigacode_tool.py', 'src/gigacode_tool.py', 'gigacode'"
                ),
            },
        },
        "required": ["buffer_id", "file_name"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["ok", "error"],
                "description": "Result status.",
            },
            "file_location": {
                "type": "string",
                "description": "Relative path within the buffer root.",
            },
            "absolute_path": {
                "type": "string",
                "description": "Absolute path on disk.",
            },
            "match_type": {
                "type": "string",
                "enum": ["exact", "basename", "partial", "multiple"],
                "description": "How the file was matched.",
            },
            "candidates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Multiple candidate paths (only when match_type is 'multiple').",
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
            "diff": {"type": "string", "description": "Unified diff of the changes made by this write operation."},
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

HYBRID_SEARCH_SCHEMA: dict[str, Any] = {
    "name": "hybrid_search",
    "description": (
        "Combine FAISS semantic search with BM25 lexical search via Reciprocal Rank Fusion. "
        "Returns file paths, line ranges, and merged relevance scores."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
            "query": {"type": "string", "description": "Natural language or keyword query."},
            "top_k": {
                "type": "integer",
                "description": "Number of results to return.",
                "default": 5,
            },
            "offset": {"type": "integer", "description": "Pagination offset.", "default": 0},
            "semantic_weight": {
                "type": "number",
                "description": "Weight for semantic rank contribution.",
                "default": 1.0,
            },
            "lexical_weight": {
                "type": "number",
                "description": "Weight for lexical rank contribution.",
                "default": 1.0,
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
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "type": {"type": "string"},
                        "name": {"type": ["string", "null"]},
                        "rrf_score": {"type": "number"},
                        "semantic_rank": {"type": "integer"},
                        "lexical_rank": {"type": "integer"},
                        "doc_id": {"type": "integer"},
                    },
                },
            },
            "cached": {"type": "boolean"},
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}

FIND_DUPLICATES_SCHEMA: dict[str, Any] = {
    "name": "find_duplicates",
    "description": "Find near-duplicate code chunks within a buffer using MinHash + LSH.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
            "threshold": {
                "type": "number",
                "description": "Jaccard similarity threshold (0.0–1.0).",
                "default": 0.85,
            },
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "duplicates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_a": {"type": "string"},
                        "start_line_a": {"type": "integer"},
                        "end_line_a": {"type": "integer"},
                        "file_b": {"type": "string"},
                        "start_line_b": {"type": "integer"},
                        "end_line_b": {"type": "integer"},
                        "similarity": {"type": "number"},
                    },
                },
            },
            "total": {"type": "integer"},
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}

PACK_CONTEXT_SCHEMA: dict[str, Any] = {
    "name": "pack_context",
    "description": (
        "Return an optimally packed set of chunks fitting within a token budget. "
        "Uses hybrid search for relevance and greedily packs by score."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
            "query": {"type": "string", "description": "Query describing the context needed."},
            "max_tokens": {
                "type": "integer",
                "description": "Target token budget.",
                "default": 8192,
            },
            "top_k": {
                "type": "integer",
                "description": "Number of candidate chunks from hybrid search.",
                "default": 20,
            },
        },
        "required": ["buffer_id", "query"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "packed_chunks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "name": {"type": ["string", "null"]},
                        "type": {"type": "string"},
                        "score": {"type": "number"},
                        "tokens": {"type": "integer"},
                    },
                },
            },
            "total_tokens": {"type": "integer"},
            "remaining_tokens": {"type": "integer"},
            "count": {"type": "integer"},
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}


INFER_TYPES_SCHEMA: dict[str, Any] = {
    "name": "infer_types",
    "description": (
        "Infer type information for a symbol (parameter types, return type, confidence). "
        "Supports 'llm' method (accurate, ~50-300ms, includes confidence) or 'ast' method (fast, ~1-5ms)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
            "symbol": {
                "type": "string",
                "description": "Symbol name to infer types for.",
            },
            "method": {
                "type": "string",
                "enum": ["llm", "ast"],
                "description": "Type inference method: 'llm' (accurate) or 'ast' (fast). Default: 'llm'.",
                "default": "llm",
            },
        },
        "required": ["buffer_id", "symbol"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "symbol": {"type": "string"},
            "parameters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                    },
                },
            },
            "return_type": {"type": ["string", "null"]},
            "is_async": {"type": "boolean"},
            "signature": {"type": "string"},
            "type_confidence": {"type": ["number", "null"], "description": "Confidence 0.0-1.0 (LLM only)"},
            "method": {"type": "string", "enum": ["llm", "ast"]},
            "cached": {"type": "boolean", "description": "Whether result was from type inference cache"},
        },
        "required": ["status"],
    },
}


# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------

GET_SYMBOL_METADATA_SCHEMA: dict[str, Any] = {
    "name": "get_symbol_metadata",
    "description": (
        "Get comprehensive metadata for a symbol: type, parameters, return type, "
        "lines of code, cyclomatic complexity, caller/callee counts, docstring, "
        "and optional type confidence scores."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
            "symbol": {
                "type": "string",
                "description": "Symbol name (may be qualified: 'ClassName.method_name').",
            },
            "include_types": {
                "type": "boolean",
                "description": "Include inferred type hints in metadata (default: true).",
                "default": True,
            },
            "type_inference_method": {
                "type": "string",
                "enum": ["llm", "ast"],
                "description": "Type inference method: 'llm' (accurate) or 'ast' (fast). Default: 'ast'.",
                "default": "ast",
            },
        },
        "required": ["buffer_id", "symbol"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "name": {"type": "string"},
            "file": {"type": "string"},
            "line": {"type": "integer"},
            "end_line": {"type": "integer"},
            "type": {"type": "string", "description": "Symbol type (function, class, method, etc.)"},
            "lines_of_code": {"type": "integer"},
            "cyclomatic_complexity": {"type": "integer"},
            "called_by_count": {"type": "integer"},
            "calls_count": {"type": "integer"},
            "docstring": {"type": ["string", "null"]},
            "parameters": {
                "type": "array",
                "items": {"type": "object", "properties": {"name": {"type": "string"}, "type": {"type": "string"}}},
            },
            "return_type": {"type": ["string", "null"]},
            "type_confidence": {"type": ["number", "null"]},
            "inference_method": {"type": "string", "enum": ["llm", "ast"]},
            "parent": {"type": ["string", "null"]},
        },
        "required": ["status"],
    },
}

ALL_SCHEMAS: list[dict[str, Any]] = [
    EMBED_CODEBASE_SCHEMA,
    SEMANTIC_SEARCH_SCHEMA,
    HYBRID_SEARCH_SCHEMA,
    SEARCH_FOR_SCHEMA,
    SEARCH_SYMBOLS_SCHEMA,
    CLUSTER_CODE_SCHEMA,
    FIND_DUPLICATES_SCHEMA,
    PACK_CONTEXT_SCHEMA,
    RELOAD_CODEBASE_SCHEMA,
    CHECK_CODEBASE_SCHEMA,
    LIST_BUFFERS_SCHEMA,
    DELETE_BUFFER_SCHEMA,
    READ_CODE_SCHEMA,
    LOOK_FOR_FILE_SCHEMA,
    WRITE_CODE_SCHEMA,
    DIFF_SCHEMA,
    DISCARD_SCHEMA,
    COMMIT_SCHEMA,
    INFER_TYPES_SCHEMA,
    GET_SYMBOL_METADATA_SCHEMA,
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
