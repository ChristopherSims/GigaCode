"""Formal JSON schemas for the GigaCode agent tool interface.

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
    "SEARCH_BATCH_SCHEMA",
    "AUTO_FORMAT_SCHEMA",
    "AUTO_LINT_SCHEMA",
    "AUTO_POLISH_SCHEMA",
    "GET_REFERENCES_SCHEMA",
    "GET_FULL_CONTEXT_SCHEMA",
    "ANALYZE_CHANGE_SCHEMA",
    "GET_TEST_COVERAGE_SCHEMA",
    "POLISH_BEFORE_COMMIT_SCHEMA",
    "TRACE_EXECUTION_PATHS_SCHEMA",
    "GET_DEPENDENCY_GRAPH_SCHEMA",
    "DETECT_CODE_SMELLS_SCHEMA",
    "SCAN_SECURITY_SCHEMA",
    "SUGGEST_REFACTORINGS_SCHEMA",
    "LINT_BUFFER_SCHEMA",
    "FORMAT_BUFFER_SCHEMA",
    "FIND_PERFORMANCE_HOTSPOTS_SCHEMA",
    "GENERATE_DOCUMENTATION_SCHEMA",
    "FIND_SIMILAR_PATTERNS_SCHEMA",
    "FIND_DEPRECATED_SCHEMA",
    "VALIDATE_CHANGES_SCHEMA",
    "EXTRACT_CONFIGURATION_SCHEMA",
    "ANALYZE_LOGGING_PATTERNS_SCHEMA",
    "ANALYZE_ERROR_HANDLING_SCHEMA",
    "GENERATE_CHANGELOG_SCHEMA",
    "DETECT_API_CHANGES_SCHEMA",
    "GET_ROLLBACK_INFO_SCHEMA",
    "GENERATE_CHANGE_TEMPLATE_SCHEMA",
    "MAP_API_ENDPOINTS_SCHEMA",
    "ANALYZE_CACHE_PATTERNS_SCHEMA",
    "ANALYZE_THREAD_SAFETY_SCHEMA",
    "DETECT_MEMORY_ISSUES_SCHEMA",
    "LINT_WITH_CONFIG_SCHEMA",
    "FORMAT_WITH_CONFIG_SCHEMA",
    "ALL_SCHEMAS",
    "TOOL_CATEGORIES",
    "SchemaFormat",
    "SchemaConfig",
    "get_schema",
    "get_all_schemas",
    "get_schemas_by_category",
    "get_read_only_tools",
    "get_write_tools",
    "to_openai_functions",
    "to_anthropic_tools",
    "to_ollama_tools",
    "to_mcp_tools",
    "export_schemas",
    "export_schemas_from_config",
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
                    "Examples: 'gigacode_tool.py', 'gigacode'"
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

SEARCH_BATCH_SCHEMA: dict[str, Any] = {
    "name": "search_batch",
    "description": (
        "Search multiple queries in one call. Embeds all queries in parallel, "
        "then searches for each independently. Returns a dict mapping each query "
        "to its results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {
                "type": "string",
                "description": "Buffer handle returned by embed_codebase.",
            },
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of natural language search queries (max 20).",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top results per query (default 5).",
                "default": 5,
            },
            "include_types": {
                "type": "boolean",
                "description": "Include inferred type hints in results (default: false).",
                "default": False,
            },
            "type_inference_method": {
                "type": "string",
                "enum": ["llm", "ast"],
                "description": "Type inference method: 'llm' (accurate) or 'ast' (fast). Default: 'llm'.",
                "default": "llm",
            },
        },
        "required": ["buffer_id", "queries"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "results": {
                "type": "object",
                "description": "Dict mapping query strings to arrays of search matches.",
            },
            "query_count": {"type": "integer"},
        },
        "required": ["status"],
    },
}

AUTO_FORMAT_SCHEMA: dict[str, Any] = {
    "name": "auto_format",
    "description": (
        "Format code using Black or ruff format. Operates on entire "
        "buffer directory by default. Use dry_run=True to preview changes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle."},
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific files to format. If null, format entire directory.",
            },
            "formatter": {
                "type": "string",
                "enum": ["black", "ruff.format"],
                "description": "Formatter to use. Default: 'black'.",
                "default": "black",
            },
            "line_length": {"type": "integer", "description": "Max line length. Default: 88.", "default": 88},
            "skip_magic_trailing_comma": {"type": "boolean", "default": False},
            "dry_run": {"type": "boolean", "description": "Preview only. Default: true.", "default": True},
            "exclude_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Glob patterns to exclude.",
            },
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "formatter": {"type": "string"},
            "formatted_files": {"type": "integer"},
            "already_formatted": {"type": "integer"},
            "changes": {"type": "array"},
            "summary": {"type": "string"},
        },
        "required": ["status"],
    },
}

AUTO_LINT_SCHEMA: dict[str, Any] = {
    "name": "auto_lint",
    "description": (
        "Lint code using Ruff. Operates on entire buffer directory by default. "
        "Optionally auto-fix fixable issues."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle."},
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific files to lint. If null, lint entire directory.",
            },
            "linter": {"type": "string", "description": "Linter (only 'ruff'). Default: 'ruff'.", "default": "ruff"},
            "select": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Rule categories (e.g., ['E', 'F', 'W']).",
            },
            "ignore": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Rule codes to ignore (e.g., ['E501']).",
            },
            "auto_fix": {"type": "boolean", "description": "Auto-fix fixable issues. Default: false.", "default": False},
            "dry_run": {"type": "boolean", "description": "Preview only. Default: true.", "default": True},
            "exclude_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Glob patterns to exclude.",
            },
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "linter": {"type": "string"},
            "files_with_issues": {"type": "integer"},
            "total_issues": {"type": "integer"},
            "issues": {"type": "array"},
            "fixed_count": {"type": "integer"},
            "unfixed_count": {"type": "integer"},
            "by_rule": {"type": "object"},
        },
        "required": ["status"],
    },
}

AUTO_POLISH_SCHEMA: dict[str, Any] = {
    "name": "auto_polish",
    "description": (
        "Format AND lint in one call. Convenience wrapper that delegates to "
        "auto_format then auto_lint. Format runs first so lint checks formatted code."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle."},
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific files. If null, polish entire directory.",
            },
            "format_with": {
                "type": "string",
                "enum": ["black", "ruff.format"],
                "description": "Formatter. Default: 'black'.",
                "default": "black",
            },
            "lint_with": {"type": "string", "description": "Linter. Default: 'ruff'.", "default": "ruff"},
            "auto_fix_lints": {"type": "boolean", "description": "Auto-fix fixable lint issues. Default: true.", "default": True},
            "line_length": {"type": "integer", "description": "Max line length. Default: 88.", "default": 88},
            "ruff_select": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ruff rule categories.",
            },
            "exclude_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Glob patterns to exclude.",
            },
            "dry_run": {"type": "boolean", "description": "Preview only. Default: true.", "default": True},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "formatting": {"type": "object"},
            "linting": {"type": "object"},
            "ready_to_commit": {"type": "boolean"},
            "summary": {"type": "string"},
        },
        "required": ["status"],
    },
}

GET_REFERENCES_SCHEMA: dict[str, Any] = {
    "name": "get_references",
    "description": (
        "Find all callers and callees for a symbol using an incremental "
        "reference map. Lazy on-demand construction with caching. Optionally "
        "expand to deeper call chains with expand_depth."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle."},
            "symbol": {"type": "string", "description": "Symbol name to find references for."},
            "direction": {
                "type": "string",
                "enum": ["both", "calls", "called_by"],
                "description": "Direction: 'both' (default), 'calls' (callees), 'called_by' (callers).",
                "default": "both",
            },
            "top_k": {"type": "integer", "description": "Max references per direction. Default: 50.", "default": 50},
            "expand_depth": {
                "type": ["integer", "null"],
                "description": "If set, expand call chain to this depth (on-demand fill).",
            },
        },
        "required": ["buffer_id", "symbol"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "symbol": {"type": "string"},
            "file": {"type": "string"},
            "line": {"type": "integer"},
            "callers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "symbol": {"type": "string"},
                        "context": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium"]},
                        "via": {"type": "string", "description": "Intermediate symbol (for expanded depth > 1)"},
                        "depth": {"type": "integer", "description": "Depth level (1 = direct)"},
                    },
                },
            },
            "callees": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "symbol": {"type": "string"},
                        "context": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium"]},
                        "via": {"type": "string"},
                        "depth": {"type": "integer"},
                    },
                },
            },
            "direction": {"type": "string"},
            "depth": {"type": "integer"},
            "cached": {"type": "boolean"},
        },
        "required": ["status"],
    },
}

GET_FULL_CONTEXT_SCHEMA: dict[str, Any] = {
    "name": "get_full_context",
    "description": (
        "Get everything about a symbol in one call: definition, callers, callees, "
        "type hints, tests, related code, and error handling. Single roundtrip "
        "instead of 5+ API calls."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle."},
            "symbol": {"type": "string", "description": "Symbol name to get full context for."},
            "include": {
                "type": "array",
                "items": {"type": "string", "enum": ["definition", "callers", "callees", "tests", "related_code", "type_hints", "errors"]},
                "description": "Sections to include. Default: all.",
            },
            "type_inference_method": {
                "type": "string",
                "enum": ["llm", "ast"],
                "description": "Type inference method. Default: 'llm'.",
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
            "definition": {"type": "object", "description": "Symbol definition with source code."},
            "callers": {"type": "array", "description": "Symbols that call this symbol."},
            "callees": {"type": "array", "description": "Symbols called by this symbol."},
            "types": {"type": "object", "description": "Inferred type information."},
            "tests": {"type": "array", "description": "Test files/functions related to this symbol."},
            "related_code": {"type": "array", "description": "Semantically related code."},
            "errors": {"type": "array", "description": "Error handling patterns involving this symbol."},
        },
        "required": ["status"],
    },
}

ANALYZE_CHANGE_SCHEMA: dict[str, Any] = {
    "name": "analyze_change",
    "description": (
        "Analyze impact of a proposed change before editing. Reports direct callers, "
        "test coverage, dependent symbols, and files affected."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle."},
            "file": {"type": "string", "description": "File that would be modified."},
            "start_line": {"type": ["integer", "null"], "description": "Start line of proposed change (1-based)."},
            "end_line": {"type": ["integer", "null"], "description": "End line of proposed change (1-based)."},
            "max_depth": {"type": "integer", "description": "Max call-chain depth. Default: 6.", "default": 6},
        },
        "required": ["buffer_id", "file"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "file": {"type": "string"},
            "affected_symbols": {"type": "array", "items": {"type": "string"}},
            "direct_callers": {"type": "array"},
            "dependent_symbols": {"type": "integer"},
            "files_affected": {"type": "integer"},
            "impacted_files": {"type": "array", "items": {"type": "string"}},
            "test_coverage": {"type": "array"},
            "has_tests": {"type": "boolean"},
            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
            "risk_score": {"type": "number"},
        },
        "required": ["status"],
    },
}

GET_TEST_COVERAGE_SCHEMA: dict[str, Any] = {
    "name": "get_test_coverage",
    "description": (
        "Get test coverage map for the codebase. Maps each source file "
        "to line ranges and the test functions that cover them."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle."},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "coverage": {
                "type": "object",
                "description": "Map of source file -> {line_range -> [test_names]}.",
            },
        },
        "required": ["status"],
    },
}

POLISH_BEFORE_COMMIT_SCHEMA: dict[str, Any] = {
    "name": "polish_before_commit",
    "description": (
        "Format, lint, and validate before committing. Convenience wrapper "
        "that chains auto_polish + commit-readiness checks (test coverage, "
        "impact analysis warnings)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle."},
            "files_to_commit": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific files. If null, all dirty files.",
            },
            "format_with": {
                "type": "string",
                "enum": ["black", "ruff.format"],
                "description": "Formatter. Default: 'black'.",
                "default": "black",
            },
            "lint_with": {"type": "string", "description": "Linter. Default: 'ruff'.", "default": "ruff"},
            "check_only": {"type": "boolean", "description": "Only validate, don't modify. Default: false.", "default": False},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "formatting": {"type": "object"},
            "linting": {"type": "object"},
            "ready_to_commit": {"type": "boolean"},
            "pre_commit_warnings": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["status"],
    },
}

TRACE_EXECUTION_PATHS_SCHEMA: dict[str, Any] = {
    "name": "trace_execution_paths",
    "description": "Trace all execution paths through a symbol using AST branch detection.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "symbol": {"type": "string", "description": "Symbol name to analyze."},
            "max_depth": {"type": "integer", "description": "Maximum call depth to trace.", "default": 3},
        },
        "required": ["buffer_id", "symbol"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "symbol": {"type": "string"},
            "paths": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "array", "items": {"type": "string"}},
                        "branches": {"type": "integer"},
                        "calls": {"type": "array", "items": {"type": "string"}},
                        "conditions": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "path_count": {"type": "integer"},
        },
        "required": ["status"],
    },
}

GET_DEPENDENCY_GRAPH_SCHEMA: dict[str, Any] = {
    "name": "get_dependency_graph",
    "description": "Get dependency graph visualization data (nodes + edges).",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "symbol": {"type": ["string", "null"], "description": "Optional symbol to scope graph around."},
            "depth": {"type": "integer", "description": "Depth of dependency graph to include.", "default": 2},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "type": {"type": "string", "description": "chunk type: function/class/module"},
                        "file": {"type": "string"},
                    },
                },
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string", "description": "source node id"},
                        "to": {"type": "string", "description": "target node id"},
                        "type": {"type": "string", "enum": ["calls", "imports"]},
                    },
                },
            },
        },
        "required": ["status"],
    },
}

DETECT_CODE_SMELLS_SCHEMA: dict[str, Any] = {
    "name": "detect_code_smells",
    "description": "Detect code smells: long functions, deep nesting, missing docstrings, complex logic, too many params.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "types": {
                "type": "array",
                "items": {"type": "string", "enum": ["long_function", "deep_nesting", "missing_docstring", "complex_logic", "too_many_params", "duplicates"]},
                "description": "Smell types to detect. Options: long_function, deep_nesting, missing_docstring, complex_logic, too_many_params, duplicates.",
            },
            "severity_min": {"type": "string", "enum": ["low", "medium", "high"], "description": "Minimum severity to report: low, medium, or high.", "default": "low"},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "smells": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "type": {"type": "string", "description": "long_function|deep_nesting|missing_docstring|complex_logic|too_many_params"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "suggestion": {"type": "string"},
                    },
                },
            },
            "total": {"type": "integer"},
        },
        "required": ["status"],
    },
}

SCAN_SECURITY_SCHEMA: dict[str, Any] = {
    "name": "scan_security",
    "description": "Scan for security vulnerabilities: eval, exec, shell injection, SQL injection, hardcoded secrets, unsafe pickle/yaml.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "severity_min": {"type": "string", "enum": ["low", "medium", "high"], "description": "Minimum severity to report: low, medium, or high.", "default": "medium"},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "vulnerabilities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "type": {"type": "string", "description": "eval_usage|exec_usage|shell_injection|os_system|unsafe_pickle|unsafe_yaml|sql_injection|hardcoded_secret|assert_usage|broad_except|wildcard_import"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "context": {"type": "string"},
                        "fix_suggestion": {"type": "string"},
                    },
                },
            },
            "total": {"type": "integer"},
        },
        "required": ["status"],
    },
}

SUGGEST_REFACTORINGS_SCHEMA: dict[str, Any] = {
    "name": "suggest_refactorings",
    "description": "Suggest safe refactorings for a symbol with risk assessment.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "symbol": {"type": "string", "description": "Symbol name to analyze."},
        },
        "required": ["buffer_id", "symbol"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "symbol": {"type": "string"},
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "description": "extract_method|simplify_branches|consolidate_calls|add_type_hints|use_guard_clauses"},
                        "lines": {"type": "string", "description": "line range like '10-20'"},
                        "symbol": {"type": "string", "description": "related symbol name"},
                        "benefit": {"type": "string"},
                        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                },
            },
        },
        "required": ["status"],
    },
}

LINT_BUFFER_SCHEMA: dict[str, Any] = {
    "name": "lint_buffer",
    "description": "Deep lint analysis with detailed aggregation by file/severity/rule. Report-only, no auto-fix.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "files": {"type": "array", "items": {"type": "string"}, "description": "Specific files to analyze. If None, analyzes entire buffer."},
            "select": {"type": "array", "items": {"type": "string"}, "description": "Lint rule categories to check (e.g. E, F, W)."},
            "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Glob patterns to exclude from analysis."},
            "group_by": {"type": "string", "enum": ["file", "severity", "rule"], "description": "How to organize results: by file, severity, or rule.", "default": "file"},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "total_issues": {"type": "integer"},
            "by_file": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line": {"type": "integer"},
                            "code": {"type": "string"},
                            "message": {"type": "string"},
                        },
                    },
                },
            },
            "by_severity": {
                "type": "object",
                "properties": {
                    "error": {"type": "integer"},
                    "warning": {"type": "integer"},
                    "info": {"type": "integer"},
                },
            },
            "by_rule": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "severity": {"type": "string"},
                    },
                },
            },
        },
        "required": ["status"],
    },
}

FORMAT_BUFFER_SCHEMA: dict[str, Any] = {
    "name": "format_buffer",
    "description": "Deep format analysis with detailed change tracking across codebase.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "files": {"type": "array", "items": {"type": "string"}, "description": "Specific files to analyze. If None, analyzes entire buffer."},
            "formatter": {"type": "string", "enum": ["black", "ruff.format"], "description": "Formatter to use: black or ruff.format.", "default": "black"},
            "line_length": {"type": "integer", "description": "Maximum line length for formatting.", "default": 88},
            "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Glob patterns to exclude from analysis."},
            "dry_run": {"type": "boolean", "description": "Preview only without making changes.", "default": True},
            "summary_only": {"type": "boolean", "description": "Return only summary statistics, not full diffs.", "default": False},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "total_files": {"type": "integer"},
            "formatted_files": {"type": "integer"},
            "already_formatted": {"type": "integer"},
            "total_lines_added": {"type": "integer"},
            "total_lines_removed": {"type": "integer"},
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "added_lines": {"type": "integer"},
                        "removed_lines": {"type": "integer"},
                        "diff": {"type": "string"},
                    },
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["status"],
    },
}

FIND_PERFORMANCE_HOTSPOTS_SCHEMA: dict[str, Any] = {
    "name": "find_performance_hotspots",
    "description": "Detect performance hotspots: N+1 queries, inefficient loops, unbounded recursion.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "hotspots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "type": {"type": "string", "description": "n_plus_one|unbounded_growth|missing_prefetch|inefficient_loop|slow_serialization|regex_in_loop|unclosed_file|full_table_scan|nested_loop"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "context": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                },
            },
            "total": {"type": "integer"},
        },
        "required": ["status"],
    },
}

GENERATE_DOCUMENTATION_SCHEMA: dict[str, Any] = {
    "name": "generate_documentation",
    "description": "Auto-generate documentation from code analysis.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "symbol": {"type": "string", "description": "Symbol name to analyze."},
            "style": {"type": "string", "enum": ["google", "numpy", "sphinx"], "description": "Docstring style: google, numpy, or sphinx.", "default": "google"},
        },
        "required": ["buffer_id", "symbol"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "symbol": {"type": "string"},
            "docstring": {"type": "string", "description": "Generated docstring ready to insert"},
            "type_hints": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Map of parameter name to inferred type",
            },
            "examples": {"type": "array", "items": {"type": "string"}, "description": "Usage examples found in test files"},
            "generated_from_code": {"type": "boolean"},
            "style": {"type": "string"},
        },
        "required": ["status"],
    },
}

FIND_SIMILAR_PATTERNS_SCHEMA: dict[str, Any] = {
    "name": "find_similar_patterns",
    "description": "Find similar code patterns using semantic + syntactic matching.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "code_snippet": {"type": "string", "description": "Code snippet to find similar patterns for."},
            "min_similarity": {"type": "number", "description": "Minimum Jaccard similarity threshold (0.0-1.0).", "default": 0.7},
            "top_k": {"type": "integer", "description": "Maximum number of results to return.", "default": 10},
        },
        "required": ["buffer_id", "code_snippet"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "syntactic_matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "similarity": {"type": "number"},
                    },
                },
            },
            "semantic_matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "score": {"type": "number"},
                    },
                },
            },
            "snippet_length": {"type": "integer"},
        },
        "required": ["status"],
    },
}

FIND_DEPRECATED_SCHEMA: dict[str, Any] = {
    "name": "find_deprecated",
    "description": "Detect usage of deprecated functions and APIs.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "deprecated": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "detection_method": {"type": "string", "description": "decorator|warning|comment|deprecation_warning"},
                        "context": {"type": "string"},
                        "symbol": {"type": "string"},
                    },
                },
            },
            "total": {"type": "integer"},
        },
        "required": ["status"],
    },
}

VALIDATE_CHANGES_SCHEMA: dict[str, Any] = {
    "name": "validate_changes",
    "description": "Validate changes before committing (static analysis + import resolution).",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "dry_run": {"type": "boolean", "description": "Preview only without making changes.", "default": True},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "type_errors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "message": {"type": "string"},
                    },
                },
            },
            "broken_imports": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "import": {"type": "string"},
                        "message": {"type": "string"},
                    },
                },
            },
            "test_impact_predictions": {"type": "array", "items": {"type": "string"}},
            "safe_to_commit": {"type": "boolean"},
            "dry_run": {"type": "boolean"},
        },
        "required": ["status"],
    },
}

EXTRACT_CONFIGURATION_SCHEMA: dict[str, Any] = {
    "name": "extract_configuration",
    "description": "Extract configuration: env vars, config files, hardcoded secrets, defaults.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "env_vars": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "used_in": {"type": "string", "description": "file:line reference"},
                        "required": {"type": "boolean"},
                        "default": {"type": ["string", "null"]},
                    },
                },
            },
            "config_files": {"type": "array", "items": {"type": "string"}},
            "hardcoded_secrets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "pattern": {"type": "string"},
                        "severity": {"type": "string", "enum": ["high"]},
                    },
                },
            },
            "default_values": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Map of env var name to default value",
            },
        },
        "required": ["status"],
    },
}

ANALYZE_LOGGING_PATTERNS_SCHEMA: dict[str, Any] = {
    "name": "analyze_logging_patterns",
    "description": "Analyze logging patterns: levels, consistency, gaps.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "total_logs": {"type": "integer"},
            "levels": {
                "type": "object",
                "properties": {
                    "debug": {"type": "integer"},
                    "info": {"type": "integer"},
                    "warning": {"type": "integer"},
                    "error": {"type": "integer"},
                    "critical": {"type": "integer"},
                },
            },
            "missing_logs_in": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "symbol": {"type": "string"},
                        "issue": {"type": "string"},
                    },
                },
            },
            "inconsistent_format": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "format_string": {"type": "string"},
                    },
                },
            },
            "patterns_detected": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["status"],
    },
}

ANALYZE_ERROR_HANDLING_SCHEMA: dict[str, Any] = {
    "name": "analyze_error_handling_patterns",
    "description": "Analyze error handling patterns: broad catches, missing finally, uncaught exceptions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "try_except_blocks": {"type": "integer"},
            "uncaught_exceptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "issue": {"type": "string"},
                    },
                },
            },
            "broad_catches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "catches": {"type": "string"},
                    },
                },
            },
            "missing_finally": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "resource": {"type": "string"},
                    },
                },
            },
            "suggestions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["status"],
    },
}

GENERATE_CHANGELOG_SCHEMA: dict[str, Any] = {
    "name": "generate_changelog",
    "description": "Generate changelog from git history + semantic analysis.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "since_commit": {"type": ["string", "null"], "description": "Git commit hash to compare against. If None, compares against HEAD."},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "commit": {"type": "string"},
                        "message": {"type": "string"},
                    },
                },
            },
            "bugfixes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "commit": {"type": "string"},
                        "message": {"type": "string"},
                    },
                },
            },
            "breaking_changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "commit": {"type": "string"},
                        "message": {"type": "string"},
                    },
                },
            },
            "other": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "commit": {"type": "string"},
                        "message": {"type": "string"},
                    },
                },
            },
            "migration_notes": {"type": "string"},
        },
        "required": ["status"],
    },
}

DETECT_API_CHANGES_SCHEMA: dict[str, Any] = {
    "name": "detect_api_changes",
    "description": "Detect API-breaking changes between commits.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "since_commit": {"type": ["string", "null"], "description": "Git commit hash to compare against. If None, compares against HEAD."},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "current_api_surface": {"type": "integer"},
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "breaking": {"type": "boolean"},
                        "parameters_added": {"type": "array", "items": {"type": "string"}},
                        "return_type_changed": {"type": "boolean"},
                        "migration_guide": {"type": "string"},
                    },
                },
            },
            "symbols": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "params": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
        "required": ["status"],
    },
}

GET_ROLLBACK_INFO_SCHEMA: dict[str, Any] = {
    "name": "get_rollback_info",
    "description": "Get rollback information for a file.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "file": {"type": "string", "description": "File path within the buffer."},
        },
        "required": ["buffer_id", "file"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "last_working_commit": {"type": ["string", "null"]},
            "commit_message": {"type": "string"},
            "commit_date": {"type": "string"},
            "diff_to_revert": {"type": "string", "description": "Unified diff to revert to last working state"},
        },
        "required": ["status"],
    },
}

GENERATE_CHANGE_TEMPLATE_SCHEMA: dict[str, Any] = {
    "name": "generate_change_template",
    "description": "Generate a change plan template for a request.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "request": {"type": "string", "description": "Natural language description of the desired change."},
        },
        "required": ["buffer_id", "request"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "request": {"type": "string"},
            "files_to_modify": {"type": "array", "items": {"type": "string"}},
            "change_strategy": {"type": "string"},
            "test_cases_needed": {"type": "array", "items": {"type": "string"}},
            "risk_assessment": {"type": "string", "description": "low, medium, or high"},
        },
        "required": ["status"],
    },
}

MAP_API_ENDPOINTS_SCHEMA: dict[str, Any] = {
    "name": "map_api_endpoints",
    "description": "Map all API endpoints (FastAPI, Flask, Django patterns).",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "endpoints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string", "description": "HTTP method"},
                        "path": {"type": "string"},
                        "handler": {"type": "string"},
                        "is_async": {"type": "boolean"},
                        "file": {"type": "string"},
                    },
                },
            },
            "total": {"type": "integer"},
        },
        "required": ["status"],
    },
}

ANALYZE_CACHE_PATTERNS_SCHEMA: dict[str, Any] = {
    "name": "analyze_cache_patterns",
    "description": "Analyze cache usage patterns: invalidation logic, stale data risks.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "caches_used": {"type": "array", "items": {"type": "string"}, "description": "Cache libraries detected: redis, memcache, lru_cache, etc."},
            "invalidation_logic": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "pattern": {"type": "string"},
                        "safe": {"type": "boolean"},
                    },
                },
            },
            "stale_data_risks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "risk_level": {"type": "string"},
                    },
                },
            },
        },
        "required": ["status"],
    },
}

ANALYZE_THREAD_SAFETY_SCHEMA: dict[str, Any] = {
    "name": "analyze_thread_safety",
    "description": "Analyze thread safety: shared state, race conditions, deadlock risks.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "shared_state": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "file": {"type": "string"},
                        "modified_by": {"type": "array", "items": {"type": "string"}},
                        "protected_by": {"type": "string", "description": "lock|atomic|none"},
                    },
                },
            },
            "race_conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "variables": {"type": "array", "items": {"type": "string"}},
                        "risk_level": {"type": "string"},
                    },
                },
            },
            "deadlock_risks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "locks_count": {"type": "integer"},
                        "suggestion": {"type": "string"},
                    },
                },
            },
        },
        "required": ["status"],
    },
}

DETECT_MEMORY_ISSUES_SCHEMA: dict[str, Any] = {
    "name": "detect_memory_issues",
    "description": "Detect memory issues: circular refs, unbounded collections, resource leaks.",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "circular_refs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "symbols": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "unbounded_collections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "symbol": {"type": "string"},
                        "growth_reason": {"type": "string"},
                    },
                },
            },
            "resource_leaks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "resource": {"type": "string"},
                        "cleanup_missing": {"type": "boolean"},
                    },
                },
            },
        },
        "required": ["status"],
    },
}

LINT_WITH_CONFIG_SCHEMA: dict[str, Any] = {
    "name": "lint_with_config",
    "description": "Lint using project configuration (ruff.toml, pyproject.toml).",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "config_file": {"type": ["string", "null"], "description": "Path to config file (ruff.toml, pyproject.toml). Auto-detected if None."},
            "files": {"type": "array", "items": {"type": "string"}, "description": "Specific files to analyze. If None, analyzes entire buffer."},
            "auto_fix": {"type": "boolean", "description": "Automatically fix lint issues instead of just reporting.", "default": False},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "config_file": {"type": "string", "description": "Config file used (auto-detected or specified)"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "severity": {"type": "string"},
                    },
                },
            },
        },
        "required": ["status"],
    },
}

FORMAT_WITH_CONFIG_SCHEMA: dict[str, Any] = {
    "name": "format_with_config",
    "description": "Format using project configuration (pyproject.toml, .black, ruff.toml).",
    "input_schema": {
        "type": "object",
        "properties": {
            "buffer_id": {"type": "string", "description": "Buffer handle returned by embed_codebase."},
            "config_file": {"type": ["string", "null"], "description": "Path to config file (ruff.toml, pyproject.toml). Auto-detected if None."},
            "files": {"type": "array", "items": {"type": "string"}, "description": "Specific files to analyze. If None, analyzes entire buffer."},
            "dry_run": {"type": "boolean", "description": "Preview only without making changes.", "default": True},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "config_file": {"type": "string", "description": "Config file used (auto-detected or specified)"},
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "added_lines": {"type": "integer"},
                        "removed_lines": {"type": "integer"},
                        "diff": {"type": "string"},
                    },
                },
            },
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
    SEARCH_BATCH_SCHEMA,
    AUTO_FORMAT_SCHEMA,
    AUTO_LINT_SCHEMA,
    AUTO_POLISH_SCHEMA,
    GET_REFERENCES_SCHEMA,
    GET_FULL_CONTEXT_SCHEMA,
    ANALYZE_CHANGE_SCHEMA,
    GET_TEST_COVERAGE_SCHEMA,
    POLISH_BEFORE_COMMIT_SCHEMA,
    TRACE_EXECUTION_PATHS_SCHEMA,
    GET_DEPENDENCY_GRAPH_SCHEMA,
    DETECT_CODE_SMELLS_SCHEMA,
    SCAN_SECURITY_SCHEMA,
    SUGGEST_REFACTORINGS_SCHEMA,
    LINT_BUFFER_SCHEMA,
    FORMAT_BUFFER_SCHEMA,
    FIND_PERFORMANCE_HOTSPOTS_SCHEMA,
    GENERATE_DOCUMENTATION_SCHEMA,
    FIND_SIMILAR_PATTERNS_SCHEMA,
    FIND_DEPRECATED_SCHEMA,
    VALIDATE_CHANGES_SCHEMA,
    EXTRACT_CONFIGURATION_SCHEMA,
    ANALYZE_LOGGING_PATTERNS_SCHEMA,
    ANALYZE_ERROR_HANDLING_SCHEMA,
    GENERATE_CHANGELOG_SCHEMA,
    DETECT_API_CHANGES_SCHEMA,
    GET_ROLLBACK_INFO_SCHEMA,
    GENERATE_CHANGE_TEMPLATE_SCHEMA,
    MAP_API_ENDPOINTS_SCHEMA,
    ANALYZE_CACHE_PATTERNS_SCHEMA,
    ANALYZE_THREAD_SAFETY_SCHEMA,
    DETECT_MEMORY_ISSUES_SCHEMA,
    LINT_WITH_CONFIG_SCHEMA,
    FORMAT_WITH_CONFIG_SCHEMA,
]

# ---------------------------------------------------------------------------
# AI Discoverability Enrichment (auto-applied at import time)
# ---------------------------------------------------------------------------

# 1. Category + Tags — tool categorization for AI filtering
_SCHEMA_CATEGORIES: dict[str, dict[str, Any]] = {
    "embed_codebase":            {"category": "indexing",    "tags": ["write", "slow", "setup"]},
    "semantic_search":           {"category": "search",      "tags": ["read-only", "fast"]},
    "hybrid_search":             {"category": "search",      "tags": ["read-only", "fast"]},
    "search_for":                {"category": "search",      "tags": ["read-only", "fast"]},
    "search_symbols":            {"category": "search",      "tags": ["read-only", "fast"]},
    "cluster_code":              {"category": "search",      "tags": ["read-only", "slow"]},
    "find_duplicates":           {"category": "search",      "tags": ["read-only", "slow"]},
    "pack_context":              {"category": "search",      "tags": ["read-only", "fast"]},
    "reload_codebase":           {"category": "indexing",    "tags": ["write", "slow"]},
    "check_codebase":            {"category": "indexing",    "tags": ["read-only", "fast"]},
    "list_buffers":              {"category": "indexing",    "tags": ["read-only", "fast"]},
    "delete_buffer":             {"category": "indexing",    "tags": ["write", "destructive"]},
    "read_code":                 {"category": "navigation",  "tags": ["read-only", "fast"]},
    "look_for_file":             {"category": "navigation",  "tags": ["read-only", "fast"]},
    "write_code":                {"category": "editing",     "tags": ["write", "mutating", "slow"]},
    "diff":                      {"category": "editing",     "tags": ["read-only", "fast"]},
    "discard":                   {"category": "editing",     "tags": ["write", "destructive"]},
    "commit":                    {"category": "editing",     "tags": ["write", "mutating", "slow"]},
    "infer_types":               {"category": "analysis",    "tags": ["read-only", "medium"]},
    "get_symbol_metadata":       {"category": "navigation",  "tags": ["read-only", "fast"]},
    "search_batch":              {"category": "search",      "tags": ["read-only", "slow"]},
    "auto_format":               {"category": "quality",     "tags": ["write", "mutating", "medium"]},
    "auto_lint":                 {"category": "quality",     "tags": ["read-only", "medium"]},
    "auto_polish":               {"category": "quality",     "tags": ["write", "mutating", "medium"]},
    "get_references":            {"category": "navigation",  "tags": ["read-only", "fast"]},
    "get_full_context":          {"category": "navigation",  "tags": ["read-only", "medium"]},
    "analyze_change":            {"category": "safety",      "tags": ["read-only", "medium"]},
    "get_test_coverage":         {"category": "navigation",  "tags": ["read-only", "medium"]},
    "polish_before_commit":      {"category": "quality",     "tags": ["write", "mutating", "medium"]},
    "trace_execution_paths":     {"category": "analysis",    "tags": ["read-only", "medium"]},
    "get_dependency_graph":      {"category": "analysis",    "tags": ["read-only", "medium"]},
    "detect_code_smells":        {"category": "analysis",    "tags": ["read-only", "fast"]},
    "scan_security":             {"category": "security",    "tags": ["read-only", "fast"]},
    "suggest_refactorings":      {"category": "analysis",    "tags": ["read-only", "fast"]},
    "lint_buffer":               {"category": "quality",     "tags": ["read-only", "medium"]},
    "format_buffer":             {"category": "quality",     "tags": ["read-only", "medium"]},
    "find_performance_hotspots": {"category": "analysis",    "tags": ["read-only", "fast"]},
    "generate_documentation":    {"category": "analysis",    "tags": ["read-only", "fast"]},
    "find_similar_patterns":     {"category": "search",      "tags": ["read-only", "medium"]},
    "find_deprecated":           {"category": "analysis",    "tags": ["read-only", "fast"]},
    "validate_changes":          {"category": "safety",      "tags": ["read-only", "medium"]},
    "extract_configuration":     {"category": "analysis",    "tags": ["read-only", "fast"]},
    "analyze_logging_patterns":  {"category": "analysis",    "tags": ["read-only", "fast"]},
    "analyze_error_handling_patterns": {"category": "analysis", "tags": ["read-only", "fast"]},
    "generate_changelog":        {"category": "analysis",    "tags": ["read-only", "fast"]},
    "detect_api_changes":        {"category": "safety",      "tags": ["read-only", "fast"]},
    "get_rollback_info":         {"category": "safety",      "tags": ["read-only", "fast"]},
    "generate_change_template":  {"category": "safety",      "tags": ["read-only", "medium"]},
    "map_api_endpoints":         {"category": "analysis",    "tags": ["read-only", "fast"]},
    "analyze_cache_patterns":    {"category": "analysis",    "tags": ["read-only", "fast"]},
    "analyze_thread_safety":     {"category": "analysis",    "tags": ["read-only", "fast"]},
    "detect_memory_issues":      {"category": "analysis",    "tags": ["read-only", "fast"]},
    "lint_with_config":          {"category": "quality",     "tags": ["read-only", "medium"]},
    "format_with_config":        {"category": "quality",     "tags": ["write", "mutating", "medium"]},
}

# 2. Read-only + Side-effects — safety annotations
_SCHEMA_SIDE_EFFECTS: dict[str, dict[str, Any]] = {
    "embed_codebase":            {"read_only": False, "side_effects": "Creates a new buffer with embedded code; allocates GPU/CPU memory for embeddings index."},
    "semantic_search":           {"read_only": True,  "side_effects": None},
    "hybrid_search":             {"read_only": True,  "side_effects": None},
    "search_for":                {"read_only": True,  "side_effects": None},
    "search_symbols":            {"read_only": True,  "side_effects": None},
    "cluster_code":              {"read_only": True,  "side_effects": None},
    "find_duplicates":           {"read_only": True,  "side_effects": None},
    "pack_context":              {"read_only": True,  "side_effects": None},
    "reload_codebase":           {"read_only": False, "side_effects": "Re-embeds changed files; updates embeddings index in-place."},
    "check_codebase":            {"read_only": True,  "side_effects": None},
    "list_buffers":              {"read_only": True,  "side_effects": None},
    "delete_buffer":             {"read_only": False, "side_effects": "Permanently deletes the buffer and its embeddings from disk. Cannot be undone."},
    "read_code":                 {"read_only": True,  "side_effects": None},
    "look_for_file":             {"read_only": True,  "side_effects": None},
    "write_code":                {"read_only": False, "side_effects": "Modifies the in-buffer source snapshot. Changes are not written to disk until commit()."},
    "diff":                      {"read_only": True,  "side_effects": None},
    "discard":                   {"read_only": False, "side_effects": "Discards in-buffer changes, reverting to the last committed state."},
    "commit":                    {"read_only": False, "side_effects": "Writes all in-buffer changes to disk files. Irreversible — use dry_run=True first."},
    "infer_types":               {"read_only": True,  "side_effects": None},
    "get_symbol_metadata":       {"read_only": True,  "side_effects": None},
    "search_batch":              {"read_only": True,  "side_effects": None},
    "auto_format":               {"read_only": False, "side_effects": "Reformats source files on disk when dry_run=False. Use dry_run=True to preview."},
    "auto_lint":                 {"read_only": True,  "side_effects": "Report-only by default. auto_fix=True will modify source files on disk."},
    "auto_polish":               {"read_only": False, "side_effects": "Formats and lints files on disk when dry_run=False. Use dry_run=True to preview."},
    "get_references":            {"read_only": True,  "side_effects": None},
    "get_full_context":          {"read_only": True,  "side_effects": None},
    "analyze_change":            {"read_only": True,  "side_effects": None},
    "get_test_coverage":         {"read_only": True,  "side_effects": None},
    "polish_before_commit":      {"read_only": False, "side_effects": "Formats and lints files when check_only=False. Use check_only=True for preview."},
    "trace_execution_paths":     {"read_only": True,  "side_effects": None},
    "get_dependency_graph":      {"read_only": True,  "side_effects": None},
    "detect_code_smells":        {"read_only": True,  "side_effects": None},
    "scan_security":             {"read_only": True,  "side_effects": None},
    "suggest_refactorings":      {"read_only": True,  "side_effects": None},
    "lint_buffer":               {"read_only": True,  "side_effects": None},
    "format_buffer":             {"read_only": True,  "side_effects": None},
    "find_performance_hotspots": {"read_only": True,  "side_effects": None},
    "generate_documentation":    {"read_only": True,  "side_effects": None},
    "find_similar_patterns":     {"read_only": True,  "side_effects": None},
    "find_deprecated":           {"read_only": True,  "side_effects": None},
    "validate_changes":          {"read_only": True,  "side_effects": None},
    "extract_configuration":     {"read_only": True,  "side_effects": None},
    "analyze_logging_patterns":  {"read_only": True,  "side_effects": None},
    "analyze_error_handling_patterns": {"read_only": True, "side_effects": None},
    "generate_changelog":        {"read_only": True,  "side_effects": None},
    "detect_api_changes":        {"read_only": True,  "side_effects": None},
    "get_rollback_info":         {"read_only": True,  "side_effects": None},
    "generate_change_template":  {"read_only": True,  "side_effects": None},
    "map_api_endpoints":         {"read_only": True,  "side_effects": None},
    "analyze_cache_patterns":    {"read_only": True,  "side_effects": None},
    "analyze_thread_safety":     {"read_only": True,  "side_effects": None},
    "detect_memory_issues":      {"read_only": True,  "side_effects": None},
    "lint_with_config":          {"read_only": True,  "side_effects": "Report-only by default. auto_fix=True will modify source files on disk."},
    "format_with_config":        {"read_only": False, "side_effects": "Reformats source files when dry_run=False. Use dry_run=True to preview."},
}

# 3. Shared error schema — documents the error/conflict response shape
_SHARED_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Error response returned when status is 'error' or 'conflict'.",
    "properties": {
        "status": {"type": "string", "enum": ["error", "conflict"], "description": "Error or conflict status."},
        "message": {"type": "string", "description": "Human-readable error description."},
        "operation": {"type": "string", "description": "The tool operation that failed."},
        "buffer_id": {"type": "string", "description": "Buffer handle, if applicable."},
    },
    "required": ["status", "message"],
}

# 4. Composition hints — which tools delegate to or compose other tools
_SCHEMA_COMPOSITION: dict[str, dict[str, Any]] = {
    "search_batch":             {"delegates_to": ["semantic_search"]},
    "auto_polish":              {"composed_of": ["auto_format", "auto_lint"]},
    "polish_before_commit":     {"composed_of": ["auto_polish", "get_test_coverage", "analyze_change"]},
    "get_full_context":         {"composed_of": ["get_references", "get_symbol_metadata", "get_test_coverage", "infer_types"]},
    "lint_buffer":              {"delegates_to": ["auto_lint"]},
    "format_buffer":            {"delegates_to": ["auto_format"]},
    "lint_with_config":         {"delegates_to": ["auto_lint"]},
    "format_with_config":       {"delegates_to": ["auto_format"]},
    "get_dependency_graph":     {"delegates_to": ["get_references"]},
    "generate_change_template": {"composed_of": ["semantic_search", "analyze_change"]},
}

# 5. Examples — JSON Schema examples keyword for few-shot prompting
_SCHEMA_EXAMPLES: dict[str, dict[str, Any]] = {
    "embed_codebase":            {"input": {"path": "./src", "pattern": "*.py"}, "output": {"status": "ok", "buffer_id": "gcbuff-abc123", "token_count": 4500}},
    "semantic_search":           {"input": {"buffer_id": "gcbuff-abc123", "query": "authentication middleware", "top_k": 5}, "output": {"status": "ok", "results": [{"file": "src/auth.py", "start_line": 10, "score": 0.92}]}},
    "hybrid_search":             {"input": {"buffer_id": "gcbuff-abc123", "query": "payment processing", "top_k": 5}, "output": {"status": "ok", "results": [{"file": "src/pay.py", "start_line": 1, "score": 0.88}]}},
    "search_for":                {"input": {"buffer_id": "gcbuff-abc123", "literal": "def authenticate"}, "output": {"status": "ok", "results": [{"file": "src/auth.py", "start_line": 15}]}},
    "search_symbols":            {"input": {"buffer_id": "gcbuff-abc123", "name": "authenticate"}, "output": {"status": "ok", "results": [{"file": "src/auth.py", "name": "authenticate", "type": "function"}]}},
    "cluster_code":              {"input": {"buffer_id": "gcbuff-abc123", "n_clusters": 5}, "output": {"status": "ok", "clusters": [{"label": "authentication", "files": ["src/auth.py"]}]}},
    "find_duplicates":           {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "duplicates": [{"files": ["src/a.py", "src/b.py"], "similarity": 0.92}]}},
    "pack_context":              {"input": {"buffer_id": "gcbuff-abc123", "query": "database connection", "max_tokens": 4000}, "output": {"status": "ok", "packed_lines": 120, "truncated": False}},
    "reload_codebase":           {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "re_embedded_files": 3}},
    "check_codebase":            {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "dirty_files": ["src/auth.py"]}},
    "list_buffers":              {"input": {}, "output": {"status": "ok", "buffers": [{"buffer_id": "gcbuff-abc123", "files": 12}]}},
    "delete_buffer":             {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok"}},
    "read_code":                 {"input": {"buffer_id": "gcbuff-abc123", "file": "src/auth.py"}, "output": {"status": "ok", "lines": ["def authenticate(user, pwd):", "    ..."], "start_line": 1}},
    "look_for_file":             {"input": {"buffer_id": "gcbuff-abc123", "glob": "**/auth*.py"}, "output": {"status": "ok", "files": ["src/auth.py", "tests/test_auth.py"]}},
    "write_code":                {"input": {"buffer_id": "gcbuff-abc123", "file": "src/auth.py", "start_line": 1, "new_lines": ["def authenticate(token):", "    ..."]}, "output": {"status": "ok", "changed_lines": 2, "diff": "--- a/src/auth.py\n+++ b/src/auth.py\n@@ -1,2 +1,2 @@"}},
    "diff":                      {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "changed_files": [{"file": "src/auth.py", "dirty": True}]}},
    "discard":                   {"input": {"buffer_id": "gcbuff-abc123", "file": "src/auth.py"}, "output": {"status": "ok"}},
    "commit":                    {"input": {"buffer_id": "gcbuff-abc123", "dry_run": True}, "output": {"status": "ok", "dry_run": True, "written_files": ["src/auth.py"]}},
    "infer_types":               {"input": {"buffer_id": "gcbuff-abc123", "file": "src/auth.py", "start_line": 1, "end_line": 20, "method": "llm"}, "output": {"status": "ok", "variables": {"user": {"type": "str", "confidence": 0.95}}, "return_type": "bool"}},
    "get_symbol_metadata":       {"input": {"buffer_id": "gcbuff-abc123", "symbol": "authenticate"}, "output": {"status": "ok", "file": "src/auth.py", "line": 10, "cyclomatic_complexity": 4, "called_by_count": 8}},
    "search_batch":              {"input": {"buffer_id": "gcbuff-abc123", "queries": ["auth middleware", "database query"]}, "output": {"status": "ok", "results": {"auth middleware": [{"file": "src/auth.py", "score": 0.9}]}}},
    "auto_format":               {"input": {"buffer_id": "gcbuff-abc123", "dry_run": True}, "output": {"status": "ok", "formatted_files": 2, "already_formatted": 10}},
    "auto_lint":                 {"input": {"buffer_id": "gcbuff-abc123", "dry_run": True}, "output": {"status": "ok", "issues": [{"file": "src/auth.py", "code": "E501", "message": "line too long"}]}},
    "auto_polish":               {"input": {"buffer_id": "gcbuff-abc123", "dry_run": True}, "output": {"status": "ok", "formatting": {"formatted_files": 2}, "linting": {"issues": 3}}},
    "get_references":            {"input": {"buffer_id": "gcbuff-abc123", "symbol": "authenticate", "direction": "both"}, "output": {"status": "ok", "callers": [{"symbol": "login", "file": "src/api.py"}], "callees": [{"symbol": "verify_token", "file": "src/auth.py"}]}},
    "get_full_context":          {"input": {"buffer_id": "gcbuff-abc123", "symbol": "authenticate"}, "output": {"status": "ok", "definition": {"file": "src/auth.py", "lines": 20}, "callers": 8, "tests": ["test_auth"]}},
    "analyze_change":            {"input": {"buffer_id": "gcbuff-abc123", "file": "src/auth.py", "start_line": 10, "end_line": 20}, "output": {"status": "ok", "risk_level": "medium", "affected_symbols": ["authenticate"], "direct_callers": 5}},
    "get_test_coverage":         {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "coverage": {"src/auth.py": {"(10, 25)": ["test_auth"]}}}},
    "polish_before_commit":      {"input": {"buffer_id": "gcbuff-abc123", "check_only": True}, "output": {"status": "ok", "ready_to_commit": True, "pre_commit_warnings": []}},
    "trace_execution_paths":     {"input": {"buffer_id": "gcbuff-abc123", "symbol": "handle_request", "max_depth": 3}, "output": {"status": "ok", "paths": [{"path": ["handle_request -> validate -> check_auth"], "branches": 3, "calls": ["validate", "check_auth"]}]}},
    "get_dependency_graph":      {"input": {"buffer_id": "gcbuff-abc123", "symbol": "process_payment", "depth": 2}, "output": {"status": "ok", "nodes": [{"id": "process_payment", "label": "process_payment", "type": "function", "file": "src/pay.py"}], "edges": [{"from": "process_payment", "to": "validate_card", "type": "calls"}]}},
    "detect_code_smells":        {"input": {"buffer_id": "gcbuff-abc123", "types": ["long_function", "deep_nesting"]}, "output": {"status": "ok", "smells": [{"file": "src/auth.py", "line": 42, "type": "long_function", "severity": "medium", "suggestion": "Consider extracting methods."}]}},
    "scan_security":             {"input": {"buffer_id": "gcbuff-abc123", "severity_min": "high"}, "output": {"status": "ok", "vulnerabilities": [{"file": "src/db.py", "line": 42, "type": "sql_injection", "severity": "high", "fix_suggestion": "Use parameterized queries"}]}},
    "suggest_refactorings":      {"input": {"buffer_id": "gcbuff-abc123", "symbol": "process_payment"}, "output": {"status": "ok", "suggestions": [{"type": "extract_method", "lines": "10-60", "benefit": "Reduce complexity", "risk": "medium"}]}},
    "lint_buffer":               {"input": {"buffer_id": "gcbuff-abc123", "group_by": "severity"}, "output": {"status": "ok", "total_issues": 17, "by_severity": {"error": 5, "warning": 12, "info": 3}}},
    "format_buffer":             {"input": {"buffer_id": "gcbuff-abc123", "dry_run": True}, "output": {"status": "ok", "formatted_files": 3, "total_lines_added": 12, "total_lines_removed": 8}},
    "find_performance_hotspots": {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "hotspots": [{"file": "src/db.py", "line": 42, "type": "n_plus_one", "severity": "high", "suggestion": "Use select_related"}]}},
    "generate_documentation":    {"input": {"buffer_id": "gcbuff-abc123", "symbol": "authenticate", "style": "google"}, "output": {"status": "ok", "docstring": '"""authenticate documentation."""', "type_hints": {"user": "str"}, "examples": []}},
    "find_similar_patterns":     {"input": {"buffer_id": "gcbuff-abc123", "code_snippet": "def validate(x):\\n    return x is not None"}, "output": {"status": "ok", "semantic_matches": [{"file": "src/validators.py", "line": 15, "score": 0.89}]}},
    "find_deprecated":           {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "deprecated": [{"file": "src/api.py", "line": 42, "detection_method": "decorator", "symbol": "old_endpoint"}]}},
    "validate_changes":          {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "type_errors": [], "broken_imports": [], "safe_to_commit": True}},
    "extract_configuration":     {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "env_vars": [{"name": "DATABASE_URL", "used_in": "src/db.py:5", "required": True, "default": None}], "hardcoded_secrets": []}},
    "analyze_logging_patterns":  {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "total_logs": 142, "levels": {"debug": 30, "info": 80, "warning": 25, "error": 7, "critical": 0}}},
    "analyze_error_handling_patterns": {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "try_except_blocks": 42, "broad_catches": [{"file": "src/db.py", "line": 42, "catches": "Exception"}]}},
    "generate_changelog":        {"input": {"buffer_id": "gcbuff-abc123", "since_commit": "v1.0.0"}, "output": {"status": "ok", "features": [{"commit": "abc1234", "message": "feat: add retry logic"}], "bugfixes": []}},
    "detect_api_changes":        {"input": {"buffer_id": "gcbuff-abc123", "since_commit": "v1.0.0"}, "output": {"status": "ok", "current_api_surface": 25, "changes": [{"symbol": "process_payment", "breaking": True}]}},
    "get_rollback_info":         {"input": {"buffer_id": "gcbuff-abc123", "file": "src/auth.py"}, "output": {"status": "ok", "last_working_commit": "abc1234", "commit_message": "feat: add MFA support"}},
    "generate_change_template":  {"input": {"buffer_id": "gcbuff-abc123", "request": "add retry logic to database calls"}, "output": {"status": "ok", "files_to_modify": ["src/db.py"], "risk_assessment": "medium"}},
    "map_api_endpoints":         {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "endpoints": [{"method": "POST", "path": "/api/v1/payment", "handler": "process_payment", "is_async": True}]}},
    "analyze_cache_patterns":    {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "caches_used": ["redis", "lru_cache"], "stale_data_risks": []}},
    "analyze_thread_safety":     {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "shared_state": [{"name": "global_cache", "protected_by": "none"}], "race_conditions": []}},
    "detect_memory_issues":      {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "unbounded_collections": [{"file": "src/collector.py", "line": 15, "growth_reason": "append in loop"}]}},
    "lint_with_config":          {"input": {"buffer_id": "gcbuff-abc123"}, "output": {"status": "ok", "config_file": "pyproject.toml", "issues": []}},
    "format_with_config":        {"input": {"buffer_id": "gcbuff-abc123", "dry_run": True}, "output": {"status": "ok", "config_file": "pyproject.toml", "formatted_files": 2}},
}


def _enrich_all_schemas() -> None:
    """Apply AI discoverability metadata to all schemas at import time.

    Adds: category, tags, read_only, side_effects, error_schema,
    delegates_to/composed_of (for wrappers), and input/output examples.
    """
    for schema in ALL_SCHEMAS:
        name = schema["name"]

        # 1. Category + Tags
        cat = _SCHEMA_CATEGORIES.get(name, {"category": "uncategorized", "tags": []})
        schema["category"] = cat["category"]
        schema["tags"] = cat["tags"]

        # 2. Read-only + Side-effects
        se = _SCHEMA_SIDE_EFFECTS.get(name, {"read_only": True, "side_effects": None})
        schema["read_only"] = se["read_only"]
        schema["side_effects"] = se["side_effects"]

        # 3. Error schema
        schema["error_schema"] = _SHARED_ERROR_SCHEMA

        # 4. Composition hints (only for wrapper tools)
        comp = _SCHEMA_COMPOSITION.get(name)
        if comp:
            schema.update(comp)

        # 5. Examples (JSON Schema examples keyword)
        ex = _SCHEMA_EXAMPLES.get(name)
        if ex:
            if "input_schema" in schema:
                schema["input_schema"]["examples"] = [ex["input"]]
            if "output_schema" in schema:
                schema["output_schema"]["examples"] = [ex["output"]]


# Apply enrichment on import
_enrich_all_schemas()

# ---------------------------------------------------------------------------
# Access helpers
# ---------------------------------------------------------------------------

# Categories for filtering
TOOL_CATEGORIES = sorted(set(s["category"] for s in ALL_SCHEMAS))


def get_schemas_by_category(category: str) -> list[dict[str, Any]]:
    """Return all schemas matching a category (e.g. 'search', 'quality', 'safety')."""
    return [s for s in ALL_SCHEMAS if s.get("category") == category]


def get_read_only_tools() -> list[dict[str, Any]]:
    """Return all schemas for tools that have no side effects (safe to call freely)."""
    return [s for s in ALL_SCHEMAS if s.get("read_only") is True]


def get_write_tools() -> list[dict[str, Any]]:
    """Return all schemas for tools that modify files or buffers."""
    return [s for s in ALL_SCHEMAS if s.get("read_only") is False]


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
    """Convert schemas to OpenAI function-calling format.

    Includes category, tags, read_only, side_effects, and composition hints
    as top-level metadata alongside the standard function definition.
    """
    functions: list[dict[str, Any]] = []
    for schema in ALL_SCHEMAS:
        func: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["input_schema"],
            },
        }
        # Attach enriched metadata (ignored by OpenAI API but useful for local routing)
        func["category"] = schema.get("category", "uncategorized")
        func["tags"] = schema.get("tags", [])
        func["read_only"] = schema.get("read_only", True)
        func["side_effects"] = schema.get("side_effects")
        if "delegates_to" in schema:
            func["delegates_to"] = schema["delegates_to"]
        if "composed_of" in schema:
            func["composed_of"] = schema["composed_of"]
        functions.append(func)
    return functions



def to_mcp_tools() -> list[dict[str, Any]]:
    """Convert schemas to MCP tool format.

    Includes enriched metadata as annotations per the MCP spec.
    """
    tools: list[dict[str, Any]] = []
    for schema in ALL_SCHEMAS:
        tool: dict[str, Any] = {
            "name": schema["name"],
            "description": schema["description"],
            "inputSchema": schema["input_schema"],
        }
        # MCP annotations for safety and categorization
        tool["annotations"] = {
            "category": schema.get("category", "uncategorized"),
            "tags": schema.get("tags", []),
            "readOnlyHint": schema.get("read_only", True),
            "sideEffects": schema.get("side_effects"),
        }
        if "delegates_to" in schema:
            tool["annotations"]["delegatesTo"] = schema["delegates_to"]
        if "composed_of" in schema:
            tool["annotations"]["composedOf"] = schema["composed_of"]
        if "error_schema" in schema:
            tool["errorSchema"] = schema["error_schema"]
        tools.append(tool)
    return tools


# ---------------------------------------------------------------------------
# Schema Format Configuration
# ---------------------------------------------------------------------------

from enum import Enum


class SchemaFormat(str, Enum):
    """Supported tool schema export formats.

    - OPENAI:   OpenAI function-calling format (also used by Ollama).
                Nested under ``{"type":"function","function":{...}}`` with
                ``parameters`` key.

    - ANTHROPIC: Anthropic/Claude tool-use format. Flat dict with
                ``input_schema`` (snake_case) at top level.

    - MCP:      Model Context Protocol format. Flat dict with
                ``inputSchema`` (camelCase) and ``annotations``.

    - OLLAMA:   Alias for OPENAI (Ollama uses the same format).
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MCP = "mcp"
    OLLAMA = "ollama"


def to_anthropic_tools(
    *,
    include_metadata: bool = True,
) -> list[dict[str, Any]]:
    """Convert schemas to Anthropic tool-use format.

    Anthropic tools use a flat structure with ``input_schema`` (snake_case)
    and optional ``cache_control`` for prompt caching.

    Args:
        include_metadata: If True, attach category/tags/read_only/side_effects
            as a ``x-gigacode`` extension dict. Default: True.

    Returns:
        List of Anthropic tool dicts ready for ``client.messages.create()``.
    """
    tools: list[dict[str, Any]] = []
    for schema in ALL_SCHEMAS:
        tool: dict[str, Any] = {
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": schema["input_schema"],
        }
        if include_metadata:
            tool["x-gigacode"] = {
                "category": schema.get("category", "uncategorized"),
                "tags": schema.get("tags", []),
                "read_only": schema.get("read_only", True),
                "side_effects": schema.get("side_effects"),
                "error_schema": schema.get("error_schema"),
            }
            if "delegates_to" in schema:
                tool["x-gigacode"]["delegates_to"] = schema["delegates_to"]
            if "composed_of" in schema:
                tool["x-gigacode"]["composed_of"] = schema["composed_of"]
        tools.append(tool)
    return tools


def to_ollama_tools(
    *,
    include_metadata: bool = True,
) -> list[dict[str, Any]]:
    """Convert schemas to Ollama tool format.

    Ollama uses the same format as OpenAI function-calling.
    This is an alias for ``to_openai_functions()`` with metadata support.

    Args:
        include_metadata: If True, attach enriched metadata as top-level
            extension fields. Default: True.

    Returns:
        List of Ollama-compatible function dicts.
    """
    # Ollama uses the OpenAI function-calling format
    functions = to_openai_functions()
    if not include_metadata:
        # Strip metadata fields
        for func in functions:
            for key in ("category", "tags", "read_only", "side_effects",
                        "delegates_to", "composed_of"):
                func.pop(key, None)
    return functions


def export_schemas(
    format: str | SchemaFormat = SchemaFormat.OPENAI,
    *,
    include_metadata: bool = True,
    category: str | None = None,
    read_only_only: bool = False,
) -> list[dict[str, Any]]:
    """Export all tool schemas in the specified format.

    Unified entry point for schema export. Picks the right converter
    based on the format parameter.

    Args:
        format: Output format — "openai", "anthropic", "mcp", or "ollama".
            Also accepts SchemaFormat enum values. Default: "openai".
        include_metadata: If True, include enriched metadata
            (category, tags, read_only, side_effects, composition hints).
            Default: True.
        category: If set, only export tools matching this category
            (e.g. "search", "security", "quality").
        read_only_only: If True, only export read-only (safe) tools.

    Returns:
        List of tool schema dicts in the requested format.

    Example:
        >>> from gigacode.tool_schema import export_schemas, SchemaFormat
        >>> openai_tools = export_schemas("openai")
        >>> anthropic_tools = export_schemas(SchemaFormat.ANTHROPIC, category="security")
        >>> mcp_tools = export_schemas("mcp", read_only_only=True)
    """
    fmt = SchemaFormat(format)

    # Filter schemas if requested
    global ALL_SCHEMAS
    source = ALL_SCHEMAS
    if category:
        source = [s for s in source if s.get("category") == category]
    if read_only_only:
        source = [s for s in source if s.get("read_only") is True]

    # Temporarily swap ALL_SCHEMAS so converters work on the filtered set
    original = ALL_SCHEMAS
    ALL_SCHEMAS = source  # type: ignore[assignment]

    try:
        if fmt == SchemaFormat.OPENAI:
            result = to_openai_functions()
        elif fmt == SchemaFormat.ANTHROPIC:
            result = to_anthropic_tools(include_metadata=include_metadata)
        elif fmt == SchemaFormat.MCP:
            result = to_mcp_tools()
        elif fmt == SchemaFormat.OLLAMA:
            result = to_ollama_tools(include_metadata=include_metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")
    finally:
        ALL_SCHEMAS = original

    return result


# ---------------------------------------------------------------------------
# Schema Config (file-based defaults)
# ---------------------------------------------------------------------------

import os


class SchemaConfig:
    """Configuration for schema export defaults.

    Reads from ``gigacode.toml`` in the working directory, or falls back
    to ``[tool.gigacode]`` section in ``pyproject.toml``, or uses built-in
    defaults.

    Supported keys in TOML:

        [tool.gigacode]
        schema_format = "openai"          # "openai" | "anthropic" | "mcp" | "ollama"
        include_metadata = true           # Attach enriched metadata to exported schemas
        default_category = null           # Filter to a single category by default
        read_only_only = false            # Only export read-only tools by default

    Example gigacode.toml:

        schema_format = "anthropic"
        include_metadata = true
    """

    _DEFAULTS: dict[str, Any] = {
        "schema_format": "openai",
        "include_metadata": True,
        "default_category": None,
        "read_only_only": False,
    }

    def __init__(self, config_path: str | None = None) -> None:
        self.schema_format: SchemaFormat = SchemaFormat.OPENAI
        self.include_metadata: bool = True
        self.default_category: str | None = None
        self.read_only_only: bool = False

        if config_path:
            self._load_toml(config_path)
        else:
            # Auto-discover config file
            cwd = os.getcwd()
            for candidate in [
                os.path.join(cwd, "gigacode.toml"),
                os.path.join(cwd, "pyproject.toml"),
            ]:
                if os.path.isfile(candidate):
                    self._load_toml(candidate)
                    break

    def _load_toml(self, path: str) -> None:
        """Load config from a TOML file."""
        try:
            with open(path, "rb") as f:
                import tomllib
                data = tomllib.load(f)
        except (ImportError, ModuleNotFoundError):
            # Python < 3.11 fallback
            try:
                import tomli as tomllib  # type: ignore[no-redef]
                with open(path, "rb") as f:
                    data = tomllib.load(f)
            except ImportError:
                return  # No TOML parser available, use defaults
        except FileNotFoundError:
            return

        # Support both top-level gigacode.toml and [tool.gigacode] in pyproject.toml
        if "schema_format" in data:
            # Flat gigacode.toml
            section = data
        else:
            section = data.get("tool", {}).get("gigacode", {})

        if "schema_format" in section:
            self.schema_format = SchemaFormat(section["schema_format"])
        if "include_metadata" in section:
            self.include_metadata = bool(section["include_metadata"])
        if "default_category" in section:
            self.default_category = section["default_category"]
        if "read_only_only" in section:
            self.read_only_only = bool(section["read_only_only"])

    def export(self) -> list[dict[str, Any]]:
        """Export schemas using this config's defaults."""
        return export_schemas(
            format=self.schema_format,
            include_metadata=self.include_metadata,
            category=self.default_category,
            read_only_only=self.read_only_only,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return current config as a dict."""
        return {
            "schema_format": self.schema_format.value,
            "include_metadata": self.include_metadata,
            "default_category": self.default_category,
            "read_only_only": self.read_only_only,
        }


def export_schemas_from_config(
    config_path: str | None = None,
) -> list[dict[str, Any]]:
    """Load config from file and export schemas in the configured format.

    Convenience function that reads ``gigacode.toml`` or ``pyproject.toml``
    and exports schemas accordingly.

    Args:
        config_path: Explicit path to a TOML config file. If None,
            auto-discovers gigacode.toml or pyproject.toml in CWD.

    Returns:
        List of tool schema dicts in the configured format.
    """
    config = SchemaConfig(config_path)
    return config.export()
