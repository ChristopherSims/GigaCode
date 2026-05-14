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
                "description": "If set, expand call chain to this depth (Phase 3 fill).",
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
            "buffer_id": {"type": "string"},
            "symbol": {"type": "string", "description": "Symbol name to trace."},
            "max_depth": {"type": "integer", "description": "Max call depth. Default: 3.", "default": 3},
        },
        "required": ["buffer_id", "symbol"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "symbol": {"type": "string"},
            "paths": {"type": "array", "items": {"type": "object"}},
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
            "buffer_id": {"type": "string"},
            "symbol": {"type": ["string", "null"], "description": "Optional symbol to scope graph around."},
            "depth": {"type": "integer", "description": "Depth of dependencies. Default: 2.", "default": 2},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "nodes": {"type": "array", "items": {"type": "object"}},
            "edges": {"type": "array", "items": {"type": "object"}},
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
            "buffer_id": {"type": "string"},
            "types": {
                "type": "array",
                "items": {"type": "string", "enum": ["long_function", "deep_nesting", "missing_docstring", "complex_logic", "too_many_params", "duplicates"]},
            },
            "severity_min": {"type": "string", "enum": ["low", "medium", "high"], "default": "low"},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "smells": {"type": "array"},
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
            "buffer_id": {"type": "string"},
            "severity_min": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "vulnerabilities": {"type": "array"},
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
            "buffer_id": {"type": "string"},
            "symbol": {"type": "string"},
        },
        "required": ["buffer_id", "symbol"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "symbol": {"type": "string"},
            "suggestions": {"type": "array"},
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
            "buffer_id": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
            "select": {"type": "array", "items": {"type": "string"}},
            "exclude_patterns": {"type": "array", "items": {"type": "string"}},
            "group_by": {"type": "string", "enum": ["file", "severity", "rule"], "default": "file"},
        },
        "required": ["buffer_id"],
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "total_issues": {"type": "integer"},
            "by_file": {"type": "object"},
            "by_severity": {"type": "object"},
            "by_rule": {"type": "object"},
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
            "buffer_id": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
            "formatter": {"type": "string", "enum": ["black", "ruff.format"], "default": "black"},
            "line_length": {"type": "integer", "default": 88},
            "exclude_patterns": {"type": "array", "items": {"type": "string"}},
            "dry_run": {"type": "boolean", "default": True},
            "summary_only": {"type": "boolean", "default": False},
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
            "changes": {"type": "array"},
            "summary": {"type": "string"},
        },
        "required": ["status"],
    },
}

FIND_PERFORMANCE_HOTSPOTS_SCHEMA: dict[str, Any] = {
    "name": "find_performance_hotspots",
    "description": "Detect performance hotspots: N+1 queries, inefficient loops, unbounded recursion.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "hotspots": {"type": "array"}, "total": {"type": "integer"}}, "required": ["status"]},
}

GENERATE_DOCUMENTATION_SCHEMA: dict[str, Any] = {
    "name": "generate_documentation",
    "description": "Auto-generate documentation from code analysis.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}, "symbol": {"type": "string"}, "style": {"type": "string", "enum": ["google", "numpy", "sphinx"], "default": "google"}}, "required": ["buffer_id", "symbol"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "symbol": {"type": "string"}, "docstring": {"type": "string"}, "type_hints": {"type": "object"}, "examples": {"type": "array"}}, "required": ["status"]},
}

FIND_SIMILAR_PATTERNS_SCHEMA: dict[str, Any] = {
    "name": "find_similar_patterns",
    "description": "Find similar code patterns using semantic + syntactic matching.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}, "code_snippet": {"type": "string"}, "min_similarity": {"type": "number", "default": 0.7}, "top_k": {"type": "integer", "default": 10}}, "required": ["buffer_id", "code_snippet"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "syntactic_matches": {"type": "array"}, "semantic_matches": {"type": "array"}}, "required": ["status"]},
}

FIND_DEPRECATED_SCHEMA: dict[str, Any] = {
    "name": "find_deprecated",
    "description": "Detect usage of deprecated functions and APIs.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "deprecated": {"type": "array"}, "total": {"type": "integer"}}, "required": ["status"]},
}

VALIDATE_CHANGES_SCHEMA: dict[str, Any] = {
    "name": "validate_changes",
    "description": "Validate changes before committing (static analysis + import resolution).",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}, "dry_run": {"type": "boolean", "default": True}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "type_errors": {"type": "array"}, "broken_imports": {"type": "array"}, "safe_to_commit": {"type": "boolean"}}, "required": ["status"]},
}

EXTRACT_CONFIGURATION_SCHEMA: dict[str, Any] = {
    "name": "extract_configuration",
    "description": "Extract configuration: env vars, config files, hardcoded secrets, defaults.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "env_vars": {"type": "array"}, "config_files": {"type": "array"}, "hardcoded_secrets": {"type": "array"}, "default_values": {"type": "object"}}, "required": ["status"]},
}

ANALYZE_LOGGING_PATTERNS_SCHEMA: dict[str, Any] = {
    "name": "analyze_logging_patterns",
    "description": "Analyze logging patterns: levels, consistency, gaps.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "total_logs": {"type": "integer"}, "levels": {"type": "object"}, "missing_logs_in": {"type": "array"}, "inconsistent_format": {"type": "array"}}, "required": ["status"]},
}

ANALYZE_ERROR_HANDLING_SCHEMA: dict[str, Any] = {
    "name": "analyze_error_handling_patterns",
    "description": "Analyze error handling patterns: broad catches, missing finally, uncaught exceptions.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "try_except_blocks": {"type": "integer"}, "broad_catches": {"type": "array"}, "missing_finally": {"type": "array"}, "suggestions": {"type": "array"}}, "required": ["status"]},
}

GENERATE_CHANGELOG_SCHEMA: dict[str, Any] = {
    "name": "generate_changelog",
    "description": "Generate changelog from git history + semantic analysis.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}, "since_commit": {"type": ["string", "null"]}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "features": {"type": "array"}, "bugfixes": {"type": "array"}, "breaking_changes": {"type": "array"}}, "required": ["status"]},
}

DETECT_API_CHANGES_SCHEMA: dict[str, Any] = {
    "name": "detect_api_changes",
    "description": "Detect API-breaking changes between commits.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}, "since_commit": {"type": ["string", "null"]}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "changes": {"type": "array"}, "current_api_surface": {"type": "integer"}}, "required": ["status"]},
}

GET_ROLLBACK_INFO_SCHEMA: dict[str, Any] = {
    "name": "get_rollback_info",
    "description": "Get rollback information for a file.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}, "file": {"type": "string"}}, "required": ["buffer_id", "file"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "last_working_commit": {"type": ["string", "null"]}, "diff_to_revert": {"type": "string"}}, "required": ["status"]},
}

GENERATE_CHANGE_TEMPLATE_SCHEMA: dict[str, Any] = {
    "name": "generate_change_template",
    "description": "Generate a change plan template for a request.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}, "request": {"type": "string"}}, "required": ["buffer_id", "request"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "files_to_modify": {"type": "array"}, "change_strategy": {"type": "string"}, "risk_assessment": {"type": "string"}}, "required": ["status"]},
}

MAP_API_ENDPOINTS_SCHEMA: dict[str, Any] = {
    "name": "map_api_endpoints",
    "description": "Map all API endpoints (FastAPI, Flask, Django patterns).",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "endpoints": {"type": "array"}, "total": {"type": "integer"}}, "required": ["status"]},
}

ANALYZE_CACHE_PATTERNS_SCHEMA: dict[str, Any] = {
    "name": "analyze_cache_patterns",
    "description": "Analyze cache usage patterns: invalidation logic, stale data risks.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "caches_used": {"type": "array"}, "invalidation_logic": {"type": "array"}, "stale_data_risks": {"type": "array"}}, "required": ["status"]},
}

ANALYZE_THREAD_SAFETY_SCHEMA: dict[str, Any] = {
    "name": "analyze_thread_safety",
    "description": "Analyze thread safety: shared state, race conditions, deadlock risks.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "shared_state": {"type": "array"}, "race_conditions": {"type": "array"}, "deadlock_risks": {"type": "array"}}, "required": ["status"]},
}

DETECT_MEMORY_ISSUES_SCHEMA: dict[str, Any] = {
    "name": "detect_memory_issues",
    "description": "Detect memory issues: circular refs, unbounded collections, resource leaks.",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "circular_refs": {"type": "array"}, "unbounded_collections": {"type": "array"}, "resource_leaks": {"type": "array"}}, "required": ["status"]},
}

LINT_WITH_CONFIG_SCHEMA: dict[str, Any] = {
    "name": "lint_with_config",
    "description": "Lint using project configuration (ruff.toml, pyproject.toml).",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}, "config_file": {"type": ["string", "null"]}, "files": {"type": "array", "items": {"type": "string"}}, "auto_fix": {"type": "boolean", "default": False}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "config_file": {"type": "string"}, "issues": {"type": "array"}}, "required": ["status"]},
}

FORMAT_WITH_CONFIG_SCHEMA: dict[str, Any] = {
    "name": "format_with_config",
    "description": "Format using project configuration (pyproject.toml, .black, ruff.toml).",
    "input_schema": {"type": "object", "properties": {"buffer_id": {"type": "string"}, "config_file": {"type": ["string", "null"]}, "files": {"type": "array", "items": {"type": "string"}}, "dry_run": {"type": "boolean", "default": True}}, "required": ["buffer_id"]},
    "output_schema": {"type": "object", "properties": {"status": {"type": "string"}, "config_file": {"type": "string"}, "changes": {"type": "array"}}, "required": ["status"]},
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
