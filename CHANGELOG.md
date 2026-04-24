# Changelog

All notable changes to GigaCode are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4] - 2024-04-24

### Added

- **Hybrid Search** (`src/hybrid_search.py`, `src/lexical_index.py`)
  - BM25 lexical index with code-aware tokenization (CamelCase / snake_case splitting).
  - Reciprocal Rank Fusion (RRF) merging semantic + lexical results.
  - New `CodeEmbeddingTool.hybrid_search()` method with configurable semantic/lexical weights.

- **Query Cache & Pagination**
  - In-memory LRU query cache (`src/query_cache.py`) for repeated searches.
  - `offset` parameter added to `semantic_search()`, `hybrid_search()`.

- **Multi-Language Tree-Sitter Support** (`src/chunker.py`)
  - Promoted 7 grammar packages to core dependencies: JavaScript, TypeScript, Rust, C, C++, Go, Java.
  - Language-specific AST node type maps for accurate chunking per language.
  - Graceful fallback with `pip install` hints when grammars are missing.

- **Async HTTP Server** (`src/gigacode_api.py`, `src/gigacode_server.py`)
  - FastAPI-based ASGI server with typed Pydantic request models.
  - Endpoints: `/buffers`, `/search/semantic`, `/search/hybrid`, `/search/literal`, `/search/symbols`, `/duplicates`, `/pack`, `/read`, `/write`, `/commit`, `/diff`, `/discard`, `/call`.
  - Backward-compatible stdlib `HTTPServer` fallback when FastAPI is not installed.

- **MCP Server** (`src/mcp_server.py`)
  - Full Model Context Protocol server using the `mcp` Python SDK.
  - Supports **stdio** transport (Claude Desktop, Cursor) and **HTTP-SSE** transport.
  - Exposes all GigaCode tools via `tools/list` and `tools/call`.

- **File Watcher & Auto-Reload** (`src/file_watcher.py`)
  - `watchdog`-based recursive file observer with polling fallback.
  - Auto-calls `reload_codebase()` on source changes with configurable debounce.

- **Duplicate Detection** (`src/duplicate_detector.py`)
  - MinHash + LSH near-duplicate detection for code chunks.
  - Configurable Jaccard similarity threshold (default 0.85).
  - New `CodeEmbeddingTool.find_duplicates()` method.

- **LLM Context Packing** (`src/context_packer.py`)
  - Token-budgeted greedy packing of hybrid-search results for LLM prompting.
  - Approximate token counting (~4 chars/token) with optional `tiktoken` exact mode.
  - New `CodeEmbeddingTool.pack_context()` method.

- **Tool Schemas**
  - Added `hybrid_search`, `find_duplicates`, and `pack_context` schemas to `src/tool_schema.py`.

- **Tests**
  - `tests/test_lexical_index.py`
  - `tests/test_hybrid_search.py`
  - `tests/test_duplicate_detector.py`
  - `tests/test_context_packer.py`
  - `tests/test_file_watcher.py`

### Changed

- `requirements.txt`: added `fastapi`, `uvicorn`, `pydantic`, `mcp`, `watchdog`, and multi-language `tree-sitter-*` grammars.
- `semantic_search()` now returns `doc_id` and supports cache hits.
- `src/gigacode_tool.py` now imports and manages lexical index + query cache instances per buffer.

## [0.3] - 2024-04-23

### Added

- **Agent Read-Write-Commit Workflow** (`src/gigacode_tool.py`)
  - `read_code()`, `write_code()`, `commit()`, `discard()`, `diff()` for safe agent editing.
  - Hash-based safety checks prevent commits when files are modified externally.
  - Deferred batch rebuilds: edits accumulate in a dirty queue; re-embedding is batched until `commit()`.

- **Literal & Symbol Search**
  - `search_for()`: grep-style substring search across buffered snapshots.
  - `search_symbols()`: merges name-based matching with semantic embedding search for functions, classes, and methods.

- **HTTP Agent Server** (`src/gigacode_server.py`)
  - Lightweight stdlib `HTTPServer` exposing all tools via JSON POST to `/call`.
  - Health (`/health`) and schema (`/schemas`) endpoints.

- **Language-Agnostic Editing** (`src/gigacode_skill.py`, `src/cross_language_rules.py`)
  - Automated code improvements: docstrings, type annotations, bare exception fixes, context managers, visibility modifiers.
  - Supports Python, JavaScript, TypeScript, Java, C, C++, Rust, Go, Ruby, PHP, C#, Swift, Kotlin, Scala, Lua, Elixir.

- **Vectorized Optimizations** (`src/gpu_index.py`, `src/embedder.py`)
  - FAISS CPU `IndexIDMap` + optional GPU mirror via `faiss.index_cpu_to_gpu`.
  - Sub-millisecond ANN search on GPU, single-digit ms on CPU.
  - Memory-mapped embedding I/O and compact JSON metadata storage.

- **Tool Schemas** (`src/tool_schema.py`)
  - Formal JSON schemas for OpenAI function calling, Anthropic tool use, and MCP export.

- **Benchmarks** (`benchmark.py`)
  - Built-in performance suite measuring embed, search, cluster, and edit latencies.

### Changed

- README rewritten with architecture diagrams, usage examples, and troubleshooting guide.
- Added MIT `LICENSE` file.

## [0.2] - 2024-04-22

### Added

- **AST-Based Chunking** (`src/chunker.py`)
  - Tree-sitter parser for function/class/method boundary extraction.
  - Sliding-window fallback for unsupported languages or parse failures.
  - Code-specific embedding model: `jina-embeddings-v2-base-code` with `all-MiniLM-L6-v2` fallback.

- **Pre-Flight Size Guard** (`src/size_guard.py`)
  - Estimates memory usage before embedding large codebases.
  - Configurable `threshold_mb` to prevent OOM errors.

- **Language Detection** (`src/language_detect.py`)
  - File extension, special filename, and shebang-based language detection.
  - Tree-sitter package name resolution for 20+ languages.

## [0.1] - 2024-04-21

### Added

- Initial implementation of GigaCode as a Vulkan GPU-accelerated code embedding agent tool.
- Core modules: `chunker`, `embedder`, `gpu_index`, `diff_engine`, `metadata_store`.
- Basic `CodeEmbeddingTool` class with `embed_codebase()`, `semantic_search()`, and `cluster_code()`.

---

[0.4]: https://github.com/yourusername/gigacode/compare/v0.3...v0.4
[0.3]: https://github.com/yourusername/gigacode/compare/v0.2...v0.3
[0.2]: https://github.com/yourusername/gigacode/compare/v0.1...v0.2
[0.1]: https://github.com/yourusername/gigacode/releases/tag/v0.1
