Noticed AI is slow at searching/parsing code so I made this.
AI agents searching for code snippits is blazingly fast now with this tool/skill.

Code benchmarks show that AI agent coding is 4x faster, however AI token rate will always be the bottleneck.<br>
Reduced token usage by about 60% on average<br>
Searching for code is orders of magnitude faster (100x +)(depends on how much GPU VRAM you want to use)<br>
Small code bases don't take up that much memory<br>

After releasing this tool I have noticed that some AI Agents now include semantic code searches. I will still develop this tool for open-source.
# GigaCode

[![Version](https://img.shields.io/badge/version-0.5.1-blue?style=for-the-badge)](VERSION)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-yellow?style=for-the-badge)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-346%20%7C%2099.5%25-brightgreen?style=for-the-badge)](#tests)
[![Status](https://img.shields.io/badge/status-development-orange?style=for-the-badge)](#)

**GPU-accelerated code embedding for AI agents.** Embed a codebase into searchable chunks, run semantic search and clustering, and edit code through a safe read-write-commit workflow.

GigaCode is optimized for AI agent loops—fast chunking, sub-millisecond search on GPU, and surgical index updates on edit. Designed as a local-only tool that runs on your machine with no network exposure.

**Version 0.5.1** — Agent intelligence, token-saving, and unified task loops

## Key Features

- **Smart context packing** — Deduplicates chunks, strips boilerplate (imports, licenses, `__all__`), excludes tests; 30-40% token savings
- **Incremental result streaming** — Progressive disclosure: signatures-only (84% savings) → details → full text
- **Query intent caching** — 3-layer cache (index → semantic → intent clusters) for 67% savings on paraphrased queries
- **Intent-based action planning** — Classifies intent (feature, bug, refactor, docs, testing, optimization) and recommends optimal action sequences
- **`solve()` unified loop** — Automatically plan and execute multi-step tasks with audit trail, rollback, and test verification
- **Diff-aware search** — Scope search to dirty files + dependencies only (10-20x speedup)
- **Conflict prediction** — Warn before commit if files were modified externally via real git log parsing
- **"Why this matters" annotations** — Explain search result relevance (keyword match, call site, import dependency, edit context)
- **Undo/redo with branching** — Git-like branches for experimental edits; surgical undo, not full rollback
- **Symbol search** — Exact/prefix/fuzzy symbol search, jump-to-definition, find-references
- **Faceted search** — Language/path/type/line-count filters with confidence scoring
- **AST-based chunking** — Functions, classes, and methods via tree-sitter (falls back to sliding windows)
- **Semantic search** with FAISS — Sub-millisecond ANN on GPU, single-digit ms on CPU
- **Persistent GPU index** — Embeddings stay in VRAM, auto-syncs on edit
- **Agent read-write-commit workflow** — Hash-based safety checks prevent external file modifications
- **Deferred batch rebuilds** — Edits accumulate in dirty queue; re-embedding batched until `commit`
- **Code-specific embeddings** — `jina-embeddings-v2-base-code` by default (falls back to `all-MiniLM-L6-v2`)
- **Large file streaming** — Handle files over 100MB without OOM
- **Batch embedding optimization** — 2-5x faster for large batches with automatic caching
- **Path traversal protection** — Validated path access prevents accidental file escapes
- **ACID transactions** — Write-ahead logging ensures data safety on abrupt termination

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install CPU version (default)
pip install .

# Or install with GPU support (requires CUDA/cuDNN)
pip install ".[gpu]"

# Development tools
pip install ".[dev]"

# Documentation
pip install ".[docs]"
```

**GPU Support:**
GigaCode supports GPU-accelerated search via FAISS. To use GPU:

1. **Check your system:**
   ```bash
   python scripts/check_gpu.py
   ```

2. **If GPU is available:**
   ```bash
   pip install ".[gpu]"
   ```
   This installs `faiss-gpu` instead of `faiss-cpu` for sub-millisecond search.

3. **If GPU is not available:**
   The default CPU version works great! FAISS on CPU delivers single-digit millisecond search.

**System Requirements:**
- **CPU-only:** Python 3.9+, any platform
- **GPU:** NVIDIA GPU with CUDA 11.8+, cuDNN 8.x, and NVIDIA drivers

### Embed and Search

```python
from gigacode.gigacode_tool import CodeEmbeddingTool

# Embed a codebase
with CodeEmbeddingTool(work_dir="./buffers", device="cpu") as tool:
    result = tool.embed_codebase("./examplecode", pattern="*.py")
    buf_id = result["buffer_id"]
    print(f"Embedded: {result['num_chunks']} chunks")

    # Semantic search
    search = tool.semantic_search(buf_id, "sorting algorithm", top_k=5)
    for match in search["matches"]:
        print(f"{match['file']}:{match['start_line']}-{match['end_line']} (score: {match['score']:.3f})")

    # Cluster related code
    clusters = tool.cluster_code(buf_id, threshold=0.75)
    for cluster in clusters["clusters"]:
        print(f"Cluster: {cluster['file']}:{cluster['start_line']}-{cluster['end_line']}")
```

### Edit Code Safely

```python
with CodeEmbeddingTool(work_dir="./buffers", device="cpu") as tool:
    result = tool.embed_codebase("./src", pattern="*.py")
    buf_id = result["buffer_id"]

    # Read lines from a file
    read = tool.read_code(buf_id, file="main.py")
    print("\n".join(read["lines"]))

    # Write changes (deferred re-embedding)
    tool.write_code(
        buf_id,
        file="main.py",
        start_line=5,
        new_lines=["    # Added by agent", "    return value"],
    )

    # Preview changes before commit
    diff = tool.diff(buf_id)
    for changed_file in diff["changed_files"]:
        print(f"Changed: {changed_file['file']} ({changed_file['buffer_lines']} lines)")

    # Commit changes to disk (rebuilds index for dirty files)
    tool.commit(buf_id, dry_run=False)

    # Or revert a file
    tool.discard(buf_id, file="main.py")
```

**Safety guarantees:**
- `commit()` aborts if the original file was modified externally since embedding
- All changes are kept in memory until `commit()` is called
- Use `dry_run=True` to preview without writing
- Path traversal attempts are blocked at API boundaries
- ACID transactions ensure data integrity on interruption

## Performance

GigaCode is optimized for fast agent loops:

- **AST chunking** reduces embedding count 5–20× vs per-line
- **FAISS ANN** search: **~0.1 ms** on GPU, **~20 ms** on CPU for 100K+ chunks
- **Deferred rebuilds**: `write_code` is **~0.5 ms** because re-embedding is batched until `commit`
- **Surgical index updates**: only dirty files are re-chunked and patched into FAISS
- **Batch optimization**: 2-5x faster embedding for large batches with LRU caching
- **Large file streaming**: Handle files >100MB without OOM using 1MB chunk streaming
- **Diff-aware search**: 10-20x speedup on large codebases by scoping to dirty files + dependencies
- **Query intent caching**: 67% savings on paraphrased queries via 3-layer cache (index → semantic → intent clusters)
- **Smart context packing**: 30-40% token savings by deduplication, boilerplate stripping, and test exclusion
- **Incremental result streaming**: 84% token savings for signatures-only vs full chunk text

### Benchmark

Run the built-in benchmark:

```bash
python benchmark.py --dir examplecode/ --search-iters 50 --edit-iters 5
```

**Example output** (`examplecode/`, 44 chunks, CPU):

```
embed_codebase : 1.57s
semantic_search: 20.6 ms median (CPU; ~0.1 ms on GPU)
cluster_code   : 1.9 ms
write_code     : 0.59 ms (deferred rebuild)
```

## Architecture

### Modular Design (v0.5.1)

```
Codebase
   |
   v
Chunker (tree-sitter AST: functions/classes, sliding-window fallback)
   |
   v
OptimizedEmbedder (batch optimization + LRU cache)
   |
   +-- BatchEmbeddingProcessor (handles large batches)
   +-- IntentCache (3-layer: index → semantic → intent clusters)
   |
   v
FAISS Index (CPU IDMap + FlatIP)
   |
   +-- GPU mirror (faiss.index_cpu_to_gpu) — rebuilt lazily on edit
   +-- SymbolIndex (exact/prefix/fuzzy symbol search)
   +-- FacetedSearcher (language/path/type filters)
   |
   v
SearchService (unified: semantic + hybrid + symbol + faceted)
   |
   +-- SemanticSearchStreaming (3-level disclosure: signatures/details/full)
   +-- ContextPacker (smart dedup + boilerplate strip + test exclusion)
   +-- DiffAwareSearch (scoped to dirty files + dependencies)
```

**Core Modules:**

- `buffer_manager.py` — Buffer registry, persistence, file I/O, state tracking
- `index_manager.py` — FAISS index caching, GPU memory management, query result caching
- `search_service.py` — Unified search with streaming + intent caching
- `batch_embedder.py` — Dynamic batch sizing, embedding result caching
- `streaming_support.py` — Large file streaming with language-aware break points
- `embedder_optimizer.py` — Transparent embedding optimization wrapper
- `context_packer.py` — Smart context packing with dedup and boilerplate strip
- `context_summarizer.py` — Hierarchical 3-level context packing
- `context_assembler.py` — Cross-file context assembly (callers, tests, interfaces)
- `symbol_index.py` — Symbol search, definitions, references
- `faceted_search.py` — Filtered search with confidence scoring
- `type_search.py` — Type signature search and interface finding
- `multi_buffer.py` — Multi-buffer orchestration, aliases, session persistence
- `intent_router.py` — Intent classification and action planning
- `solver.py` — Automated solve loop with audit trail and rollback
- `diff_aware_search.py` — Diff-aware scoped search
- `conflict_predictor.py` — Conflict prediction via git log analysis
- `why_annotator.py` — "Why this matters" search result annotations
- `undo_redo.py` — Undo/redo stack with git-like branching
- `intent_cache.py` — Intent clustering cache for paraphrased queries
- `refactor_engine.py` — Higher-level refactor operations
- `dependency_graph.py` — Call chain tracing, cycle detection
- `dead_code_detector.py` — Unused symbol detection
- `todo_tracker.py` — TODO/FIXME/HACK extraction
- `quality_scorer.py` — Cyclomatic complexity, docstring coverage
- `conversation_memory.py` — Multi-turn key-value memory
- `git_utils.py` — Git status, diff, blame
- `resource_budget.py` — Pre-embed cost estimation
- `agent_profile.py` — Profile-based chunking strategies
- `phases_integration.py` — Phases 4-10 mixin integration layer
- `path_utils.py` — Path validation for security
- `tool_security.py` — Unified access control, audit logging, rate limiting
- `response_adapters.py` — Response translation between services

## Tool Schemas & Agent Integration

All tools expose formal JSON schemas for agent integration:

```python
from gigacode.gigacode_tool import CodeEmbeddingTool
schemas = CodeEmbeddingTool.get_tool_schemas()
```

**Available tools:**

| Tool | Purpose |
|------|---------|
| `embed_codebase` | Index a codebase into a buffer |
| `check_codebase` | Verify buffer matches disk state |
| `reload_codebase` | Reload buffer if files match hash |
| `semantic_search` | Find code by semantic similarity |
| `semantic_search_streaming` | Search with progressive disclosure (signatures/details/full) |
| `expand_match` | Expand a search match to higher disclosure level |
| `hybrid_search` | Combined semantic + lexical (BM25) search with RRF fusion |
| `faceted_search` | Filtered search with language/path/type/line-count filters |
| `plan_actions` | Classify intent and recommend optimal action sequence |
| `solve` | Automatically plan and execute multi-step coding tasks |
| `search_since_last_edit` | Diff-aware search scoped to dirty files + dependencies |
| `predict_conflicts` | Warn if files were modified externally before commit |
| `annotate_search_results` | Add "why this matters" explanations to search results |
| `search_symbols` | Symbol search (exact/prefix/fuzzy) |
| `get_symbol_definition` | Jump-to-definition for symbols |
| `get_symbol_references` | Find all references to a symbol |
| `search_by_type` | Search by type signature |
| `find_implementations` | Find all implementations of an interface |
| `cluster_code` | Group related code chunks |
| `find_duplicates` | Near-duplicate detection via MinHash + LSH |
| `read_code` | Read file from buffer |
| `write_code` | Edit file in buffer (deferred) |
| `diff` | View pending changes |
| `discard` | Revert file to disk state |
| `commit` | Write changes and rebuild index |
| `undo` | Undo last N operations |
| `redo` | Redo last N undone operations |
| `branch` | Create new branch for experimental edits |
| `checkout` | Switch to a branch |
| `list_branches` | List all branches |
| `pack_context` | Token-budgeted packing of search results |
| `pack_context_smart` | Smart packing with dedup, boilerplate stripping, test exclusion |
| `pack_context_hierarchical` | 3-level hierarchical context (file → chunk → lines) |
| `get_context` | Get cross-file context (callers + tests + interfaces + imports + semantic neighbors) for a symbol |
| `related_code` | Alias for `get_context` — find callers, tests, and interfaces for a symbol |
| `refactor_rename` | Rename symbol across codebase |
| `add_import` | Add import statement to file |
| `remove_import` | Remove import statement from file |
| `git_status` | Git status for buffer directory |
| `git_diff` | Git diff for buffer |
| `git_blame` | Git blame for file |
| `trace_call_chain` | Find call chain between two symbols |
| `find_circular_dependencies` | Detect circular import/reference cycles |
| `find_dead_code` | Find unused symbols in buffer |
| `extract_todos` | Extract TODO/FIXME/HACK/XXX items with priority |
| `score_code_quality` | Score cyclomatic complexity, docstring coverage, nesting depth |
| `estimate_budget` | Pre-embed cost estimation |
| `get_memory_usage` | Current memory usage for buffer |
| `get_audit_log` | Query audit log entries |
| `remember` | Store key-value in conversation memory |
| `recall` | Retrieve from conversation memory |
| `create_alias` | Create buffer alias |
| `resolve_alias` | Resolve buffer alias to ID |
| `list_buffers` | List all buffers |
| `delete_buffer` | Remove a buffer |
| `health_check` | Health check for buffer |
| `get_cache_stats` | Index and query cache statistics |

Schemas are exportable to **OpenAI function-calling** and **MCP** formats via [gigacode/tool_schema.py](gigacode/tool_schema.py).

### HTTP Server

Run a lightweight JSON HTTP server for agent APIs (localhost-only by default):

```bash
python -m gigacode.gigacode_server --work-dir ./buffers --port 8765
```

Or embed first, then serve:

```python
from gigacode.gigacode_server import run_server
from gigacode.gigacode_tool import CodeEmbeddingTool

tool = CodeEmbeddingTool(work_dir="./buffers", device="cpu")
result = tool.embed_codebase("./my_project", pattern="*.py")
print(f"Buffer: {result['buffer_id']}")
run_server(tool, port=8765)
```

**Example request:**

```bash
curl -X POST http://localhost:8765 \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "semantic_search",
    "args": {
      "buffer_id": "<uuid>",
      "query": "def fetch_data",
      "top_k": 10
    }
  }'
```

### MCP Server

Run a Model Context Protocol server for Claude Desktop and Cursor:

```bash
python -m gigacode.mcp_server --work-dir ./buffers
```

Uses stdio transport by default (no network exposure). All GigaCode tools available as MCP tools.

## Language-Agnostic Code Refactoring

GigaCode can automatically improve code across supported languages:

```bash
# Analyze and improve a single file
python gigacode/gigacode_skill.py example.js --language javascript
python gigacode/gigacode_skill.py src/main.rs
```

**Supported improvements:**
- Add documentation comments / docstrings
- Add type annotations / function signatures
- Fix bare exception handlers
- Use context managers for resource handling
- Add explicit visibility modifiers

Language detection is automatic from file extension or shebang.

## Tests

Run the test suite:

```bash
pytest tests/ -v
```
or
```bash
pytest tests/ -q --tb=line
```
**Test coverage:**
- Chunking (AST parsing, sliding windows)
- Embedding models and FAISS index
- Read-write-commit workflow and safety checks
- Incremental diff and hash validation
- Cross-language rules
- Language detection
- Path validation and security
- Streaming support for large files
- Batch embedding optimization
- Audit logging and access control

**Current Status:** 451 tests, 100% pass rate (v0.5.1)

## Project Structure

| File | Purpose |
|------|---------|
| [gigacode/gigacode_tool.py](gigacode/gigacode_tool.py) | Main agent-facing API (65+ methods) |
| [gigacode/gigacode_server.py](gigacode/gigacode_server.py) | Lightweight HTTP server (localhost-only) |
| [gigacode/mcp_server.py](gigacode/mcp_server.py) | Model Context Protocol server |
| [gigacode/gigacode_skill.py](gigacode/gigacode_skill.py) | Language-agnostic code refactoring |
| [gigacode/tool_schema.py](gigacode/tool_schema.py) | JSON schemas for tools (OpenAI, MCP) |
| [gigacode/buffer_manager.py](gigacode/buffer_manager.py) | Buffer management and persistence |
| [gigacode/index_manager.py](gigacode/index_manager.py) | FAISS index caching and management |
| [gigacode/search_service.py](gigacode/search_service.py) | Unified search with streaming + intent caching |
| [gigacode/batch_embedder.py](gigacode/batch_embedder.py) | Batch embedding with optimization |
| [gigacode/streaming_support.py](gigacode/streaming_support.py) | Large file streaming |
| [gigacode/embedder_optimizer.py](gigacode/embedder_optimizer.py) | Optimized embedding wrapper |
| [gigacode/path_utils.py](gigacode/path_utils.py) | Path validation and security |
| [gigacode/tool_security.py](gigacode/tool_security.py) | Access control and audit logging |
| [gigacode/response_adapters.py](gigacode/response_adapters.py) | Response translation |
| [gigacode/language_detect.py](gigacode/language_detect.py) | Language detection from extension/shebang |
| [gigacode/cross_language_rules.py](gigacode/cross_language_rules.py) | Language-agnostic refactoring rules |
| [gigacode/chunker.py](gigacode/chunker.py) | AST-based code chunking (tree-sitter / sliding window) |
| [gigacode/embedder.py](gigacode/embedder.py) | Sentence-transformers embedding model |
| [gigacode/gpu_index.py](gigacode/gpu_index.py) | FAISS CPU+GPU index manager |
| [gigacode/diff_engine.py](gigacode/diff_engine.py) | Incremental diff with hash verification |
| [gigacode/size_guard.py](gigacode/size_guard.py) | Codebase size threshold checks |
| [gigacode/metadata_store.py](gigacode/metadata_store.py) | Compact JSON metadata I/O |
| [gigacode/context_packer.py](gigacode/context_packer.py) | Smart context packing with dedup and boilerplate strip |
| [gigacode/context_summarizer.py](gigacode/context_summarizer.py) | Hierarchical context packing |
| [gigacode/context_assembler.py](gigacode/context_assembler.py) | Cross-file context assembly |
| [gigacode/refactor_engine.py](gigacode/refactor_engine.py) | Higher-level refactor operations |
| [gigacode/faceted_search.py](gigacode/faceted_search.py) | Filtered search with confidence scoring |
| [gigacode/symbol_index.py](gigacode/symbol_index.py) | Symbol search, definitions, references |
| [gigacode/type_search.py](gigacode/type_search.py) | Type signature search and interface finding |
| [gigacode/multi_buffer.py](gigacode/multi_buffer.py) | Multi-buffer orchestration, aliases, virtual buffers |
| [gigacode/resource_budget.py](gigacode/resource_budget.py) | Pre-embed cost estimation, confidence scoring |
| [gigacode/git_utils.py](gigacode/git_utils.py) | Git status, diff, blame, show |
| [gigacode/dependency_graph.py](gigacode/dependency_graph.py) | Call chain tracing, cycle detection |
| [gigacode/dead_code_detector.py](gigacode/dead_code_detector.py) | Unused symbol detection |
| [gigacode/todo_tracker.py](gigacode/todo_tracker.py) | TODO/FIXME/HACK extraction |
| [gigacode/quality_scorer.py](gigacode/quality_scorer.py) | Cyclomatic complexity, docstring coverage |
| [gigacode/conversation_memory.py](gigacode/conversation_memory.py) | Multi-turn key-value memory |
| [gigacode/audit_logger.py](gigacode/audit_logger.py) | Queryable audit logging |
| [gigacode/intent_router.py](gigacode/intent_router.py) | Intent classification and action planning |
| [gigacode/solver.py](gigacode/solver.py) | Automated solve loop with audit trail |
| [gigacode/diff_aware_search.py](gigacode/diff_aware_search.py) | Diff-aware scoped search |
| [gigacode/agent_profile.py](gigacode/agent_profile.py) | Profile-based chunking strategies |
| [gigacode/conflict_predictor.py](gigacode/conflict_predictor.py) | Conflict prediction via git log analysis |
| [gigacode/why_annotator.py](gigacode/why_annotator.py) | "Why this matters" search result annotations |
| [gigacode/undo_redo.py](gigacode/undo_redo.py) | Undo/redo stack with git-like branching |
| [gigacode/intent_cache.py](gigacode/intent_cache.py) | Intent clustering cache for paraphrased queries |
| [gigacode/phases_integration.py](gigacode/phases_integration.py) | Phases 4-10 mixin integration layer |
| [benchmark.py](benchmark.py) | Performance benchmarking |
| [scripts/check_gpu.py](scripts/check_gpu.py) | GPU detection and configuration |
| [scripts/profile_performance.py](scripts/profile_performance.py) | Performance profiling harness |
| [scripts/streaming_integration_guide.py](scripts/streaming_integration_guide.py) | Streaming integration patterns |
| [tests/](tests/) | Pytest test suite |
| [examplecode/](examplecode/) | Example codebase |

## Incremental Updates & Persistence

### Dirty Queue

When you edit files, changes accumulate in memory:

1. `write_code()` updates the snapshot (fast: ~0.5 ms)
2. Chunks are **not** re-embedded until `commit()`
3. On `commit()`, only dirty files are re-chunked and re-embedded

### Buffer Storage

Buffers persist in `.gcbuff/` directories:

```
work_dir/
├── registry.json          # Buffer metadata
└── <uuid>.gcbuff/
    ├── embeddings.npy     # Embedding vectors
    ├── chunks.json        # Code chunks
    ├── index.faiss        # FAISS index
    └── metadata_snapshot.json # File metadata only
```

**Reload without re-embedding:**

```python
# If files haven't changed (hash match), skip re-embedding
result = tool.reload_codebase(buffer_id)
```

### State Tracking

Buffer state machine tracks lifecycle:

- **READY**: Buffer is ready for queries
- **DIRTY**: Files have uncommitted changes
- **REBUILDING**: Index is being rebuilt after commit

Transitions are validated to prevent invalid state combinations.

## Requirements

- **Python 3.9+**
- **PyTorch** (CPU or CUDA)
- **sentence-transformers** (embedding models)
- **faiss-gpu** (recommended) or **faiss-cpu** (vector search)
- **NumPy** (numeric arrays)
- **tree-sitter** + language grammars (AST parsing)

For GPU support, ensure CUDA 11.8+ and cuDNN 8.x are installed.

**Version pinning:**
All dependencies are pinned with tilde-equal constraints (~=) in `pyproject.toml` and `requirements.txt` for reproducible environments.

## Troubleshooting

### Out-of-memory errors

- Reduce `threshold_mb` when embedding large codebases
- Use CPU mode: `device="cpu"`
- Enable streaming for large files: automatic for files >50MB
- Increase system virtual memory

### Slow embedding

- Tree-sitter requires language grammars; ensure `tree-sitter-python`, etc. are installed
- Use `pattern="*.py"` to narrow file scope
- Consider splitting large projects into multiple buffers
- Batch embedding is automatic for large batches (>100 texts)

### FAISS GPU not found

```bash
# Check GPU availability
python scripts/check_gpu.py

# Reinstall faiss-gpu with proper CUDA
pip uninstall faiss-cpu faiss-gpu
pip install ".[gpu]"
```

### Commit fails with "file modified externally"

GigaCode detected that a file changed on disk since embedding. Use `discard()` to revert or `reload_codebase()` to re-embed.

### Import errors or version conflicts

Create a clean environment and install with explicit dependency pinning:

```bash
python3 -m venv .venv-clean
source .venv-clean/bin/activate
pip install -r requirements.txt
```

## Local-Only Design

GigaCode is designed to run entirely on your machine with no network exposure:

- **HTTP server** defaults to `127.0.0.1` (localhost only)
- **MCP server** uses stdio transport (no network listening)
- **No remote authentication** required
- **No cloud dependencies** — all processing local
- **All data stays on disk** in your work directory

For development-only HTTP servers, the server will warn if exposed to the network.

## Contributing

Contributions are welcome! Please:

1. Fork and create a feature branch
2. Add tests for new functionality
3. Run `pytest tests/ -v` to verify
4. Submit a pull request

## License

MIT
