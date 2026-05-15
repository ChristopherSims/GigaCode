# GigaCode

[![Version](https://img.shields.io/badge/version-0.6.1-blue?style=for-the-badge)](VERSION)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-yellow?style=for-the-badge)](pyproject.toml)

**GPU-accelerated code embedding and semantic search for AI agents.** Embed a codebase into searchable chunks, run semantic search, navigate references, detect code smells and security vulnerabilities, and edit code through a safe read-write-commit workflow -- all from a single tool interface with 54 agent-discoverable capabilities.

GigaCode is optimized for AI agent loops -- fast chunking, sub-millisecond search on GPU, surgical index updates on edit, and full tool schema export in OpenAI, Anthropic, MCP, and Ollama formats. Runs locally with no network exposure.

---

## Quick Start

### Installation

```bash
pip install .

# Or with GPU support (requires CUDA/cuDNN)
pip install ".[gpu]"

# Development tools
pip install ".[dev]"
```

**System Requirements:**
- CPU-only: Python 3.10+, any platform
- GPU: NVIDIA GPU with CUDA 11.8+, cuDNN 8.x

### Embed and Search

```python
from gigacode.gigacode_tool import CodeEmbeddingTool

with CodeEmbeddingTool(work_dir="./buffers", device="cpu") as tool:
    # Embed a codebase
    result = tool.embed_codebase("./src", pattern="*.py")
    buf_id = result["buffer_id"]

    # Semantic search
    search = tool.semantic_search(buf_id, "authentication middleware", top_k=5)
    for match in search["matches"]:
        print(f"{match['file']}:{match['start_line']} (score: {match['score']:.3f})")

    # Get full context for a symbol
    ctx = tool.get_full_context(buf_id, "authenticate")
    print(f"Callers: {ctx['callers']}, Tests: {ctx['tests']}")
```

### Edit Code Safely

```python
with CodeEmbeddingTool(work_dir="./buffers", device="cpu") as tool:
    result = tool.embed_codebase("./src", pattern="*.py")
    buf_id = result["buffer_id"]

    # Write changes (deferred -- kept in memory)
    tool.write_code(buf_id, file="main.py", start_line=5,
                    new_lines=["    return value"])

    # Preview before commit
    diff = tool.diff(buf_id)

    # Commit to disk (rebuilds index for dirty files)
    tool.commit(buf_id, dry_run=False)
```

Safety guarantees:
- `commit()` aborts if the original file was modified externally since embedding
- All changes kept in memory until `commit()` is called
- Use `dry_run=True` to preview without writing
- Path traversal attempts are blocked at API boundaries

---

## 54 Agent-Discoverable Tools

All tools expose formal JSON schemas with full input/output type definitions, categorization, side-effect annotations, and usage examples. Exportable in 4 formats for AI agent integration.

### Indexing

| Tool | Purpose | Side Effects |
|------|---------|-------------|
| `embed_codebase` | Index a codebase into a searchable buffer | Creates buffer, allocates memory |
| `reload_codebase` | Re-embed changed files in-place | Updates index |
| `check_codebase` | Verify buffer matches disk state | None (read-only) |
| `list_buffers` | List all active buffers | None (read-only) |
| `delete_buffer` | Remove a buffer and its data | Destructive -- cannot undo |

### Search

| Tool | Purpose | Side Effects |
|------|---------|-------------|
| `semantic_search` | Find code by natural language query | None |
| `hybrid_search` | Semantic + lexical (BM25) with RRF fusion | None |
| `search_for` | Literal string search | None |
| `search_symbols` | Exact/prefix/fuzzy symbol search | None |
| `cluster_code` | Group related code chunks | None |
| `find_duplicates` | Near-duplicate detection (MinHash + LSH) | None |
| `pack_context` | Token-budgeted context packing | None |
| `search_batch` | Run up to 20 search queries at once | None |
| `find_similar_patterns` | Semantic + syntactic similar code search | None |

### Navigation

| Tool | Purpose | Side Effects |
|------|---------|-------------|
| `read_code` | Read file from buffer | None |
| `look_for_file` | Find files by glob pattern | None |
| `get_references` | Callers, callees, and references for a symbol | None |
| `get_full_context` | Definition + callers + tests + types + errors | None |
| `get_symbol_metadata` | Complexity, callers, docstring, type info | None |
| `get_test_coverage` | Source-to-test line range mapping | None |

### Editing

| Tool | Purpose | Side Effects |
|------|---------|-------------|
| `write_code` | Edit file in buffer (deferred) | Modifies in-buffer snapshot |
| `diff` | View pending changes | None |
| `discard` | Revert file to disk state | Discards in-buffer changes |
| `commit` | Write changes to disk and rebuild index | Writes to disk -- use dry_run first |

### Analysis

| Tool | Purpose | Side Effects |
|------|---------|-------------|
| `trace_execution_paths` | AST branch detection + call graph traversal | None |
| `get_dependency_graph` | Nodes + edges for graph visualization | None |
| `detect_code_smells` | Long functions, deep nesting, missing docstrings | None |
| `find_performance_hotspots` | N+1 queries, nested loops, resource leaks | None |
| `generate_documentation` | Auto-generate docstrings (Google/NumPy/Sphinx) | None |
| `find_deprecated` | Detect @deprecated and DeprecationWarning usage | None |
| `analyze_logging_patterns` | Log levels, missing logs, inconsistencies | None |
| `analyze_error_handling_patterns` | Broad catches, missing finally, silent pass | None |
| `map_api_endpoints` | Enumerate FastAPI/Flask endpoints | None |
| `analyze_cache_patterns` | Cache invalidation logic, stale data risks | None |
| `analyze_thread_safety` | Shared state, race conditions, deadlocks | None |
| `detect_memory_issues` | Circular refs, unbounded collections, leaks | None |
| `infer_types` | Type inference (AST or LLM) with confidence | None |

### Safety

| Tool | Purpose | Side Effects |
|------|---------|-------------|
| `analyze_change` | Pre-edit impact assessment with risk scoring | None |
| `scan_security` | eval, exec, injection, hardcoded secrets | None |
| `suggest_refactorings` | Safe refactoring suggestions with risk | None |
| `validate_changes` | Syntax + import resolution before commit | None |
| `detect_api_changes` | Breaking changes between commits | None |
| `get_rollback_info` | Git history + diff-to-revert | None |
| `generate_change_template` | Change plan from natural language | None |
| `generate_changelog` | Git log categorized as features/bugfixes/breaking | None |
| `extract_configuration` | Env vars, config files, hardcoded secrets | None |

### Quality

| Tool | Purpose | Side Effects |
|------|---------|-------------|
| `auto_format` | Format code with Black or ruff format | Modifies files when dry_run=False |
| `auto_lint` | Lint with Ruff (report-only or auto-fix) | Modifies files when auto_fix=True |
| `auto_polish` | Combined format + lint convenience wrapper | Modifies files when dry_run=False |
| `lint_buffer` | Deep lint analysis grouped by file/severity/rule | None |
| `format_buffer` | Deep format analysis with change tracking | None |
| `polish_before_commit` | Format + lint + impact + large diff check | Modifies files when check_only=False |
| `lint_with_config` | Config-aware linting (ruff.toml, pyproject.toml) | See auto_lint |
| `format_with_config` | Config-aware formatting | See auto_format |

---

## AI Agent Integration

### Schema Export Formats

Export tool schemas in the format your AI framework expects:

```python
from gigacode.tool_schema import export_schemas, SchemaFormat

# OpenAI / Ollama format
openai_tools = export_schemas("openai")

# Anthropic / Claude format
anthropic_tools = export_schemas("anthropic")

# MCP (Model Context Protocol) format
mcp_tools = export_schemas("mcp")

# Filter by category or safety
search_only = export_schemas("anthropic", category="search")
safe_only = export_schemas("openai", read_only_only=True)
```

### Config File

Create `gigacode.toml` in your project root:

```toml
schema_format = "anthropic"    # "openai" | "anthropic" | "mcp" | "ollama"
include_metadata = true        # Attach category, tags, side-effects
read_only_only = false        # Only export safe tools
# default_category = "search" # Filter to a single category
```

Or add a `[tool.gigacode]` section to `pyproject.toml`.

### HTTP API

```bash
python -m gigacode.gigacode_server --work-dir ./buffers --port 8765
```

Or use the FastAPI app directly:

```python
from gigacode.gigacode_tool import CodeEmbeddingTool
from gigacode.gigacode_api import create_app

tool = CodeEmbeddingTool(work_dir="./buffers", device="cpu")
app = create_app(tool)
```

Schema export endpoints:

```bash
# Export schemas in any format
curl "http://localhost:8765/schemas?format=anthropic&category=search"
curl "http://localhost:8765/schemas?format=mcp&read_only_only=true"

# List categories
curl "http://localhost:8765/schemas/categories"

# Read current config
curl "http://localhost:8765/schemas/config"
```

Production server with auth and rate limiting:

```python
from gigacode.gigacode_api import create_production_app

app = create_production_app(tool, api_key="my-secret-key", rate_limit_calls=100)
```

### MCP Server

```bash
python -m gigacode.mcp_server --work-dir ./buffers
```

Uses stdio transport by default (no network exposure). All 54 tools available as MCP tools with annotations for categorization and safety.

---

## Performance

- AST chunking reduces embedding count 5-20x vs per-line
- FAISS ANN search: ~0.1ms on GPU, ~20ms on CPU for 100K+ chunks
- `write_code` is ~0.5ms because re-embedding is deferred until `commit`
- Surgical index updates: only dirty files are re-chunked and patched into FAISS
- Batch optimization: 2-5x faster embedding for large batches with LRU caching

### Benchmark

```bash
python benchmark.py --dir examplecode/ --search-iters 50 --edit-iters 5
```

Example output (44 chunks, CPU):

```
embed_codebase : 1.57s
semantic_search: 20.6 ms median (CPU; ~0.1 ms on GPU)
cluster_code   : 1.9 ms
write_code     : 0.59 ms (deferred rebuild)
```

---

## Architecture

### Tool Categories

```
Codebase
   |
   v
Chunker (tree-sitter AST: functions/classes, sliding-window fallback)
   |
   v
OptimizedEmbedder (batch optimization + LRU cache)
   |
   v
FAISS Index (CPU IDMap + FlatIP)
   |
   +-- GPU mirror (faiss.index_cpu_to_gpu) -- rebuilt lazily on edit
   +-- SymbolIndex (exact/prefix/fuzzy symbol search)
   +-- ReferenceMap (incremental caller/callee graph)
   +-- TypeInferenceCache (session-scoped LRU with write-invalidation)
   |
   v
SearchService (semantic + hybrid + symbol + faceted + batch)
   |
   +-- ContextPacker (smart dedup + boilerplate strip)
   +-- DiffAwareSearch (scoped to dirty files + dependencies)
   +-- IntentCache (3-layer: index -> semantic -> intent clusters)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `gigacode_tool.py` | Main agent-facing API (54+ methods) |
| `tool_schema.py` | JSON schemas with format export (OpenAI/Anthropic/MCP/Ollama) |
| `gigacode_api.py` | FastAPI REST server with Pydantic models |
| `gigacode_server.py` | Lightweight HTTP server (localhost-only) |
| `mcp_server.py` | Model Context Protocol server |
| `search_service.py` | Unified search with streaming + intent caching |
| `reference_map.py` | Incremental caller/callee graph (3-phase strategy) |
| `execution_paths.py` | AST branch detection + call graph traversal |
| `type_inference_cache.py` | Session-scoped LRU cache with write-invalidation |
| `code_quality.py` | Auto-format (Black/ruff), auto-lint (Ruff), auto-polish |
| `pydantic_models.py` | Typed request/response models for all endpoints |
| `dependency_graph.py` | Call chain tracing, cycle detection |
| `dead_code_detector.py` | Unused symbol detection |
| `duplicate_detector.py` | MinHash + LSH near-duplicate detection |
| `impact_analyzer.py` | Pre-edit impact assessment with risk scoring |
| `quality_scorer.py` | Cyclomatic complexity, docstring coverage |
| `context_assembler.py` | Cross-file context assembly |
| `buffer_manager.py` | Buffer registry, persistence, file I/O |
| `index_manager.py` | FAISS index caching and management |
| `git_utils.py` | Git status, diff, blame, run_git |

---

## Buffer Persistence

Buffers persist in `.gcbuff/` directories:

```
work_dir/
  registry.json              # Buffer metadata
  <uuid>.gcbuff/
    embeddings.npy           # Embedding vectors
    chunks.json              # Code chunks
    index.faiss              # FAISS index
    metadata_snapshot.json   # File metadata
```

Reload without re-embedding if files have not changed (hash match):

```python
result = tool.reload_codebase(buffer_id)
```

### State Machine

Buffer states: **READY** -> **DIRTY** (after write_code) -> **REBUILDING** (during commit) -> **READY**

Transitions are validated to prevent invalid combinations.

---

## Requirements

- Python 3.10+
- PyTorch (CPU or CUDA)
- sentence-transformers (embedding models)
- faiss-gpu (recommended) or faiss-cpu (vector search)
- NumPy
- tree-sitter + language grammars (AST parsing)

For GPU support, ensure CUDA 11.8+ and cuDNN 8.x are installed.

---

## Local-Only Design

GigaCode runs entirely on your machine with no network exposure:

- HTTP server defaults to 127.0.0.1 (localhost only)
- MCP server uses stdio transport (no network listening)
- No cloud dependencies -- all processing is local
- All data stays on disk in your work directory

---

## Contributing

Contributions are welcome. Please:

1. Fork and create a feature branch
2. Add tests for new functionality
3. Run `pytest tests/ -v` to verify
4. Submit a pull request

## License

MIT
