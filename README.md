# GigaCode

[![Version](https://img.shields.io/badge/version-0.6.1-blue?style=for-the-badge)](VERSION)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-yellow?style=for-the-badge)](pyproject.toml)

**GPU-accelerated code embedding and semantic search for AI agents.**
Embed a codebase into searchable chunks, run semantic search, navigate references,
detect code smells and security vulnerabilities, and edit code through a safe
read-write-commit workflow -- all from a single tool interface with 67 agent-discoverable capabilities.

GigaCode is optimized for AI agent loops -- fast chunking, sub-millisecond search on GPU,
surgical index updates on edit, and full tool schema export in OpenAI, Anthropic, MCP,
and Ollama formats. Runs locally with no network exposure.

## Foreword
Noticed AI is slow at searching/parsing code so I made this.
AI agents searching for code snippets is blazingly fast now with this tool/skill.

Code benchmarks show that AI agent coding is 4x faster, however AI token rate will always be the bottleneck.
Reduced token usage by about 60% on average.
Searching for code is orders of magnitude faster (100x +) (depends on how much GPU VRAM you want to use).
Small code bases do not take up that much memory.

After releasing this tool I have noticed that some AI Agents now include semantic code searches.
I will still develop this tool for open-source.

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
from gigacode.gigacode_tool import CodeEmbeddingTool

with CodeEmbeddingTool(work_dir="./buffers", device="cpu") as tool:
    # Embed a codebase
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
- **write_code** only touches the in-buffer snapshot -- disk is unchanged until commit()
- **diff** shows a unified diff of all pending changes before you commit
- **discard** reverts all in-buffer changes instantly
- **commit** rebuilds the embedding index for only the changed files (not the whole codebase)

## Tool Reference

**67 agent-discoverable tools** across 8 categories.
**50 read-only** (safe to call anytime) | **17 mutating** (modify buffer or disk).

| Category | Tools | Read-Only | Mutating |
|----------|-------|:---------:|:--------:|
| Analysis | 16 | 16 | 0 |
| Editing | 11 | 2 | 9 |
| Indexing | 5 | 2 | 3 |
| Navigation | 6 | 6 | 0 |
| Quality | 8 | 4 | 4 |
| Safety | 7 | 6 | 1 |
| Search | 13 | 13 | 0 |
| Security | 1 | 1 | 0 |

### Read-Only vs Mutating

| Side Effect | Count | Description |
|-------------|:-----:|-------------|
| Read-only | 50 | Safe to call anytime. Only reads buffer state. |
| Write | 17 | Modifies in-buffer snapshot or disk files. Preview with dry_run=True. |
| Destructive | 2 | Permanently removes data (delete_buffer, discard). Use with care. |

## Schema Export Formats

GigaCode exports tool schemas in 4 formats for direct integration with major AI frameworks:

| Format | Use Case | Export |
|--------|----------|--------|
| **OpenAI** | Function calling | `tool_schema.to_openai_functions()` |
| **Anthropic** | Tool use | `tool_schema.to_anthropic_tools()` |
| **MCP** | Model Context Protocol | `tool_schema.to_mcp_tools()` |
| **Ollama** | Local LLM tooling | `tool_schema.to_ollama_tools()` |

All formats include typed input/output schemas, categories, side-effect annotations,
and usage examples for automatic AI agent discovery.

## Configuration

Create `gigacode.toml` in your project root to configure schema export defaults:

```toml
[schemas]
format = "openai"          # openai | anthropic | mcp | ollama
include_metadata = true    # categories, tags, side_effects
default_category = "all"
read_only_only = false     # filter to read-only tools only
```

See `gigacode.toml.example` for all options.

## Architecture

```
CodeEmbeddingTool
  |-- BufferManager       (state, dirty tracking, registry)
  |-- IndexManager        (Faiss index, embeddings, chunk store)
  |-- SearchService       (semantic, hybrid, faceted, type-aware)
  |-- ReferenceMap        (lazy incremental call graph)
  |-- CodeQuality         (format, lint, polish)
  |-- UndoRedoService     (surgical undo/redo with branching)
  |-- ProfileAdapter      (agent-profile-based chunking)
  |-- SecurityScanner     (security vulnerability detection)
  |-- gigacode_api        (FastAPI server + OpenAPI schemas)
  |-- gigacode_server     (MCP server for Claude Desktop)
```

## API Server

Start the FastAPI server:

```bash
# Default (port 8765)
python -m gigacode.gigacode_server --work-dir ./buffers

# Custom port
python -m gigacode.gigacode_server --work-dir ./buffers --port 8080
```

Or import the app directly:

```python
from gigacode.gigacode_api import create_app
from gigacode.gigacode_tool import CodeEmbeddingTool

tool = CodeEmbeddingTool(work_dir="./buffers")
app = create_app(tool)
```

## MCP Server (Claude Desktop)

```bash
# stdio transport (for Claude Desktop integration)
python -m gigacode.mcp_server --work-dir ./buffers

# HTTP-SSE transport
python -m gigacode.mcp_server --work-dir ./buffers --transport sse --port 8766
```

## License

MIT License -- see [LICENSE](LICENSE).
