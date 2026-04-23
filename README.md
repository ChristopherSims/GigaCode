Noticed AI is slow at searching/parsing code so I made this.
AI agents searching for code snippits is blazingly fast now with this tool/skill.

Code benchmarks show that AI agent coding is 4x faster, however AI token rate will always be the bottleneck
# GigaCode

**GPU-accelerated code embedding for AI agents.** Embed a codebase into searchable chunks, run semantic search and clustering, and edit code through a safe read-write-commit workflow.

GigaCode is optimized for AI agent loops—fast chunking, sub-millisecond search on GPU, and surgical index updates on edit.

## Key Features

- **AST-based chunking** — functions, classes, and methods extracted via tree-sitter (falls back to sliding windows)
- **Semantic search** with FAISS — sub-millisecond approximate nearest neighbor on GPU, single-digit ms on CPU
- **Persistent GPU index** — embeddings stay in VRAM, auto-syncs on edit
- **Agent read-write-commit workflow** — hash-based safety checks prevent external file modifications
- **Deferred batch rebuilds** — edits accumulate in a dirty queue; re-embedding batched until `commit`
- **Code-specific embeddings** — `jina-embeddings-v2-base-code` by default (falls back to `all-MiniLM-L6-v2`)
- **Incremental updates** — only changed files are re-chunked and re-embedded
- **Language-agnostic editing** — cross-language rules for docstrings, type annotations, and refactoring

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** For GPU search, install `faiss-gpu` instead of `faiss-cpu` in `requirements.txt` and ensure CUDA/cuDNN are available.

### Embed and Search

```python
from src.gigacode_tool import CodeEmbeddingTool

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

## Performance

GigaCode is optimized for fast agent loops:

- **AST chunking** reduces embedding count 5–20× vs per-line
- **FAISS ANN** search: **~0.1 ms** on GPU, **~20 ms** on CPU for 100K+ chunks
- **Deferred rebuilds**: `write_code` is **~0.5 ms** because re-embedding is batched until `commit`
- **Surgical index updates**: only dirty files are re-chunked and patched into FAISS

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

```
Codebase
   |
   v
Chunker (tree-sitter AST: functions/classes, sliding-window fallback)
   |
   v
Embedder (code-specific sentence-transformers model)
   |
   v
FAISS Index (CPU IDMap + FlatIP)
   |
   +-- GPU mirror (faiss.index_cpu_to_gpu) — rebuilt lazily on edit
   |
   +-- search() uses GPU if available & clean, else CPU
```

## Tool Schemas & Agent Integration

All tools expose formal JSON schemas for agent integration:

```python
from src.gigacode_tool import CodeEmbeddingTool
schemas = CodeEmbeddingTool.get_tool_schemas()
```

**Available tools:**

| Tool | Purpose |
|------|---------|
| `embed_codebase` | Index a codebase into a buffer |
| `check_codebase` | Verify buffer matches disk state |
| `reload_codebase` | Reload buffer if files match hash |
| `semantic_search` | Find code by semantic similarity |
| `cluster_code` | Group related code chunks |
| `read_code` | Read file from buffer |
| `write_code` | Edit file in buffer (deferred) |
| `diff` | View pending changes |
| `discard` | Revert file to disk state |
| `commit` | Write changes and rebuild index |
| `list_buffers` | List all buffers |
| `delete_buffer` | Remove a buffer |

Schemas are exportable to **OpenAI function-calling** and **MCP** formats via [src/tool_schema.py](src/tool_schema.py).

### HTTP Server

Run a lightweight JSON HTTP server for agent APIs:

```bash
python -m src.gigacode_server --work-dir ./buffers --port 8765
```

Or embed first, then serve:

```python
from src.gigacode_server import run_server
from src.gigacode_tool import CodeEmbeddingTool

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

## Language-Agnostic Code Refactoring

GigaCode can automatically improve code across supported languages:

```bash
# Analyze and improve a single file
python src/gigacode_skill.py example.js --language javascript
python src/gigacode_skill.py src/main.rs
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

**Test coverage:**
- Chunking (AST parsing, sliding windows)
- Embedding models and FAISS index
- Read-write-commit workflow and safety checks
- Incremental diff and hash validation
- Cross-language rules
- Language detection

## Project Structure

| File | Purpose |
|------|---------|
| [src/gigacode_tool.py](src/gigacode_tool.py) | Main agent-facing API |
| [src/gigacode_server.py](src/gigacode_server.py) | Lightweight HTTP server |
| [src/gigacode_skill.py](src/gigacode_skill.py) | Language-agnostic code refactoring |
| [src/tool_schema.py](src/tool_schema.py) | JSON schemas for tools (OpenAI, MCP) |
| [src/language_detect.py](src/language_detect.py) | Language detection from extension/shebang |
| [src/cross_language_rules.py](src/cross_language_rules.py) | Language-agnostic refactoring rules |
| [src/chunker.py](src/chunker.py) | AST-based code chunking (tree-sitter / sliding window) |
| [src/embedder.py](src/embedder.py) | Sentence-transformers embedding model |
| [src/gpu_index.py](src/gpu_index.py) | FAISS CPU+GPU index manager |
| [src/diff_engine.py](src/diff_engine.py) | Incremental diff with hash verification |
| [src/size_guard.py](src/size_guard.py) | Codebase size threshold checks |
| [src/metadata_store.py](src/metadata_store.py) | Compact JSON metadata I/O |
| [benchmark.py](benchmark.py) | Performance benchmarking |
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
    └── source_snapshot.json # File snapshots
```

**Reload without re-embedding:**

```python
# If files haven't changed (hash match), skip re-embedding
result = tool.reload_codebase(buffer_id)
```

## Requirements

- **Python 3.10+**
- **PyTorch** (CPU or CUDA)
- **sentence-transformers** (embedding models)
- **faiss-gpu** (recommended) or **faiss-cpu** (vector search)
- **NumPy** (numeric arrays)
- **tree-sitter** + language grammars (AST parsing)

For GPU support, ensure CUDA 11.8+ and cuDNN 8.x are installed.

## Troubleshooting

### Out-of-memory errors

- Reduce `threshold_mb` when embedding large codebases
- Use CPU mode: `device="cpu"`
- Increase system virtual memory

### Slow embedding

- Tree-sitter requires language grammars; ensure `tree-sitter-python`, etc. are installed
- Use `pattern="*.py"` to narrow file scope
- Consider splitting large projects into multiple buffers

### FAISS GPU not found

```bash
# Reinstall faiss-gpu with proper CUDA
pip uninstall faiss-cpu faiss-gpu
pip install faiss-gpu
```

### Commit fails with "file modified externally"

GigaCode detected that a file changed on disk since embedding. Use `discard()` to revert or `reload_codebase()` to re-embed.

## Contributing

Contributions are welcome! Please:

1. Fork and create a feature branch
2. Add tests for new functionality
3. Run `pytest tests/ -v` to verify
4. Submit a pull request

## License

MIT
