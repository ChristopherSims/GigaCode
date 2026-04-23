# GigaCode

GPU-accelerated code embedding for AI agents. Embed a codebase into searchable chunks, run semantic search and clustering, and edit code through a safe read-write-commit workflow.

## Features

- **AST-based chunking** — functions, classes, and methods extracted via tree-sitter (falls back to sliding windows)
- **Semantic search** with FAISS — approximate nearest neighbor on GPU when available, CPU fallback
- **Persistent GPU index** — embeddings stay in VRAM for sub-millisecond search (auto-syncs on edit)
- **Agent read-write-commit** workflow with hash-based safety checks
- **Deferred batch rebuilds** — edits accumulate in a dirty queue; re-embedding happens on commit or threshold
- **Code-specific embeddings** — defaults to `jina-embeddings-v2-base-code` (falls back to `all-MiniLM-L6-v2`)
- **Incremental updates** — only changed files are re-chunked and re-embedded

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Embed and search

```python
from src.gigacode_tool import CodeEmbeddingTool

with CodeEmbeddingTool(work_dir="./buffers", device="cpu") as tool:
    result = tool.embed_codebase("./examplecode", pattern="*.py")
    buf_id = result["buffer_id"]

    search = tool.semantic_search(buf_id, "sorting algorithm", top_k=5)
    for m in search["matches"]:
        print(m["file"], m["start_line"], m["end_line"], m["score"])

    clusters = tool.cluster_code(buf_id, threshold=0.75)
    for c in clusters["clusters"]:
        print(c["file"], c["start_line"], c["end_line"], c["size"])
```

### Edit code through the buffer

```python
with CodeEmbeddingTool(work_dir="./buffers", device="cpu") as tool:
    result = tool.embed_codebase("./src", pattern="*.py")
    buf_id = result["buffer_id"]

    # Read
    read = tool.read_code(buf_id, file="main.py")
    for line in read["lines"]:
        print(line)

    # Write (fast — only updates snapshot, defers re-embed)
    tool.write_code(
        buf_id,
        file="main.py",
        start_line=5,
        new_lines=["    # Added by agent", "    pass"],
    )

    # Preview changes
    diff = tool.diff(buf_id)
    for f in diff["changed_files"]:
        print(f"Changed: {f['file']} ({f['buffer_lines']} lines)")

    # Commit to disk (rebuilds index for dirty files, then writes)
    tool.commit(buf_id, dry_run=False)
```

- `discard(buf_id, file="main.py")` reverts a file to its on-disk state
- `commit` aborts if the original file was modified externally since embedding

## Performance

GigaCode is optimized for fast agent loops:

- **AST chunking** reduces embedding count 5-20x vs per-line
- **FAISS ANN** search is sub-millisecond even for 100K+ chunks (GPU) or single-digit ms (CPU)
- **Deferred rebuilds** — `write_code` is ~0.5 ms because re-embedding is batched until `commit`
- **Surgical index updates** — only dirty files are re-chunked and patched into the FAISS index

Run the benchmark:

```bash
python benchmark.py --dir examplecode/ --search-iters 50 --edit-iters 5
```

Example output on `examplecode/` (CPU, 44 chunks):

```
embed_codebase : 1.57s
semantic_search: ~20.6 ms median (CPU FAISS; ~0.1 ms expected on GPU)
cluster_code   : ~1.9 ms
write_code     : ~0.59 ms median (deferred rebuild)
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

## Tool Schemas

All tools expose formal JSON schemas for agent integration:

```python
from src.gigacode_tool import CodeEmbeddingTool
schemas = CodeEmbeddingTool.get_tool_schemas()
```

Available tools:
- `embed_codebase` / `check_codebase` / `reload_codebase`
- `semantic_search` / `cluster_code`
- `read_code` / `write_code` / `diff` / `discard` / `commit`
- `list_buffers` / `delete_buffer`

Also exportable to OpenAI function-calling and MCP formats via `src.tool_schema`.

## Language-Agnostic Editing

Edit any language GigaCode can parse:

```bash
python src/gigacode_skill.py example.js --language javascript
python src/gigacode_skill.py src/main.rs
```

Supported improvements (language-aware via tree-sitter or regex fallback):
- Add documentation comments / docstrings
- Add type annotations / signatures
- Fix bare exception / catch blocks
- Use context managers for resource handling
- Add explicit visibility modifiers

## Incremental Updates

The dirty-queue tracks edited files. On `commit()`, only dirty files are:
1. Re-chunked via tree-sitter
2. Re-embedded in a batch
3. Old chunks removed from the FAISS index
4. New chunks added with fresh IDs
5. Files written to disk

## Persistence

Buffers are stored in `.gcbuff/` directories under the working directory:

```
work_dir/
├── registry.json
└── <uuid>.gcbuff/
    ├── embeddings.npy
    ├── chunks.json
    ├── index.faiss
    └── source_snapshot.json
```

Use `reload_codebase(buffer_id)` to skip re-embedding when file hashes match.

## Tests

```bash
pytest tests/ -v
```

## File Overview

| File | Purpose |
|------|---------|
| `src/gigacode_tool.py` | Main agent interface |
| `src/gigacode_skill.py` | Language-agnostic code editing agent |
| `src/tool_schema.py` | Formal JSON schemas for all tools |
| `src/language_detect.py` | Language detection from extension / shebang |
| `src/cross_language_rules.py` | Language-agnostic editing rules |
| `src/chunker.py` | AST-based code chunking (tree-sitter / sliding window) |
| `src/embedder.py` | Code embedding model wrapper |
| `src/gpu_index.py` | FAISS CPU+GPU index manager |
| `src/diff_engine.py` | Incremental diff with parallel hashing |
| `src/size_guard.py` | Size threshold guard |
| `src/metadata_store.py` | Compact JSON metadata I/O |
| `benchmark.py` | Performance benchmark script |
| `tests/` | Test suite |
| `examplecode/` | Example codebase for testing |

## Requirements

- Python 3.10+
- PyTorch
- sentence-transformers
- faiss-gpu (recommended) or faiss-cpu
- NumPy
- tree-sitter + language grammars

## License

MIT
