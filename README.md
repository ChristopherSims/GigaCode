# GigaCode

GPU-accelerated code embedding for AI agents. Embed a codebase into searchable buffers, run semantic search and clustering, and edit code through a safe read-write-commit workflow.

## Features

- **Semantic search** over code with sentence-transformers embeddings
- **Region clustering** to find similar code blocks
- **Agent read-write-commit** workflow with hash-based safety checks
- **Memory-mapped buffers** for fast search without loading everything into RAM
- **Incremental updates** — only changed files are re-embedded on edit
- **Vulkan compute backend** (with CPU fallback)
- **Language-agnostic** — tree-sitter, tiktoken, or regex tokenization

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Embed and search

```python
from src.agent_tool import CodeEmbeddingTool

with CodeEmbeddingTool(work_dir="./buffers", device="cpu") as tool:
    result = tool.embed_codebase("./examplecode", pattern="*.py")
    buf_id = result["buffer_id"]

    search = tool.semantic_search(buf_id, "sorting algorithm", top_k=5)
    for m in search["matches"]:
        print(m["file"], m["line"], m["score"])

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

    # Write
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

    # Commit to disk
    tool.commit(buf_id, dry_run=False)
```

- `discard(buf_id, file="main.py")` reverts a file to its on-disk state
- `commit` aborts if the original file was modified externally since embedding

## Performance

GigaCode is optimized for fast agent loops:

- **Vectorized flattening** — embeddings are stacked with NumPy instead of per-line Python loops
- **Compact JSON I/O** — metadata is persisted without pretty-printing overhead
- **Memory-mapped search** — `embeddings.bin` is accessed via `np.memmap`, so search only touches the pages the CPU needs
- **Surgical rebuilds** — editing a file re-embeds only that file and splices head + new + tail bytes directly to disk
- **Larger default batch size** — embedding batch size defaults to 256 for fewer forward passes

Run the benchmark:

```bash
python benchmark.py --dir examplecode/ --search-iters 10 --edit-iters 5
```

Example output on `examplecode/` (CPU, 382 lines):

```
embed_codebase : 1.71s
semantic_search: ~11ms median
cluster_code   : ~3ms
write_code     : ~115ms median (dominated by re-embedding the changed file)
```

## Architecture

```
Codebase
   |
   v
Tokenizer (tree-sitter / tiktoken / regex)
   |
   v
Embedder (sentence-transformers)
   |
   v
Flatten + Size Guard
   |
   v
VkBuffer (GPU) or NumPy memmap (CPU fallback)
   |
   +-- Compute Shader: similarity_search.comp (Top-K NNS)
   +-- Compute Shader: cluster_regions.comp (region clustering)
```

## Tool Schemas

All tools expose formal JSON schemas for agent integration:

```python
from src.agent_tool import CodeEmbeddingTool
schemas = CodeEmbeddingTool.get_tool_schemas()
```

Available tools:
- `embed_codebase` / `check_codebase` / `reload_codebase`
- `semantic_search` / `cluster_code`
- `read_code` / `write_code` / `diff` / `discard` / `commit`
- `list_buffers` / `delete_buffer`

Also exportable to OpenAI function-calling and MCP formats via `src.tool_schema`.

## Language-Agnostic Editing

Edit any language GigaCode can tokenize:

```bash
python src/agent_skill.py example.js --language javascript
python src/agent_skill.py src/main.rs
```

Supported improvements (language-aware via tree-sitter or regex fallback):
- Add documentation comments / docstrings
- Add type annotations / signatures
- Fix bare exception / catch blocks
- Use context managers for resource handling
- Add explicit visibility modifiers

## Incremental Updates

The diff engine tracks per-line SHA-256 hashes. On re-ingest, only changed lines are re-embedded and patched into the buffer.

## Persistence

Buffers are stored in `.vkbuff/` directories under the working directory:

```
work_dir/
├── registry.json
└── <uuid>.vkbuff/
    ├── embeddings.bin
    ├── offsets.bin
    ├── metadata.json
    ├── file_index.json
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
| `src/agent_tool.py` | Main agent interface |
| `src/agent_skill.py` | Language-agnostic code editing agent |
| `src/tool_schema.py` | Formal JSON schemas for all tools |
| `src/language_detect.py` | Language detection from extension / shebang |
| `src/cross_language_rules.py` | Language-agnostic editing rules |
| `src/tokenizer.py` | Code tokenization (tree-sitter / tiktoken / regex) |
| `src/embedder.py` | Sentence-transformers wrapper |
| `src/flatten.py` | Vectorized buffer serialization |
| `src/diff_engine.py` | Incremental diff with parallel hashing |
| `src/size_guard.py` | Size threshold guard |
| `src/metadata_store.py` | Compact JSON metadata I/O |
| `src/vulkan_context.py` | Vulkan device + CPU fallback compute |
| `src/buffer_manager.py` | Buffer allocation / upload / patch |
| `shaders/similarity_search.comp` | GLSL Top-K dot-product shader |
| `shaders/cluster_regions.comp` | GLSL clustering shader |
| `shaders/compile_shaders.py` | SPIR-V compilation script |
| `benchmark.py` | Performance benchmark script |
| `tests/test_buffer_rw.py` | Read/write/commit round-trip tests |
| `examplecode/` | Example codebase for testing |

## Requirements

- Python 3.10+
- PyTorch
- sentence-transformers
- NumPy
- Optional: tree-sitter + language grammars, tiktoken, Vulkan bindings

## License

MIT
