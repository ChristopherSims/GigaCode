# GigaCode — GPU-Accelerated Code Embedding Agent Tool

Embed source code into GPU/CPU buffers and run semantic search + clustering
without ever exposing raw source text to the agent context.

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
VkBuffer (GPU) or Numpy Array (CPU fallback)
   |
   +-- Compute Shader: similarity_search.comp (Top-K NNS)
   +-- Compute Shader: cluster_regions.comp (region clustering)
```

## Setup

### System dependencies (Ubuntu/Debian)

```bash
sudo apt install nvidia-driver-535 vulkan-tools libvulkan1
```

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Compile shaders (optional — GPU path)

```bash
cd shaders
python compile_shaders.py
```

Requires `glslc` or `glslangValidator` in your PATH.

## Usage

```python
from src.agent_tool import CodeEmbeddingTool

with CodeEmbeddingTool(work_dir="./buffers", device="cpu") as tool:
    # Embed
    result = tool.embed_codebase("./examplecode", pattern="*.py")
    buf_id = result["buffer_id"]

    # Search
    search = tool.semantic_search(buf_id, "sorting algorithm", top_k=5)
    for m in search["matches"]:
        print(m["file"], m["line"], m["score"])

    # Cluster
    clusters = tool.cluster_code(buf_id, threshold=0.75)
    for c in clusters["clusters"]:
        print(c["file"], c["start_line"], c["end_line"], c["size"])
```

## Tool Contract (Agent Interface)

The tool **never returns raw source code**. Responses contain only:

- `buffer_id` — opaque handle
- `file` — relative file path
- `line` — 1-based line number
- `score` — similarity score
- `size` — cluster token count

If a codebase exceeds the size threshold (default 500 MB), the tool returns a
warning with a suggestion to narrow the scope.

## Incremental Updates

The diff engine tracks per-line SHA-256 hashes. On re-ingest, only changed
lines are re-embedded and patched into the buffer.

## Tests

```bash
pytest tests/ -v
```

## Files

| File | Purpose |
|------|---------|
| `src/agent_tool.py` | Main agent interface |
| `src/tokenizer.py` | Code tokenization |
| `src/embedder.py` | Sentence-transformers wrapper |
| `src/flatten.py` | Buffer serialization |
| `src/diff_engine.py` | Incremental diff |
| `src/size_guard.py` | Size threshold guard |
| `src/metadata_store.py` | JSON metadata I/O |
| `src/vulkan_context.py` | Vulkan device + compute fallback |
| `src/buffer_manager.py` | Buffer alloc / upload / patch |
| `shaders/similarity_search.comp` | GLSL Top-K dot-product shader |
| `shaders/cluster_regions.comp` | GLSL clustering shader |
| `shaders/compile_shaders.py` | SPIR-V compilation script |
| `examplecode/` | Example codebase for testing |

## License

MIT
