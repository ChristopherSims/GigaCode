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

### Formal Tool Schemas (Phase 6.1)

```python
from src.agent_tool import CodeEmbeddingTool
schemas = CodeEmbeddingTool.get_tool_schemas()
```

Schemas are available for:
- `embed_codebase`
- `semantic_search`
- `cluster_code`
- `update_codebase`
- `check_codebase`
- `list_buffers`
- `delete_buffer`

Also exportable to OpenAI function-calling format and MCP tool format via
`src.tool_schema`.

## Language-Agnostic Code Editing

GigaCode can now edit source files in **any programming language**:

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

The diff engine tracks per-line SHA-256 hashes. On re-ingest, only changed
lines are re-embedded and patched into the buffer.

## Persistence & Reload (Phase 6.4)

Buffers are stored in `.vkbuff/` directories under the working directory:

```
work_dir/
├── registry.json
└── <uuid>.vkbuff/
    ├── embeddings.bin
    ├── offsets.bin
    ├── metadata.json
    └── file_index.json
```

Use `reload_codebase(buffer_id)` to skip re-embedding when file hashes match.

## Tests

```bash
pytest tests/ -v
```

## Files

| File | Purpose |
|------|---------|
| `src/agent_tool.py` | Main agent interface |
| `src/agent_skill.py` | Language-agnostic code editing agent |
| `src/tool_schema.py` | Formal JSON schemas for all tools (Phase 6.1) |
| `src/language_detect.py` | Language detection from extension / shebang |
| `src/cross_language_rules.py` | Language-agnostic editing rules |
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
