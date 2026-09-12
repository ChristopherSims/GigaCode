# GigaCode

[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-yellow?style=for-the-badge)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/github/actions/workflow/status/gigacode-ai/gigacode/ci.yml?style=for-the-badge)](https://github.com/gigacode-ai/gigacode/actions)

**GPU-accelerated code embedding and semantic search for AI agents.**

Embed a codebase into searchable chunks, run semantic search, navigate references, detect code smells and security vulnerabilities, and edit code through a safe read-write-commit workflow — all from a single tool with 67 agent-discoverable capabilities.

Optimized for AI agent loops: fast AST chunking, sub-millisecond search on GPU, surgical index updates on edit, and full tool schema export in OpenAI, Anthropic, MCP, and Ollama formats.

---

## Install

GigaCode is distributed as source and as prebuilt wheels attached to
[GitHub releases](https://github.com/gigacode-ai/gigacode/releases). It is not
published to PyPI.

```bash
# From a checkout
git clone https://github.com/gigacode-ai/gigacode.git
cd gigacode

# Core only (chunking + lexical search, ~50MB deps)
pip install .

# With semantic search (torch + sentence-transformers + faiss, ~3GB deps)
pip install ".[embed]"

# With API server
pip install ".[embed,server]"

# With GPU acceleration (requires CUDA; Linux)
pip install ".[embed,gpu]"

# Everything
pip install ".[all]"

# Development / tests
pip install ".[test]"
```

From a downloaded release wheel instead of a checkout:

```bash
pip install ./gigacode-<version>-py3-none-any.whl
pip install './gigacode-<version>-py3-none-any.whl[embed,server]'
```

**System Requirements:**
- Python 3.10+
- CPU-only: any platform
- GPU: NVIDIA with CUDA 11.8+, cuDNN 8.x
- First `embed_codebase()` call downloads a ~160MB embedding model

## Quick Start

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

    # Edit code safely (buffer only — disk unchanged until commit)
    tool.write_code(buf_id, file="main.py", start_line=5,
                    new_lines=["    return value"])
    tool.diff(buf_id)       # preview
    tool.commit(buf_id)     # write to disk
```

## CLI

```bash
# Start API server
gigacode-server --work-dir ./buffers --port 8765

# Start MCP server (for Claude Desktop)
gigacode-mcp --work-dir ./buffers

# Code editing skill
gigacode-skill example.py

# Or via python -m
python -m gigacode --work-dir ./buffers
```

## MCP Client Configuration

MCP servers expose a curated **read-only** tool profile by default. Add `--tool-profile editing` (or `full`) to enable `write_code`, `commit`, and other mutating tools.

**Local stdio (Claude Desktop, most agents):**

```json
{
  "mcpServers": {
    "gigacode": {
      "command": "gigacode-mcp",
      "args": ["--work-dir", "/absolute/path/to/buffers"]
    }
  }
}
```

**Remote Streamable HTTP (requires auth for non-local bind):**

```bash
gigacode-mcp --transport streamable-http --host 0.0.0.0 --port 8766 \
  --api-key "$GIGACODE_API_KEY" --tool-profile read_only
```

```json
{
  "mcpServers": {
    "gigacode": {
      "url": "http://127.0.0.1:8766/mcp",
      "headers": { "X-API-Key": "your-api-key" }
    }
  }
}
```

See [SECURITY.md](SECURITY.md#transports-and-network-deployment) for authentication, workspace boundaries, and deployment assumptions.

## Docker

```bash
# CPU
docker compose up

# GPU (requires nvidia-docker)
docker build -f Dockerfile.gpu -t gigacode:gpu .
docker run --gpus all -p 8765:8765 gigacode:gpu
```

## Tool Categories

67 tools across 8 categories. 50 read-only, 17 mutating.

| Category | Tools | Read-Only | Mutating |
|----------|:-----:|:---------:|:--------:|
| Analysis | 16 | 16 | 0 |
| Editing | 11 | 2 | 9 |
| Indexing | 5 | 2 | 3 |
| Navigation | 6 | 6 | 0 |
| Quality | 8 | 4 | 4 |
| Safety | 7 | 6 | 1 |
| Search | 13 | 13 | 0 |
| Security | 1 | 1 | 0 |

## Schema Export Formats

| Format | Use Case | Export Function |
|--------|----------|-----------------|
| **OpenAI** | Function calling | `to_openai_functions()` |
| **Anthropic** | Tool use | `to_anthropic_tools()` |
| **MCP** | Model Context Protocol | `to_mcp_tools()` |
| **Ollama** | Local LLM tooling | `to_ollama_tools()` |

```python
from gigacode.tool_schema import export_schemas, SchemaFormat

# Get schemas for your AI framework
tools = export_schemas("openai")
tools = export_schemas("anthropic", category="security", read_only_only=True)
```

## Configuration

Create `gigacode.toml` in your project root:

```toml
[schemas]
format = "openai"
include_metadata = true
default_category = "all"
read_only_only = false
```

See `gigacode.toml.example` for all options.

## Links

- [Full tool reference](tools.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Documentation](https://github.com/gigacode-ai/gigacode#readme)

## License

MIT — see [LICENSE](LICENSE).
