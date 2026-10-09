# GigaCode

[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-yellow?style=for-the-badge)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/github/actions/workflow/status/gigacode-ai/gigacode/ci.yml?style=for-the-badge)](https://github.com/gigacode-ai/gigacode/actions)

**GPU-accelerated code embedding and semantic search for AI agents.**

Embed a codebase into searchable chunks, run semantic search, navigate references, detect code smells and security vulnerabilities, and edit code through a safe read-write-commit workflow — all from a single tool with 76 agent-discoverable capabilities.

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

## Structural context, without embeddings

Use `get_task_context(task)` for bounded, evidence-backed orientation. Use
`code_navigate(action="symbol", file="auth.py", symbol="AuthService.login",
include_source=true)` to read one exact symbol, or choose `structure`, `callers`,
`callees`, `references`, `children`, `summary`, `dependencies`, `dependents`
or `graph`. Specify `root` when there is no active project.

The shared index supports Python, JS/TS/TSX, C/C++, YAML, unrendered Helm and
Dockerfiles. Ambiguous names and unresolved calls remain explicit.
Read-only Git tools and individual navigation operations are discoverable
through `tool_search`. See [context navigation](docs/context_navigation.md)
for examples, freshness, cache locations and resolution limits.

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

MCP servers use a curated **read-only** profile by default. For coding agents,
add `--tool-profile agent_core` (or `editing`). The compact published surface
is `code_find`, `code_edit`, `code_navigate`, `get_task_context`, `tool_search`,
and `tool_call`, with editing tools
excluded from read-only profiles.

The normal workflow is **`code_find` → `code_edit` → run tests**. No discovery,
explicit embedding, or `post_edit` call is required. For small edits, pass the
returned `file` with a unique `old_text` and its `new_text` replacement. Include
indentation and enough context to match once; no narrower read is required.
Use `expected_hash=file_hash` when available. Each successful edit returns a
fresh `file_hash` for subsequent edits. Alternatively, copy anchors and use
`new_lines` to replace the entire inclusive range, including declarations and
decorators; this mode requires `expected_hash`. Python syntax is
checked before mutation; `dry_run=true` previews without changing buffer or
disk. Intentional definition deletion requires `allow_definition_removal=true`.
Other languages retain text/hash/anchor guards but do not have a Python AST check.

Source paths accept either separator, project-prefixed paths, in-root absolute
paths, and unique basenames. Responses use root-relative `/` paths. Outside-root
paths, traversal and ambiguous basenames are rejected without guessing.

Tool arguments must be native JSON objects, not XML tool-call wrappers.
`old_text` and `new_text` contain source strings, not serialized lists or
`arg_key`/`arg_value` argument markup. XML that is part of the source is allowed.
Malformed calls are rejected before execution with compact JSON recovery
messages; the server never repairs or guesses edit arguments.

Exact identifiers and file names use source lookup rather than semantic
guesses. Behavioral searches return candidates; inspect their declarations
before editing. Set `GIGACODE_DIRECT_TOOLS=off` for legacy deferred-only
discovery, or use `--eager-tools` to publish the full selected profile.

See [coding benchmark methodology](docs/coding_benchmarks.md) for repeat runs,
correctness checks, token accounting, and GPU measurement limits. These changes
do not establish a token or latency improvement until new agent runs are measured.

The [`examples/agent_setup/`](examples/agent_setup/) directory ships ready
agent configurations for popular harnesses (OpenCode, Factory Droid,
Claude Code, Codex CLI, Hermes, Oh-My-Pi, Cursor, Windsurf, Gemini CLI,
VS Code, Zed, Goose, Cline), including the recommended `AGENTS.md` guidance
paragraph (single-search workflow, no-repeat rule, buffer recovery semantics).

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

76 tools across 9 categories. 56 read-only, 20 mutating.

| Category | Tools | Read-Only | Mutating |
|----------|:-----:|:---------:|:--------:|
| Analysis | 16 | 16 | 0 |
| Editing | 11 | 2 | 9 |
| Indexing | 5 | 2 | 3 |
| Navigation | 6 | 6 | 0 |
| Quality | 8 | 4 | 4 |
| Safety | 7 | 6 | 1 |
| Search | 16 | 16 | 0 |
| Agent | 6 | 3 | 3 |
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
