# Agent setup examples

Ready-to-use MCP configurations for wiring the GigaCode MCP server into
popular agent harnesses. Every folder below contains a minimal config for one
harness; copy or merge it into the target file, then replace the
`/absolute/path/to/.gigacode-buffers` placeholder with a real absolute path
for your buffer work directory.

All examples run the server with `--tool-profile agent_core`. The compact MCP
listing publishes `code_find`, `code_edit`, `tool_search`, and `tool_call`;
additional capabilities are discovered on demand. `code_find` → `code_edit`
is the normal coding loop, with no explicit embed or finishing call.
Use `read_only` to disable editing, or `editing`/`full` for other capabilities.
Set `GIGACODE_DIRECT_TOOLS=off` for legacy deferred-only discovery.
Token/latency improvements still require repeated agent measurements.

Pair the config with the recommended agent guidance from
[`AGENTS.md.example`](AGENTS.md.example) (workflow, no-repeat rule, buffer
recovery semantics) for best token efficiency.

| Harness | Folder / file | Merge into |
|---|---|---|
| OpenCode | [`opencode/opencode.json`](opencode/opencode.json) | `.opencode/opencode.json` (project) or `~/.config/opencode/opencode.json` |
| Factory (Droid) | [`factory/mcp.json`](factory/mcp.json) | `.factory/mcp.json` (project) or `~/.factory/mcp.json` (user) |
| Claude Code | [`claude-code/.mcp.json`](claude-code/.mcp.json) | `.mcp.json` (project root) or `claude mcp add` |
| Codex CLI | [`codex/config.toml`](codex/config.toml) | `~/.codex/config.toml` (`[mcp_servers.gigacode]`) |
| Hermes Agent | [`hermes/config.yaml`](hermes/config.yaml) | `mcp_servers:` block in `~/.hermes/config.yaml` |
| Oh-My-Pi (omp) | [`oh-my-pi/mcp.json`](oh-my-pi/mcp.json) | `.omp/mcp.json` (project) or `~/.omp/agent/mcp.json` — omp also auto-imports the OpenCode config |
| Cursor | [`cursor/mcp.json`](cursor/mcp.json) | `.cursor/mcp.json` |
| Windsurf | [`windsurf/mcp_config.json`](windsurf/mcp_config.json) | `~/.codeium/windsurf/mcp_config.json` |
| Gemini CLI | [`gemini-cli/settings.json`](gemini-cli/settings.json) | `mcpServers:` block in `.gemini/settings.json` |
| VS Code (Copilot) | [`vscode/mcp.json`](vscode/mcp.json) | `.vscode/mcp.json` (`servers:` key) |
| Zed | [`zed/settings.json`](zed/settings.json) | `context_servers:` block in `~/.zed/settings.json` |
| Goose | [`goose/config.yaml`](goose/config.yaml) | `mcpServers:` block in `~/.config/goose/config.yaml` |
| Cline | [`cline/cline_mcp_settings.json`](cline/cline_mcp_settings.json) | `cline_mcp_settings.json` via the Cline MCP Servers UI |

Remote / shared setups: `gigacode-mcp --transport streamable-http --port 8766
--tool-profile agent_core --api-key "$GIGACODE_API_KEY"`, then point any
harness at `http://127.0.0.1:8766/mcp` with an `X-API-Key` header (see the
main [README](../../README.md#mcp-client-configuration) for the remote shape
and security notes).

Notes:

- Prefer `read_only` or an `enabledTools`-style allowlist (Factory) / `autoApprove`
  (Cline) when prompts are untrusted. `code_edit` persists immediately unless
  `dry_run=true`; legacy buffer writes require `commit`. Read access remains
  directory-scoped.
- The buffer work directory is agent-state, not project state: keep it out of
  source control (it is already covered by the `buffers/` entry in this repo's
  `.gitignore`).
