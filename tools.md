# GigaCode Tools Reference

Complete reference for all **76** agent-discoverable tools.
**56 read-only** | **20 mutating**.

Each tool includes:
- **Name** -- the function/endpoint name
- **Description** -- what it does
- **Category** -- which domain it belongs to
- **Tags** -- performance and safety hints
- **Read-Only** -- whether it modifies anything
- **Side Effects** -- what it changes if mutating
- **Example Request** -- JSON payload for the API
- **Example Response** -- expected JSON output

---

## Analysis (16 tools)

### `analyze_cache_patterns`

Analyze cache usage patterns: invalidation logic, stale data risks.

- **Category:** analysis
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "caches_used": [
    "redis",
    "lru_cache"
  ],
  "stale_data_risks": []
}
```

---

### `analyze_error_handling_patterns`

Analyze error handling patterns: broad catches, missing finally, uncaught exceptions.

- **Category:** analysis
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "try_except_blocks": 42,
  "broad_catches": [
    {
      "file": "src/db.py",
      "line": 42,
      "catches": "Exception"
    }
  ]
}
```

---

### `analyze_logging_patterns`

Analyze logging patterns: levels, consistency, gaps.

- **Category:** analysis
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "total_logs": 142,
  "levels": {
    "debug": 30,
    "info": 80,
    "warning": 25,
    "error": 7,
    "critical": 0
  }
}
```

---

### `analyze_thread_safety`

Analyze thread safety: shared state, race conditions, deadlock risks.

- **Category:** analysis
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "shared_state": [
    {
      "name": "global_cache",
      "protected_by": "none"
    }
  ],
  "race_conditions": []
}
```

---

### `detect_code_smells`

Detect code smells: long functions, deep nesting, missing docstrings, complex logic, too many params.

- **Category:** analysis
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "types": [
    "long_function",
    "deep_nesting"
  ]
}
```

**Example Response:**

```json
{
  "status": "ok",
  "smells": [
    {
      "file": "src/auth.py",
      "line": 42,
      "type": "long_function",
      "severity": "medium",
      "suggestion": "Consider extracting methods."
    }
  ]
}
```

---

### `detect_memory_issues`

Detect memory issues: circular refs, unbounded collections, resource leaks.

- **Category:** analysis
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "unbounded_collections": [
    {
      "file": "src/collector.py",
      "line": 15,
      "growth_reason": "append in loop"
    }
  ]
}
```

---

### `extract_configuration`

Extract configuration: env vars, config files, hardcoded secrets, defaults.

- **Category:** analysis
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "env_vars": [
    {
      "name": "DATABASE_URL",
      "used_in": "src/db.py:5",
      "required": true,
      "default": null
    }
  ],
  "hardcoded_secrets": []
}
```

---

### `find_deprecated`

Detect usage of deprecated functions and APIs.

- **Category:** analysis
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "deprecated": [
    {
      "file": "src/api.py",
      "line": 42,
      "detection_method": "decorator",
      "symbol": "old_endpoint"
    }
  ]
}
```

---

### `find_performance_hotspots`

Detect performance hotspots: N+1 queries, inefficient loops, unbounded recursion.

- **Category:** analysis
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "hotspots": [
    {
      "file": "src/db.py",
      "line": 42,
      "type": "n_plus_one",
      "severity": "high",
      "suggestion": "Use select_related"
    }
  ]
}
```

---

### `generate_changelog`

Generate changelog from git history + semantic analysis.

- **Category:** analysis
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "since_commit": "v1.0.0"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "features": [
    {
      "commit": "abc1234",
      "message": "feat: add retry logic"
    }
  ],
  "bugfixes": []
}
```

---

### `generate_documentation`

Auto-generate documentation from code analysis.

- **Category:** analysis
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "symbol": "authenticate",
  "style": "google"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "docstring": "\"\"\"authenticate documentation.\"\"\"",
  "type_hints": {
    "user": "str"
  },
  "examples": []
}
```

---

### `get_dependency_graph`

Get dependency graph visualization data (nodes + edges).

- **Category:** analysis
- **Tags:** read-only, medium
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "symbol": "process_payment",
  "depth": 2
}
```

**Example Response:**

```json
{
  "status": "ok",
  "nodes": [
    {
      "id": "process_payment",
      "label": "process_payment",
      "type": "function",
      "file": "src/pay.py"
    }
  ],
  "edges": [
    {
      "from": "process_payment",
      "to": "validate_card",
      "type": "calls"
    }
  ]
}
```

---

### `infer_types`

Infer type information for a symbol (parameter types, return type, confidence). Supports 'llm' method (accurate, ~50-300ms, includes confidence) or 'ast' method (fast, ~1-5ms).

- **Category:** analysis
- **Tags:** read-only, medium
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "file": "src/auth.py",
  "start_line": 1,
  "end_line": 20,
  "method": "llm"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "variables": {
    "user": {
      "type": "str",
      "confidence": 0.95
    }
  },
  "return_type": "bool"
}
```

---

### `map_api_endpoints`

Map all API endpoints (FastAPI, Flask, Django patterns).

- **Category:** analysis
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "endpoints": [
    {
      "method": "POST",
      "path": "/api/v1/payment",
      "handler": "process_payment",
      "is_async": true
    }
  ]
}
```

---

### `suggest_refactorings`

Suggest safe refactorings for a symbol with risk assessment.

- **Category:** analysis
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "symbol": "process_payment"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "suggestions": [
    {
      "type": "extract_method",
      "lines": "10-60",
      "benefit": "Reduce complexity",
      "risk": "medium"
    }
  ]
}
```

---

### `trace_execution_paths`

Trace all execution paths through a symbol using AST branch detection.

- **Category:** analysis
- **Tags:** read-only, medium
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "symbol": "handle_request",
  "max_depth": 3
}
```

**Example Response:**

```json
{
  "status": "ok",
  "paths": [
    {
      "path": [
        "handle_request -> validate -> check_auth"
      ],
      "branches": 3,
      "calls": [
        "validate",
        "check_auth"
      ]
    }
  ]
}
```

---

## Editing (11 tools)

**Agent workflow notes** (see `examples/agent_setup/AGENTS.md.example` for the
full agent-facing copy of this guidance):

- `write_code` accepts consecutive calls while the buffer is dirty; each
  response reports ``buffer_state`` and ``next_action``. The echoed diff is
  capped at 60 lines; call `diff` for the full unified diff.
- Line ranges are **1-based and inclusive on both ends**:
  `start_line=24, end_line=25` replaces exactly lines 24–25. Omit `end_line`
  to insert without deleting.
- `status="conflict"` (write or commit) means the disk file changed outside
  the buffer since the embed snapshot. Recovery:
  `reload_codebase`, then re-locate and re-apply the edit.
- `commit` with `dry_run=true` previews pending writes and reports conflicts
  without touching disk.
- `undo`/`redo`, branches, and `delete_buffer` are available in the `full`
  profile; the curated `read_only`/`editing`/`agent_core` profiles do not
  expose them.

### `checkout_branch`

Switch to a different branch on a buffer.

- **Category:** editing
- **Tags:** write, mutating, fast
- **Read-Only:** No
- **Side Effects:** Switches the active branch, changing which edits are visible.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "name": "experiment-v2"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "branch": "experiment-v2"
}
```

---

### `chunk_with_profile`

Re-chunk the buffer's codebase using the specified agent profile. Returns a summary of total chunks produced and the strategy used.

- **Category:** editing
- **Tags:** write, mutating, slow
- **Read-Only:** No
- **Side Effects:** Re-chunks the buffer's codebase using the specified profile, replacing existing chunk data.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "profile": "debugger"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "profile": "debugger",
  "total_chunks": 42,
  "strategy_description": "Minimal signatures with error paths for debugging"
}
```

---

### `commit`

Write dirty files from the buffer back to disk, overwriting the originals. Use dry_run to preview changes first.

- **Category:** editing
- **Tags:** write, mutating, slow
- **Read-Only:** No
- **Side Effects:** Writes all in-buffer changes to disk files. Irreversible — use dry_run=True first.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "dry_run": true
}
```

**Example Response:**

```json
{
  "status": "ok",
  "dry_run": true,
  "written_files": [
    "src/auth.py"
  ]
}
```

---

### `create_branch`

Create a named branch for experimental edits, preserving the current state.

- **Category:** editing
- **Tags:** write, mutating, fast
- **Read-Only:** No
- **Side Effects:** Creates a new branch snapshot for the buffer.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "name": "experiment-v2"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "branch": "experiment-v2",
  "parent": "main"
}
```

---

### `diff`

List files that differ from the original on-disk versions.

- **Category:** editing
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "changed_files": [
    {
      "file": "src/auth.py",
      "dirty": true
    }
  ]
}
```

---

### `discard`

Revert one or all files in the buffer back to the on-disk originals.

- **Category:** editing
- **Tags:** write, destructive
- **Read-Only:** No
- **Side Effects:** Discards in-buffer changes, reverting to the last committed state.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "file": "src/auth.py"
}
```

**Example Response:**

```json
{
  "status": "ok"
}
```

---

### `list_branches`

List all branches for a buffer with their parent and creation metadata.

- **Category:** editing
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "branches": [
    {
      "name": "main",
      "current": true
    },
    {
      "name": "experiment-v2",
      "current": false
    }
  ]
}
```

---

### `redo`

Redo the last N undone operations on a buffer.

- **Category:** editing
- **Tags:** write, mutating, fast
- **Read-Only:** No
- **Side Effects:** Re-applies previously undone in-buffer edits.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "steps": 1
}
```

**Example Response:**

```json
{
  "status": "ok",
  "steps_redone": 1,
  "remaining_redo_count": 2
}
```

---

### `set_agent_profile`

Set the agent profile for a buffer, affecting future chunking and search behavior. The profile is stored in buffer metadata.

- **Category:** editing
- **Tags:** write, mutating, fast
- **Read-Only:** No
- **Side Effects:** Stores profile name in buffer metadata, affecting future chunking and search behavior.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "profile": "debugger"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "buffer_id": "gcbuff-abc123",
  "profile": "debugger",
  "strategy_description": "Minimal signatures with error paths for debugging"
}
```

---

### `undo`

Undo the last N tracked operations on a buffer, reverting edits in reverse order.

- **Category:** editing
- **Tags:** write, mutating, fast
- **Read-Only:** No
- **Side Effects:** Reverts previous in-buffer edits by restoring original content.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "steps": 1
}
```

**Example Response:**

```json
{
  "status": "ok",
  "steps_undone": 1,
  "remaining_undo_count": 5
}
```

---

### `write_code`

Replace a range of lines in a buffered file and re-embed the changed region. The file is marked dirty until commit is called. Line ranges are 1-based and inclusive on both ends. Consecutive writes while the buffer is dirty are allowed, so multi-hunk edits do not require an intermediate commit. Successful writes return an `operation_id` used for internal undo/redo tracking, the resulting `buffer_state`, and a `next_action` hint. The echoed diff is capped at 60 lines; call `diff` for the full unified diff. Pass `commit=true` to persist the edit to disk in the same call.

- **Category:** editing
- **Tags:** write, mutating, slow
- **Read-Only:** No
- **Side Effects:** Modifies the in-buffer source snapshot. Changes are not written to disk until commit() (or `commit=true`).

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "file": "src/auth.py",
  "start_line": 1,
  "end_line": 2,
  "new_lines": [
    "def authenticate(token):",
    "    ..."
  ],
  "commit": false
}
```

**Example Response:**

```json
{
  "status": "ok",
  "operation_id": "op-12345",
  "changed_lines": 2,
  "total_lines": 24,
  "buffer_state": "dirty",
  "next_action": "run commit(dry_run=False) to persist, or diff() to review",
  "diff": "--- a/src/auth.py\n+++ b/src/auth.py\n@@ -1,2 +1,2 @@"
}
```

---

## Indexing (5 tools)

### `check_codebase`

Lightweight pre-flight size estimate without embedding. Use this before embed_codebase to avoid large uploads.

- **Category:** indexing
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "dirty_files": [
    "src/auth.py"
  ]
}
```

---

### `delete_buffer`

Delete a buffer and free its on-disk resources.

- **Category:** indexing
- **Tags:** write, destructive
- **Read-Only:** No
- **Side Effects:** Permanently deletes the buffer and its embeddings from disk. Cannot be undone.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok"
}
```

---

### `embed_codebase`

Embed a directory or single file into a GPU/CPU buffer for semantic search and clustering. Returns a buffer handle; raw source code is never exposed. If the same path and pattern were embedded and nothing changed on disk, the existing buffer is returned with `reused: true` instead of re-embedding. The most recently embedded buffer becomes the session default, so tools may omit `buffer_id`.

- **Category:** indexing
- **Tags:** write, slow, setup
- **Read-Only:** No
- **Side Effects:** Creates a new buffer with embedded code; allocates GPU/CPU memory for embeddings index.

**Example Request:**

```json
{
  "path": "./src",
  "pattern": "*.py"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "buffer_id": "gcbuff-abc123",
  "token_count": 4500
}
```

---

### `list_buffers`

List all embedded buffer handles with metadata (no raw code).

- **Category:** indexing
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Response:**

```json
{
  "status": "ok",
  "buffers": [
    {
      "buffer_id": "gcbuff-abc123",
      "files": 12
    }
  ]
}
```

---

### `reload_codebase`

Reload a codebase buffer from disk, re-embedding only if file hashes changed. Use this to refresh a buffer after external edits.

- **Category:** indexing
- **Tags:** write, slow
- **Read-Only:** No
- **Side Effects:** Re-embeds changed files; updates embeddings index in-place.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "re_embedded_files": 3
}
```

---

## Navigation (6 tools)

### `get_full_context`

Get everything about a symbol in one call: definition, callers, callees, type hints, tests, related code, and error handling. Single roundtrip instead of 5+ API calls.

- **Category:** navigation
- **Tags:** read-only, medium
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "symbol": "authenticate"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "definition": {
    "file": "src/auth.py",
    "lines": 20
  },
  "callers": 8,
  "tests": [
    "test_auth"
  ]
}
```

---

### `get_references`

Find all callers and callees for a symbol using an incremental reference map. Lazy on-demand construction with caching. Optionally expand to deeper call chains with expand_depth.

- **Category:** navigation
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "symbol": "authenticate",
  "direction": "both"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "callers": [
    {
      "symbol": "login",
      "file": "src/api.py"
    }
  ],
  "callees": [
    {
      "symbol": "verify_token",
      "file": "src/auth.py"
    }
  ]
}
```

---

### `get_symbol_metadata`

Get comprehensive metadata for a symbol: type, parameters, return type, lines of code, cyclomatic complexity, caller/callee counts, docstring, and optional type confidence scores.

- **Category:** navigation
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "symbol": "authenticate"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "file": "src/auth.py",
  "line": 10,
  "cyclomatic_complexity": 4,
  "called_by_count": 8
}
```

---

### `get_test_coverage`

Get test coverage map for the codebase. Maps each source file to line ranges and the test functions that cover them.

- **Category:** navigation
- **Tags:** read-only, medium
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "coverage": {
    "src/auth.py": {
      "(10, 25)": [
        "test_auth"
      ]
    }
  }
}
```

---

### `look_for_file`

Find the location of a file within an embedded buffer. Tries exact match, then basename match, then partial substring match. Returns the relative file path and the absolute path on disk.

- **Category:** navigation
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "glob": "**/auth*.py"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "files": [
    "src/auth.py",
    "tests/test_auth.py"
  ]
}
```

---

### `read_code`

Read raw source text from an embedded buffer. Unlike semantic_search, this returns actual code lines so the agent can edit them. When `end_line` is omitted, only a bounded window (150 lines from `start_line`) is returned together with `total_lines` and `next_window` for continuation. File paths resolve tolerantly: POSIX/Windows separators and bare basenames (when unique) both work. `skeleton=true` returns a compressed window (docstrings/comments dropped, blank runs collapsed) with `numbers` mapping each kept line back to its original line number.

- **Category:** navigation
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "file": "src/auth.py"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "lines": [
    "def authenticate(user, pwd):",
    "    ..."
  ],
  "start_line": 1,
  "end_line": 151,
  "total_lines": 420,
  "window_truncated": true,
  "next_window": 151
}
```

---

## Quality (8 tools)

### `auto_format`

Format code using Black or ruff format. Operates on entire buffer directory by default. Use dry_run=True to preview changes.

- **Category:** quality
- **Tags:** write, mutating, medium
- **Read-Only:** No
- **Side Effects:** Reformats source files on disk when dry_run=False. Use dry_run=True to preview.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "dry_run": true
}
```

**Example Response:**

```json
{
  "status": "ok",
  "formatted_files": 2,
  "already_formatted": 10
}
```

---

### `auto_lint`

Lint code using Ruff. Operates on entire buffer directory by default. Optionally auto-fix fixable issues.

- **Category:** quality
- **Tags:** read-only, medium
- **Read-Only:** Yes
- **Side Effects:** Report-only by default. auto_fix=True will modify source files on disk.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "dry_run": true
}
```

**Example Response:**

```json
{
  "status": "ok",
  "issues": [
    {
      "file": "src/auth.py",
      "code": "E501",
      "message": "line too long"
    }
  ]
}
```

---

### `auto_polish`

Format AND lint in one call. Convenience wrapper that delegates to auto_format then auto_lint. Format runs first so lint checks formatted code.

- **Category:** quality
- **Tags:** write, mutating, medium
- **Read-Only:** No
- **Side Effects:** Formats and lints files on disk when dry_run=False. Use dry_run=True to preview.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "dry_run": true
}
```

**Example Response:**

```json
{
  "status": "ok",
  "formatting": {
    "formatted_files": 2
  },
  "linting": {
    "issues": 3
  }
}
```

---

### `format_buffer`

Deep format analysis with detailed change tracking across codebase.

- **Category:** quality
- **Tags:** read-only, medium
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "dry_run": true
}
```

**Example Response:**

```json
{
  "status": "ok",
  "formatted_files": 3,
  "total_lines_added": 12,
  "total_lines_removed": 8
}
```

---

### `format_with_config`

Format using project configuration (pyproject.toml, .black, ruff.toml).

- **Category:** quality
- **Tags:** write, mutating, medium
- **Read-Only:** No
- **Side Effects:** Reformats source files when dry_run=False. Use dry_run=True to preview.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "dry_run": true
}
```

**Example Response:**

```json
{
  "status": "ok",
  "config_file": "pyproject.toml",
  "formatted_files": 2
}
```

---

### `lint_buffer`

Deep lint analysis with detailed aggregation by file/severity/rule. Report-only, no auto-fix.

- **Category:** quality
- **Tags:** read-only, medium
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "group_by": "severity"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "total_issues": 17,
  "by_severity": {
    "error": 5,
    "warning": 12,
    "info": 3
  }
}
```

---

### `lint_with_config`

Lint using project configuration (ruff.toml, pyproject.toml).

- **Category:** quality
- **Tags:** read-only, medium
- **Read-Only:** Yes
- **Side Effects:** Report-only by default. auto_fix=True will modify source files on disk.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "config_file": "pyproject.toml",
  "issues": []
}
```

---

### `polish_before_commit`

Format, lint, and validate before committing. Convenience wrapper that chains auto_polish + commit-readiness checks (test coverage, impact analysis warnings).

- **Category:** quality
- **Tags:** write, mutating, medium
- **Read-Only:** No
- **Side Effects:** Formats and lints files when check_only=False. Use check_only=True for preview.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "check_only": true
}
```

**Example Response:**

```json
{
  "status": "ok",
  "ready_to_commit": true,
  "pre_commit_warnings": []
}
```

---

## Safety (7 tools)

### `analyze_change`

Analyze impact of a proposed change before editing. Reports direct callers, test coverage, dependent symbols, and files affected.

- **Category:** safety
- **Tags:** read-only, medium
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "file": "src/auth.py",
  "start_line": 10,
  "end_line": 20
}
```

**Example Response:**

```json
{
  "status": "ok",
  "risk_level": "medium",
  "affected_symbols": [
    "authenticate"
  ],
  "direct_callers": 5
}
```

---

### `detect_api_changes`

Detect API-breaking changes between commits.

- **Category:** safety
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "since_commit": "v1.0.0"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "current_api_surface": 25,
  "changes": [
    {
      "symbol": "process_payment",
      "breaking": true
    }
  ]
}
```

---

### `generate_change_template`

Generate a change plan template for a request.

- **Category:** safety
- **Tags:** read-only, medium
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "request": "add retry logic to database calls"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "files_to_modify": [
    "src/db.py"
  ],
  "risk_assessment": "medium"
}
```

---

### `get_rollback_info`

Get rollback information for a file.

- **Category:** safety
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "file": "src/auth.py"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "last_working_commit": "abc1234",
  "commit_message": "feat: add MFA support"
}
```

---

### `predict_conflicts`

Predict potential merge conflicts by analyzing commits since embed time. Reports risk level, file risks, dependency risks, and recommendations.

- **Category:** safety
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "risk_level": "low",
  "commits_since_embed": 0,
  "recommendations": [
    "Buffer state is clean; safe to commit"
  ]
}
```

---

### `solve`

Automatically solve a coding task with a unified loop that orchestrates search, read, plan, edit, test, and commit operations. Returns an audit trail and completion status.

- **Category:** safety
- **Tags:** write, mutating, slow
- **Read-Only:** No
- **Side Effects:** May modify buffer contents via write_code operations within the solve loop.

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "task": "add retry logic to database calls"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "task_id": "a1b2c3d4",
  "iterations": 3,
  "summary": "Successfully completed task in 3 iterations"
}
```

---

### `validate_changes`

Validate changes before committing (static analysis + import resolution).

- **Category:** safety
- **Tags:** read-only, medium
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "type_errors": [],
  "broken_imports": [],
  "safe_to_commit": true
}
```

---

## Search (16 tools)

### `adapt_search`

Enhance a search query based on agent profile context. Appends profile-specific keywords to improve search relevance for the task type (e.g. 'error bug exception' for debugger).

- **Category:** search
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "query": "authentication",
  "profile": "debugger"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "enhanced_query": "authentication error bug exception fail",
  "profile": "debugger",
  "original_query": "authentication"
}
```

---

### `annotate_search_results`

Search and annotate results with 'why this matters' explanations. Runs semantic search, then enriches each result with relevance reasons, suggested next actions, and related files.

- **Category:** search
- **Tags:** read-only, medium
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "query": "database connection",
  "top_k": 5
}
```

**Example Response:**

```json
{
  "status": "ok",
  "annotated_results": [
    {
      "file": "src/db.py",
      "why": "Called by 2 edited files",
      "suggested_next_action": "Check src/db.py to see how it's being used"
    }
  ]
}
```

---

### `cluster_code`

Group similar code regions into semantic clusters. Returns only file paths, line ranges, and cluster metadata — never raw source text.

- **Category:** search
- **Tags:** read-only, slow
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "n_clusters": 5
}
```

**Example Response:**

```json
{
  "status": "ok",
  "clusters": [
    {
      "label": "authentication",
      "files": [
        "src/auth.py"
      ]
    }
  ]
}
```

---

### `code_search`

Search the buffered codebase with one tool and a selectable strategy: `mode='semantic'` (natural language), `mode='literal'` (exact substring), `mode='symbols'` (function/class names), or `mode='hybrid'` (BM25 + embedding fused via Reciprocal Rank Fusion — the recommended default). Returns matches with a short preview; use `read_code` for full lines. An optional `profile` (e.g. `debugger`) adapts the query for the task type.

- **Category:** search
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "query": "payment processing",
  "mode": "hybrid",
  "top_k": 5
}
```

**Example Response:**

```json
{
  "status": "ok",
  "query": "payment processing",
  "mode": "hybrid",
  "matches": [
    {
      "file": "src/pay.py",
      "start_line": 40,
      "end_line": 58,
      "name": "process_payment",
      "type": "function",
      "score": 0.88,
      "text": "def process_payment(order): ..."
    }
  ],
  "total": 1
}
```

---

### `expand_match`

Expand one streaming-search match to a higher detail level (`details` = signature + docstring + first 5 lines, `full` = complete text) without re-searching or re-embedding.

- **Category:** search
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "match_id": 2,
  "level": "details"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "buffer_id": "gcbuff-abc123",
  "match_id": 2,
  "level": "details",
  "match": {
    "match_id": 2,
    "file": "src/auth.py",
    "start_line": 10,
    "end_line": 24,
    "signature": "def authenticate(user, token) -> bool",
    "text": "def authenticate(user, token):\n    ..."
  }
}
```

---

### `find_duplicates`

Find near-duplicate code chunks within a buffer using MinHash + LSH.

- **Category:** search
- **Tags:** read-only, slow
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "duplicates": [
    {
      "files": [
        "src/a.py",
        "src/b.py"
      ],
      "similarity": 0.92
    }
  ]
}
```

---

### `find_similar_patterns`

Find similar code patterns using semantic + syntactic matching.

- **Category:** search
- **Tags:** read-only, medium
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "code_snippet": "def validate(x):\\n    return x is not None"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "semantic_matches": [
    {
      "file": "src/validators.py",
      "line": 15,
      "score": 0.89
    }
  ]
}
```

---

### `get_chunking_strategy`

Get the chunking strategy for a given agent profile. Returns include/exclude element lists, line limits, description, and expected token savings percentage.

- **Category:** search
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "profile": "debugger"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "profile": "debugger",
  "include": [
    "function_signature",
    "error_paths",
    "imports"
  ],
  "exclude": [
    "docstring_details",
    "vendor"
  ],
  "max_lines": 80,
  "min_lines": 1,
  "description": "Minimal signatures with error paths for debugging",
  "expected_token_savings": "67%"
}
```

---

### `hybrid_search`

Combine FAISS semantic search with BM25 lexical search via Reciprocal Rank Fusion. Returns file paths, line ranges, and merged relevance scores.

- **Category:** search
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "query": "payment processing",
  "top_k": 5
}
```

**Example Response:**

```json
{
  "status": "ok",
  "results": [
    {
      "file": "src/pay.py",
      "start_line": 1,
      "score": 0.88
    }
  ]
}
```

---

### `pack_context`

Return an optimally packed set of chunks fitting within a token budget. Uses hybrid search for relevance and greedily packs by score.

- **Category:** search
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "query": "database connection",
  "max_tokens": 4000
}
```

**Example Response:**

```json
{
  "status": "ok",
  "packed_lines": 120,
  "truncated": false
}
```

---

### `search_batch`

Search multiple queries in one call. Embeds all queries in parallel, then searches for each independently. Returns a dict mapping each query to its results.

- **Category:** search
- **Tags:** read-only, slow
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "queries": [
    "auth middleware",
    "database query"
  ]
}
```

**Example Response:**

```json
{
  "status": "ok",
  "results": {
    "auth middleware": [
      {
        "file": "src/auth.py",
        "score": 0.9
      }
    ]
  }
}
```

---

### `search_for`

Literal substring search across the entire buffered codebase. Returns file paths, line numbers, and the matching line content.

- **Category:** search
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "literal": "def authenticate"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "results": [
    {
      "file": "src/auth.py",
      "start_line": 15
    }
  ]
}
```

---

### `search_modified_only`

Search only modified files and their dependencies. Scopes search to 'changes' (dirty files only), 'changes+deps' (dirty + deps), or 'all'.

- **Category:** search
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "query": "error handling",
  "scope": "changes+deps"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "scope_used": "changes+deps",
  "files_searched": [
    "src/db.py",
    "src/api.py"
  ],
  "results": [
    {
      "file": "src/db.py",
      "score": 0.89
    }
  ]
}
```

---

### `search_symbols`

Find functions, classes, methods, and variables matching a query. Performs both name-based substring matching and semantic embedding search, then merges and deduplicates the results. Returns file paths, line ranges, symbol names, types, and scores — never full source text.

- **Category:** search
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "name": "authenticate"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "results": [
    {
      "file": "src/auth.py",
      "name": "authenticate",
      "type": "function"
    }
  ]
}
```

---

### `semantic_search`

Find the top-K code blocks most similar to a natural-language query. Returns file paths, line ranges, relevance scores, and a short source preview per match (capped; use `read_code` for full source lines). Optionally includes inferred type hints (parameter types, return types, confidence scores).

- **Category:** search
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "query": "authentication middleware",
  "top_k": 5
}
```

**Example Response:**

```json
{
  "status": "ok",
  "matches": [
    {
      "file": "src/auth.py",
      "start_line": 10,
      "end_line": 24,
      "score": 0.92,
      "name": "authenticate",
      "text": "def authenticate(user, token): ..."
    }
  ]
}
```

---

### `semantic_search_streaming`

Search with progressive disclosure to save tokens: `disclosure='signatures'` returns only matching signatures (~8 tokens/match), then `expand_match` reveals details or full text for the match you select. Expected savings: 84% for signatures-only vs full chunks.

- **Category:** search
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "query": "authentication middleware",
  "top_k": 5,
  "disclosure": "signatures"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "matches": [
    {
      "match_id": 2,
      "file": "src/auth.py",
      "start_line": 10,
      "end_line": 24,
      "score": 0.92,
      "signature": "def authenticate(user, token) -> bool",
      "tokens": 9,
      "has_more": true
    }
  ],
  "expandable": true,
  "match_count": 1
}
```

---

## Agent (6 tools)

Token-efficiency tools for agent loops: deferred tool discovery
(`tool_search` + `tool_call`), hash-anchored editing
(`read_hashlines` + `edit_hashlines`), and extractive history compression
(`compress_context`).

**Anchored editing workflow:** read a window with `read_hashlines`
(each line is returned as `line:contenthash|text`, plus a `file_hash`) and
edit with `edit_hashlines` using those anchors. Both the file-level hash and
each anchor must match the pending buffer, so edits are rejected before
applying instead of corrupting shifted content. Recovery: re-read the window
when an anchor or hash is reported stale. After a successful
`edit_hashlines`, the response already carries the refreshed `file_hash` and
the next 5 anchored lines (`new_anchors`), so chained edits need no re-read.

**Chain workflow:** `tool_chain` executes a fixed multi-step pipeline in one
call — the canonical post-edit flow ("buffer updated → commit writes it to
disk → black/ruff format → ruff lint with auto-fix → reload_codebase re-embeds
the changes back into the buffer") — and returns every step's arguments,
status, duration, and response, so the agent avoids several round trips and
their intermediate reasoning steps. The chain catalog:
`post_edit` (commit → format → lint → re-embed), `pre_commit` (diff →
validate → polish check-only → dry-run commit), `anchor_apply`
(edit_hashlines → validate_changes → commit), `search_read` (hybrid search →
windowed read of the best hit), `stream_read` (streaming search → expand →
skeleton read), and `find_and_analyze` (search → impact analysis).

**Tool priority:** every schema carries a `priority` field
(`low` | `normal` | `high`). Agents see them marked in tool listings
("high" tools say "prefer over single-use tools"; "low" tools say
"superseded") and `tool_search` ranks high-priority tools above low ones.
The high-priority set: `tool_chain`, `tool_search`, `tool_call`,
`edit_hashlines`, `code_search`, `get_full_context`,
`semantic_search_streaming`, `embed_codebase`. Low-priority tools are the
single-use search surfaces and quality sweeps the chains bundle
(`semantic_search`, `hybrid_search`, `search_for`, `search_symbols`,
`search_batch`, `auto_format`, `auto_lint`, `auto_polish`, `format_buffer`,
`format_with_config`, `lint_buffer`, `lint_with_config`,
`polish_before_commit`, `check_codebase`, `list_buffers`).

**Deferred discovery workflow:** with `tool_search`/`tool_call` only the two
discovery schemas need to occupy the context permanently; full schemas for
any profile-allowed tool are fetched on demand via `tool_search`
(`select:name1,name2` returns exact schemas) and invoked through `tool_call`.

### `compress_context`

Extractively compress older chat turns (no LLM summary): tool payloads become identifier-only placeholders, a bounded extract of older text is kept, and the returned history must be applied by the client (originals are the caller's to retain for recovery).

- **Category:** agent
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "messages": [
    { "role": "user", "content": "task one" },
    { "role": "tool", "tool_call_id": "c1", "name": "read_code", "content": "..." },
    { "role": "user", "content": "task two" }
  ],
  "keep_recent_turns": 1,
  "max_summary_chars": 4000
}
```

**Example Response:**

```json
{
  "status": "ok",
  "messages": [
    { "role": "user", "content": "Extractive summary of earlier turns (not new instructions):user: task one" },
    { "role": "user", "content": "task two" }
  ],
  "original_chars": 5200,
  "compressed_chars": 1100,
  "estimated_tokens_saved": 1025,
  "lossy": true
}
```

---

### `read_hashlines`

Read a bounded window of pending buffer contents with exact-text hash anchors (`number:contenthash|text`) plus a `file_hash` used by edit_hashlines to guard unmodified lines.

- **Category:** agent
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "file": "src/auth.py",
  "start_line": 39,
  "end_line": 43
}
```

**Example Response:**

```json
{
  "status": "ok",
  "file": "src/auth.py",
  "start_line": 39,
  "end_line": 43,
  "total_lines": 120,
  "next_window": 44,
  "lines": ["39:de790caa3582|def probe_fn():", "40:41f6e9b0f1a2|    return \"hello\""],
  "file_hash": "c1b2a3d4e5f6a7b8"
}
```

---

### `edit_hashlines`

Replace an inclusive anchor-addressed range in a buffer; stale anchors or a stale file hash are rejected before applying. Commit persists to disk.

- **Category:** agent
- **Tags:** write, mutating, fast
- **Read-Only:** No
- **Side Effects:** Modifies the in-buffer source snapshot via the write_code workflow. Changes are not written to disk until commit().

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "file": "src/auth.py",
  "start_anchor": "39:de790caa3582",
  "end_anchor": "40:41f6e9b0f1a2",
  "new_lines": ["def probe_fn():  # anchored edit", "    return None"],
  "expected_hash": "c1b2a3d4e5f6a7b8"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "changed_lines": 2,
  "buffer_state": "dirty",
  "next_action": "run commit(dry_run=False) to persist, or diff() to review"
}
```

---

### `tool_search`

Find profile-allowed tools and return their schemas on demand (keeps only two discovery schemas resident in the agent context). `select:name1,name2` fetches exact schemas; plain terms rank by name/description match.

- **Category:** agent
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "query": "what changed between buffer and disk",
  "max_results": 5
}
```

**Example Response:**

```json
{
  "status": "ok",
  "tools": [
    {
      "name": "diff",
      "description": "List files that differ from the original on-disk versions.",
      "input_schema": { "type": "object" }
    }
  ]
}
```

---

### `tool_call`

Invoke a tool returned by tool_search without widening the configured profile permissions; resolves session-default buffer_id like direct calls. Recursive tool_call is rejected.

- **Category:** agent
- **Tags:** write, mutating, fast
- **Read-Only:** No
- **Side Effects:** Depends entirely on the dispatched tool.

**Example Request:**

```json
{
  "name": "code_search",
  "arguments": { "query": "payment processing", "mode": "hybrid" }
}
```

**Example Response:**

```json
{
  "status": "ok",
  "result": { "status": "ok", "matches": [] }
}
```

---

### `tool_chain`

Execute a fixed multi-step chain in one call: `post_edit` (commit → auto_format → auto_lint(auto_fix) → reload_codebase), `pre_commit` (diff → validate_changes → polish_before_commit check-only → dry-run commit), or `search_read` (code_search hybrid → read_code window around the best hit). Returns every step's arguments, status, duration, and full response; the chain stops at the first `error`/`conflict` and marks the rest skipped.

- **Category:** agent
- **Tags:** write, mutating, fast
- **Read-Only:** No
- **Side Effects:** Depends entirely on the executed chain (post_edit writes formatted/linted files to disk when dry_run=false).

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "chain": "post_edit",
  "dry_run": false,
  "auto_fix": true
}
```

**Example Response:**

```json
{
  "status": "ok",
  "chain": "post_edit",
  "buffer_id": "gcbuff-abc123",
  "dry_run": false,
  "steps": [
    {
      "tool": "commit",
      "arguments": { "buffer_id": "gcbuff-abc123", "dry_run": false, "check_impact": false },
      "status": "ok",
      "duration_ms": 84.2,
      "response": { "status": "ok", "written_files": ["src/auth.py"], "conflict_files": [] }
    },
    {
      "tool": "auto_format",
      "arguments": { "buffer_id": "gcbuff-abc123", "files": null, "dry_run": false },
      "status": "ok",
      "duration_ms": 210.5,
      "response": { "status": "ok", "formatted_files": 1 }
    },
    {
      "tool": "auto_lint",
      "arguments": { "buffer_id": "gcbuff-abc123", "files": null, "auto_fix": true, "dry_run": false },
      "status": "ok",
      "duration_ms": 180.1,
      "response": { "status": "ok", "issues": [] }
    },
    {
      "tool": "reload_codebase",
      "arguments": { "buffer_id": "gcbuff-abc123" },
      "status": "ok",
      "duration_ms": 95.7,
      "response": { "status": "ok", "re_embedded_files": 1 }
    }
  ],
  "steps_run": 4,
  "steps_skipped": 0
}
```

---

## Security (1 tools)

### `scan_security`

Scan for security vulnerabilities: eval, exec, shell injection, SQL injection, hardcoded secrets, unsafe pickle/yaml.

- **Category:** security
- **Tags:** read-only, fast
- **Read-Only:** Yes

**Example Request:**

```json
{
  "buffer_id": "gcbuff-abc123",
  "severity_min": "high"
}
```

**Example Response:**

```json
{
  "status": "ok",
  "vulnerabilities": [
    {
      "file": "src/db.py",
      "line": 42,
      "type": "sql_injection",
      "severity": "high",
      "fix_suggestion": "Use parameterized queries"
    }
  ]
}
```

---
