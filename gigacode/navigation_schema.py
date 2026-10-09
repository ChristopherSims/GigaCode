"""Small published context surface; individual operations remain discoverable."""

ROOT = {"type": ["string", "null"], "description": "Project directory. Defaults to the active buffer or working project; never requires embeddings."}
BUFFER = {"type": ["string", "null"]}
FILE = {"type": "string", "description": "Root-relative source path; equivalent safe separators/prefixes are accepted."}
SYMBOL = {"type": "string", "description": "Exact qualified name or returned symbol ID. Ambiguous names are never guessed."}
LIMIT = {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}
OFFSET = {"type": "integer", "minimum": 0, "default": 0}
REF = {"type": "string", "default": "HEAD"}


def schema(name, description, properties, required=()):
    return {
        "name": name, "description": description,
        "input_schema": {"type": "object", "properties": {"root": ROOT, "buffer_id": BUFFER, **properties},
                         "required": list(required), "additionalProperties": False},
        "output_schema": {"type": "object"}, "read_only": True,
        "category": "navigation", "tags": ["read-only", "deterministic"],
        "priority": "normal", "description_limit": 600,
    }


NAVIGATION_SCHEMAS = [
    schema("code_navigate",
           "Use native JSON. Deterministic AST navigation without embeddings: symbol, callers, callees, "
           "references, children, structure, summary, dependencies, dependents or graph. file scopes the "
           "query; symbol accepts a qualified name or ID. Bodies are opt-in. Responses are bounded; "
           "ambiguous/unresolved relationships are explicit. For task-wide orientation use get_task_context.",
           {"action": {"type": "string", "enum": ["symbol", "callers", "callees", "references", "children",
                                                  "structure", "summary", "dependencies", "dependents", "graph"]},
            "file": FILE, "symbol": SYMBOL, "include_source": {"type": "boolean", "default": False},
            "limit": LIMIT, "offset": OFFSET}, ["action"]),
    schema("get_task_context",
           "Use native JSON. Build bounded, evidence-backed task context without LLM analysis or embeddings: "
           "likely files/symbols, summaries, import-derived architecture, related tests and optional Git status. "
           "Candidates are not a guarantee of completeness. Read a selected symbol with code_navigate "
           "action=symbol, include_source=true. Run tests after edits.",
           {"task": {"type": "string", "minLength": 1, "maxLength": 10000},
            "limit": {**LIMIT, "maximum": 20, "default": 10},
            "include_git": {"type": "boolean", "default": True}}, ["task"]),
    schema("get_symbol", "Exact symbol metadata, without source bodies or semantic guessing.",
           {"symbol": SYMBOL, "file": FILE}, ["symbol"]),
    schema("read_symbol", "Read just one exact, file-scoped symbol. Optional relationships are names/locations, not source dumps.",
           {"file": FILE, "symbol": SYMBOL, "limit": LIMIT,
            "include": {"type": "array", "items": {"type": "string", "enum": ["callers", "callees", "references", "dependencies"]}}},
           ["file", "symbol"]),
]
for name in ("get_callers", "get_callees", "get_children"):
    NAVIGATION_SCHEMAS.append(schema(name, "Bounded AST relationship navigation; use exact symbols/IDs and inspect resolution.",
                                     {"symbol": SYMBOL, "file": FILE, "limit": LIMIT, "offset": OFFSET}, ["symbol"]))
for name in ("get_file_structure", "dependencies", "dependents"):
    NAVIGATION_SCHEMAS.append(schema(name, "Bounded source-backed structure/dependency metadata; no source body or embedding required.",
                                     {"file": FILE, "limit": LIMIT, "offset": OFFSET}, ["file"]))
for name in ("file_summary", "dependency_graph"):
    NAVIGATION_SCHEMAS.append(schema(name, "Compact, current source-backed summary or transitive dependency graph.",
                                     {"file": FILE, "limit": LIMIT}, ["file"]))
for name in ("git_status", "changed_files"):
    NAVIGATION_SCHEMAS.append(schema(name, "Read-only, project-scoped staged/unstaged/untracked status with pagination.",
                                     {"limit": LIMIT, "offset": OFFSET}))
NAVIGATION_SCHEMAS.extend([
    schema("git_diff", "Read-only bounded diff against a resolved revision. staged=true reads the index; default reads working tree against HEAD.",
           {"file": FILE, "against": REF, "staged": {"type": "boolean", "default": False},
            "max_chars": {"type": "integer", "minimum": 100, "maximum": 20000, "default": 8000}}),
    schema("recent_commits", "Project-scoped recent commit IDs/dates/subjects, without author identities or source.",
           {"limit": LIMIT, "offset": OFFSET}),
    schema("file_history", "Paginated commit history for one project file.",
           {"file": FILE, "limit": LIMIT, "offset": OFFSET}, ["file"]),
    schema("blame", "Read-only, bounded line provenance without author identities or source text.",
           {"file": FILE, "line": {"type": "integer", "minimum": 1, "default": 1},
            "limit": LIMIT, "commit": REF}, ["file"]),
    schema("changed_symbols", "Map before/after Git changes to AST symbols, including deletions and new definitions. No LLM summary. Renames may appear as delete/add.",
           {"against": REF, "staged": {"type": "boolean", "default": False}, "limit": LIMIT, "offset": OFFSET}),
])
NAVIGATION_TOOL_NAMES = frozenset(s["name"] for s in NAVIGATION_SCHEMAS)
