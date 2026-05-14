# GigaCode API Improvements for AI Agents (v5)

**Date:** May 10, 2026  
**Purpose:** Comprehensive feature roadmap to accelerate AI-driven code analysis, refactoring, and understanding.

---

## Executive Summary

These 30+ features transform GigaCode from a semantic search engine into a **complete code understanding and manipulation platform** for AI agents. Each feature is designed to reduce API roundtrips, provide richer context, and enable autonomous code changes with confidence.

**Impact Areas:**
- 🔍 **Code Navigation** (8 features)
- 📊 **Metadata & Analysis** (10 features)
- 🔄 **Change Management** (6 features)
- 🛡️ **Safety & Quality** (8 features)

---

## Core Features (Must-Have)

### 1. **Reference Map** (Who calls what)
```python
results = tool.get_references(buffer_id, symbol="authenticate_user", direction="both")
# Returns: [{file, line, type: "calls" | "called_by", symbol_name, context}]
```
**Priority:** ⭐⭐⭐⭐⭐  
**Benefit:** Instantly find all callers/callees—eliminates guesswork when understanding dependencies.  
**Implementation:** Incremental call graph construction using a hybrid three-phase strategy:  
1. **Lazy/On-Demand Indexing** — On first query for a symbol, build only its direct caller/callee neighborhood via AST + symbol_index. Cache results for fast repeat lookups. No upfront full-graph cost.  
2. **Incremental Update on File Changes** — Detect changed files via git diff or file watcher; invalidate and rebuild only the affected subgraph (symbols defined in or imported by changed files). Avoids full rebuild on every change.  
3. **Background Fill** — After serving the initial query, asynchronously expand the call graph to unvisited symbols in a background task. Subsequent queries hit a progressively more complete graph without blocking the caller.

---

### 2. **Context Bundle** (Everything at once)
```python
context = tool.get_full_context(
    buffer_id,
    symbol="process_payment",
    include=["definition", "callers", "tests", "related_code", "type_hints", "errors"],
    type_inference_method="llm"  # or "ast" for faster response
)
# Returns: {definition: {...}, callers: [...], tests: [...], types: {...}, error_handling: [...]}
```
**Priority:** ⭐⭐⭐⭐⭐  
**Benefit:** Single roundtrip to understand a symbol completely instead of 5 API calls.  
**Implementation:** Combine get_symbol_definition + get_references + search_symbols. Type inference uses LLM by default for comprehensive understanding, or AST for speed.

---

### 3. **Type Information in Search Results**
```python
results = tool.semantic_search(
    buffer_id, 
    "validate input", 
    top_k=5, 
    include_types=True,
    type_inference_method="llm"  # or "ast" for faster results
)
# Returns: {
#   file, start_line, end_line, text, signature,
#   types: {parameters, return},
#   type_confidence: 0.95,  # Only for LLM inference
#   inference_method: "llm"
# }
```
**Priority:** ⭐⭐⭐⭐⭐  
**Benefit:** Know parameter types and return types instantly without reading full code.  
**Implementation:**
- **AST-based (default, faster):** Extract type hints from AST during chunking; store in metadata. ~1ms per result.
- **LLM-assisted (more accurate):** Use LLM to infer types from untyped code. ~50-200ms per result. Higher accuracy for dynamic code.

---

### 4. **Impact Analysis**
```python
impact = tool.analyze_change(buffer_id, file="src/auth.py", start_line=42, end_line=58)
# Returns: {direct_callers: [...], test_coverage: [...], dependent_symbols: 3, files_affected: 7}
```
**Priority:** ⭐⭐⭐⭐⭐  
**Benefit:** Before editing, know what breaks. Prevents cascading failures.  
**Implementation:** Use call graph + test discovery + file dependency analysis.

---

### 5. **Batch Search** (Multiple queries in one call)
```python
results = tool.search_batch(
    buffer_id,
    queries=["database connection", "error handling", "caching logic"],
    top_k=5
)
# Returns: {query1: [...], query2: [...], query3: [...]}
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** Understand a feature by searching all components in parallel; faster than sequential calls.  
**Implementation:** Embed all queries in parallel; batch FAISS search.

---

### 6. **Diff-Aware Search**
```python
results = tool.search_modified_only(
    buffer_id,
    query="authentication",
    since_commit="main"  # or since_last_commit, since_date
)
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** When making changes, search only modified files—focuses attention on relevant code.  
**Implementation:** Use git_utils to detect changed files; filter search results.

---

### 7. **Test Coverage Map**
```python
coverage = tool.get_test_coverage(buffer_id)
# Returns: {"src/auth.py": {(42, 58): ["test_login", "test_mfa"]}, "src/db.py": {(10, 20): []}}
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** Know which code is tested before editing; untested code highlights risk areas.  
**Implementation:** Run coverage tool; map line ranges to test names.

---

### 8. **Symbol Metadata**
```python
meta = tool.get_symbol_metadata(
    buffer_id, 
    "process_payment",
    include_types=True,
    type_inference_method="llm"  # or "ast"
)
# Returns: {file, line, type, parameters, return_type, lines_of_code, cyclomatic_complexity, 
#           called_by_count, calls_count, last_modified, test_count, docstring}
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** Quick metrics reveal if a function is stable/fragile/central to codebase. Type information helps assess safety.  
**Implementation:** AST analysis + git history + test discovery + optional type inference (LLM or AST).

---

### 9. **Return the Diff**
```python
result = tool.write_code(buffer_id, file, start_line, new_lines)
# Also returns: {status, diff: "- old\n+ new", impact_summary, affected_tests}
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** See changes immediately without calling diff() separately; understand impact.  
**Implementation:** Store original before write; generate diff on response.

---

### 10. **Execution Paths** (Trace code flow)
```python
paths = tool.trace_execution_paths(buffer_id, symbol="handle_request", max_depth=3)
# Returns: [
#   {path: ["handle_request → validate_input → check_auth"], branches: 3},
#   {path: ["handle_request → process_data → cache"], branches: 2}
# ]
```
**Priority:** ⭐⭐⭐  
**Benefit:** For complex logic, know all execution branches before editing.  
**Implementation:** Control flow graph + AST branch detection.

---

## Advanced Navigation (Nice-to-Have)

### 11. **Dependency Graph Visualization**
```python
graph = tool.get_dependency_graph(buffer_id, symbol="process_payment", depth=2)
# Returns: {nodes: [{id, label, type}], edges: [{from, to, type: "calls" | "imports"}]}
```
**Priority:** ⭐⭐⭐  
**Benefit:** Visualize relationships; understand architecture at a glance.  
**Implementation:** Use NetworkX; serialize for GraphML or JSON export.

---

### 12. **Dead Code Detection**
```python
dead = tool.find_dead_code(buffer_id, threshold=0.95)
# Returns: [{file, symbol, reason: "never_called" | "unused_import" | "unreachable", confidence}]
```
**Priority:** ⭐⭐⭐  
**Benefit:** Identify safe-to-delete code before refactoring.  
**Implementation:** Call graph analysis + import tracking.

---

### 13. **Code Smell Detection**
```python
smells = tool.detect_code_smells(buffer_id, types=["long_function", "deep_nesting", "duplicates"])
# Returns: [{file, line, type, severity: "low" | "medium" | "high", suggestion}]
```
**Priority:** ⭐⭐⭐  
**Benefit:** Automatically flag refactoring opportunities.  
**Implementation:** AST metrics + text similarity.

---

### 14. **Performance Hotspots**
```python
hotspots = tool.find_performance_hotspots(buffer_id)
# Returns: [{file, symbol, reason: "n_plus_one" | "inefficient_loop" | "unbounded_recursion", severity}]
```
**Priority:** ⭐⭐⭐  
**Benefit:** Identify performance-critical code for optimization.  
**Implementation:** Pattern matching + complexity analysis.

---

### 15. **Security Vulnerability Scanning**
```python
vulns = tool.scan_security(buffer_id, severity_min="medium")
# Returns: [{file, line, type: "sql_injection" | "xss" | "unsafe_pickle", fix_suggestion}]
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** Catch security issues before they reach production.  
**Implementation:** Pattern library + regex rules + data flow analysis.

---

### 16. **Auto-Documentation Generation**
```python
docs = tool.generate_documentation(buffer_id, symbol="process_payment", style="google" | "numpy")
# Returns: {docstring, type_hints, examples, generated_from_code: True}
```
**Priority:** ⭐⭐⭐  
**Benefit:** Auto-generate accurate documentation from code analysis.  
**Implementation:** AST + LLM integration (optional).

---

### 17. **Type Inference Improvements**
```python
inferred = tool.infer_types(
    buffer_id, 
    file="src/utils.py", 
    start_line=42, 
    end_line=58,
    method="llm"  # or "ast" for fast, lightweight inference
)
# Returns: {
#   variables: {name: type, confidence: 0.92},
#   return_type: str,
#   return_confidence: 0.88,
#   parameter_types: {param_name: type},
#   method: "llm",
#   inferred_at: "2026-05-10T14:32:00Z"
# }
```
**Priority:** ⭐⭐⭐  
**Benefit:** Detect type issues in untyped code; understand code intent.  
**Implementation:**
- **AST-based (faster, ~2ms):** Extracts explicit type hints and basic type inference from syntax patterns
- **LLM-assisted (more accurate, ~100-300ms):** Analyzes code semantics to infer types in complex/dynamic code; includes confidence scores

---

### 18. **Find Similar Code Patterns**
```python
similar = tool.find_similar_patterns(buffer_id, code_snippet="def validate(x):\n    return x is not None")
# Returns: [{file, start_line, end_line, similarity_score, diff}]
```
**Priority:** ⭐⭐⭐  
**Benefit:** Find duplicate logic for consolidation.  
**Implementation:** Semantic + syntactic matching.

---

### 19. **Code Quality Score**
```python
score = tool.get_code_quality_score(buffer_id, file="src/auth.py")
# Returns: {overall: 0.78, maintainability: 0.85, testability: 0.65, documentation: 0.72}
```
**Priority:** ⭐⭐⭐  
**Benefit:** Quantify code health; track improvements over time.  
**Implementation:** Multi-factor scoring algorithm.

---

### 20. **Deprecated Function Detection**
```python
deprecated = tool.find_deprecated(buffer_id)
# Returns: [{file, symbol, deprecated_since, replacement_symbol, migration_path}]
```
**Priority:** ⭐⭐⭐  
**Benefit:** Identify code that uses outdated APIs.  
**Implementation:** Comment parsing + version tracking.

---

## Change Management Features

### 21. **Refactoring Suggestions**
```python
suggestions = tool.suggest_refactorings(buffer_id, symbol="process_payment")
# Returns: [
#   {type: "extract_method", lines: (10, 20), benefit: "Reduces complexity", risk: "low"},
#   {type: "consolidate_duplicates", files: ["src/a.py", "src/b.py"], reduction: "30 LOC"}
# ]
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** AI suggests safe refactorings with risk assessment.  
**Implementation:** Pattern library + impact analysis.

---

### 22. **Pre-Commit Validation**
```python
validation = tool.validate_changes(buffer_id, dry_run=True)
# Returns: {type_errors: [], broken_imports: [], test_failures: [...], safe_to_commit: True}
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** Validate changes before committing (without running full test suite).  
**Implementation:** Static type checking + import resolution + test impact prediction.

---

### 23. **Rollback Information**
```python
rollback = tool.get_rollback_info(buffer_id, file="src/auth.py")
# Returns: {last_working_commit, diff_to_revert, test_failures_fixed, features_added}
```
**Priority:** ⭐⭐⭐  
**Benefit:** Understand what changed and why—helps with debugging regressions.  
**Implementation:** Git integration + semantic analysis.

---

### 24. **Change Request Templates**
```python
template = tool.generate_change_template(buffer_id, request="add retry logic to database calls")
# Returns: {files_to_modify: [...], change_strategy: "...", test_cases_needed: [...], risk_assessment: "..."}
```
**Priority:** ⭐⭐⭐  
**Benefit:** AI creates a plan before making changes.  
**Implementation:** Semantic search + pattern matching + impact analysis.

---

### 25. **API Contract Changes**
```python
changes = tool.detect_api_changes(buffer_id, since_commit="main")
# Returns: [{symbol, breaking: bool, parameters_added: [...], return_type_changed: bool, migration_guide: "..."}]
```
**Priority:** ⭐⭐⭐  
**Benefit:** Detect breaking changes that would affect consumers.  
**Implementation:** Symbol comparison + type analysis.

---

### 26. **Changelog Generation**
```python
changelog = tool.generate_changelog(buffer_id, since_commit="v1.0.0")
# Returns: {features: [...], bugfixes: [...], breaking_changes: [...], migration_notes: "..."}
```
**Priority:** ⭐⭐⭐  
**Benefit:** Auto-generate release notes from code changes.  
**Implementation:** Git log parsing + semantic analysis.

---

## Safety & Quality Features

### 27. **Configuration Extraction**
```python
config = tool.extract_configuration(buffer_id)
# Returns: {env_vars: [...], config_files: [...], hardcoded_secrets: [...], default_values: {...}}
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** Understand what needs to be configured; detect hardcoded secrets.  
**Implementation:** AST + regex patterns + credential scanning.

---

### 28. **Environment Variable Tracking**
```python
env_vars = tool.get_environment_variables(buffer_id)
# Returns: {
#   "DATABASE_URL": {used_in: ["src/db.py:42"], required: True, default: None},
#   "DEBUG_MODE": {used_in: [...], required: False, default: "False"}
# }
```
**Priority:** ⭐⭐⭐  
**Benefit:** Know all environment dependencies; generate .env templates.  
**Implementation:** AST traversal + os.getenv detection.

---

### 29. **Logging Pattern Analysis**
```python
logs = tool.analyze_logging_patterns(buffer_id)
# Returns: {
#   total_logs: 142,
#   levels: {debug: 30, info: 80, warning: 25, error: 7},
#   missing_logs_in: [{"file": "src/critical.py", "symbols": [...]}],
#   inconsistent_format: [{file, line, format_string}]
# }
```
**Priority:** ⭐⭐⭐  
**Benefit:** Ensure consistent logging for debugging/monitoring.  
**Implementation:** AST log detection + pattern matching.

---

### 30. **Error Handling Pattern Detection**
```python
errors = tool.analyze_error_handling(buffer_id)
# Returns: {
#   try_except_blocks: 42,
#   uncaught_exceptions: [{file, line, exception_type}],
#   broad_catches: [{file, line, catches: "Exception"}],
#   missing_finally: [{file, line, resource}],
#   suggestions: [...]
# }
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** Ensure robust error handling; prevent silent failures.  
**Implementation:** AST exception analysis + resource tracking.

---

## Advanced Analysis Features

### 31. **Database Schema Detection**
```python
schema = tool.extract_database_schema(buffer_id)
# Returns: {tables: [...], relationships: [...], queries: [...], migrations_needed: [...]}
```
**Priority:** ⭐⭐⭐  
**Benefit:** Understand data model without reading SQL files.  
**Implementation:** SQLAlchemy models + raw SQL parsing + ORM analysis.

---

### 32. **API Endpoint Mapping**
```python
endpoints = tool.map_api_endpoints(buffer_id)
# Returns: [
#   {method: "POST", path: "/api/v1/payment", handler: "process_payment", auth: "jwt", rate_limit: "100/min", is_async: True},
#   ...
# ]
```
**Priority:** ⭐⭐⭐  
**Benefit:** Know all exposed endpoints; find security issues. FastAPI support includes async detection.  
**Implementation:** FastAPI decorator parsing (@app.post, @app.get, etc.) + async/await detection + dependency injection analysis.

---

### 33. **Cache Invalidation Patterns**
```python
cache = tool.analyze_cache_patterns(buffer_id)
# Returns: {
#   caches_used: ["redis", "local_lru"],
#   invalidation_logic: [{cache, triggers: [...], safe: bool}],
#   stale_data_risks: [{file, line, risk_level}]
# }
```
**Priority:** ⭐⭐⭐  
**Benefit:** Prevent stale cache bugs.  
**Implementation:** Cache library detection + invalidation tracking.

---

### 34. **Thread Safety Analysis**
```python
threading = tool.analyze_thread_safety(buffer_id)
# Returns: {
#   shared_state: [{name, modified_by: [...], protected_by: "lock" | "atomic" | "none"}],
#   race_conditions: [{file, line, variables, risk_level}],
#   deadlock_risks: [...]
# }
```
**Priority:** ⭐⭐⭐  
**Benefit:** Catch concurrency bugs early.  
**Implementation:** Variable tracking + lock detection + deadlock analysis.

---

### 35. **Memory Leak Detection**
```python
memory = tool.detect_memory_issues(buffer_id, language="python")
# Returns: {
#   circular_refs: [{file, symbols}],
#   unbounded_collections: [{file, line, collection_name, growth_reason}],
#   resource_leaks: [{file, line, resource, cleanup_missing: bool}]
# }
```
**Priority:** ⭐⭐⭐  
**Benefit:** Identify memory issues in long-running processes.  
**Implementation:** Reference tracking + resource lifecycle analysis.

---

## Automated Code Quality Tools

### 36. **Format Code (Black)** — Directory-Wide Batch Operations
```python
result = tool.auto_format(
    buffer_id,
    files=None,  # If None, format entire buffer directory
    formatter="black",  # or "ruff.format"
    line_length=88,
    skip_magic_trailing_comma=False,
    dry_run=True,  # Preview before applying
    exclude_patterns=["*.pb2.py", "migrations/"]  # Optional exclusions
)
# Returns: {
#   status: "ok",
#   total_files: 42,
#   formatted_files: 15,
#   already_formatted: 27,
#   changes: [
#     {file: "src/auth.py", added_lines: 2, removed_lines: 1, diff: "..."},
#     ...
#   ],
#   summary: "Formatted 15 files (2 blank lines, 5 long lines wrapped, ...)"
# }
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** Format entire codebase in one call; prepare entire PRs for review.  
**Implementation:** Wrap Black to work on directories (like `black .`); return aggregated results and diffs.

---

### 37. **Lint Code (Ruff)** — Directory-Wide Batch Operations
```python
result = tool.auto_lint(
    buffer_id,
    files=None,  # If None, lint entire buffer directory
    linter="ruff",
    select=["E", "F", "W"],  # Error, undefined names, warnings
    ignore=["E501"],  # Line too long (handled by formatter)
    auto_fix=True,  # Auto-fix fixable issues
    dry_run=True,
    exclude_patterns=["tests/", "*.pb2.py"]  # Optional exclusions
)
# Returns: {
#   status: "ok",
#   total_files: 42,
#   files_with_issues: 8,
#   total_issues: 47,
#   issues: [
#     {file: "src/auth.py", line: 42, code: "F841", message: "unused variable", fixed: True},
#     {file: "src/db.py", line: 58, code: "E302", message: "expected 2 blank lines", fixed: False}
#   ],
#   fixed_count: 2,
#   unfixed_count: 1,
#   by_rule: {"F841": {count: 5, files: [...]}, "E302": {count: 8, files: [...]}},
#   auto_fixed_code_available: True
# }
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** Scan entire codebase for issues; auto-fix what can be fixed; understand health across project.  
**Implementation:** Wrap Ruff to work on directories (like `ruff check .`); aggregate results by file/rule/severity.

---

### 38. **Combined Format & Lint** — Full Directory Polish in One Call
```python
result = tool.auto_polish(
    buffer_id,
    files=None,  # If None, polish entire buffer directory
    format_with="black",
    lint_with="ruff",
    auto_fix_lints=True,
    line_length=88,
    ruff_select=["E", "F", "W"],
    exclude_patterns=["tests/", "migrations/"],
    dry_run=True
)
# Returns: {
#   status: "ok",
#   formatting: {
#     total_files: 42,
#     formatted_files: 15,
#     changes: [{file, added_lines, removed_lines, diff}, ...]
#   },
#   linting: {
#     total_files: 42,
#     files_with_issues: 8,
#     total_issues: 47,
#     fixed: 2,
#     unfixed: 1,
#     by_rule: {"F841": {count: 5}, "E302": {count: 8}}
#   },
#   ready_to_commit: True,
#   summary: "Formatted 15 files, fixed 2 lint issues, 1 manual fix needed"
# }
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** Single call to format AND lint entire codebase—perfect for preparing PRs.  
**Implementation:** Convenience wrapper that delegates to `auto_format()` then `auto_lint()` in sequence. Format runs first (so lint checks the formatted code), then results are merged. ~20 lines of delegation logic.

---

### 39. **Lint Entire Buffer/Project** (Detailed Analysis)
```python
results = tool.lint_buffer(
    buffer_id,
    files=None,  # If None, lint all files in buffer
    linter="ruff",
    select=None,  # Use default or custom rules
    exclude_patterns=["tests/", "*.pb2.py"],
    group_by="file" | "severity" | "rule",  # Organization preference
    auto_fix=False  # Don't auto-fix, just report
)
# Returns: {
#   total_issues: 47,
#   by_file: {
#     "src/auth.py": {count: 5, issues: [{line, code, message}]},
#     "src/db.py": {count: 12, issues: [...]}
#   },
#   by_severity: {error: 8, warning: 25, info: 14},
#   by_rule: {
#     "F841": {count: 5, files: ["src/auth.py", ...]},
#     "E302": {count: 8, files: [...]}
#   }
# }
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** Deep analysis of linting health; understand issues by file/rule/severity; no auto-fixing (report-only).  
**Implementation:** Ruff on entire directory with detailed aggregation.

---

### 40. **Format Entire Buffer/Project** (Detailed Analysis)
```python
results = tool.format_buffer(
    buffer_id,
    files=None,  # If None, format all files in buffer
    formatter="black",
    line_length=88,
    exclude_patterns=["migrations/", "*.pb2.py"],
    dry_run=True,
    summary_only=False  # If True, return only stats (not full diffs)
)
# Returns: {
#   total_files: 42,
#   formatted_files: 15,
#   already_formatted: 27,
#   changes: [
#     {file: "src/auth.py", added_lines: 2, removed_lines: 1, diff: "..."},
#     {file: "src/utils.py", added_lines: 5, removed_lines: 3, diff: "..."}
#   ],
#   total_lines_added: 47,
#   total_lines_removed: 23,
#   summary: "Formatted 15 files (2 blank lines, 5 long lines wrapped, 8 imports reorganized)"
# }
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** Deep analysis of formatting across codebase; understand exactly what changed; prepare entire PRs.  
**Implementation:** Black on entire directory with detailed change tracking.

---

### 41. **Linting with Custom Configuration**
```python
result = tool.auto_lint_with_config(
    buffer_id,
    config_file=".ruff.toml",  # Read from pyproject.toml or .ruff.toml
    file="src/auth.py",
    auto_fix=True,
    dry_run=True
)
# Returns: {
#   status: "ok",
#   config_used: {line_length: 88, select: ["E", "F"], ignore: ["E501"]},
#   issues: [...],
#   auto_fixed_code: "..."
# }
```
**Priority:** ⭐⭐⭐  
**Benefit:** Respect project's linting rules instead of defaults.  
**Implementation:** Read pyproject.toml/setup.cfg; pass to Ruff.

---

### 42. **Formatting with Custom Configuration**
```python
result = tool.auto_format_with_config(
    buffer_id,
    config_file="pyproject.toml",
    file="src/auth.py",
    dry_run=True
)
# Returns: {
#   status: "ok",
#   config_used: {line_length: 88, target_version: "py39", ...},
#   formatted_code: "...",
#   diff: "...",
#   changes_count: 3
# }
```
**Priority:** ⭐⭐⭐  
**Benefit:** Honor project-specific formatting preferences.  
**Implementation:** Read Black/Ruff config; apply with custom settings.

---

### 43. **Pre-Commit Code Polish**
```python
result = tool.polish_before_commit(
    buffer_id,
    files_to_commit=["src/auth.py", "src/db.py"],  # Or None for all changed
    format_with="black",
    lint_with="ruff",
    check_only=False,  # If True, only validate (don't modify)
)
# Returns: {
#   status: "ok" | "needs_fixes",
#   formatting: {applied: 3},
#   linting: {fixed: 2, unfixed: 1},
#   ready_to_commit: True,
#   pre_commit_warnings: [...]  # e.g., "Large diff detected (200 lines)"
# }
```
**Priority:** ⭐⭐⭐⭐  
**Benefit:** Workflow: Write → Format → Lint → Commit (all in one call).  
**Implementation:** Convenience wrapper that chains `auto_format()` + `auto_lint()` + commit-readiness checks. Delegates to the separate tools rather than re-implementing their logic.

---

### **Design Principle: Directory-First Operations**
All formatting/linting tools operate on **entire directories by default**, matching native tool behavior:

```python
# Format entire project (default: files=None)
tool.auto_format(buffer_id)  # Format everything

# Lint entire project (default: files=None)
tool.auto_lint(buffer_id)  # Lint everything

# Polish entire project (default: files=None)
tool.auto_polish(buffer_id)  # Format + lint everything
```

**Why directory-first?**
- ✅ Matches native `black .`, `ruff check .` behavior
- ✅ Prepares entire PRs in one call
- ✅ Consistent formatting across codebase
- ✅ Single API call for AI agents (simpler than single-file mode)
- ✅ Dry-run preview before applying

### **Design Principle: Layered Tool Architecture**
Separate tools are stable, feature-rich primitives. Combined tools are convenience wrappers that delegate to them:

**Layer 1 — Primitives** (independent, full-featured):
- `auto_format()` — format only, all formatter options exposed
- `auto_lint()` — lint only, all linter options exposed
- `format_buffer()` — deep format analysis, report-only
- `lint_buffer()` — deep lint analysis, report-only

**Layer 2 — Convenience Wrappers** (delegate to Layer 1):
- `auto_polish()` → calls `auto_format()` then `auto_lint()`, merges results
- `polish_before_commit()` → calls `auto_polish()` + adds commit-readiness checks

**Why both layers?**
- ✅ Agents that need only linting (or only formatting) use Layer 1 directly — no confusing `auto_polish(format_with=None)` hacks
- ✅ Combined tools evolve independently without bloating the primitive signatures
- ✅ Each primitive can grow its own options (e.g., `target_version` for format, `select`/`ignore` rule sets for lint) without exploding the combined tool's parameter list
- ✅ Composable workflows: format → review → lint → fix → format again (combined can't express this)
- ✅ Wrapper implementation is ~20 lines of delegation — negligible maintenance cost

---

### Usage Example for AI Agents

```python
# Workflow 1: Make changes
result = tool.write_code(buffer_id, file, start_line, new_lines)

# Workflow 2: Auto-polish before commit (single call)
polish = tool.auto_polish(buffer_id, file, dry_run=True)
if polish['ready_to_commit']:
    tool.commit(buffer_id)  # Proceed
else:
    # Review unfixed linting issues
    for issue in polish['linting']['unfixed']:
        print(f"Needs manual fix: {issue}")
```

---

## Type Inference Strategy: AST vs. LLM

All tools that support type inference should allow AI agents to choose the inference method:

### **AST-Based Type Inference** (Fast, Lightweight)
```python
# Use this for real-time, latency-sensitive operations
results = tool.semantic_search(buffer_id, query, include_types=True, type_inference_method="ast")

results = tool.infer_types(buffer_id, file, method="ast")

context = tool.get_full_context(buffer_id, symbol, type_inference_method="ast")
```
**Characteristics:**
- ⚡ **Speed:** ~1-5ms per operation
- 📝 **Accuracy:** High for explicitly typed code, medium for untyped Python
- 💾 **Memory:** Minimal (AST-based, no model inference)
- ✅ **Use When:** Quick type hints needed, latency < 100ms required, mostly typed codebase

**Returns:**
```python
{
    "type": "str",
    "confidence": None,  # AST doesn't provide confidence
    "method": "ast",
    "inferred_from": "explicit_annotation"  # or "pattern_matching"
}
```

---

### **LLM-Assisted Type Inference** (Accurate, Context-Aware) — **DEFAULT**
```python
# Use this for comprehensive understanding and complex type inference
results = tool.semantic_search(buffer_id, query, include_types=True, type_inference_method="llm")

results = tool.infer_types(buffer_id, file, method="llm")

context = tool.get_full_context(buffer_id, symbol, type_inference_method="llm")
```
**Characteristics:**
- 🎯 **Speed:** ~50-300ms per operation (depends on code complexity)
- 📊 **Accuracy:** Very high, even for untyped/dynamic Python; understands intent
- 🧠 **Intelligence:** Analyzes semantics, data flow, and usage patterns
- 💾 **Model:** Uses the same local embedding model as semantic search (no API calls, no cost)
- ✅ **Use When:** Comprehensive analysis needed, latency < 500ms acceptable, complex/untyped code

**Returns:**
```python
{
    "type": "Dict[str, List[int]]",
    "confidence": 0.94,  # LLM provides confidence scores
    "method": "llm",
    "reasoning": "Variable used as dict with list values; inferred from usage patterns"
}
```

**Model Configuration:**
Type inference uses the same embedding model configured during tool initialization:
```python
# Global embedder choice (affects both search AND type inference)
# Use local embedding models only—no API calls, no cost, full privacy
tool = CodeEmbeddingTool(
    work_dir="./project",
    model_name="all-MiniLM-L6-v2",  # Fast local model, or use larger ones like "multilingual-e5-large"
    device="cuda"  # or "cpu" for CPU-only
)
# Both semantic_search(...) and infer_types(...) now use the same local embedder
```
No per-request model switching is needed—the embedder is the same for both features. All embedding happens locally, ensuring privacy and cost-effectiveness.

---

### **Confidence Score Caching Strategy**

LLM-assisted type inference produces confidence scores (0.0-1.0) that are expensive to recompute (50-300ms per symbol) and consumed by multiple features (Features 2, 3, 8, 17). A caching layer avoids redundant inference across repeated queries for the same symbol.

**Cache Design:**

```python
class TypeInferenceCache:
    """Session-scoped LRU cache for type inference confidence scores."""

    def __init__(self, max_entries=500):
        self._cache: OrderedDict[str, InferredType] = OrderedDict()
        self._file_to_symbols: dict[str, set[str]] = {}  # reverse index for invalidation
        self._max_entries = max_entries

    def get(self, symbol_key: str) -> Optional[InferredType]:
        """Return cached inference or None (cold-path passthrough)."""
        ...

    def put(self, symbol_key: str, file: str, inferred: InferredType):
        """Cache result; evict LRU entry if at capacity. Track file→symbol mapping."""
        ...

    def invalidate_file(self, file: str):
        """Evict all entries for symbols defined in a modified file."""
        for symbol_key in self._file_to_symbols.get(file, set()):
            self._cache.pop(symbol_key, None)
        self._file_to_symbols.pop(file, None)
```

**Rules:**

| Rule | Detail |
|------|--------|
| **Scope** | Session-scoped per `buffer_id`. No persistence, no cross-session sharing. Cache dies when the buffer session ends. |
| **Capacity** | LRU eviction at 500 symbols. Sufficient for typical AI agent sessions (~20-50 unique symbol queries). |
| **Invalidation** | Any `write_code` call on a file evicts all cached type entries for symbols in that file. Uses the `symbol_index` reverse mapping (file → defined symbols) to identify affected entries. |
| **TTL** | None. Session scope + write-invalidation removes the need for time-based expiry. |
| **Passthrough** | Cache miss = run inference immediately. No blocking, no pre-warming, no background pre-computation. |
| **AST entries** | Not cached. AST inference is already ~1-5ms; caching provides negligible benefit and adds complexity. Only LLM-assisted confidence scores are cached. |

**Integration with `write_code`:**

```python
def write_code(self, buffer_id, file, start_line, new_lines):
    # ... existing write logic ...
    self._type_cache.invalidate_file(file)  # Bust cache for modified file
    return {status, diff, impact_summary, affected_tests}
```

**Integration with type-returning features:**

```python
def infer_types(self, buffer_id, file, method="llm"):
    cache_key = f"{buffer_id}:{file}:{symbol}"
    if method == "llm":
        cached = self._type_cache.get(cache_key)
        if cached:
            return cached
    result = self._run_llm_inference(...)
    if method == "llm":
        self._type_cache.put(cache_key, file, result)
    return result
```

---

---

## Framework Integration Strategy: FastAPI + Async

GigaCode should provide **generic core APIs** with **FastAPI-specific adapters and examples**.

### **Why FastAPI?**
✅ **Async/Await Native** — Aligns with streaming search results and non-blocking code analysis  
✅ **Modern Python** — Supports type hints deeply (Pydantic models)  
✅ **Performance** — Lower latency for AI agent interactions  
✅ **Standard Adoption** — Becoming the default for Python async web services  

### **Core Pattern: Async Search with Streaming**
```python
# FastAPI endpoint for streaming search results
@app.post("/api/v1/search/semantic")
async def semantic_search_streaming(
    buffer_id: str,
    query: str,
    top_k: int = 5,
    include_types: bool = True,
    type_inference_method: str = "llm"  # or "ast"
):
    """Stream semantic search results as they're found."""
    async def generate():
        async for result in tool.semantic_search_stream(
            buffer_id, query, top_k=top_k, 
            include_types=include_types,
            type_inference_method=type_inference_method
        ):
            yield f"data: {json.dumps(result)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### **Async Tool Methods**
All blocking operations should have async variants:
```python
# Async search (returns async generator)
async for result in tool.semantic_search_stream(buffer_id, query):
    print(result)  # Process results as they arrive

# Async batch operations
results = await tool.search_batch_async(buffer_id, queries=["q1", "q2", "q3"])

# Async type inference
types = await tool.infer_types_async(buffer_id, file, method="llm")
```

### **FastAPI Dependency Integration**
GigaCode tools can be used as FastAPI dependencies:
```python
from fastapi import Depends
from gigacode import CodeEmbeddingTool

async def get_tool() -> CodeEmbeddingTool:
    """Dependency for injecting GigaCode tool."""
    return CodeEmbeddingTool(work_dir="./project")

@app.post("/api/v1/refactor")
async def suggest_refactor(
    buffer_id: str,
    symbol: str,
    tool: CodeEmbeddingTool = Depends(get_tool)
):
    suggestions = await tool.suggest_refactorings_async(buffer_id, symbol)
    return {"suggestions": suggestions}
```

### **Pydantic Model Integration**
All request/response bodies use Pydantic for validation:
```python
from pydantic import BaseModel
from typing import Optional

class SemanticSearchRequest(BaseModel):
    buffer_id: str
    query: str
    top_k: int = 5
    include_types: bool = True
    type_inference_method: str = "llm"  # Validated to "llm" | "ast"
    
    class Config:
        json_schema_extra = {
            "example": {
                "buffer_id": "my-project",
                "query": "authenticate user",
                "top_k": 5,
                "include_types": True,
                "type_inference_method": "llm"
            }
        }

class SearchResult(BaseModel):
    file: str
    start_line: int
    end_line: int
    text: str
    signature: Optional[str]
    type_inference_method: str
    type_confidence: Optional[float]

class SemanticSearchResponse(BaseModel):
    results: list[SearchResult]
    buffer_id: str
    query: str
    elapsed_ms: int
    cache_hit: bool
```

### **Middleware for Rate Limiting & Monitoring**
```python
from fastapi import Request
from gigacode import RateLimiter

rate_limiter = RateLimiter(calls=100, period=60)  # 100 calls/min

@app.middleware("http")
async def add_rate_limit(request: Request, call_next):
    if not rate_limiter.allow():
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"}
        )
    return await call_next(request)
```

### **Implementation Priority**

| Phase | Task | Details |
|-------|------|----------|
| **Phase 1** | Async Core APIs | `semantic_search_stream()`, `write_code_async()`, `search_batch_async()` |
| **Phase 1** | FastAPI Examples | Sample endpoints for all Phase 1 features |
| **Phase 2** | Advanced Streaming | Stream results for batch operations, type inference |
| **Phase 2** | Pydantic Models | Full request/response validation |
| **Phase 3** | FastAPI App Template | Pre-built starter app with auth, rate limiting, logging |
| **Phase 3** | Dependency Injection | Best practices for tool lifecycle management |

### **Generic Core (Framework-Agnostic)**
Keep core `CodeEmbeddingTool` APIs synchronous and non-blocking. FastAPI adapters wrap async variants:

```python
# Core API (sync, generator-based, local embeddings only)
def semantic_search(buffer_id, query, **kwargs):
    yield SearchMatch(...)  # Non-blocking generator, uses local model

# FastAPI adapter (async, local embeddings only)
async def semantic_search_stream(buffer_id, query, **kwargs):
    async for result in tool.semantic_search(buffer_id, query, **kwargs):
        yield result  # No API calls, all local processing
```

**All embeddings are computed locally using sentence-transformers models (e.g., all-MiniLM-L6-v2, multilingual-e5-large). Zero external API calls, zero embedding costs, 100% privacy.**

---

## Tool Schema & Documentation Standards

**All GigaCode tools must follow MCP server standards for schema definition and documentation.** This ensures AI agents can discover, understand, and use tools autonomously.

### **Python Tool Definition Standard**

Every tool method must have:
1. **Comprehensive docstring** explaining what it does
2. **Type hints** on all parameters and return types
3. **Pydantic model** for complex inputs (for validation + schema generation)
4. **Return type definition** so AI agents know what they get back

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class SemanticSearchParams(BaseModel):
    """Parameters for semantic search."""
    buffer_id: str = Field(..., description="ID of the code buffer to search")
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(5, description="Number of results to return (default: 5)")
    include_types: bool = Field(False, description="Include type information in results")
    type_inference_method: str = Field("llm", description="Type inference: 'llm' (accurate) or 'ast' (fast)")

class SearchResult(BaseModel):
    """A single search result."""
    file: str = Field(..., description="File path containing the match")
    start_line: int = Field(..., description="Starting line number (1-indexed)")
    end_line: int = Field(..., description="Ending line number (inclusive)")
    text: str = Field(..., description="Full source code of the match")
    signature: Optional[str] = Field(None, description="Function/class signature if applicable")
    type_confidence: Optional[float] = Field(None, description="Type confidence (0.0-1.0, LLM only)")

class SemanticSearchResponse(BaseModel):
    """Response from semantic search."""
    results: List[SearchResult] = Field(..., description="List of search results")
    buffer_id: str = Field(..., description="The buffer ID searched")
    query: str = Field(..., description="The query executed")
    elapsed_ms: int = Field(..., description="Query execution time in milliseconds")
    cache_hit: bool = Field(..., description="Whether result was from cache")

def semantic_search(
    buffer_id: str,
    query: str,
    top_k: int = 5,
    include_types: bool = False,
    type_inference_method: str = "llm"
) -> SemanticSearchResponse:
    """Search codebase semantically using embeddings.
    
    Find code similar to a natural language query by embedding both and 
    computing semantic similarity. Supports optional type inference for 
    parameter/return type hints.
    
    Args:
        buffer_id: ID of the code buffer to search
        query: Natural language search query (e.g., "authenticate user")
        top_k: Maximum number of results to return (default: 5)
        include_types: Include inferred type hints in results (default: False)
        type_inference_method: How to infer types:
            - "llm": Use local embedder for semantic type inference (~50-300ms, more accurate)
            - "ast": Use AST patterns only (~1-5ms, less accurate for untyped code)
    
    Returns:
        SemanticSearchResponse: List of matching code blocks with metadata
    
    Example:
        >>> results = tool.semantic_search(
        ...     "my-project",
        ...     "validate email addresses",
        ...     top_k=3,
        ...     include_types=True,
        ...     type_inference_method="llm"
        ... )
        >>> for r in results.results:
        ...     print(f"{r.file}:{r.start_line} - {r.signature}")
    """
    ...  # Implementation
```

### **API Schema Endpoint**

Expose an `/api/v1/schema` endpoint that returns all tool definitions in MCP-compliant format:

```python
@app.get("/api/v1/schema")
async def get_api_schema():
    """Return all tool definitions in MCP server format.
    
    AI agents use this to discover available tools, their parameters,
    return types, and usage examples.
    """
    return {
        "tools": [
            {
                "name": "semantic_search",
                "description": "Search codebase semantically using embeddings",
                "inputSchema": SemanticSearchParams.model_json_schema(),
                "returnSchema": SemanticSearchResponse.model_json_schema(),
                "examples": [
                    {
                        "input": {
                            "buffer_id": "my-project",
                            "query": "authenticate user",
                            "top_k": 5,
                            "include_types": True,
                            "type_inference_method": "llm"
                        },
                        "description": "Find authentication-related code with type hints"
                    }
                ]
            },
            {
                "name": "get_references",
                "description": "Find all references to a symbol (callers/callees)",
                "inputSchema": GetReferencesParams.model_json_schema(),
                "returnSchema": GetReferencesResponse.model_json_schema(),
                "examples": [...]
            },
            # ... all other tools
        ],
        "version": "1.0.0",
        "apiVersion": "2026-05-10",
        "standards": ["MCP 1.0", "JSON Schema Draft 7", "OpenAPI 3.0"]
    }
```

### **Tool Docstring Standard (Summary)**

Every tool MUST have:

| Element | Format | Example |
|---------|--------|---------|
| **Name** | Clear, action-oriented | `semantic_search`, `get_references`, `auto_format` |
| **Description** | 1-2 lines, purpose | "Search codebase semantically using embeddings" |
| **Parameters** | Pydantic model with descriptions | `buffer_id: str = Field(..., description="...")` |
| **Return Type** | Pydantic model with descriptions | `class SemanticSearchResponse(BaseModel): ...` |
| **Examples** | At least one usage example | `>>> tool.semantic_search("my-project", "auth")` |
| **Notes** | Performance, limitations, alternatives | "Type inference: ~50-300ms (LLM) vs ~1-5ms (AST)" |

### **MCP Server Compliance Checklist**

- ✅ All tools have names, descriptions, and purposes
- ✅ All parameters have type hints and descriptions
- ✅ All return types are defined (Pydantic models)
- ✅ Input schemas use JSON Schema format (from Pydantic)
- ✅ Return schemas use JSON Schema format (from Pydantic)
- ✅ Tools have usage examples in docstrings
- ✅ `/api/v1/schema` endpoint returns all tool definitions
- ✅ Tools are organized by capability (search, analysis, modification, etc.)
- ✅ Performance characteristics documented (speed, accuracy, cost)
- ✅ Tradeoffs clearly explained (e.g., AST vs. LLM type inference)

### **Implementation Priority**

| Phase | Task | Details |
|-------|------|----------|
| **Phase 1** | Define Pydantic models for all Phase 1 features | Input + output schemas |
| **Phase 1** | Write comprehensive docstrings | Every tool gets description + examples |
| **Phase 1** | Implement `/api/v1/schema` endpoint | Tool discovery for AI agents |
| **Phase 2** | Validate against MCP spec | Ensure compliance with standards |
| **Phase 2** | Generate OpenAPI docs from schemas | Auto-generated API documentation |
| **Phase 3** | Create SDK documentation | Show agents how to use each tool |

---

### **Recommendation for AI Agents**

| Scenario | Recommended Method | Rationale |
|----------|-------------------|-----------|
| **Real-time code completion** | `ast` | Speed critical; user waiting |
| **Pre-commit validation** | `llm` | ~200ms acceptable; need accuracy |
| **Semantic search** (interactive) | `llm` default, allow override to `ast` | More accurate results; user expects comprehensive info |
| **Batch analysis** (async) | `llm` | No latency constraint; best accuracy |
| **Type checking before refactor** | `llm` | Safety critical; want high confidence |
| **Quick symbol lookup** | `ast` | Speed > accuracy for quick reference |

**Implementation Detail:** Features 3, 8, 17, 2, and any that return type information should accept:
```python
type_inference_method="llm"  # Default
# OR
type_inference_method="ast"  # Faster alternative
```

---

## Summary Table

| # | Feature | Priority | Effort | Impact | Status |
|----|---------|----------|--------|--------|--------|
| 1 | Reference Map | ⭐⭐⭐⭐⭐ | Medium | High | 🎯 Next |
| 2 | Context Bundle | ⭐⭐⭐⭐⭐ | High | Very High | 🎯 Next |
| 3 | Type Info in Search | ⭐⭐⭐⭐⭐ | Low | High | ✅ Implemented |
| 4 | Impact Analysis | ⭐⭐⭐⭐⭐ | High | Very High | 🎯 Next |
| 5 | Batch Search | ⭐⭐⭐⭐ | Low | Medium | 💡 Proposed |
| 6 | Diff-Aware Search | ⭐⭐⭐⭐ | Medium | High | 💡 Proposed |
| 7 | Test Coverage Map | ⭐⭐⭐⭐ | High | Very High | 🎯 Next |
| 8 | Symbol Metadata | ⭐⭐⭐⭐ | Medium | High | 💡 Proposed |
| 9 | Return the Diff | ⭐⭐⭐⭐ | Low | High | 🔧 In Progress |
| 10 | Execution Paths | ⭐⭐⭐ | High | Medium | 💡 Proposed |
| 11-35 | Advanced Features | ⭐⭐-⭐⭐⭐ | Medium-High | Medium-High | 💡 Roadmap |
| 36 | Format Code (Black) | ⭐⭐⭐⭐ | Low | High | 🎯 Next |
| 37 | Lint Code (Ruff) | ⭐⭐⭐⭐ | Low | High | 🎯 Next |
| 38 | Combined Polish | ⭐⭐⭐⭐ | Medium | Very High | 🎯 Next |
| 39 | Lint Entire Buffer | ⭐⭐⭐⭐ | Medium | High | 💡 Proposed |
| 40 | Format Entire Buffer | ⭐⭐⭐⭐ | Medium | High | 💡 Proposed |
| 41 | Lint with Config | ⭐⭐⭐ | Low | Medium | 💡 Proposed |
| 42 | Format with Config | ⭐⭐⭐ | Low | Medium | 💡 Proposed |
| 43 | Pre-Commit Polish | ⭐⭐⭐⭐ | Medium | Very High | 🎯 Next |

---

## Implementation Priorities

### Phase 1 (Weeks 1-2) — Core Foundations + FastAPI
1. Type Info in Search Results
2. Return the Diff
3. Symbol Metadata
4. Batch Search
5. Auto Format (Black) - Feature 36
6. Auto Lint (Ruff) - Feature 37
7. **NEW:** Async variants: `semantic_search_stream()`, `write_code_async()`, `search_batch_async()`
8. **NEW:** FastAPI example endpoints for Phase 1 features

### Phase 2 (Weeks 3-4) — Navigation & Formatting + FastAPI
1. Reference Map
2. Context Bundle
3. Impact Analysis
4. Test Coverage Map
5. Combined Polish (auto_polish) - Feature 38
6. Pre-Commit Polish - Feature 43
7. **NEW:** Advanced streaming for batch operations
8. **NEW:** Pydantic models for all request/response types
9. **NEW:** FastAPI dependency injection examples

### Phase 3 (Weeks 5-8) — Advanced Analysis + FastAPI Polish
1. Execution Paths
2. Dependency Graph
3. Code Smell Detection
4. Security Scanning
5. Refactoring Suggestions
6. Lint Entire Buffer - Feature 39
7. Format Entire Buffer - Feature 40
8. **NEW:** FastAPI app starter template with auth + rate limiting
9. **NEW:** Middleware for monitoring and error handling

### Phase 4 (Weeks 9+) — Polish & Optimization
1. Advanced features 11-35 based on adoption feedback
2. Config-aware formatting/linting (Features 41-42)
3. Optimization and integration refinements

---

## Expected Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| API roundtrips per code change | 8-10 | 2-3 | 70% reduction |
| Time to understand a symbol | 5 min | 30 sec | 10x faster |
| Safe refactoring confidence | 60% | 95% | +58% |
| Bug detection (pre-commit) | 40% | 85% | +112% |
| Code review time | 30 min | 10 min | 3x faster |
| Code formatting consistency | Manual | 100% auto | Eliminates all style debates |
| Linting errors caught pre-commit | 30% | 95% | 3x improvement |
| Time spent on code review style issues | 10 min per PR | < 1 min | 10x faster |

---

## Notes for Implementation

- **Priority Labeling:** ⭐⭐⭐⭐⭐ = Must-Have, ⭐⭐⭐⭐ = Should-Have, ⭐⭐⭐ = Nice-to-Have
- **Status:** 🎯 Next = Ready to implement, 🔧 In Progress = Under development, 💡 Proposed = Future consideration
- **AI-Agent Focus:** All features designed for autonomous code understanding and manipulation
- **Backward Compatibility:** All new features should extend existing API without breaking changes

---

## Questions for Roadmapping

1. Which Phase 1 features should we prioritize first?
2. ✅ **RESOLVED:** Reference map built **incrementally** using a hybrid three-phase strategy: (1) lazy/on-demand indexing for first query, (2) incremental update on file changes, (3) background fill for progressive completeness. See Feature #1 implementation notes.
3. ✅ **RESOLVED:** Type inference should support **both AST and LLM methods**. Default to LLM-assisted for accuracy; allow AI agents to override to AST for speed (~1-5ms vs. ~50-300ms). See "Type Inference Strategy" section above.
4. Which security vulnerabilities should be covered in v1?
5. ✅ **RESOLVED:** Framework integration with **FastAPI + async** as the primary pattern. Keep core API generic; provide FastAPI-specific adapters and examples. Rationale: FastAPI's async/await aligns with GigaCode's streaming responses and non-blocking search. See "Framework Integration Strategy" section below.
6. ✅ **RESOLVED:** Implement **both separate AND combined tools** using a layered architecture. Layer 1 primitives (`auto_format`, `auto_lint`, `format_buffer`, `lint_buffer`) are independent and full-featured. Layer 2 wrappers (`auto_polish`, `polish_before_commit`) delegate to Layer 1 and merge results. Rationale: forcing `auto_polish(format_with=None)` for lint-only is confusing; primitives can grow independent options without bloating combined signatures; wrapper cost is ~20 lines. See "Design Principle: Layered Tool Architecture".
7. Should `auto_polish()` also run a pre-commit hook integration (e.g., pre-commit framework)?
8. ✅ **RESOLVED:** Formatting/linting **ALWAYS operate on entire directories by default**, just like native `ruff` and `black` tools. Tools like `auto_format()`, `auto_lint()`, and `auto_polish()` work on directories when `files=None` (the default). Single-file operations are available but not the primary use case. This aligns with how developers actually use formatters/linters—for entire projects, not one file at a time. Features 36-40 all support batch operations on directories with detailed aggregation and dry-run previews.
9. ✅ **RESOLVED:** The LLM model for type inference is **the same local embedder model used for semantic search**. No separate configuration needed. The embedding model is chosen once during buffer initialization (e.g., `CodeEmbeddingTool(model_name="all-MiniLM-L6-v2", device="cuda")`) and is reused for both semantic search AND type inference. **All embedding happens locally—no API calls, no cost, full privacy.** If users want to switch models, they configure the embedder globally, which affects both features simultaneously.
10. ✅ **RESOLVED:** Type inference confidence scores **should be cached** using a session-scoped LRU strategy (cap: 500 symbols, no TTL, no persistence). Invalidation is write-driven: any `write_code` call evicts all cached entries for symbols in the modified file via the `symbol_index` reverse mapping. Only LLM-assisted confidence scores are cached; AST inference (~1-5ms) is too cheap to warrant caching. See "Confidence Score Caching Strategy" section.

---

**Version:** 5.0  
**Last Updated:** May 10, 2026  
**Next Review:** June 1, 2026
