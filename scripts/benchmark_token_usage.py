"""Token-usage benchmark: GigaCode MCP server vs plain opencode CLI agent.

Runs repeated, paired coding tasks through a real CLI agent against fresh
sandboxes:

  arm "gigacode" : project copy + compact direct or legacy deferred MCP surface
  arm "plain"    : identical project copy, NO MCP server

Token accounting comes from opencode's storage DB (authoritative per-message
usage, including every tool result payload the model sees).  Tool-call counts
use opencode's built-in tool names (read/grep/glob/bash/edit/write/...);
everything else (gigacode_*) is counted as MCP tool use.

Deliverables (default --out bench_results):
  raw/<runkey>.jsonl        raw opencode JSON event stream per run
  raw/<runkey>.stderr.log   opencode stderr (incl. server/MCP diagnostics)
  runs/<runkey>.json        per-run structured measurement
  measurements.json         all runs accumulated across invocations
  comparison.json           machine-readable comparison summary
  comparison_log.txt        human-readable comparison log

Usage:
    python scripts/benchmark_token_usage.py
    python scripts/benchmark_token_usage.py --arm gigacode
    python scripts/benchmark_token_usage.py --tasks add_docstring --out X
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OPENCODE_EXE = Path(
    os.environ.get("OPENCODE_EXE")
    or shutil.which("opencode.exe")
    or shutil.which("opencode")
    or "opencode"
)
MODEL = os.environ.get("BENCH_MODEL", "opencode-go/glm-5.3-flash")
SURFACE = "direct"
EMBEDDER = "hashing"
BENCHMARK_VERSION = 3
MCP_SERVER_SCRIPT = REPO_ROOT / "scripts" / "mcp_bench_server.py"
CODEBASE_SRC = REPO_ROOT / "examplecode"

RUN_TIMEOUT_SEC = 900
DB_FLUSH_WAIT_SEC = 30
REGULAR_TOOLS = {
    "read",
    "grep",
    "glob",
    "bash",
    "write",
    "edit",
    "patch",
    "multiedit",
    "todowrite",
    "webfetch",
    "list",
    "task",
    "skill",
}

# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------
# A "suite" pins the codebase under test plus a fixed set of search+edit tasks.
# Each task pins its target for behavior/scope checks. Legacy regex hints are
# retained as metadata, not used to grade semantically equivalent implementations:
#   kind "window":   scope = up to `lines` lines after "def <symbol>" / "class <symbol>"
#   kind "file":     scope = whole target file
#   must:     list of regexes, at least one must match
#   must_not: list of regexes, none may match

GIGACODE_AGENTS_SECTION = """## Code tools
A `gigacode` MCP server may be connected in this workspace. Its schema surface
is deliberately tiny: only `gigacode_tool_search` (discover tools) and
`gigacode_tool_call` (invoke a discovered tool) are published; every other tool
is discovered on demand and invoked through `gigacode_tool_call`. The two
published tool descriptions carry ready-made workflow recipes — follow them
instead of re-deriving a multi-call plan:

- search+read one target: `tool_chain` chain='anchor_read' — one call yields
  matches, anchors and `file_hash` (set skeleton=false only if you need raw text)
- edit+persist strictly: end with `tool_chain` chain='anchor_apply' passing
  only `new_lines` and **dry_run=false** — file/anchors/expected_hash are
  auto-resumed from the stored `anchor_read` result
- optional formatting/linting: `tool_chain` chain='post_edit'

Policy: code knowledge and edits go through gigacode tools, NOT through shell
text utilities. Running tests or git status with bash is fine; using
grep/sed/cat to find or change code is a protocol violation. The accuracy of
your edits is graded, so pay for the anchored, verified path.

Do NOT call `embed_codebase` yourself: the server auto-embeds the working
project when the first code tool runs, so go straight to `anchor_read`.

If no `gigacode_*` tools are available, simply use your built-in tools.
"""

GIGACODE_DIRECT_SECTION = """## Code tools
Send tool arguments as native JSON objects. Never emit XML tool-call wrappers
or arg_key/arg_value delimiters. old_text/new_text are plain source strings,
not serialized lists or argument markup. XML that belongs to source is allowed.
When gigacode is available, use code_find to locate/read code, then code_edit
to validate and persist the change. No discovery, embed or post_edit step is
required. Prefer file + old_text + new_text for a small unique replacement.
Use returned file_hash as expected_hash when available. Copy old_text exactly,
including indentation; include context if repeated. Paths may be root-relative,
project-prefixed or absolute inside the root; either separator works.
Anchored new_lines replaces the ENTIRE inclusive range, including declarations.
Inspect the declaration before editing; search candidates may be uncertain.
Run relevant tests. Formatting/linting is optional. If gigacode is absent,
use your built-in tools.
"""
TASK_PROMPT_TEMPLATE = (
    "You are working in a Python repository located in the 'project' folder "
    "(relative to the current directory). The absolute path of the project root is: "
    "{project_root}\n"
    "{layout}\n"
    "IMPORTANT: When you create or modify files, write them inside the 'project' "
    "folder.\n"
    "Finish as soon as the task is done; keep your final answer to at most 3 short "
    "sentences.\n\n"
    "Task: "
)

TASKS_EXAMPLECODE: list[dict[str, Any]] = [
    {
        "id": "add_input_validation",
        "prompt": (
            "There is a function that sorts an unsorted collection by repeatedly "
            "choosing a pivot and partitioning the remaining elements around it. "
            "It currently trusts whatever the caller passes in. Add argument "
            "checking: raise a clear, type-specific built-in error when the "
            "input is not a list, and leave the sorting behavior itself "
            "untouched."
        ),
        "check": {
            "file": "sorting_algorithms.py",
            "symbol": "quick_sort",
            "kind": "window",
            "lines": 14,
            "must": [r"\bTypeError\b"],
        },
    },
    {
        "id": "add_docstring",
        "prompt": (
            "Somewhere in this workspace is a tiny helper that answers, for an "
            "integer larger than one, whether it is divisible only by one and "
            "itself. Its docstring has no runnable example. Add a one-line "
            "doctest-style usage (a '>>> ...' call with its expected True output) right "
            "under the docstring text, and change nothing else."
        ),
        "check": {
            "file": "math_utils.py",
            "symbol": "is_prime",
            "kind": "window",
            "lines": 10,
            "must": [r">>>"],
        },
    },
    {
        "id": "rename_in_file",
        "prompt": (
            "In the module that parses key/value pairs out of raw text, rename the "
            "function that parses key-value pairs from its current name to "
            "'parse_kv_config'. Update the function name itself and any uses of the "
            "old name within the same file. Touch nothing else."
        ),
        "check": {
            "file": "file_parser.py",
            "kind": "file",
            "must": [r"def parse_kv_config"],
            "must_not": [r"def parse_key_value_pairs"],
        },
    },
    {
        "id": "write_error_handling",
        "prompt": (
            "The moving-average helper raises an exception when the window size is "
            "larger than the number of values. Change it to return an empty list in "
            "that case instead of crashing. Only modify that one function."
        ),
        "check": {
            "file": "math_utils.py",
            "symbol": "moving_average",
            "kind": "window",
            "lines": 20,
            "must": [r"return \[\]", r"return list\(\)", r"if [^\n]+:\n\s+return avgs"],
        },
    },
    {
        "id": "add_function",
        "prompt": (
            "Add a new function 'camel_to_kebab' in the same module where the "
            "camelCase-to-snake_case helper lives. It should behave like that helper "
            "but convert to kebab-case (words joined by '-'). Put it directly below "
            "the existing camelCase helper."
        ),
        "check": {
            "file": "string_utils.py",
            "kind": "file",
            "must": [r"def camel_to_kebab"],
        },
    },
]

TASKS_RICH: list[dict[str, Any]] = [
    {
        "id": "add_method_validation",
        "prompt": (
            "A class that wraps human-readable text offers a method that strips a "
            "given substring off the end of its stored text when it matches. "
            "The method currently trusts whatever the caller passes. Add argument "
            "checking: raise a ValueError with a clear message when the suffix "
            "argument is an empty string. Change nothing else."
        ),
        "check": {
            "file": "rich/text.py",
            "symbol": "remove_suffix",
            "kind": "window",
            "lines": 12,
            "must": [r"\bValueError\b"],
        },
    },
    {
        "id": "assert_to_return",
        "prompt": (
            "A small helper picks the first non-missing boolean from several "
            "candidate values, but it currently crashes with an assertion error when "
            "called with no values at all. Change it to return False in that case "
            "instead of crashing. Only modify that one function."
        ),
        "check": {
            "file": "rich/_pick.py",
            "symbol": "pick_bool",
            "kind": "window",
            "lines": 14,
            "must": [r"return False"],
            "must_not": [r"assert\s+values"],
        },
    },
    {
        "id": "rename_filesize_helper",
        "prompt": (
            "In the module that formats file sizes for human readers, rename the "
            "internal string-formatting helper (the one that assembles the final "
            "human-readable size string) to 'format_size'. Update the function "
            "definition and any uses of the old name within the same module. Touch "
            "nothing else."
        ),
        "check": {
            "file": "rich/filesize.py",
            "kind": "file",
            "must": [r"def format_size"],
            "must_not": [r"def _to_str"],
        },
    },
    {
        "id": "add_doctest_example",
        "prompt": (
            "Find the helper that converts a six-digit hexadecimal color string into "
            "a color triplet (red, green, blue), and add a doctest-style example "
            "(a line like '>>> f(...)' with its expected output) to its docstring. "
            "Change nothing else in the file."
        ),
        "check": {
            "file": "rich/color.py",
            "symbol": "parse_rgb_hex",
            "kind": "window",
            "lines": 10,
            "must": [r">>>"],
        },
    },
    {
        "id": "needle_validation",
        "prompt": (
            "There is a helper that mixes one color into another color by a "
            "fractional weight, where a weight of 0.0 yields the first color and "
            "1.0 yields the second color. Add input validation to it: raise a "
            "ValueError with a clear message when the fractional weight is outside "
            "the inclusive range 0.0 to 1.0. Change nothing else."
        ),
        "check": {
            "file": "rich/color.py",
            "symbol": "blend_rgb",
            "kind": "window",
            "lines": 14,
            "must": [r"\bValueError\b"],
        },
    },
]

TASK_SUITES: dict[str, dict[str, Any]] = {
    "examplecode": {
        "codebase": REPO_ROOT / "examplecode",
        "copy_ignore": None,
        "layout": "Repo layout: project/*.py and project/ml/*.py.",
        "agents_md_body": (
            "You are working on a small Python utility library located in the "
            "`project` directory: `project/*.py` plus a `project/ml/` subpackage.\n\n"
            "## Guidelines\n"
            "- Pure Python only; no third-party imports.\n"
            "- Keep the existing code style (small functions, docstrings).\n"
            "- Work only inside the `project` directory."
        ),
        "tasks": TASKS_EXAMPLECODE,
    },
    "rich": {
        "codebase": REPO_ROOT / "bench_repos" / "rich",
        "copy_ignore": [".git", ".github", "__pycache__"],
        "layout": (
            "Repo layout: the 'rich' terminal-formatting library — package sources "
            "under project/rich/, tests under project/tests/ (do not modify tests)."
        ),
        "agents_md_body": (
            "You are working on `rich` (github.com/Textualize/rich), a Python "
            "library for terminal formatting, located in the `project` directory.\n"
            "Package sources are under `project/rich/`; tests under "
            "`project/tests/` (read-only for you).\n\n"
            "## Guidelines\n"
            "- Match the file's existing code style and typing conventions.\n"
            "- Work only inside the `project` directory; never modify "
            "`project/tests/`."
        ),
        "tasks": TASKS_RICH,
    },
}

from scripts.context_benchmark_tasks import TASKS_CONTEXT  # noqa: E402

TASK_SUITES["context"] = {
    "codebase": REPO_ROOT / "scripts" / "benchmark_fixtures" / "context",
    "copy_ignore": [".git", "__pycache__", ".pytest_cache"],
    "layout": "Python checkout API modules, tests/, deployment/values.yaml and frontend/settings.ts. Tests are read-only.",
    "agents_md_body": "Work only inside project. Preserve existing style. Never modify tests or unrelated files.",
    "tasks": TASKS_CONTEXT,
}


def agents_md_for_suite(suite: str) -> str:
    body = TASK_SUITES[suite]["agents_md_body"]
    guidance = GIGACODE_DIRECT_SECTION if SURFACE == "direct" else GIGACODE_AGENTS_SECTION
    if suite == "context" and SURFACE == "direct":
        guidance += (
            "\nFor multi-file orientation use get_task_context. For exact symbols, "
            "callers, dependencies, summaries and configuration structure use code_navigate. "
            "These read-only tools do not require embeddings. Bodies are opt-in with "
            "action=symbol, include_source=true. Use code_edit for minimal persisted edits.\n"
        )
    return f"# Benchmark workspace\n\n{body}\n\n{guidance}\n"


def db_path() -> Path:
    return Path.home() / ".local" / "share" / "opencode" / "opencode.db"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_sandbox(sandbox: Path, suite_name: str) -> Path:
    suite = TASK_SUITES[suite_name]
    project = sandbox / "project"
    if project.exists():
        raise FileExistsError(f"Refusing to overwrite existing benchmark project: {project}")
    ignore = (
        shutil.ignore_patterns(*suite["copy_ignore"]) if suite["copy_ignore"] else None
    )
    shutil.copytree(suite["codebase"], project, ignore=ignore)
    if suite_name == "examplecode":
        # The shipped helper already returns [] for oversized windows. Give
        # both arms the same failing baseline so this task is not a no-op.
        target = project / "math_utils.py"
        text = target.read_text(encoding="utf-8")
        text = text.replace(
            '    """Calculate the moving average over a sliding window."""',
            '    """Calculate the moving average over a sliding window."""\n'
            '    if window > len(values):\n'
            '        raise ValueError("window exceeds number of values")',
        )
        target.write_text(text, encoding="utf-8")
    (sandbox / "AGENTS.md").write_text(agents_md_for_suite(suite_name), encoding="utf-8")
    return project


def write_mcp_config(sandbox: Path, buffers_dir: Path, python_exe: str) -> None:
    cfg_dir = sandbox / ".opencode"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {
            "gigacode": {
                "type": "local",
                "command": [
                    python_exe,
                    str(MCP_SERVER_SCRIPT),
                    str(buffers_dir),
                    "agent_core",
                ],
                "cwd": str(sandbox),
                "enabled": True,
                "timeout": 120000,
                "environment": {
                    "GIGACODE_DIRECT_TOOLS": "on" if SURFACE == "direct" else "off",
                    "GIGACODE_BENCH_EMBEDDER": EMBEDDER,
                },
            }
        },
    }
    (cfg_dir / "opencode.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def run_opencode(
    sandbox: Path, prompt: str, events_file: Path, stderr_file: Path, timeout: int
) -> tuple[int, float, str | None]:
    started = time.monotonic()
    cmd = [
        str(OPENCODE_EXE),
        "run",
        "--format",
        "json",
        "--auto",
        "-m",
        MODEL,
        prompt,
    ]
    session_id: str | None = None
    with open(events_file, "w", encoding="utf-8") as ef, open(
        stderr_file, "w", encoding="utf-8"
    ) as sf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(sandbox),
            stdout=ef,
            stderr=sf,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
            )
            proc.wait(timeout=30)
    duration = time.monotonic() - started
    try:
        for line in events_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"sessionID"' in line and session_id is None:
                try:
                    obj = json.loads(line)
                    session_id = obj.get("sessionID") or obj.get("part", {}).get(
                        "sessionID"
                    )
                except json.JSONDecodeError:
                    continue
            if session_id:
                break
    except OSError:
        pass
    return proc.returncode, duration, session_id


def find_session_by_marker(db_file: Path, marker: str) -> str | None:
    con = sqlite3.connect(db_file)
    try:
        row = con.execute(
            "SELECT session_id FROM part WHERE data LIKE ? ORDER BY time_created DESC LIMIT 1",
            (f"%{marker}%",),
        ).fetchone()
        if row:
            return row[0]
        row = con.execute(
            "SELECT session_id FROM message WHERE data LIKE ? ORDER BY time_created DESC LIMIT 1",
            (f"%{marker}%",),
        ).fetchone()
        return row[0] if row else None
    finally:
        con.close()


def session_fingerprint(db_file: Path, session_id: str) -> tuple:
    con = sqlite3.connect(db_file)
    try:
        msgs = list(
            con.execute(
                "SELECT data FROM message WHERE session_id=?", (session_id,)
            )
        )
        parts = list(
            con.execute(
                "SELECT data FROM part WHERE session_id=?", (session_id,)
            )
        )
    finally:
        con.close()
    n_assist = 0
    total_in = 0
    total_out = 0
    for (data,) in msgs:
        try:
            d = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue
        if d.get("role") == "assistant":
            n_assist += 1
            toks = d.get("tokens") or {}
            total_in += int(toks.get("input") or 0)
            total_out += int(toks.get("output") or 0)
    n_tool_parts = 0
    tool_outputs_len = 0
    for (pdata,) in parts:
        try:
            pd = json.loads(pdata)
        except (json.JSONDecodeError, TypeError):
            continue
        if pd.get("type") == "tool":
            n_tool_parts += 1
            out = (pd.get("state") or {}).get("output")
            if isinstance(out, str):
                tool_outputs_len += len(out)
    return (n_assist, total_in, total_out, n_tool_parts, tool_outputs_len)


def wait_for_flush(db_file: Path, session_id: str, max_wait: int) -> tuple:
    deadline = time.monotonic() + max_wait
    prev = session_fingerprint(db_file, session_id)
    while time.monotonic() < deadline:
        time.sleep(2.0)
        cur = session_fingerprint(db_file, session_id)
        if cur == prev:
            return cur
        prev = cur
    return prev


def measure_session(db_file: Path, session_id: str) -> dict[str, Any]:
    con = sqlite3.connect(db_file)
    con.row_factory = sqlite3.Row
    try:
        msgs = list(
            con.execute(
                "SELECT id, data FROM message WHERE session_id=? ORDER BY time_created",
                (session_id,),
            )
        )
        parts = list(
            con.execute(
                "SELECT message_id, data FROM part WHERE session_id=? "
                "ORDER BY time_created, id",
                (session_id,),
            )
        )
    finally:
        con.close()

    parts_by_msg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in parts:
        try:
            parts_by_msg[p["message_id"]].append(json.loads(p["data"]))
        except (json.JSONDecodeError, TypeError):
            continue

    steps = 0
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_write = 0
    total_cost = 0.0
    tool_calls = 0
    regular_tool_calls = 0
    mcp_tool_calls = 0
    mcp_breakdown: dict[str, int] = {}
    regular_breakdown: dict[str, int] = {}
    mcp_output_chars = 0
    regular_output_chars = 0

    for m in msgs:
        try:
            d = json.loads(m["data"])
        except (json.JSONDecodeError, TypeError):
            continue
        if d.get("role") != "assistant":
            continue
        steps += 1
        toks = d.get("tokens") or {}
        input_tokens += int(toks.get("input") or 0)
        output_tokens += int(toks.get("output") or 0)
        cache = toks.get("cache") or {}
        cache_read += int(cache.get("read") or 0)
        cache_write += int(cache.get("write") or 0)
        total_cost += float(d.get("cost") or 0.0)
        for pd in parts_by_msg.get(m["id"], []):
            if pd.get("type") != "tool":
                continue
            name = str(pd.get("tool") or "unknown")
            state = pd.get("state") or {}
            out = state.get("output") or ""
            out_len = len(out) if isinstance(out, str) else 0
            tool_calls += 1
            if name in REGULAR_TOOLS:
                regular_tool_calls += 1
                regular_breakdown[name] = regular_breakdown.get(name, 0) + 1
                regular_output_chars += out_len
            else:
                mcp_tool_calls += 1
                mcp_breakdown[name] = mcp_breakdown.get(name, 0) + 1
                mcp_output_chars += out_len

    return {
        "steps": steps,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "total_tokens_input_plus_output": input_tokens + output_tokens,
        "cost_usd": round(total_cost, 6),
        "tool_calls_total": tool_calls,
        "tool_calls_regular": regular_tool_calls,
        "tool_calls_mcp": mcp_tool_calls,
        "regular_tool_breakdown": dict(sorted(regular_breakdown.items())),
        "mcp_tool_breakdown": dict(sorted(mcp_breakdown.items())),
        "regular_tool_output_chars": regular_output_chars,
        "mcp_tool_output_chars": mcp_output_chars,
    }


def _window_after(text: str, needle: str, lines_ahead: int = 12) -> str:
    idx = text.find(needle)
    if idx < 0:
        return ""
    lines = text[idx:].splitlines()
    return "\n".join(lines[: lines_ahead + 1])


def verify_task(
    task: dict[str, Any], project_root: Path,
    before: dict[str, str] | None = None,
) -> dict[str, Any]:
    check = task["check"]
    target = project_root / check["file"]
    result: dict[str, Any] = {
        "expected_file": check["file"],
        "expected_symbol": check.get("symbol") or "(file-level check)",
        "file_exists": target.exists(),
        "edit_applied": False,
        "detail": "",
    }
    if not target.exists():
        result["detail"] = "target file missing"
        return result
    if before is None:
        result["edit_applied"] = False
        result["detail"] = "Missing baseline; correctness and scope cannot be verified."
        return result
    from scripts.benchmark_checks import check_behavior, check_scope, source_snapshot

    failure = check_scope(task, before, source_snapshot(project_root))
    if failure is None:
        failure = check_behavior(task, project_root)
    if failure:
        result["edit_applied"] = False
        result["detail"] = failure
    else:
        result["edit_applied"] = True
        result["detail"] = "Behavior, syntax, and edit scope verified."
    return result


EDIT_TOOL_SUFFIXES = ("edit_hashlines", "write_code", "commit", "code_edit")

# Deferred tool_call inputs count as editing only when the wrapped tool or
# chain actually performs an edit or commit.
EDIT_ARG_MARKERS = (
    '"chain": "anchor_apply"',
    '"chain": "post_edit"',
    '"edit_hashlines"',
    '"write_code"',
    '"commit"',
)


def _is_editing_tool_call(tool_name: str, tool_input_json: str) -> bool:
    """True when a recorded opencode tool call performs a code edit.

    Regular harness tools match by name; deferred MCP calls count when the
    wrapped tool or chain performs an edit/commit.
    """
    base = tool_name.rsplit("gigacode_", 1)[-1]
    if tool_name.startswith("gigacode_"):
        if base == "tool_call":
            try:
                payload = json.loads(tool_input_json)
            except (TypeError, json.JSONDecodeError):
                return False
            name = payload.get("name")
            args = payload.get("arguments") or {}
            if args.get("dry_run", name == "tool_chain"):
                return False
            return name in EDIT_TOOL_SUFFIXES or (name == "tool_chain" and args.get("chain") == "anchor_apply")
        return base in EDIT_TOOL_SUFFIXES
    return base in ("edit", "write", "patch", "multiedit")


def _read_events(events_file: Path) -> list[dict[str, Any]]:
    """Robustly parse an opencode raw event stream."""
    events: list[dict[str, Any]] = []
    try:
        for line in events_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return events


def _detect_gpu_mode(stderr_file: Path) -> str | None:
    """Read the bench server's GPU residency announcement from its stderr."""
    try:
        text = stderr_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"GIGACODE_BENCH .*?gpu_mode=([a-z\-]+)", text)
    return match.group(1) if match else None


def _successful_edit(tool_name: str, state: dict[str, Any]) -> bool:
    if state.get("status") != "completed":
        return False
    payload = state.get("input") or {}
    if not _is_editing_tool_call(tool_name, json.dumps(payload)):
        return False
    args = payload.get("arguments", payload)
    if args.get("dry_run"):
        return False
    if not tool_name.startswith("gigacode_"):
        return True
    try:
        out = state.get("output")
        output = json.loads(out) if isinstance(out, str) else out
    except (ValueError, TypeError):
        return False
    if not isinstance(output, dict) or output.get("status") != "ok" or output.get("dry_run"):
        return False
    if output.get("applied") is True or output.get("written_files"):
        return True
    for step in output.get("steps") or []:
        response = step.get("response") or {}
        if step.get("tool") == "commit" and response.get("written_files") and not response.get("dry_run"):
            return True
    return False


def measure_edit_speed(events_file: Path, run_start_ms: float | None = None) -> dict[str, Any]:
    """Derive edit timing from the opencode event stream.

    Metrics (seconds, on wall clock):
      time_to_edit    first event -> end of first successful editing call
      edit_span       span from first edit start to last edit end
      time_after_edit remaining time from the last edit to run end
    """
    events: list[dict[str, Any]] = _read_events(events_file)
    if not events:
        return {"time_to_edit_s": None}

    edit_calls: list[tuple[int, int]] = []  # (start_ms, end_ms) of edit calls
    for ev in events:
        part = ev.get("part") or {}
        if part.get("type") != "tool" or part.get("tool") is None:
            continue
        state = part.get("state") or {}
        if state.get("status") not in {"completed", "error"}:
            continue
        timing = (state.get("time") or {})
        start = timing.get("start")
        end = timing.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        try:
            tool_input = json.dumps(state.get("input") or {}, ensure_ascii=False)
        except (TypeError, ValueError):
            tool_input = ""
        if _successful_edit(str(part.get("tool", "")), state):
            edit_calls.append((int(start), int(end)))
    if not events or not edit_calls:
        return {"time_to_edit_s": None}

    t0 = (run_start_ms if run_start_ms is not None else int(events[0].get("timestamp", 0))) / 1000
    t_end = max(int(e.get("timestamp", 0)) for e in events) / 1000
    first_end = min(end for _, end in edit_calls) / 1000
    edit_span = max(end for _, end in edit_calls) / 1000 - min(
        start for start, _ in edit_calls
    ) / 1000
    last_edit_end = max(end for _, end in edit_calls) / 1000
    return {
        "time_to_edit_s": max(round(first_end - t0, 2), 0.0),
        "edit_span_s": round(edit_span, 2),
        "edit_time_s": round(sum(e - s for s, e in edit_calls) / 1000, 2),
        "time_after_edit_s": max(round(t_end - last_edit_end, 2), 0.0),
        "edit_calls": len(edit_calls),
        "observations": len(events),
    }


def measure_latency(events_file: Path, run_start_ms: float, duration: float) -> dict[str, Any]:
    """Separate startup, retrieval, edits and all-tool wait using interval unions."""
    events = _read_events(events_file)
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for ev in events:
        part = ev.get("part") or {}
        state = part.get("state") or {}
        timing = state.get("time") or {}
        start, end = timing.get("start"), timing.get("end")
        if part.get("type") != "tool" or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if end < start:
            continue
        buckets["all_tools"].append((start, end))
        tool = str(part.get("tool") or "")
        args = state.get("input") or {}
        name = args.get("name", tool.removeprefix("gigacode_"))
        chain = (args.get("arguments") or {}).get("chain")
        if name in {"code_find", "code_search", "read_code", "read_hashlines", "semantic_search", "embed_codebase"} or chain in {"anchor_read", "search_read", "stream_read"}:
            buckets["retrieval"].append((start, end))
        if _successful_edit(tool, state):
            buckets["edit"].append((start, end))
    def union_seconds(intervals):
        total = 0.0
        last_end = float("-inf")
        for start, end in sorted(intervals):
            total += max(0, end - max(start, last_end))
            last_end = max(last_end, end)
        return round(total / 1000, 3)
    totals = {f"{key}_s": union_seconds(buckets[key]) for key in ("all_tools", "retrieval", "edit")}
    first = min((e.get("timestamp", run_start_ms) for e in events), default=None)
    totals["startup_to_first_event_s"] = max(0, round((first - run_start_ms) / 1000, 3)) if first is not None else None
    totals["non_tool_wall_s"] = max(0, round(duration - totals["all_tools_s"], 3))
    totals["note"] = "Non-tool wall time includes startup, model generation, IPC and harness work; not a pure model timer."
    return totals


def measure_tool_ms(events_file: Path) -> dict[str, Any]:
    """Sum server-side tool time split out of the agent's wall clock (secs).

    For each completed MCP (gigacode_*) call, ``state.time`` spans request ->
    response; summing gives the total time the tool stack spent working.
    Everything else in the run (model steps, IPC, retries) is loop overhead.
    """
    events = _read_events(events_file)
    if not events:
        return {"tool_ms_s": None}
    tool_ms = 0.0
    calls = 0
    for ev in events:
        part = ev.get("part") or {}
        if part.get("type") != "tool":
            continue
        tool_name = str(part.get("tool") or "")
        if not tool_name.startswith("gigacode_"):
            continue
        state = part.get("state") or {}
        if state.get("status") not in {"completed", "error"}:
            continue
        timing = state.get("time") or {}
        start = timing.get("start")
        end = timing.get("end")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            tool_ms += (end - start) / 1000
            calls += 1
    if not calls:
        return {"tool_ms_s": 0.0, "tool_calls": 0}
    return {"tool_ms_s": round(tool_ms, 2), "tool_calls": calls}


def run_one(
    arm: str,
    suite_name: str,
    task: dict[str, Any],
    sandbox: Path,
    out_dir: Path,
    run_id: str,
    python_exe: str,
    timeout: int,
) -> dict[str, Any]:
    task_id = task["id"]
    runkey = f"{arm}__{task_id}__{run_id}"
    raw_dir = out_dir / "raw"
    runs_dir = out_dir / "runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    events_file = raw_dir / f"{runkey}.jsonl"
    stderr_file = raw_dir / f"{runkey}.stderr.log"

    project_root = build_sandbox(sandbox, suite_name)
    from scripts.benchmark_checks import source_snapshot

    before = source_snapshot(project_root)
    implementation = hashlib.sha256()
    for path in sorted((REPO_ROOT / "gigacode").glob("*.py")) + [
        Path(__file__), MCP_SERVER_SCRIPT, REPO_ROOT / "scripts" / "benchmark_checks.py",
    ]:
        implementation.update(path.name.encode("utf-8"))
        implementation.update(path.read_bytes())
    configuration = {
        "implementation_sha256": implementation.hexdigest(),
        "baseline_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
        "model": MODEL, "surface": SURFACE, "embedder": EMBEDDER,
        "embedding_device": os.environ.get("GIGACODE_BENCH_DEVICE", "cpu"),
        "embedding_model": os.environ.get("GIGACODE_BENCH_EMBED_MODEL"),
        "gpu_requested": os.environ.get("GIGACODE_BENCH_GPU", "auto"),
        "guidance_sha256": hashlib.sha256(agents_md_for_suite(suite_name).encode()).hexdigest(),
    }
    config_id = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    if arm == "gigacode":
        buffers_dir = sandbox / f"buffers_{task_id}"
        if buffers_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing buffers: {buffers_dir}")
        write_mcp_config(sandbox, buffers_dir, python_exe)
    else:
        cfg = sandbox / ".opencode" / "opencode.json"
        if cfg.exists():
            cfg.unlink()

    marker = f"gigabench-{run_id}-{task_id}-{arm}"
    prompt = f"[{marker}]\n\n" + TASK_PROMPT_TEMPLATE.format(
        project_root=str(project_root),
        layout=TASK_SUITES[suite_name]["layout"],
    ) + task["prompt"]

    print(f"[{runkey}] starting opencode ({MODEL}) ...", flush=True)
    run_start_ms = time.time() * 1000
    started_at = now_iso()
    exit_code, duration, sid_from_events = run_opencode(
        sandbox, prompt, events_file, stderr_file, timeout
    )
    print(f"[{runkey}] exit={exit_code} duration={duration:.1f}s", flush=True)

    db_file = db_path()
    session_id = find_session_by_marker(db_file, marker) or sid_from_events
    measurement: dict[str, Any] = {}
    if session_id:
        wait_for_flush(db_file, session_id, DB_FLUSH_WAIT_SEC)
        measurement = measure_session(db_file, session_id)
    else:
        print(f"[{runkey}] WARNING: no session found in DB", flush=True)

    verification = verify_task(task, project_root, before)
    stderr_tail = ""
    try:
        stderr_tail = stderr_file.read_text(encoding="utf-8", errors="replace")[-4000:]
    except OSError:
        pass

    record = {
        "runkey": runkey,
        "run_id": run_id,
        "suite": suite_name,
        "arm": arm,
        "task_id": task_id,
        "model": MODEL,
        "started_at": started_at,
        "benchmark_version": BENCHMARK_VERSION,
        "surface": SURFACE,
        "embedder": EMBEDDER,
        "config_id": config_id,
        "configuration": configuration,
        "duration_sec": round(duration, 2),
        "exit_code": exit_code,
        "session_id": session_id,
        "sandbox": f"{sandbox.parent.name}/{sandbox.name}",
        "edit_speed": measure_edit_speed(events_file, run_start_ms),
        "latency": measure_latency(events_file, run_start_ms, duration),
        "tool_ms": measure_tool_ms(events_file),
        "gpu_mode": _detect_gpu_mode(stderr_file),
        "verification": verification,
        "measurement": measurement,
        "stderr_tail": stderr_tail[-1500:],
    }
    (runs_dir / f"{runkey}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def load_measurements(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "measurements.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"runs": {}}
    return {"runs": {}}


def save_run(out_dir: Path, record: dict[str, Any]) -> None:
    suite = TASK_SUITES[record["suite"]]
    codebase = Path(suite["codebase"])
    n_files = len(list(codebase.rglob("*.py")))
    data = load_measurements(out_dir)
    data["meta"] = {
        "model": MODEL,
        "opencode_exe": str(OPENCODE_EXE),
        "updated_at": now_iso(),
        "suite": record["suite"],
        "codebase": f"{codebase.name} ({n_files} py files)",
        "task_definitions": {t["id"]: t["prompt"] for t in suite["tasks"]},
        "benchmark_version": BENCHMARK_VERSION,
        "surface": SURFACE,
        "embedder": EMBEDDER,
        "config_id": record.get("config_id"),
        "configuration": record.get("configuration"),
    }
    data.setdefault("runs", {})[record["runkey"]] = record
    (out_dir / "measurements.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def pct_change(base: float, new: float) -> float | None:
    if not base:
        return None
    return round((new - base) / base * 100.0, 1)


def fmt(n: Any) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def latest_runs(data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in data.get("runs", {}).values():
        key = (rec["arm"], rec["task_id"])
        if key not in best or rec.get("started_at", "") > best[key].get("started_at", ""):
            best[key] = rec
    return best


def repeated_runs(data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Aggregate comparable repetitions, never mix suites or benchmark versions."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    meta = data.get("meta") or {}
    for rec in data.get("runs", {}).values():
        if any(rec.get(key) != meta.get(key) for key in (
            "suite", "model", "benchmark_version", "surface", "embedder", "config_id",
        )):
            continue
        groups[(rec["arm"], rec["task_id"])].append(rec)
    best = {}
    for key, records in groups.items():
        aggregate = dict(records[-1])
        for field in ("measurement", "edit_speed", "tool_ms", "latency"):
            fields = {k for r in records for k in (r.get(field) or {})}
            values = {}
            for metric in fields:
                samples = [
                    (r.get(field) or {}).get(metric) for r in records
                    if isinstance((r.get(field) or {}).get(metric), (int, float))
                ]
                if samples:
                    median = statistics.median(samples)
                    values[metric] = median
            aggregate[field] = values
        aggregate["duration_sec"] = statistics.median(r["duration_sec"] for r in records)
        successes = sum(bool((r.get("verification") or {}).get("edit_applied")) for r in records)
        aggregate["verification"] = {"edit_applied": successes == len(records)}
        aggregate["sample_count"] = len(records)
        aggregate["success_count"] = successes
        aggregate["samples"] = [
            {
                "runkey": r["runkey"], "run_id": r.get("run_id"),
                "input_tokens": (r.get("measurement") or {}).get("input_tokens"),
                "duration_sec": r.get("duration_sec"), "edit_speed": r.get("edit_speed"),
                "verified": bool((r.get("verification") or {}).get("edit_applied")),
                "latency": r.get("latency"),
                "measurement": r.get("measurement"),
                "tool_ms": r.get("tool_ms"),
                "verification": r.get("verification"),
            }
            for r in records
        ]
        # Tool mixes/output volume remain reproducible across all repetitions.
        aggregate["measurement"]["regular_tool_breakdown"] = {}
        aggregate["measurement"]["mcp_tool_breakdown"] = {}
        for metric in ("regular_tool_breakdown", "mcp_tool_breakdown"):
            for r in records:
                for tool, count in (r.get("measurement", {}).get(metric) or {}).items():
                    aggregate["measurement"][metric][tool] = aggregate["measurement"][metric].get(tool, 0) + count
        best[key] = aggregate
    return best


def summarize_arm(best: dict[tuple[str, str], dict[str, Any]], arm: str) -> dict[str, Any]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "steps": 0,
        "tool_calls_regular": 0,
        "tool_calls_mcp": 0,
        "mcp_tool_output_chars": 0,
        "regular_tool_output_chars": 0,
        "cost_usd": 0.0,
    }
    per_task: dict[str, dict[str, Any]] = {}
    time_totals: dict[str, Any] = {"time_to_edit_s": [], "edit_span_s": [], "tool_ms_s": [], "edit_time_s": []}
    successes = 0
    verified_tasks = 0
    sample_count = 0
    count = 0
    for (a, task_id), rec in best.items():
        if a != arm:
            continue
        m = rec.get("measurement", {})
        count += 1
        sample_count += rec.get("sample_count", 1)
        successes += rec.get("success_count", int(bool(rec.get("verification", {}).get("edit_applied"))))
        verified_tasks += int(bool(rec.get("verification", {}).get("edit_applied")))
        for k in ("input_tokens", "output_tokens", "cache_read_tokens", "steps",
                  "tool_calls_regular", "tool_calls_mcp",
                  "mcp_tool_output_chars", "regular_tool_output_chars"):
            totals[k] += m.get(k) or 0
        totals["cost_usd"] = round(totals["cost_usd"] + float(m.get("cost_usd") or 0.0), 6)
        per_task[task_id] = {
            "input_tokens": m.get("input_tokens") or 0,
            "output_tokens": m.get("output_tokens") or 0,
            "cache_read_tokens": m.get("cache_read_tokens") or 0,
            "processed_input_tokens": (m.get("input_tokens") or 0)
            + (m.get("cache_read_tokens") or 0),
            "steps": m.get("steps") or 0,
            "tool_calls_regular": m.get("tool_calls_regular") or 0,
            "tool_calls_mcp": m.get("tool_calls_mcp") or 0,
            "regular_tool_breakdown": m.get("regular_tool_breakdown", {}),
            "mcp_tool_breakdown": m.get("mcp_tool_breakdown", {}),
            "regular_tool_output_chars": m.get("regular_tool_output_chars") or 0,
            "mcp_tool_output_chars": m.get("mcp_tool_output_chars") or 0,
            "edit_speed": rec.get("edit_speed") or {},
            "duration_sec": rec.get("duration_sec"),
            "exit_code": rec.get("exit_code"),
            "verification": rec.get("verification"),
            "session_id": rec.get("session_id"),
            "sample_count": rec.get("sample_count", 1),
            "success_count": rec.get("success_count", int(bool(rec.get("verification", {}).get("edit_applied")))),
            "samples": rec.get("samples", []),
            "latency": rec.get("latency"),
        }
        speed = rec.get("edit_speed") or {}
        t_edit = speed.get("time_to_edit_s")
        span = speed.get("edit_span_s")
        edit_time = speed.get("edit_time_s")
        if isinstance(t_edit, (int, float)):
            time_totals["time_to_edit_s"].append(float(t_edit))
        if isinstance(span, (int, float)):
            time_totals["edit_span_s"].append(float(span))
        if isinstance(edit_time, (int, float)):
            time_totals["edit_time_s"].append(float(edit_time))
        tool_ms = (rec.get("tool_ms") or {}).get("tool_ms_s")
        if isinstance(tool_ms, (int, float)):
            time_totals["tool_ms_s"].append(float(tool_ms))
        gpu_mode = rec.get("gpu_mode")
        if gpu_mode:
            time_totals.setdefault("gpu_modes", set()).add(str(gpu_mode))
    totals["tasks_run"] = count
    totals["tasks_edit_applied"] = verified_tasks
    totals["runs_verified"] = successes
    totals["samples_run"] = sample_count
    totals["processed_input_tokens"] = totals["input_tokens"] + totals["cache_read_tokens"]
    # Timing aggregates (seconds): median across tasks.
    for key in ("time_to_edit_s", "edit_span_s", "tool_ms_s", "edit_time_s"):
        values = time_totals[key]
        totals[key] = round(statistics.median(values), 2) if values else None
    totals["gpu_modes"] = sorted(time_totals.get("gpu_modes", set()))
    return {"totals": totals, "per_task": per_task}


def paired_deltas(plain: list[dict], gigacode: list[dict]) -> dict:
    """Preserve paired variation instead of comparing only two aggregate values."""
    base = {s["run_id"]: s for s in plain if s.get("run_id")}
    deltas = []
    for sample in gigacode:
        p = base.get(sample.get("run_id"))
        if p is None:
            continue
        row = {"run_id": sample["run_id"], "both_verified": p["verified"] and sample["verified"]}
        for key in ("input_tokens", "duration_sec"):
            if isinstance(p.get(key), (int, float)) and isinstance(sample.get(key), (int, float)):
                row[key] = sample[key] - p[key]
        deltas.append(row)
    summary = {}
    for key in ("input_tokens", "duration_sec"):
        values = sorted(d[key] for d in deltas if key in d)
        if values:
            summary[key] = {
                "pairs": len(values), "median_delta": statistics.median(values),
                "min_delta": min(values), "max_delta": max(values),
            }
    return {"samples": deltas, "summary": summary}


def write_comparison(out_dir: Path) -> None:
    data = load_measurements(out_dir)
    best = repeated_runs(data)
    arms = {a: summarize_arm(best, a) for a in ("plain", "gigacode")}
    task_ids = sorted({tid for (_, tid) in best.keys()})

    per_task_cmp: dict[str, Any] = {}
    for tid in task_ids:
        p = arms["plain"]["per_task"].get(tid)
        g = arms["gigacode"]["per_task"].get(tid)
        if not p or not g:
            continue
        per_task_cmp[tid] = {
            "plain": p,
            "gigacode": g,
            "delta": {
                "input_tokens": g["input_tokens"] - p["input_tokens"],
                "input_tokens_pct": pct_change(p["input_tokens"], g["input_tokens"]),
                "output_tokens": g["output_tokens"] - p["output_tokens"],
                "output_tokens_pct": pct_change(p["output_tokens"], g["output_tokens"]),
                "cache_read_tokens_pct": pct_change(
                    p["cache_read_tokens"], g["cache_read_tokens"]
                ),
            },
            "paired_deltas": paired_deltas(p.get("samples", []), g.get("samples", [])),
        }

    pt = arms["plain"]["totals"]
    gt = arms["gigacode"]["totals"]
    comparison = {
        "generated_at": now_iso(),
        "model": data.get("meta", {}).get("model", MODEL),
        "configuration": data.get("meta", {}),
        "tasks": task_ids,
        "arms": arms,
        "per_task_comparison": per_task_cmp,
        "totals_comparison": {
            "input_tokens": {
                "plain": pt["input_tokens"],
                "gigacode": gt["input_tokens"],
                "delta_pct": pct_change(pt["input_tokens"], gt["input_tokens"]),
            },
            "output_tokens": {
                "plain": pt["output_tokens"],
                "gigacode": gt["output_tokens"],
                "delta_pct": pct_change(pt["output_tokens"], gt["output_tokens"]),
            },
            "cache_read_tokens": {
                "plain": pt["cache_read_tokens"],
                "gigacode": gt["cache_read_tokens"],
                "delta_pct": pct_change(pt["cache_read_tokens"], gt["cache_read_tokens"]),
            },
            "processed_input_tokens": {
                "plain": pt["processed_input_tokens"],
                "gigacode": gt["processed_input_tokens"],
                "delta_pct": pct_change(pt["processed_input_tokens"], gt["processed_input_tokens"]),
            },
            "tool_calls_regular": {
                "plain": pt["tool_calls_regular"],
                "gigacode": gt["tool_calls_regular"],
            },
            "tool_calls_mcp": {
                "plain": pt["tool_calls_mcp"],
                "gigacode": gt["tool_calls_mcp"],
            },
            "edit_success": {
                "plain": f'{pt["runs_verified"]}/{pt["samples_run"]}',
                "gigacode": f'{gt["runs_verified"]}/{gt["samples_run"]}',
            },
        },
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    lines: list[str] = []
    suite_desc = data.get("meta", {}).get("codebase", "unknown codebase")
    suite_name = data.get("meta", {}).get("suite", "?")
    lines.append("=" * 78)
    lines.append("GigaCode MCP Token-Usage Benchmark - comparison log")
    lines.append("=" * 78)
    lines.append(f"generated    : {comparison['generated_at']}")
    lines.append("agent        : opencode (headless `opencode run --format json --auto`)")
    lines.append(f"model        : {comparison['model']}")
    lines.append(f"suite        : {suite_name} (codebase: {suite_desc})")
    lines.append("workload     : search+edit tasks (see measurements.json.task_definitions)")
    lines.append("arm 'plain'  : opencode built-ins only (read/grep/glob/bash/edit/write)")
    lines.append(
        f"arm 'gigacode': opencode + gigacode MCP stdio server, {SURFACE} surface"
    )
    lines.append("                direct coding tools plus discovery, or legacy deferred-only")
    lines.append(
        "token source : opencode storage DB (per-message usage incl. all tool payloads)"
    )
    lines.append("")
    lines.append("-" * 78)
    lines.append("PER-TASK RESULTS  (medians across repetitions; success = all checks passed)")
    lines.append("t_edit = process start -> first confirmed disk edit; ed_s = edit-call time;")
    lines.append("after = last edit -> run end")
    lines.append("-" * 78)
    header = (
        f"{'task':<24}{'arm':<10}{'input':>9}{'output':>8}{'cacheR':>10}"
        f"{'steps':>7}{'t_edit':>7}{'ed_s':>7}{'after':>7}{'reg':>5}{'mcp':>5}  success"
    )
    lines.append(header)
    for tid in task_ids:
        for arm in ("plain", "gigacode"):
            r = arms[arm]["per_task"].get(tid)
            if not r:
                continue
            v = r.get("verification") or {}
            succ = f"{r['success_count']}/{r['sample_count']}"
            speed = r.get("edit_speed") or {}
            t_edit = speed.get("time_to_edit_s")
            after = speed.get("time_after_edit_s")
            edit_time = speed.get("edit_time_s")
            t_edit_s = f"{t_edit:>6.1f}" if isinstance(t_edit, (int, float)) else f"{'-':>6}"
            edit_s = f"{edit_time:>6.1f}" if isinstance(edit_time, (int, float)) else f"{'-':>6}"
            after_s = f"{after:>6.1f}" if isinstance(after, (int, float)) else f"{'-':>6}"
            lines.append(
                f"{tid:<24}{arm:<10}{r['input_tokens']:>9,}{r['output_tokens']:>8,}"
                f"{r['cache_read_tokens']:>10,}{r['steps']:>7}{t_edit_s}{edit_s}{after_s}"
                f"{r['tool_calls_regular']:>5}{r['tool_calls_mcp']:>5}  {succ}"
            )
        if tid in per_task_cmp:
            d = per_task_cmp[tid]["delta"]
            lines.append(
                f"{'  -> delta input':<34}"
                f"{d['input_tokens']:>+9,} ({d['input_tokens_pct']:>6}%)"
            )
        lines.append("")

    lines.append("-" * 78)
    lines.append("TOTALS (sum across tasks)")
    lines.append("-" * 78)
    for label, key in (
        ("input tokens (uncached)", "input_tokens"),
        ("output tokens", "output_tokens"),
        ("cache-read tokens", "cache_read_tokens"),
        ("processed input", "processed_input_tokens"),
        ("assistant turns", "steps"),
        ("regular tool calls", "tool_calls_regular"),
        ("mcp tool calls", "tool_calls_mcp"),
    ):
        pl = pt[key]
        gl = gt[key]
        extra = ""
        if key in ("input_tokens", "output_tokens", "cache_read_tokens"):
            dp = pct_change(pl, gl)
            extra = f"   delta: {gl - pl:+,} ({dp}% vs plain)" if dp is not None else ""
        lines.append(f"{label:<22} plain={pl:<12,} gigacode={gl:<12,}{extra}")
    lines.append(
        f"{'verified runs':<22} plain={pt['runs_verified']}/{pt['samples_run']}"
        f"    gigacode={gt['runs_verified']}/{gt['samples_run']}"
    )

    def _fmt_speed(value: Any) -> str:
        return f"{value:.1f}s" if isinstance(value, (int, float)) else "n/a"
    lines.append(
        f"{'time to first edit':<22} plain={_fmt_speed(pt.get('time_to_edit_s')):<12}"
        f" gigacode={_fmt_speed(gt.get('time_to_edit_s')):<12} (median)"
    )
    lines.append(
        f"{'total edit-call time':<22} plain={_fmt_speed(pt.get('edit_time_s')):<12}"
        f" gigacode={_fmt_speed(gt.get('edit_time_s')):<12} (median; sum of edit calls)"
    )
    lines.append(
        f"{'edit window span':<22} plain={_fmt_speed(pt.get('edit_span_s')):<12}"
        f" gigacode={_fmt_speed(gt.get('edit_span_s')):<12} (median)"
    )
    lines.append(
        f"{'MCP-call time':<22} plain={_fmt_speed(pt.get('tool_ms_s')):<12}"
        f" gigacode={_fmt_speed(gt.get('tool_ms_s')):<12} (median; includes transport/queue)"
    )
    for label, metric in (
        ("startup -> first event", "startup_to_first_event_s"),
        ("retrieval-call time", "retrieval_s"),
        ("non-tool wall time", "non_tool_wall_s"),
    ):
        medians = {}
        for arm in ("plain", "gigacode"):
            values = [
                r.get("latency", {}).get(metric)
                for r in arms[arm]["per_task"].values() if r.get("latency")
            ]
            values = [v for v in values if isinstance(v, (int, float))]
            medians[arm] = statistics.median(values) if values else None
        lines.append(f"{label:<22} plain={_fmt_speed(medians['plain']):<12} gigacode={_fmt_speed(medians['gigacode']):<12} (median)")
    lines.append(f"{'provider cost medians':<22} plain=${pt['cost_usd']:.6f} gigacode=${gt['cost_usd']:.6f} (sum of per-task medians)")
    lines.append(
        f"{'index backend hint':<22} plain={','.join(pt.get('gpu_modes') or ['-']) or '-':<12}"
        f" gigacode={','.join(gt.get('gpu_modes') or ['-']) or '-'}"
    )
    lines.append("")
    lines.append("-" * 78)
    lines.append("TOOL-CALL MIX (per arm, all tasks)")
    lines.append("-" * 78)
    for arm in ("plain", "gigacode"):
        lines.append(f"[{arm}]")
        reg: dict[str, int] = defaultdict(int)
        mcp: dict[str, int] = defaultdict(int)
        for r in arms[arm]["per_task"].values():
            for k, v in (r.get("regular_tool_breakdown") or {}).items():
                reg[k] += v
            for k, v in (r.get("mcp_tool_breakdown") or {}).items():
                mcp[k] += v
        for k in sorted(reg):
            lines.append(f"  regular {k:<16} {reg[k]}")
        for k in sorted(mcp):
            lines.append(f"  mcp     {k:<16} {mcp[k]}")
        if not reg and not mcp:
            lines.append("  (no tool calls recorded)")
    lines.append("")
    lines.append("-" * 78)
    lines.append("TOOL OUTPUT VOLUME (chars of tool results returned to the model)")
    lines.append("-" * 78)
    lines.append(
        f"{'regular tool outputs':<24} plain={pt['regular_tool_output_chars']:,}"
        f"  gigacode={gt['regular_tool_output_chars']:,}"
    )
    lines.append(
        f"{'mcp tool outputs':<24} plain={pt['mcp_tool_output_chars']:,}"
        f"  gigacode={gt['mcp_tool_output_chars']:,}"
    )
    lines.append("")
    lines.append("NOTES")
    lines.append("- Tokens/timing are per-task medians across comparable repetitions; totals")
    lines.append("  sum those medians. Raw sample values and success counts are in comparison.json.")
    lines.append("- Single-run results and small sample counts are not evidence of a stable speedup.")
    lines.append("- Non-tool wall time includes startup/model/harness/IPC work, not just model inference.")
    lines.append("- The project folder is re-copied fresh before every run, so tasks are")
    lines.append("  independent; each run's verification only reflects its own edits.")
    lines.append("- Input tokens include the fixed system prompt, AGENTS.md, tool schemas")
    lines.append("  (surface is recorded per run) and every tool result.")
    lines.append("- Processed input = uncached input + cache-read tokens, NOT dollar billing.")
    lines.append("  Report provider cost_usd separately; zero/unknown cost is not a free-run claim.")
    lines.append("- The gigacode arm uses the repo's stock MCP server code path")
    lines.append(f"  (scripts/mcp_bench_server.py) with embedder={EMBEDDER}. Hashing is lexical,")
    lines.append("  not a neural semantic model; GPU impact requires a separate real-model run.")
    (out_dir / "comparison_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global SURFACE, EMBEDDER
    parser = argparse.ArgumentParser(description="GigaCode token-usage benchmark")
    parser.add_argument(
        "--suite",
        choices=sorted(TASK_SUITES),
        default="examplecode",
        help="Task suite / codebase to benchmark (default: examplecode)",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["plain", "gigacode"],
        default=["plain", "gigacode"],
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None, help="Task ids (default: all in suite)"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "bench_results")
    parser.add_argument("--timeout", type=int, default=RUN_TIMEOUT_SEC)
    parser.add_argument("--repeats", type=int, default=3, help="Paired repetitions per task (default: 3)")
    parser.add_argument("--seed", type=int, default=55, help="Reproducible task/arm order")
    parser.add_argument("--surface", choices=["direct", "deferred"], default="direct")
    parser.add_argument("--embedder", choices=["hashing", "model"], default="hashing")
    parser.add_argument(
        "--sandbox-root",
        type=Path,
        default=None,
        help="Defaults to <temp>/gigacode_token_bench_<suite>",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    SURFACE, EMBEDDER = args.surface, args.embedder

    if args.sandbox_root is None:
        args.sandbox_root = (
            Path(tempfile.gettempdir()) / f"gigacode_token_bench_{args.suite}"
        )
    suite = TASK_SUITES[args.suite]
    known_ids = {t["id"] for t in suite["tasks"]}
    unknown = set(args.tasks or []) - known_ids
    if unknown:
        parser.error(f"unknown task ids for suite {args.suite!r}: {sorted(unknown)}")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in suite["tasks"] if args.tasks is None or t["id"] in args.tasks]
    python_exe = sys.executable

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(
        f"Benchmark run {run_id}: suite={args.suite} arms={args.arms} "
        f"tasks={[t['id'] for t in tasks]}",
        flush=True,
    )
    print(f"opencode: {OPENCODE_EXE} (exists={OPENCODE_EXE.exists()})", flush=True)
    print(f"model   : {MODEL}", flush=True)

    rng = random.Random(args.seed)
    for repetition in range(args.repeats):
        ordered_tasks = list(tasks)
        rng.shuffle(ordered_tasks)
        for task in ordered_tasks:
            ordered_arms = list(args.arms)
            if repetition % 2:
                ordered_arms.reverse()
            for arm in ordered_arms:
                sandbox = args.sandbox_root / run_id / f"r{repetition + 1}" / task["id"] / arm
                sandbox.mkdir(parents=True, exist_ok=False)
                record = run_one(
                    arm, args.suite, task, sandbox, out_dir,
                    f"{run_id}-r{repetition + 1}", python_exe, args.timeout,
                )
                record["repetition"] = repetition + 1
                record["seed"] = args.seed
                save_run(out_dir, record)
                v = record["verification"]
                m = record["measurement"]
                print(
                    f"[{record['runkey']}] in={m.get('input_tokens', 0):,} "
                    f"out={m.get('output_tokens', 0):,} steps={m.get('steps', 0)} "
                    f"verified={v.get('edit_applied')}", flush=True,
                )

    write_comparison(out_dir)
    print(f"\nWrote {out_dir / 'comparison_log.txt'}", flush=True)
    summary = (out_dir / "comparison_log.txt").read_text(encoding="utf-8")
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
