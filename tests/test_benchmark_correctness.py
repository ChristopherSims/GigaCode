"""Benchmark correctness and repeat aggregation without paid model calls."""

import json

import pytest

from scripts.benchmark_checks import source_snapshot
from scripts.benchmark_token_usage import (
    TASKS_EXAMPLECODE,
    TASKS_RICH,
    build_sandbox,
    measure_edit_speed,
    measure_latency,
    repeated_runs,
    verify_task,
    write_comparison,
)

FIXES = {
    "add_input_validation": (
        "def quick_sort(arr):",
        "def quick_sort(arr):\n    if not isinstance(arr, list):\n        raise TypeError('list required')",
    ),
    "add_docstring": (
        '    """Check if n is a prime number."""',
        '    """Check if n is a prime number.\n\n    >>> is_prime(7)\n    True\n    """',
    ),
    "rename_in_file": ("parse_key_value_pairs", "parse_kv_config"),
    "write_error_handling": ('raise ValueError("window exceeds number of values")', "return []"),
    "add_function": (
        "\n\ndef truncate",
        "\n\ndef camel_to_kebab(name):\n"
        '    return camel_to_snake(name).replace("_", "-")\n\n\ndef truncate',
    ),
    "add_method_validation": (
        "        if self.plain.endswith(suffix):",
        "        if suffix == '':\n            raise ValueError('empty suffix')\n"
        "        if self.plain.endswith(suffix):",
    ),
    "assert_to_return": (
        '    assert values, "1 or more values required"',
        "    if not values:\n        return False",
    ),
    "rename_filesize_helper": ("_to_str", "format_size"),
    "add_doctest_example": (
        '    """Parse six hex characters in to RGB triplet."""',
        '    """Parse six hex characters in to RGB triplet.\n\n'
        "    >>> parse_rgb_hex('ff0000')\n"
        '    ColorTriplet(red=255, green=0, blue=0)\n    """',
    ),
    "needle_validation": (
        '    """Blend one RGB color in to another."""',
        '    """Blend one RGB color in to another."""\n'
        "    if not 0.0 <= cross_fade <= 1.0:\n        raise ValueError('invalid weight')",
    ),
}


@pytest.mark.parametrize(
    "suite,task",
    [
        *[("examplecode", t) for t in TASKS_EXAMPLECODE],
        *[("rich", t) for t in TASKS_RICH],
    ],
    ids=[t["id"] for t in TASKS_EXAMPLECODE + TASKS_RICH],
)
def test_valid_fixes_pass_all_checks(tmp_path, suite, task):
    root = build_sandbox(tmp_path, suite)
    before = source_snapshot(root)
    target = root / task["check"]["file"]
    old, new = FIXES[task["id"]]
    text = target.read_text(encoding="utf-8")
    assert old in text
    target.write_text(text.replace(old, new), encoding="utf-8")
    verification = verify_task(task, root, before)
    assert verification["edit_applied"], verification["detail"]


def test_equivalent_boolean_fallback_passes_without_literal_return(tmp_path):
    task = next(t for t in TASKS_RICH if t["id"] == "assert_to_return")
    root = build_sandbox(tmp_path, "rich")
    before = source_snapshot(root)
    path = root / task["check"]["file"]
    text = path.read_text(encoding="utf-8").replace(
        '    assert values, "1 or more values required"',
        "    if not values:\n        result = False\n        return result",
    )
    path.write_text(text, encoding="utf-8")
    result = verify_task(task, root, before)
    assert result["edit_applied"], result


def test_oversized_window_task_has_failing_baseline(tmp_path):
    root = build_sandbox(tmp_path, "examplecode")
    assert "window exceeds number of values" in (root / "math_utils.py").read_text()
    before = source_snapshot(root)
    task = next(t for t in TASKS_EXAMPLECODE if t["id"] == "write_error_handling")
    assert not verify_task(task, root, before)["edit_applied"]
    path = root / "math_utils.py"
    path.write_text(
        path.read_text().replace(
            'raise ValueError("window exceeds number of values")',
            "return []",
        )
    )
    assert verify_task(task, root, before)["edit_applied"]
    (root / "new.py").write_text("pass\n")
    assert not verify_task(task, root, before)["edit_applied"]


def test_regex_success_does_not_accept_wrong_behavior(tmp_path):
    root = build_sandbox(tmp_path, "examplecode")
    before = source_snapshot(root)
    task = next(t for t in TASKS_EXAMPLECODE if t["id"] == "add_input_validation")
    path = root / "sorting_algorithms.py"
    path.write_text(
        path.read_text().replace("def quick_sort(arr):", "def quick_sort(arr):\n    # TypeError")
    )
    # Confirm the expected identifier exists but behavior is still wrong.
    assert "TypeError" in path.read_text()
    assert not verify_task(task, root, before)["edit_applied"]


def test_missing_baseline_is_not_verified(tmp_path):
    root = build_sandbox(tmp_path, "examplecode")
    assert not verify_task(TASKS_EXAMPLECODE[0], root)["edit_applied"]


def test_existing_sandbox_is_not_overwritten(tmp_path):
    root = build_sandbox(tmp_path, "examplecode")
    protected = root / "notes.txt"
    protected.write_text("keep this")
    with pytest.raises(FileExistsError):
        build_sandbox(tmp_path, "examplecode")
    assert protected.read_text() == "keep this"


def test_non_python_edit_is_detected(tmp_path):
    root = build_sandbox(tmp_path, "examplecode")
    before = source_snapshot(root)
    task = next(t for t in TASKS_EXAMPLECODE if t["id"] == "write_error_handling")
    target = root / "math_utils.py"
    target.write_text(
        target.read_text().replace(
            'raise ValueError("window exceeds number of values")',
            "return []",
        )
    )
    (root / "notes.txt").write_text("unexpected change")
    assert not verify_task(task, root, before)["edit_applied"]


def _mcp_event(output, args=None, tool="gigacode_tool_call", start=1000, end=2000):
    return {
        "timestamp": end,
        "part": {
            "type": "tool",
            "tool": tool,
            "state": {
                "status": "completed",
                "time": {"start": start, "end": end},
                "input": args
                or {"name": "tool_chain", "arguments": {"chain": "anchor_apply", "dry_run": False}},
                "output": json.dumps(output),
            },
        },
    }


@pytest.mark.parametrize(
    "output,args",
    [
        ({"status": "error"}, None),
        ({"status": "conflict"}, None),
        ({"status": "ok", "applied": False}, None),
        (
            {"status": "ok", "applied": True},
            {"name": "tool_chain", "arguments": {"chain": "anchor_apply", "dry_run": True}},
        ),
        (
            {"status": "ok", "written_files": ["a.py"]},
            {"name": "tool_chain", "arguments": {"chain": "post_edit", "dry_run": False}},
        ),
    ],
)
def test_no_false_edit_timing(tmp_path, output, args):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(_mcp_event(output, args)))
    assert measure_edit_speed(path)["time_to_edit_s"] is None


def test_direct_timing_includes_process_start(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(
            _mcp_event(
                {"status": "ok", "applied": True},
                {"file": "a.py"},
                tool="gigacode_code_edit",
            )
        )
    )
    assert measure_edit_speed(path, run_start_ms=0)["time_to_edit_s"] == 2


def test_latency_does_not_double_count_parallel_calls(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(ev)
            for ev in [
                _mcp_event({"status": "ok"}, {"query": "x"}, "gigacode_code_find", 1000, 3000),
                _mcp_event({"status": "ok"}, {"query": "y"}, "gigacode_code_find", 2000, 4000),
            ]
        )
    )
    latency = measure_latency(path, 0, 5)
    assert latency["all_tools_s"] == 3
    assert latency["retrieval_s"] == 3
    assert latency["non_tool_wall_s"] == 2


def test_repeats_keep_all_samples_and_filter_configuration(tmp_path):
    meta = {
        "suite": "rich",
        "model": "test",
        "benchmark_version": 2,
        "surface": "direct",
        "embedder": "hashing",
    }
    runs = {}
    for i, tokens in enumerate([10, 30, 20]):
        runs[str(i)] = {
            **meta,
            "runkey": str(i),
            "arm": "gigacode",
            "task_id": "task",
            "measurement": {"input_tokens": tokens},
            "duration_sec": tokens,
            "verification": {"edit_applied": i != 1},
        }
    runs["wrong-suite"] = {**runs["0"], "suite": "examplecode"}
    runs["old-version"] = {**runs["0"], "benchmark_version": 1}
    data = {"meta": meta, "runs": runs}
    aggregate = repeated_runs(data)[("gigacode", "task")]
    assert aggregate["measurement"]["input_tokens"] == 20
    assert aggregate["sample_count"] == 3
    assert aggregate["success_count"] == 2
    assert len(aggregate["samples"]) == 3
    (tmp_path / "measurements.json").write_text(json.dumps(data))
    write_comparison(tmp_path)
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert comparison["totals_comparison"]["edit_success"]["gigacode"] == "2/3"
    assert "processed_input_tokens" in comparison["totals_comparison"]
    assert "billed_input_tokens" not in comparison["totals_comparison"]


def test_paired_repeat_cli_without_model_calls(tmp_path, monkeypatch):
    import scripts.benchmark_token_usage as bench

    calls = []

    def fake_run(arm, suite_name, task, sandbox, out_dir, run_id, python_exe, timeout):
        calls.append((arm, sandbox, run_id))
        return {
            "runkey": f"{arm}-{run_id}",
            "run_id": run_id,
            "suite": suite_name,
            "model": bench.MODEL,
            "benchmark_version": bench.BENCHMARK_VERSION,
            "surface": bench.SURFACE,
            "embedder": bench.EMBEDDER,
            "arm": arm,
            "task_id": task["id"],
            "duration_sec": 1,
            "measurement": {"input_tokens": 5 if run_id.endswith("r1") else 6},
            "verification": {"edit_applied": True},
        }

    monkeypatch.setattr(bench, "run_one", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "--suite",
            "examplecode",
            "--tasks",
            "add_input_validation",
            "--repeats",
            "2",
            "--out",
            str(tmp_path / "results"),
            "--sandbox-root",
            str(tmp_path / "sandboxes"),
        ],
    )
    assert bench.main() == 0
    assert [arm for arm, _, _ in calls] == ["plain", "gigacode", "gigacode", "plain"]
    assert len({path for _, path, _ in calls}) == 4
    data = json.loads((tmp_path / "results" / "comparison.json").read_text())
    assert data["arms"]["plain"]["per_task"]["add_input_validation"]["input_tokens"] == 5.5
    assert data["totals_comparison"]["edit_success"]["plain"] == "2/2"
    assert len(data["per_task_comparison"]["add_input_validation"]["paired_deltas"]["samples"]) == 2
