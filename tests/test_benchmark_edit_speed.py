"""Edit-speed extraction from opencode event streams (benchmark reporting)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from benchmark_token_usage import measure_edit_speed, measure_tool_ms


def _event(kind: str, ts: int, **part: dict) -> dict:
    return {"type": kind, "timestamp": ts, "part": {"type": kind, **part}}


def _run(tmp_path: Path, events: list[dict]) -> dict:
    f = tmp_path / "events.jsonl"
    f.write_text("\n".join(e and _dump(e) for e in events), encoding="utf-8")
    return measure_edit_speed(f)


def _dump(obj: dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def test_regular_edit_tool_measured(tmp_path: Path) -> None:
    events = [
        _event("step-start", 0, index=0),
        _event("tool", 0, tool=None),  # noise
        _event(
            "tool-use",
            1000,
            part=None,
        ),
        _event("step-start", 5000, index=1),
        _event("text", 6000, text="done"),
    ]
    # Build a realistic stream manually: tool_use with edit tool + timing.
    events = [
        {"type": "step_start", "timestamp": 0, "part": {"type": "step-start"}},
        {
            "type": "tool_use",
            "timestamp": 1500,
            "part": {
                "type": "tool",
                "tool": "edit",
                "state": {
                    "status": "completed",
                    "input": {"path": "a.py"},
                    "time": {"start": 1000, "end": 3000},
                },
            },
        },
        {"type": "text", "timestamp": 6000, "part": {"type": "text", "text": "done"}},
    ]
    f = tmp_path / "ev.jsonl"
    f.write_text("\n".join(_dump(e) for e in events), encoding="utf-8")
    speed = measure_edit_speed(f)
    assert speed["time_to_edit_s"] == 3.0  # first edit end 3000ms; t0 = 0
    assert speed["edit_span_s"] == 2.0
    assert speed["time_after_edit_s"] == 3.0
    assert speed["edit_calls"] == 1


def test_deferred_tool_call_counts_anchor_apply(tmp_path: Path) -> None:
    events = [
        {"type": "step_start", "timestamp": 0, "part": {"type": "step-start"}},
        {
            "type": "tool_use",
            "timestamp": 1000,
            "part": {
                "type": "tool",
                "tool": "gigacode_tool_call",
                "state": {
                    "status": "completed",
                    "input": {
                        "name": "tool_chain",
                        "arguments": {"chain": "anchor_apply", "dry_run": False},
                    },
                    "output": '{"status":"ok","steps":[{"tool":"commit","response":{"written_files":["a.py"],"dry_run":false}}]}',
                    "time": {"start": 2000, "end": 9000},
                },
            },
        },
        # anchor_read must NOT count as an edit
        {
            "type": "tool_use",
            "timestamp": 3000,
            "part": {
                "type": "tool",
                "tool": "gigacode_tool_call",
                "state": {
                    "status": "completed",
                    "input": {
                        "name": "tool_chain",
                        "arguments": {"chain": "anchor_read"},
                    },
                    "time": {"start": 4000, "end": 5000},
                },
            },
        },
        {"type": "step_finish", "timestamp": 12000, "part": {"type": "step-finish"}},
    ]
    f = tmp_path / "ev.jsonl"
    f.write_text("\n".join(_dump(e) for e in events), encoding="utf-8")
    speed = measure_edit_speed(f)
    assert speed["edit_calls"] == 1
    assert speed["time_to_edit_s"] == 9.0  # 9000 - 0


def test_no_edit_events_yields_none(tmp_path: Path) -> None:
    events = [
        {"type": "step_start", "timestamp": 0, "part": {"type": "step-start"}},
        {
            "type": "tool_use",
            "timestamp": 1000,
            "part": {
                "type": "tool",
                "tool": "read",
                "state": {
                    "status": "completed",
                    "input": {},
                    "time": {"start": 1000, "end": 2000},
                },
            },
        },
    ]
    f = tmp_path / "ev.jsonl"
    f.write_text("\n".join(_dump(e) for e in events), encoding="utf-8")
    assert measure_edit_speed(f) == {"time_to_edit_s": None}


def test_edit_time_sums_individual_call_durations(tmp_path: Path) -> None:
    events = [
        {"type": "step_start", "timestamp": 0, "part": {"type": "step-start"}},
        {
            "type": "tool_use",
            "timestamp": 1000,
            "part": {
                "type": "tool",
                "tool": "edit",
                "state": {
                    "status": "completed",
                    "input": {},
                    "time": {"start": 1000, "end": 3000},
                },
            },
        },
        {
            "type": "tool_use",
            "timestamp": 5000,
            "part": {
                "type": "tool",
                "tool": "gigacode_tool_call",
                "state": {
                    "status": "completed",
                    "input": {"name": "tool_chain", "arguments": {"chain": "anchor_apply", "dry_run": False}},
                    "output": '{"status":"ok","steps":[{"tool":"commit","response":{"written_files":["a.py"],"dry_run":false}}]}',
                    "time": {"start": 5000, "end": 6200},
                },
            },
        },
        {"type": "step_finish", "timestamp": 7000, "part": {"type": "step-finish"}},
    ]
    f = tmp_path / "ev.jsonl"
    f.write_text("\n".join(_dump(e) for e in events), encoding="utf-8")
    speed = measure_edit_speed(f)
    assert speed["edit_time_s"] == 3.2  # 2.0 + 1.2
    assert speed["edit_calls"] == 2
    assert speed["edit_span_s"] == 5.2  # 1000..6200
    assert speed["time_after_edit_s"] == 0.8


def test_measure_tool_ms_sums_only_gigacode_calls(tmp_path: Path) -> None:
    def _call(tool_name: str, status: str, start: int, end: int) -> dict:
        return {
            "type": "tool_use",
            "timestamp": start,
            "part": {
                "type": "tool",
                "tool": tool_name,
                "state": {
                    "status": status,
                    "input": {},
                    "time": {"start": start, "end": end},
                },
            },
        }

    events = [
        _call("gigacode_tool_call", "completed", 1000, 3000),
        _call("gigacode_tool_call", "error", 4000, 5000),  # failed requests still cost time
        _call("bash", "completed", 6000, 11000),  # non-MCP: excluded
        _call("gigacode_tool_search", "completed", 12000, 13500),
    ]
    f = tmp_path / "ev.jsonl"
    f.write_text("\n".join(_dump(e) for e in events), encoding="utf-8")
    result = measure_tool_ms(f)
    assert result["tool_calls"] == 3
    assert result["tool_ms_s"] == 4.5

    empty = tmp_path / "none.jsonl"
    empty.write_text("", encoding="utf-8")
    assert measure_tool_ms(empty) == {"tool_ms_s": None}
