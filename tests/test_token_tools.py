"""Token-saving tools must preserve permissions and reject stale edits."""

import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from gigacode.gigacode_tool import CodeEmbeddingTool
from gigacode.token_tools import compress_messages, line_anchor, resolve_anchor
from gigacode.tool_schema import get_profile_tool_names


def make_tool(profile="editing"):
    tool = object.__new__(CodeEmbeddingTool)
    tool._allowed_tools = get_profile_tool_names(profile)
    tool._hashline_lock = threading.RLock()
    tool._recent_calls = {}
    tool._last_buffer_id = None
    tool._resolve_buffer = Mock(return_value=("buffer", None))
    tool._load_source_snapshot = Mock(return_value={"a.py": ["one", "two", "three"]})
    tool._get_buffer_info = Mock(return_value={"root": str(Path.cwd()), "buffer_dir": str(Path.cwd() / "__test_buffers__")})
    tool.write_code = Mock(return_value={"status": "ok"})
    return tool


def test_anchors_bind_coordinate_and_text():
    anchor = line_anchor(2, "two")
    assert resolve_anchor(anchor, ["one", "two"]) == 2
    with pytest.raises(ValueError):
        resolve_anchor(anchor, ["one", "changed"])
    with pytest.raises(ValueError):
        resolve_anchor("0:bad", ["one"])


def test_hashline_read_and_edit():
    tool = make_tool()
    read = tool.read_hashlines(None, "a.py", 2, 3)
    assert read["lines"] == [f"{line_anchor(2, 'two')}|two", f"{line_anchor(3, 'three')}|three"]
    result = tool.edit_hashlines(
        None,
        "a.py",
        line_anchor(2, "two"),
        line_anchor(3, "three"),
        ["replacement"],
        read["file_hash"],
    )
    assert result["status"] == "ok"
    tool.write_code.assert_called_once_with("buffer", "a.py", 2, ["replacement"], end_line=3)


def test_read_hashlines_drops_anchors_on_request():
    tool = make_tool()
    plain = tool.read_hashlines(None, "a.py", 2, 3, include_anchors=False)
    assert plain["lines"] == ["two", "three"]
    assert plain["anchors_omitted"] is True
    anchored = tool.read_hashlines(None, "a.py", 2, 3)
    assert anchored["lines"][0] == f"{line_anchor(2, 'two')}|two"
    # An anchored edit still requires anchors: a plain read is not enough.


def test_tool_search_resolves_vague_queries():
    tool = make_tool("editing")
    vague = "safely replace a few lines guarding against the file shifting under me"
    result = tool.tool_search(vague)
    assert result["status"] == "ok"
    names = [t["name"] for t in result["tools"]]
    assert names and "edit_hashlines" in names
    browse = tool.tool_search("read the source of a file, bounded to some lines", 20)
    names = [t["name"] for t in browse["tools"]]
    assert {"read_hashlines", "read_code", "look_for_file"} & set(names)


def test_discovery_pair_carries_recipes_and_bash_nudge():
    from gigacode.tool_schema import get_schema

    for name in ("tool_search", "tool_call"):
        schema = get_schema(name)
        assert schema["description_limit"] == 2400
        assert "search_read" in schema["description"]
        assert "anchor_apply" in schema["description"]
        assert "post_edit" in schema["description"]
        assert "shell" in schema["description"].lower()


def test_discovery_pair_descriptions_survive_mcp_trimming():
    pytest.importorskip("mcp")
    from gigacode.mcp_server import _build_mcp_tools

    tool = make_tool()
    tool.deferred_tools = True
    tool.direct_tools = False
    by_name = {t.name: t.description for t in _build_mcp_tools(tool)}
    assert len(by_name["tool_search"]) > 240
    assert "search_read" in by_name["tool_search"]
    assert "posterized" not in by_name["tool_call"]


def test_mcp_output_cap_trims_payloads_json_safe():
    import json

    from gigacode.mcp_server import _cap_response

    small = {"status": "ok", "value": 1}
    assert _cap_response(small) == small

    big = {
        "status": "ok",
        "dump": "x" * 5000,
        "rows": [f"row {i} padded content" for i in range(400)],
    }
    capped = _cap_response(big)
    assert "note" in capped
    assert "truncated" in capped["dump"]
    assert len(capped["rows"]) < 400 and "truncated" in capped["rows"][-1]
    json.dumps(capped)  # must stay valid JSON
    assert len(json.dumps(capped)) < len(json.dumps(big))
    assert big["status"] == "ok" and capped["status"] == "ok"


def test_no_buffer_returns_actionable_error_not_crash():
    import types

    tool = make_tool("agent_core")
    tool._last_buffer_id = None
    tool._resolve_buffer = types.MethodType(CodeEmbeddingTool._resolve_buffer, tool)
    result = tool.tool_call("read_code", {"file": "x.py"})
    assert result["status"] == "error"
    assert "embed_codebase" in result["message"]


def test_wandering_loop_gets_budget_nudge():
    tool = make_tool("agent_core")
    tool._wander_nudge_every = 3
    tool.read_code = Mock(side_effect=lambda *a, **k: {"status": "ok", "lines": []})
    r1 = tool.tool_call("read_code", {"buffer_id": "b", "file": "a.py"})
    assert "budget_nudge" not in r1
    tool.tool_call("read_code", {"buffer_id": "b", "file": "a.py"})
    r3 = tool.tool_call("read_code", {"buffer_id": "b", "file": "a.py"})
    nudge = r3.get("budget_nudge")
    assert nudge and nudge["steps_since_edit"] == 3
    assert nudge["next_call"]["arguments"]["chain"] == "anchor_read"


def test_budget_nudge_resets_after_a_landed_edit():
    tool = make_tool("agent_core")
    tool._wander_nudge_every = 2
    tool.write_code = Mock(side_effect=lambda *a, **k: {"status": "ok"})
    tool.read_code = Mock(side_effect=lambda *a, **k: {"status": "ok", "lines": []})
    tool.tool_call("read_code", {"buffer_id": "b", "file": "a.py"})
    tool.tool_call("read_code", {"buffer_id": "b", "file": "a.py"})
    r = tool.tool_call("read_code", {"buffer_id": "b", "file": "a.py"})
    assert r.get("budget_nudge")
    edit = tool.tool_call(
        "write_code",
        {"buffer_id": "b", "file": "a.py", "start_line": 1, "new_lines": ["x = 1"]},
    )
    assert edit["status"] == "ok" and "budget_nudge" not in edit
    r = tool.tool_call("read_code", {"buffer_id": "b", "file": "a.py"})
    r2 = tool.tool_call("read_code", {"buffer_id": "b", "file": "a.py"})
    # Counter restarted: first read clean, nudge re-arms at the threshold (2).
    assert "budget_nudge" not in r
    assert r2.get("budget_nudge", {}).get("steps_since_edit") == 2


def test_budget_nudge_recommends_anchor_apply_when_read_is_fresh():
    tool = make_tool("agent_core")
    tool._wander_nudge_every = 2
    tool.read_code = Mock(side_effect=lambda *a, **k: {"status": "ok", "lines": []})
    tool._last_anchor_read = {"buffer_id": "b", "file": "a.py", "file_hash": "h",
                              "start_anchor": "1:abc", "end_anchor": "1:abc"}
    tool.tool_call("read_code", {"buffer_id": "b", "file": "a.py"})
    tool.tool_call("read_code", {"buffer_id": "b", "file": "a.py"})
    r = tool.tool_call("read_code", {"buffer_id": "b", "file": "a.py"})
    assert r["budget_nudge"]["next_call"]["arguments"]["chain"] == "anchor_apply"


def test_auto_embed_when_buffer_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGACODE_AUTO_EMBED", "on")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "a.py").write_text("x = 1\nz = 2\n", encoding="utf-8")
    tool = CodeEmbeddingTool(tmp_path / "buffers", use_gpu=False, tool_profile="agent_core")
    try:
        # The agent never calls embed_codebase; the tool resolves (and
        # embeds, once) the working project automatically.
        result = tool.read_code(None, file="a.py")
        assert result["status"] == "ok", result
        first_buffer = result.get("buffer_id") or tool._last_buffer_id
        assert first_buffer
        # A second buffer-less call reuses the same buffer (no re-embed).
        again = tool.read_code(None, file="a.py", start_line=1, end_line=2)
        assert again.get("status") == "ok", again
        assert (again.get("buffer_id") or tool._last_buffer_id) == first_buffer
    finally:
        tool.close()


def test_auto_embed_off_returns_error(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGACODE_AUTO_EMBED", "off")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "a.py").write_text("x = 1\nz = 2\n", encoding="utf-8")
    tool = CodeEmbeddingTool(tmp_path / "buffers", use_gpu=False, tool_profile="agent_core")
    try:
        result = tool.read_code(None, file="a.py")
        assert result["status"] == "error"
        assert "embed_codebase" in result["message"]
    finally:
        tool.close()


def test_internal_tool_failure_never_crashes_caller():
    tool = make_tool("agent_core")
    tool.read_code = Mock(side_effect=RuntimeError("unexpected explosion"))
    result = tool.tool_call("read_code", {"buffer_id": "b", "file": "x.py"})
    assert result["status"] == "error"
    assert "Internal tool failure" in result["message"]


def test_chain_deadline_skips_remaining_steps():
    import time as _t

    tool = make_tool("agent_core")

    def slow_step(*_a, **_k):
        _t.sleep(0.3)
        return {"status": "ok"}

    tool.commit = Mock(side_effect=slow_step)
    tool.auto_format = Mock(return_value={"status": "ok"})
    tool.auto_lint = Mock(return_value={"status": "ok"})
    tool.reload_codebase = Mock(return_value={"status": "ok"})
    tool._chain_deadline_sec = 0.05
    chain = tool.tool_chain(chain="post_edit", buffer_id="b")
    assert chain["status"] in ("ok", "warning")
    remaining = [s for s in chain["steps"] if s.get("skipped")]
    assert remaining, "over-budget steps must be skipped, not executed"
    assert any("deadline" in s["reason"] for s in remaining)
    tool.auto_format.assert_not_called()


def test_hashline_rejects_interior_changes_and_invalid_ranges():
    tool = make_tool()
    read = tool.read_hashlines(None, "a.py")
    tool._load_source_snapshot.return_value["a.py"][1] = "changed"
    result = tool.edit_hashlines(
        None, "a.py", line_anchor(1, "one"), line_anchor(3, "three"), [], read["file_hash"]
    )
    assert result["status"] == "error"
    tool.write_code.assert_not_called()
    assert tool.read_hashlines(None, "a.py", 0)["status"] == "error"


def test_search_and_call_enforce_profile():
    tool = make_tool("read_only")
    assert tool.tool_search("select:write_code,edit_hashlines")["tools"] == []
    assert tool.tool_call("edit_hashlines", {})["status"] == "error"
    assert tool.tool_call("tool_call", {})["status"] == "error"
    assert tool.tool_search("select:read_hashlines", 1)["tools"][0]["name"] == "read_hashlines"
    assert tool.tool_call("compress_context", {"messages": []})["status"] == "ok"


def test_compression_preserves_recent_tool_pairs_and_instructions():
    messages = [
        {"role": "system", "content": "Do not delete files."},
        {"role": "user", "content": "Fix the bug."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "old", "function": {"name": "read_code"}}],
        },
        {"role": "tool", "tool_call_id": "old", "content": "source " * 5000},
        {"role": "user", "content": "Run tests."},
        {"role": "assistant", "tool_calls": [{"id": "new", "function": {"name": "test"}}]},
        {"role": "tool", "tool_call_id": "new", "content": "passed"},
    ]
    result = compress_messages(messages, keep_recent_turns=1)
    assert result["compressed_chars"] < result["original_chars"]
    assert result["messages"][0] == messages[0]
    assert result["messages"][-3:] == messages[-3:]
    assert "Fix the bug." in result["messages"][1]["content"]
    assert messages[3]["content"] == "source " * 5000


def test_short_context_is_unchanged():
    messages = [{"role": "user", "content": "Hello"}]
    assert compress_messages(messages)["messages"] == messages
    with pytest.raises(ValueError):
        compress_messages(messages, keep_recent_turns=0)


def test_deferred_mcp_surface():
    pytest.importorskip("mcp")
    from gigacode.mcp_server import _build_mcp_tools

    tool = make_tool()
    # Deferred is the DEFAULT: a tool object that never opts out publishes
    # nothing but the discovery pair, so the full schema is never exposed.
    assert {t.name for t in _build_mcp_tools(tool)} == {
        "tool_search", "tool_call", "code_find", "code_edit", "code_navigate", "get_task_context",
    }
    tool.direct_tools = False
    assert {t.name for t in _build_mcp_tools(tool)} == {"tool_search", "tool_call"}
    tool.deferred_tools = False  # explicit eager opt-out
    assert "edit_hashlines" in {t.name for t in _build_mcp_tools(tool)}


def test_hashline_buffer_workflow(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    tool = CodeEmbeddingTool(tmp_path / "buffers", use_gpu=False, tool_profile="editing")
    try:
        embedded = tool.embed_codebase(source, pattern="*.py")
        assert embedded["status"] == "ok"
        buffer_id = embedded["buffer_id"]
        snapshot = tool._load_source_snapshot(buffer_id)
        file = next(iter(snapshot))
        read = tool.read_hashlines(buffer_id, file)
        anchor = read["lines"][0].split("|", 1)[0]
        edited = tool.edit_hashlines(buffer_id, file, anchor, anchor, ["x = 10"], read["file_hash"])
        assert edited["status"] == "ok"
        updated = tool.read_hashlines(buffer_id, file)
        assert updated["lines"][0].endswith("|x = 10")
        assert (
            tool.edit_hashlines(buffer_id, file, anchor, anchor, [], read["file_hash"])["status"]
            == "error"
        )
        assert (source / "a.py").read_text(encoding="utf-8") == "x = 1\ny = 2\n"
    finally:
        tool.close()
