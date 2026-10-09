"""Tests for hash-anchored edits and compressed (skeleton) reads."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gigacode.skeleton import skeletonize
from gigacode.token_tools import compress_messages, line_anchor, resolve_anchor

PYTHON_SRC = '''"""Module docstring line one.
Module docstring line two.
"""
import math


# A comment-only line that should vanish
def alpha(x):
    """Function docstring."""
    if x <= 0:
        # inline comment kept? no: comment-only line dropped
        return 0
    return x


def beta():
    return 1


def gamma():
    pass
'''

PYTHON_SRC_LINES = PYTHON_SRC.splitlines()


def test_skeletonize_python_drops_docstrings_comments_blanks():
    kept, numbers = skeletonize(PYTHON_SRC_LINES, "mod.py")
    text = "\n".join(kept)
    assert "Module docstring" not in text
    assert "Function docstring" not in text
    assert "A comment-only line" not in text
    assert "import math" in text
    assert "def alpha(x):" in text
    assert "def gamma():" in text
    assert numbers == [4, 5, 8, 10, 12, 13, 14, 16, 17, 18, 20, 21]
    assert len(numbers) == len(kept)
    assert all(1 <= n <= len(PYTHON_SRC_LINES) for n in numbers)


def test_skeletonize_js_comments():
    src = ["// license header", "", "function a() {};", "", "", "function b() {};"]
    kept, numbers = skeletonize(src, "mod.js")
    assert kept == ["", "function a() {};", "", "function b() {};"]
    assert numbers == [2, 3, 4, 6]


def test_skeletonize_syntax_error_fallback():
    broken = ["def broken(:", "# comment line", "", "x = 1"]
    kept, numbers = skeletonize(broken, "broken.py")
    assert "# comment line" not in kept
    assert "x = 1" in kept
    assert len(kept) == len(numbers)


def test_line_anchor_roundtrip():
    lines = ["one", "two", "three"]
    anchor = line_anchor(2, lines[1])
    assert resolve_anchor(anchor, lines) == 2


def test_line_anchor_rejects_drift():
    lines = ["one", "TWO-changed", "three"]
    anchor = line_anchor(2, "two")
    with pytest.raises(ValueError):
        resolve_anchor(anchor, lines)


def test_compress_messages_compacts_tool_results():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task one " + "x" * 1000},
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "read_code"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "read_code", "content": "y" * 5000},
        {"role": "user", "content": "task two"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "task three"},
    ]
    result = compress_messages(messages, keep_recent_turns=2)
    compressed = result["messages"]
    assert result["status"] == "ok"
    assert compressed[0]["content"] == "sys"
    joined = json_dumps(compressed)
    # Old tool payload must not be carried verbatim
    assert "y" * 100 not in joined
    assert "task three" in joined
    assert result["estimated_tokens_saved"] > 0


def test_compress_messages_no_shrink_returns_original():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    result = compress_messages(messages, keep_recent_turns=3)
    assert result["messages"] == messages


def test_compress_messages_rejects_bad_input():
    with pytest.raises(ValueError):
        compress_messages("nope")
    with pytest.raises(ValueError):
        compress_messages([{"role": "bogus", "content": ""}])


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def _mktool(tmp_path: Path, profile: str = "agent_core"):
    import shutil

    shutil.copytree(
        Path(__file__).resolve().parent.parent / "examplecode",
        tmp_path / "src",
    )
    from gigacode.gigacode_tool import CodeEmbeddingTool

    tool = CodeEmbeddingTool(
        work_dir=tmp_path / f"work_{profile}",
        device="cpu",
        use_gpu=False,
        tool_profile=profile,
    )
    res = tool.embed_codebase(tmp_path / "src", pattern="*.py")
    assert res["status"] == "ok"
    return tool, res["buffer_id"]


def test_edit_hashlines_returns_refreshed_anchors(tmp_path: Path) -> None:
    """After an anchored edit, the response carries the new file hash and the
    next anchors so a following chained edit needs no re-read."""
    tool, buf = _mktool(tmp_path)
    read = tool.read_hashlines(buf, "string_utils.py", start_line=9, end_line=13)
    assert read["status"] == "ok"
    anchors = [line.split("|") for line in read["lines"]]
    (_sa, sa_text), (_ea, ea_text) = anchors[0], anchors[-1]
    first_edit = tool.edit_hashlines(
        buf,
        "string_utils.py",
        start_anchor=_sa,
        end_anchor=_ea,
        new_lines=[f"# replaces {len(read['lines'])} lines ({sa_text.strip()!r} .. {ea_text.strip()!r})"],
        expected_hash=read["file_hash"],
    )
    assert first_edit["status"] == "ok"
    assert "file_hash" in first_edit and first_edit["file_hash"] != read["file_hash"]
    assert first_edit.get("new_anchors")
    assert first_edit["next_action"].startswith("chain")

    second = tool.edit_hashlines(
        buf,
        "string_utils.py",
        start_anchor=first_edit["new_anchors"][0]["anchor"],
        end_anchor=first_edit["new_anchors"][0]["anchor"],
        new_lines=[first_edit["new_anchors"][0]["text"] + "  # chained"],
        expected_hash=first_edit["file_hash"],
    )
    assert second["status"] == "ok"
    tool.close()


def test_tool_chain_post_edit_dry_run(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    chain = tool.tool_chain(chain="post_edit", buffer_id=buf, dry_run=True)
    assert chain["status"] in ("ok", "conflict", "error")
    assert chain["chain"] == "post_edit"
    names = [s["tool"] for s in chain["steps"]]
    assert names == ["commit", "auto_format", "auto_lint", "reload_codebase"]
    assert all(not s.get("skipped") for s in chain["steps"] if s["tool"] == "commit")
    for s in chain["steps"]:
        assert "duration_ms" in s and s.get("tool")
    assert chain["steps_compacted"] is True
    # Only the final step keeps its full response; interior steps are slimmed.
    assert "response" in chain["steps"][-1]
    assert all("response" not in s for s in chain["steps"][:-1])
    tool.close()


def test_tool_chain_search_read(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    chain = tool.tool_chain(chain="search_read", buffer_id=buf, query="quicksort implementation")
    assert chain["status"] == "ok"
    assert [s["tool"] for s in chain["steps"]] == ["code_search", "read_code"]
    assert chain["steps_compacted"] is True
    # Interior search response is compacted away; the read window survives.
    assert "response" not in chain["steps"][0]
    window = chain["steps"][-1]["response"]
    assert window["status"] == "ok"
    assert window.get("file")
    tool.close()


def test_tool_chain_unknown_and_missing_query(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    bad = tool.tool_chain(chain="nope", buffer_id=buf)
    assert bad["status"] == "error"
    missing = tool.tool_chain(chain="search_read", buffer_id=buf)
    assert missing["status"] == "error"
    tool.close()


def test_tool_chain_pre_commit_preview(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    chain = tool.tool_chain(chain="pre_commit", buffer_id=buf)
    assert chain["status"] in ("ok", "conflict")
    names = [s["tool"] for s in chain["steps"]]
    assert names == ["diff", "validate_changes", "polish_before_commit", "commit"]
    assert chain["steps"][-1]["response"].get("dry_run") is True
    tool.close()


def test_tool_chain_stream_read(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    chain = tool.tool_chain(
        chain="stream_read", buffer_id=buf, query="quick sort partition swap"
    )
    assert chain["status"] in ("ok", "warning")
    tools_used = [s.get("tool") for s in chain["steps"] if s.get("tool")]
    assert tools_used[0] == "semantic_search_streaming"
    assert "expand_match" in tools_used and "read_code" in tools_used
    last = chain["steps"][-1]["response"]
    assert last.get("skeleton") is True or last.get("lines")
    tool.close()


def test_tool_chain_find_and_analyze(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    chain = tool.tool_chain(
        chain="find_and_analyze", buffer_id=buf, query="moving average window"
    )
    assert chain["status"] in ("ok", "warning")
    tools_used = [s.get("tool") for s in chain["steps"] if s.get("tool")]
    assert tools_used == ["code_search", "analyze_change"]
    impact = chain["steps"][-1]["response"]
    assert impact.get("file")
    tool.close()


def test_tool_chain_anchor_apply_end_to_end(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    read = tool.read_hashlines(buf, "string_utils.py", start_line=10, end_line=11)
    first_anchor = read["lines"][0].split("|")[0]
    last_anchor = read["lines"][0].split("|")[0]
    chain = tool.tool_chain(
        chain="anchor_apply",
        buffer_id=buf,
        file="string_utils.py",
        start_anchor=first_anchor,
        end_anchor=last_anchor,
        new_lines=[read["lines"][0].split("|", 1)[1] + "  # anchored chain apply probe"],
        expected_hash=read["file_hash"],
        dry_run=True,
    )
    assert chain["status"] in ("ok", "conflict")
    assert [s.get("tool") for s in chain["steps"]] == [
        "edit_hashlines",
        "validate_changes",
        "commit",
    ]
    assert chain["steps"][0]["status"] == "ok"
    assert chain["steps"][-1]["response"].get("dry_run") is True
    tool.close()


def test_anchor_read_repeat_served_from_memory(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    first = tool.tool_chain(chain="anchor_read", buffer_id=buf, query="alpha beta probe")
    assert first["status"] in ("ok", "warning")
    second = tool.tool_chain(chain="anchor_read", buffer_id=buf, query="alpha beta probe")
    assert second.get("cached") is True
    assert "not re-executed" in second["note"]
    assert second["chain"] == "anchor_read"
    tool.close()


def test_anchor_read_then_anchor_apply_resumes_coordinates(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    read_chain = tool.tool_chain(chain="anchor_read", buffer_id=buf, query="camel_to_snake")
    assert read_chain.get("next_call"), "anchor_read must carry ready next_call"
    assert read_chain["next_call"]["arguments"]["chain"] == "anchor_apply"
    last = read_chain["steps"][-1]["response"]
    # Resume: no file/start_anchor/end_anchor/expected_hash supplied.
    probe = tool.tool_chain(
        chain="anchor_apply",
        buffer_id=buf,
        new_lines=[
            line.split("|", 1)[1] + ("  # steady resume probe" if i == 0 else "")
            for i, line in enumerate(last["lines"])
        ],
        dry_run=False,
    )
    assert probe["status"] == "ok"
    assert probe["resumed_last_read"] is True
    # Interior step args are compacted; verify the applied file via disk.
    probe_file = tmp_path / "src" / Path(last["file"].replace("\\", "/"))
    assert "# steady resume probe" in probe_file.read_text(encoding="utf-8")
    tool.close()


def test_anchor_apply_missing_coordinates_returns_hint(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    result = tool.tool_chain(
        chain="anchor_apply", buffer_id=buf, new_lines=["x"], dry_run=False
    )
    assert result["status"] == "error"
    assert "anchor_read" in result["hint"]
    tool.close()


def test_tool_search_repeat_served_from_memory(tmp_path: Path) -> None:
    tool, _buf = _mktool(tmp_path)
    first = tool.tool_search(query="edit anchored lines", max_results=10)
    assert first["status"] == "ok"
    second = tool.tool_search(query="edit anchored lines", max_results=10)
    assert second.get("cached") is True
    assert "not re-searched" in second["note"]
    tool.close()


def test_tool_chain_anchor_read_flows_to_apply(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    read_chain = tool.tool_chain(
        chain="anchor_read", buffer_id=buf, query="string casing conversion helpers"
    )
    assert read_chain["status"] in ("ok", "warning")
    tools_used = [s["tool"] for s in read_chain["steps"] if not s.get("skipped")]
    assert tools_used == ["code_search", "read_hashlines"]
    assert read_chain.get("next_hint") and "anchor_apply" in read_chain["next_hint"]
    last = read_chain["steps"][-1]["response"]
    assert last["status"] == "ok"
    assert "|" in last["lines"][0]
    assert last.get("file_hash")
    tool.close()


def test_anchor_apply_resume_requires_prior_read(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    result = tool.tool_chain(
        chain="anchor_apply", buffer_id=buf, new_lines=["x"], dry_run=False
    )
    assert result["status"] == "error"
    assert "anchor_read" in result["hint"]
    tool.close()


def test_anchor_apply_dry_run_preview_carries_persist_hint(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    read = tool.read_hashlines(buf, "string_utils.py", start_line=10, end_line=11)
    anchor = read["lines"][0].split("|")[0]
    dry = tool.tool_chain(
        chain="anchor_apply",
        buffer_id=buf,
        file="string_utils.py",
        start_anchor=anchor,
        end_anchor=anchor,
        new_lines=[read["lines"][0].split("|", 1)[1] + "  # dry probe"],
        expected_hash=read["file_hash"],
        dry_run=True,
    )
    assert dry.get("next_hint") and "dry_run=false" in dry["next_hint"]
    tool.close()


def test_edit_hashlines_repeat_guard_fires(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)
    stale_read = tool.read_hashlines(buf, "string_utils.py", start_line=10, end_line=12)
    anchor = stale_read["lines"][0].split("|")[0]
    tool.edit_hashlines(
        buf,
        "string_utils.py",
        start_anchor=anchor,
        end_anchor=anchor,
        new_lines=["# first probe edit"],
        expected_hash=stale_read["file_hash"],
    )
    repeat = tool.edit_hashlines(
        buf,
        "string_utils.py",
        start_anchor=anchor,
        end_anchor=anchor,
        new_lines=["# first probe edit"],
        expected_hash=stale_read["file_hash"],
    )
    assert repeat.get("repeat_hint"), "second identical anchored edit must carry repeat hint"
    assert "read_hashlines" in repeat["repeat_hint"]
    tool.close()


def test_tool_search_ranks_priority_first(tmp_path: Path) -> None:
    tool, buf = _mktool(tmp_path)

    # agent_core deliberately excludes the low-priority single-use search tools
    names = [t["name"] for t in tool.tool_search(query="search", max_results=20)["tools"]]
    assert "code_search" in names
    assert "semantic_search" not in names and "search_for" not in names
    tool.close()


def test_tool_search_full_profile_orders_high_before_low(tmp_path: Path) -> None:
    tool, _buf = _mktool(tmp_path, profile="full")
    tools = tool.tool_search(query="search", max_results=20)["tools"]
    index_of = {t["name"]: i for i, t in enumerate(tools)}
    assert "code_search" in index_of and "semantic_search" in index_of
    assert index_of["code_search"] < index_of["semantic_search"]
    assert index_of.get("semantic_search_streaming", 10**9) < index_of["semantic_search"]
    tool.close()


def test_all_schemas_have_priority() -> None:
    from gigacode.tool_schema import ALL_SCHEMAS

    for schema in ALL_SCHEMAS:
        assert schema.get("priority") in ("low", "normal", "high"), schema["name"]
