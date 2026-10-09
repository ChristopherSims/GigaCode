"""Regression coverage for safe, low-turn coding workflows."""

import json
import sys
from pathlib import Path

import pytest

from gigacode.coding_safety import exact_source_matches, validate_replacement
from gigacode.gigacode_tool import CodeEmbeddingTool
from scripts.mcp_bench_server import HashEmbedder


@pytest.fixture
def coding_tool(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "utils.py").write_text(
        "def previous():\n    return True\n\n\n"
        "def moving_average(values, window):\n"
        '    """Moving average over a sliding window."""\n'
        "    if window > len(values):\n"
        "        raise ValueError('too large')\n"
        "    return [sum(values[i:i+window]) / window for i in range(len(values)-window+1)]\n",
        encoding="utf-8",
    )
    (root / "other.py").write_text("value = 0\n", encoding="utf-8")
    tool = CodeEmbeddingTool(
        work_dir=tmp_path / "buffers",
        device="cpu",
        use_gpu=False,
        tool_profile="agent_core",
        embedder=HashEmbedder(),
    )
    result = tool.embed_codebase(root)
    assert result["status"] == "ok"
    yield tool, result["buffer_id"], root
    tool.close()


def test_identifier_no_semantic_guess(coding_tool):
    tool, buf, _ = coding_tool
    found = tool.code_search(buf, "def moving_average")
    assert found["matches"][0]["file"] == "utils.py"
    assert found["matches"][0]["start_line"] == 5
    assert found["matches"][0]["confidence"] == "exact"
    assert tool.code_search(buf, "nonexistent_symbol")["matches"] == []


def test_definition_beats_earlier_reference():
    snapshot = {
        "a.py": ["def caller():", "    return needle()", "", "def needle():", "    return 1"]
    }
    matches = exact_source_matches(snapshot, "needle", 3)["matches"]
    assert matches[0]["start_line"] == 4


def test_explicit_declaration_with_description_still_finds_symbol():
    snapshot = {"text.py": ['class Text:', '    def rstrip(self):', '        """Strip whitespace at end."""', "        pass"]}
    result = exact_source_matches(snapshot, "def rstrip suffix", 1)
    assert result["matches"][0]["start_line"] == 2
    assert result["matches"][0]["definition_match"]


def test_file_lookup_and_filter(coding_tool):
    tool, buf, _ = coding_tool
    assert tool.code_search(buf, "utils.py")["matches"][0]["file"] == "utils.py"
    assert tool.code_find("moving_average", buf, file="missing.py")["status"] == "error"


def test_anchor_read_separates_previous_function(coding_tool):
    tool, buf, _ = coding_tool
    read = tool.tool_chain("anchor_read", buf, query="moving_average")
    assert read["editable_range"]["start_line"] == 5
    lines = read["steps"][-1]["response"]["lines"]
    assert "def moving_average" in lines[0]
    assert not any("return True" in line for line in lines)


def test_body_only_resume_rejected_without_mutation(coding_tool):
    tool, buf, root = coding_tool
    original = (root / "utils.py").read_text(encoding="utf-8")
    snapshot = tool._load_source_snapshot(buf)
    tool.tool_chain("anchor_read", buf, query="moving_average")
    result = tool.tool_chain("anchor_apply", buf, new_lines=["    return []"], dry_run=False)
    assert result["status"] == "error"
    assert tool._load_source_snapshot(buf) == snapshot
    assert (root / "utils.py").read_text(encoding="utf-8") == original


def _edit_args(read):
    return {
        "file": read["file"],
        "start_anchor": read["start_anchor"],
        "end_anchor": read["end_anchor"],
        "expected_hash": read["file_hash"],
    }


def test_direct_two_call_edit_and_preview(coding_tool):
    tool, buf, root = coding_tool
    read = tool.code_find("moving_average", buf)
    replacement = [line.split("|", 1)[1] for line in read["lines"]]
    replacement[3] = "        return []"
    snapshot = tool._load_source_snapshot(buf)
    original = (root / "utils.py").read_text(encoding="utf-8")
    preview = tool.code_edit(**_edit_args(read), buffer_id=buf, new_lines=replacement, dry_run=True)
    assert preview["status"] == "ok" and preview["applied"] is False
    assert tool._load_source_snapshot(buf) == snapshot
    assert (root / "utils.py").read_text(encoding="utf-8") == original
    edit = tool.code_edit(**_edit_args(read), buffer_id=buf, new_lines=replacement)
    assert edit["applied"] is True
    assert "        return []" in (root / "utils.py").read_text(encoding="utf-8")
    assert (
        tool.code_edit(**_edit_args(read), buffer_id=buf, new_lines=replacement)["status"]
        == "conflict"
    )


def test_invalid_syntax_never_changes_state(coding_tool):
    tool, buf, root = coding_tool
    read = tool.code_find("moving_average", buf)
    snapshot = tool._load_source_snapshot(buf)
    original = (root / "utils.py").read_text(encoding="utf-8")
    result = tool.code_edit(**_edit_args(read), buffer_id=buf, new_lines=["def broken(:"])
    assert result["applied"] is False
    assert tool._load_source_snapshot(buf) == snapshot
    assert (root / "utils.py").read_text(encoding="utf-8") == original


def test_cached_read_restores_correct_target(coding_tool):
    tool, buf, _ = coding_tool
    first = tool.tool_chain("anchor_read", buf, query="moving_average")
    tool.tool_chain("anchor_read", buf, query="previous")
    cached = tool.tool_chain("anchor_read", buf, query="moving_average")
    assert cached["cached"]
    assert tool._last_anchor_read == first["editable_range"]
    empty = tool.tool_chain("anchor_read", buf, query="nonexistent_symbol")
    assert "next_call" not in empty
    assert tool._last_anchor_read is None


def test_file_override_does_not_inherit_other_file_anchors(coding_tool):
    tool, buf, _ = coding_tool
    tool.tool_chain("anchor_read", buf, query="previous")
    result = tool.tool_chain(
        "anchor_apply", buf, file="other.py", new_lines=["pass"], dry_run=False
    )
    assert result["status"] == "error"
    assert "start_anchor" in result["message"]


def test_explicit_file_with_inherited_range_still_preserves_definition(coding_tool):
    tool, buf, _ = coding_tool
    tool.tool_chain("anchor_read", buf, query="moving_average")
    result = tool.tool_chain(
        "anchor_apply",
        buf,
        file="utils.py",
        new_lines=["average = 0"],
        dry_run=False,
    )
    assert result["status"] == "error"
    assert "remove a definition" in result["message"]


def test_noop_direct_edit_is_not_applied(coding_tool):
    tool, buf, _ = coding_tool
    read = tool.code_find("previous", buf)
    result = tool.code_edit(
        **_edit_args(read),
        buffer_id=buf,
        new_lines=[line.split("|", 1)[1] for line in read["lines"]],
    )
    assert result["status"] == "ok"
    assert result["applied"] is False


def test_direct_edit_does_not_persist_unrelated_pending_work(coding_tool):
    tool, buf, root = coding_tool
    staged = tool.write_code(buf, "other.py", 1, ["value = 2"])
    assert staged["status"] == "ok"
    read = tool.code_find("previous", buf)
    result = tool.code_edit(
        **_edit_args(read),
        buffer_id=buf,
        new_lines=["def previous():", "    return False"],
    )
    assert result["status"] == "blocked"
    assert (root / "other.py").read_text(encoding="utf-8") == "value = 0\n"
    assert (
        (root / "utils.py")
        .read_text(encoding="utf-8")
        .startswith("def previous():\n    return True")
    )


def test_direct_large_target_requires_narrow_read(coding_tool, monkeypatch):
    tool, buf, _ = coding_tool
    monkeypatch.setattr("gigacode.mcp_server._MCP_OUTPUT_CHAR_CAP", 200)
    result = tool.code_find("moving_average", buf)
    assert result["editable"] is False
    assert "start_anchor" not in result


def test_explicit_definition_removal_is_opt_in():
    lines = ["def a():", "    return 1", "", "def b():", "    return 2"]
    assert validate_replacement(lines, "a.py", 4, 5, [], True)
    assert validate_replacement(lines, "a.py", 4, 5, [], False) is None


@pytest.mark.parametrize("line", ["return 1", "break", "continue", "await task()"])
def test_compiler_context_errors_rejected(line):
    assert validate_replacement(["pass"], "a.py", 1, 1, [line])


def test_disabled_rank_sources_do_not_leak_zero_score_hits():
    from gigacode.hybrid_search import reciprocal_rank_fusion

    results = reciprocal_rank_fusion(
        [{"doc_id": 1, "score": 1.0}],
        [{"doc_id": 2, "score": 1.0}],
        semantic_weight=0,
        lexical_weight=1,
    )
    assert [r["doc_id"] for r in results] == [2]


def test_scoped_behavior_search_avoids_global_retrieval(coding_tool, monkeypatch):
    tool, buf, _ = coding_tool

    def forbidden(*args, **kwargs):
        raise AssertionError("Scoped search must not run global semantic retrieval")

    monkeypatch.setattr(tool, "hybrid_search", forbidden)
    read = tool.code_find("sliding window average", buf, file="utils.py")
    assert read["editable"]
    assert read["start_line"] == 5


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_stdio_direct_coding_surface(tmp_path):
    """Exercise actual discovery, auto-bootstrap, persistence and stale errors."""
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    root = tmp_path / "project"
    root.mkdir()
    source = root / "helpers.py"
    source.write_text("def answer():\n    return 1\n", encoding="utf-8")
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            str(Path(__file__).resolve().parents[1] / "scripts" / "mcp_bench_server.py"),
            str(tmp_path / "buffers"),
            "agent_core",
        ],
        cwd=str(tmp_path),
        env={
            "GIGACODE_AUTO_EMBED": "on",
            "GIGACODE_DIRECT_TOOLS": "on",
            "GIGACODE_BENCH_GPU": "off",
        },
    )
    with anyio.fail_after(30):
        async with stdio_client(params) as (receive, send):
            async with ClientSession(receive, send) as session:
                await session.initialize()
                listing = await session.list_tools()
                assert {t.name for t in listing.tools} == {
                    "code_find",
                    "code_edit",
                    "code_navigate",
                    "get_task_context",
                    "tool_search",
                    "tool_call",
                }
                descriptions = {t.name: t.description for t in listing.tools}
                assert "native JSON" in descriptions["code_edit"]
                assert "No discovery, commit or post_edit call needed." in descriptions["code_edit"]
                found = await session.call_tool("code_find", {"query": "answer"})
                assert not found.isError
                read = json.loads(found.content[0].text)
                assert read["editable"] and read["start_line"] == 1
                arguments = {
                    **_edit_args(read),
                    "new_lines": ["def answer():", "    return 42"],
                }
                edited = await session.call_tool("code_edit", arguments)
                assert not edited.isError
                assert json.loads(edited.content[0].text)["applied"]
                assert source.read_text(encoding="utf-8") == "def answer():\n    return 42\n"
                stale = await session.call_tool("code_edit", arguments)
                assert stale.isError
                assert json.loads(stale.content[0].text)["status"] == "conflict"
                text_edit = await session.call_tool("code_edit", {
                    "file": "project/helpers.py",
                    "old_text": "    return 42",
                    "new_text": "    return 43",
                })
                assert not text_edit.isError
                response = json.loads(text_edit.content[0].text)
                assert response["applied"] and "file_hash" in response
                assert "steps" not in response
                assert source.read_text(encoding="utf-8") == "def answer():\n    return 43\n"
                malformed = await session.call_tool("code_edit", {
                    "file": "project/helpers.py",
                    "new_lines": ['return 44</arg_value><arg_key>old_text</arg_key><arg_value>PRIVATE_SENTINEL'],
                })
                assert malformed.isError
                error = json.loads(malformed.content[0].text)
                assert "native JSON" in error["message"]
                assert "PRIVATE_SENTINEL" not in malformed.content[0].text
                assert len(malformed.content[0].text) < 512
                assert source.read_text(encoding="utf-8") == "def answer():\n    return 43\n"
                malformed_path = await session.call_tool("code_edit", {
                    "file": "file</arg_key><arg_value>project/helpers.py",
                    "old_text": "return 43", "new_text": "return 44",
                })
                assert malformed_path.isError
                error = json.loads(malformed_path.content[0].text)
                assert error["code"] == "invalid_arguments"
                assert "native JSON" in error["message"]
                assert source.read_text(encoding="utf-8") == "def answer():\n    return 43\n"
                xml_source = await session.call_tool("code_edit", {
                    "file": "project/helpers.py",
                    "old_text": "return 43", "new_text": 'return "<arg_key>value</arg_key>"',
                })
                assert not xml_source.isError
                assert "<arg_key>value</arg_key>" in source.read_text(encoding="utf-8")
                navigation = await session.call_tool("code_navigate", {
                    "action": "symbol", "file": "helpers.py", "symbol": "answer", "include_source": True,
                })
                assert not navigation.isError
                assert "<arg_key>value</arg_key>" in json.loads(navigation.content[0].text)["source"]
                context = await session.call_tool("get_task_context", {"task": "Update answer", "include_git": False})
                assert not context.isError
                assert json.loads(context.content[0].text)["likely_files"][0]["file"] == "helpers.py"
                deferred = await session.call_tool("tool_call", {
                    "name": "file_summary", "arguments": {"file": "helpers.py"},
                })
                assert not deferred.isError
