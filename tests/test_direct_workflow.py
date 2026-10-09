"""Concrete path and small-edit regressions from the real agent benchmark."""

import os

import pytest

from gigacode.coding_safety import text_replacement
from gigacode.gigacode_tool import CodeEmbeddingTool
from gigacode.path_utils import SourcePathError, resolve_source_path
from scripts.mcp_bench_server import HashEmbedder


@pytest.fixture
def nested_tool(tmp_path):
    root = tmp_path / "project"
    package = root / "package"
    package.mkdir(parents=True)
    source = package / "helpers.py"
    source.write_text(
        'def target(value):\n    """Return the requested value."""\n    return value\n\n'
        "def other():\n    return 0\n",
        encoding="utf-8",
    )
    tool = CodeEmbeddingTool(
        work_dir=tmp_path / "buffers", device="cpu", use_gpu=False,
        tool_profile="agent_core", embedder=HashEmbedder(),
    )
    result = tool.embed_codebase(root)
    assert result["status"] == "ok"
    yield tool, result["buffer_id"], root, source
    tool.close()


def _path(spelling, root, source):
    return {
        "posix": "package/helpers.py",
        "windows": r"package\helpers.py",
        "project_posix": "project/package/helpers.py",
        "project_windows": r"project\package\helpers.py",
        "dot": "./package/helpers.py",
        "absolute": str(source),
        "absolute_other_separator": str(source).replace("\\", "/") if os.name == "nt" else str(source).replace("/", "\\"),
        "basename": "helpers.py",
    }[spelling]


@pytest.mark.parametrize("spelling", [
    "posix", "windows", "project_posix", "project_windows", "dot",
    "absolute", "absolute_other_separator", "basename",
])
def test_equivalent_paths_work_in_one_read_and_one_small_edit(nested_tool, spelling):
    tool, buf, root, source = nested_tool
    path = _path(spelling, root, source)
    read = tool.code_find("target", buf, file=path)
    assert read["status"] == "ok" and read["editable"]
    assert read["file"] == "package/helpers.py"
    edit = tool.code_edit(
        path, buffer_id=buf, old_text="    return value",
        new_text="    return value + 1", expected_hash=read["file_hash"],
    )
    assert edit["status"] == "ok" and edit["applied"]
    assert edit["file"] == "package/helpers.py"
    assert edit["file_hash"] != read["file_hash"]
    assert "    return value + 1" in source.read_text(encoding="utf-8")
    assert not (root / "project").exists()


def test_file_only_read_and_filename_query(nested_tool):
    tool, buf, _, _ = nested_tool
    assert tool.code_find(file=r"project\package\helpers.py", buffer_id=buf)["editable"]
    assert tool.code_find("project/package/helpers.py", buf)["editable"]
    match = tool.code_find("target in `project/package/helpers.py`", buf)
    assert match["editable"] and match["file"] == "package/helpers.py"
    qualified = tool.code_find("package.helpers.target", buf)
    assert qualified["editable"] and qualified["confidence"] == "exact"
    assert tool.code_find("package.helpers.nonexistent", buf)["editable"] is False


def test_legacy_read_write_and_anchor_chain_share_path_resolution(nested_tool):
    tool, buf, _, source = nested_tool
    read = tool.read_code(buf, file="project/package/helpers.py")
    assert read["status"] == "ok"
    anchored = tool.read_hashlines(buf, r"package\helpers.py", 3, 3)
    write = tool.tool_chain(
        "anchor_apply", buf, file=str(source),
        start_anchor=anchored["lines"][0].split("|", 1)[0],
        end_anchor=anchored["lines"][0].split("|", 1)[0],
        new_lines=["    return value + 2"], expected_hash=anchored["file_hash"],
        dry_run=False,
    )
    assert write["status"] == "ok"
    assert "return value + 2" in source.read_text(encoding="utf-8")


def test_snapshot_native_keys_are_preserved(nested_tool):
    tool, buf, _, _ = nested_tool
    keys = set(tool._load_source_snapshot(buf))
    assert tool.write_code(buf, "project/package/helpers.py", 3, ["    return value + 3"], end_line=3)["status"] == "ok"
    assert set(tool._load_source_snapshot(buf)) == keys
    assert len(keys) == 1


@pytest.mark.parametrize("file", ["../outside.py", "package/../../outside.py"])
def test_traversal_is_rejected_without_mutation(nested_tool, file):
    tool, buf, _, source = nested_tool
    before = source.read_bytes()
    result = tool.code_edit(file, buffer_id=buf, old_text="return value", new_text="return 7")
    assert result["code"] == "path_outside_root"
    assert source.read_bytes() == before


def test_absolute_outside_path_is_not_reinterpreted_as_basename(nested_tool, tmp_path):
    tool, buf, _, source = nested_tool
    outside = tmp_path / "helpers.py"
    outside.write_text("outside = True\n")
    result = tool.code_find(file=str(outside), buffer_id=buf)
    assert result["code"] == "path_outside_root"
    assert "outside = True" not in source.read_text()


def test_ambiguous_basename_has_compact_candidates(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(SourcePathError) as raised:
        resolve_source_path("helper.py", root, ["a/helper.py", r"b\helper.py"])
    assert raised.value.code == "ambiguous_file"
    assert raised.value.candidates == ["a/helper.py", "b/helper.py"]


def test_resolver_accepts_windows_snapshot_keys_on_any_platform(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    key = r"rich\color.py"
    assert resolve_source_path("project/rich/color.py", root, [key]) == key
    assert resolve_source_path("rich/color.py", root, [key]) == key


def test_outside_symlink_is_rejected(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("outside = True\n")
    try:
        (root / "linked.py").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not supported by this host.")
    with pytest.raises(SourcePathError) as raised:
        resolve_source_path("linked.py", root, ["linked.py"])
    assert raised.value.code == "path_outside_root"


def test_small_text_edit_can_use_unique_text_guard_without_hash(nested_tool):
    tool, buf, _, source = nested_tool
    edit = tool.code_edit(
        "project/package/helpers.py", buffer_id=buf,
        old_text="    return value", new_text="    return value + 1",
    )
    assert edit["applied"] and "file_hash" in edit
    assert source.read_text().count("return value + 1") == 1


def test_text_preview_is_non_mutating_and_can_be_applied_unchanged(nested_tool):
    tool, buf, _, source = nested_tool
    original = source.read_bytes()
    snapshot = tool._load_source_snapshot(buf)
    args = {"file": "package/helpers.py", "buffer_id": buf, "old_text": "return value", "new_text": "return value + 1"}
    preview = tool.code_edit(**args, dry_run=True)
    assert preview["status"] == "ok" and preview["applied"] is False
    assert source.read_bytes() == original and tool._load_source_snapshot(buf) == snapshot
    assert tool.code_edit(**args)["applied"]


@pytest.mark.parametrize("old,new", [
    ("missing text", "return 1"),
    ("    ", ""),
    ("return value", "return ("),
])
def test_failed_text_guards_and_syntax_do_not_change_state(nested_tool, old, new):
    tool, buf, _, source = nested_tool
    before = source.read_bytes()
    snapshot = tool._load_source_snapshot(buf)
    result = tool.code_edit("package/helpers.py", buffer_id=buf, old_text=old, new_text=new)
    assert result["status"] == "error" and result["applied"] is False
    assert source.read_bytes() == before and tool._load_source_snapshot(buf) == snapshot


def test_text_edit_noop_and_fresh_hash_avoid_additional_reads(nested_tool):
    tool, buf, _, _ = nested_tool
    first = tool.code_edit("package/helpers.py", buffer_id=buf, old_text="return value", new_text="return value")
    assert first["status"] == "ok" and first["applied"] is False
    second = tool.code_edit(
        "package/helpers.py", buffer_id=buf, old_text="return value",
        new_text="return value + 1", expected_hash=first["file_hash"],
    )
    assert second["applied"]
    third = tool.code_edit(
        "package/helpers.py", buffer_id=buf, old_text="return value + 1",
        new_text="return value + 2", expected_hash=second["file_hash"],
    )
    assert third["applied"]


@pytest.mark.parametrize("lines,old,new,expected", [
    (["a = 1", "b = 2"], "1", "3", ["a = 3", "b = 2"]),
    (["a = 1", "b = 2"], "b = 2", "# before\nb = 2", ["a = 1", "# before", "b = 2"]),
    (["a = 1"], "a = 1", "a = 1\n# after", ["a = 1", "# after"]),
    (["a = 1", "b = 2"], "a = 1\n", "", ["b = 2"]),
    (["a = 1", "b = 2"], "a = 1\r\nb = 2", "a = 3\r\nb = 2", ["a = 3", "b = 2"]),
])
def test_text_replacement_preserves_unrelated_lines(lines, old, new, expected):
    start, end, replacement = text_replacement(lines, old, new)
    assert lines[:start - 1] + replacement + lines[end:] == expected


def test_edit_schema_enforces_exactly_one_edit_mode():
    from jsonschema import Draft202012Validator

    from gigacode.tool_schema import get_schema

    validator = Draft202012Validator(get_schema("code_edit")["input_schema"])
    assert validator.is_valid({"file": "a.py", "old_text": "one", "new_text": "two"})
    assert validator.is_valid({"file": "a.py", "start_anchor": "1:abc", "end_anchor": "1:abc", "new_lines": [], "expected_hash": "hash"})
    assert not validator.is_valid({"file": "a.py"})
    assert not validator.is_valid({"file": "a.py", "old_text": "one", "new_text": "two", "start_anchor": "1:abc"})
    assert not validator.is_valid({"file": "a.py", "old_text": "one", "new_text": "two", "arg_key": "injected"})


def test_xml_argument_path_is_rejected_not_repaired(nested_tool):
    tool, buf, _, source = nested_tool
    before = source.read_bytes()
    result = tool.code_edit(
        "file</arg_key><arg_value>project/package/helpers.py", buffer_id=buf,
        old_text="return value", new_text="return 7",
    )
    assert result["code"] == "invalid_arguments" and "native JSON" in result["message"]
    assert source.read_bytes() == before


def test_search_cache_reused_and_invalidated_after_edit(nested_tool):
    tool, buf, _, _ = nested_tool
    first = tool._source_search_index(buf)
    assert tool._source_search_index(buf) is first
    result = tool.code_edit("package/helpers.py", buffer_id=buf, old_text="target(value)", new_text="renamed(value)")
    assert result["applied"]
    second = tool._source_search_index(buf)
    assert second is not first
    assert tool.code_find("renamed", buf)["editable"]
    assert tool.code_find("target", buf)["editable"] is False


def test_anchor_read_cache_is_not_stale_after_edit(nested_tool):
    tool, buf, _, _ = nested_tool
    first = tool.tool_chain("anchor_read", buf, query="target")
    edit = tool.code_edit("package/helpers.py", buffer_id=buf, old_text="return value", new_text="return value + 1")
    assert edit["applied"]
    second = tool.tool_chain("anchor_read", buf, query="target")
    assert not second.get("cached")
    assert second["editable_range"]["file_hash"] != first["editable_range"]["file_hash"]
