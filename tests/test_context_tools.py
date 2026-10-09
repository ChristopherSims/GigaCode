"""Public context APIs must work cold, stay bounded and track live source/Git."""

import json
import subprocess

import pytest

from gigacode.gigacode_tool import CodeEmbeddingTool
from scripts.mcp_bench_server import HashEmbedder


@pytest.fixture
def context_tool(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "auth.py").write_text(
        '"""JWT authentication."""\nfrom fastapi import APIRouter\n'
        "class AuthService:\n"
        "    def login(self, user):\n        return user\n"
        "    def forward(self, user):\n        return self.login(user)\n",
        encoding="utf-8",
    )
    (root / "routes.py").write_text(
        "from auth import AuthService\n"
        "def login():\n    return AuthService()\n",
        encoding="utf-8",
    )
    (root / "tests").mkdir()
    (root / "tests/test_auth.py").write_text("from auth import AuthService\n")
    tool = CodeEmbeddingTool(tmp_path / "buffers", use_gpu=False, device="cpu",
                             tool_profile="agent_core", embedder=HashEmbedder())
    yield tool, root
    tool.close()


def test_cold_navigation_does_not_embed(context_tool, monkeypatch):
    tool, root = context_tool
    monkeypatch.setattr(tool._embedder, "encode", lambda *a, **kw: pytest.fail("Must not embed"))
    result = tool.get_file_structure("auth.py", root=str(root))
    assert result["status"] == "ok" and len(result["items"]) == 3
    assert tool._last_buffer_id is None
    read = tool.read_symbol("auth.py", "AuthService.login", include=["callers"], root=str(root))
    assert read["source"].startswith("    def login")
    assert "def forward" not in read["source"]
    assert read["callers"]["items"][0]["symbol"] == "AuthService.forward"
    assert tool.get_children("AuthService", root=str(root))["total"] == 2
    assert "source" not in tool.get_symbol("AuthService", root=str(root))


def test_symbol_ambiguity_returns_candidates_not_a_guessed_body(context_tool):
    tool, root = context_tool
    result = tool.code_navigate("symbol", symbol="login", include_source=True, root=str(root))
    assert result["status"] == "ambiguous" and result["candidates"]["total"] == 2
    assert "source" not in result
    assert tool.get_symbol("Missing.login", root=str(root))["status"] == "error"


def test_dependency_apis_and_task_broker(context_tool):
    tool, root = context_tool
    assert tool.dependencies("routes.py", root=str(root))["items"][0]["files"] == ["auth.py"]
    assert tool.dependents("auth.py", root=str(root))["total"] == 2
    graph = tool.dependency_graph("routes.py", root=str(root))
    assert "auth.py" in graph["nodes"] and graph["edges"]
    result = tool.get_task_context("Add OAuth authentication to the API", root=str(root), include_git=False)
    assert result["status"] == "ok"
    assert result["likely_files"][0]["file"] == "auth.py"
    assert result["relevant_tests"] == ["tests/test_auth.py"]
    cache = list(tool.work_dir.glob("navigation/*/navigation_summaries.json"))
    assert len(cache) == 1
    assert json.loads(cache[0].read_text())["revision"] == result["revision"]
    assert not (root / ".ai").exists()


def test_index_refreshes_on_external_edit_and_deletion(context_tool):
    tool, root = context_tool
    first = tool.get_symbol("AuthService.login", root=str(root))
    index = tool._context_index(root=str(root))
    parses = index.parse_count
    assert tool.get_symbol("AuthService.login", root=str(root))["source_hash"] == first["source_hash"]
    assert index.parse_count == parses
    (root / "auth.py").write_text("def oauth():\n    return True\n")
    second = tool.get_symbol("oauth", root=str(root))
    assert second["status"] == "ok" and second["source_hash"] != first["source_hash"]
    assert index.parse_count == parses + 1
    assert tool.get_symbol("AuthService.login", root=str(root))["status"] == "error"
    (root / "auth.py").unlink()
    assert tool.get_symbol("oauth", root=str(root))["status"] == "error"


@pytest.mark.parametrize("file", ["../outside.py", "project/../../outside.py"])
def test_navigation_cannot_escape_root(context_tool, file):
    tool, root = context_tool
    assert tool.get_file_structure(file, root=str(root))["status"] == "error"


def test_pagination_and_invalid_limits(context_tool):
    tool, root = context_tool
    first = tool.get_file_structure("auth.py", limit=1, root=str(root))
    assert first["next_offset"] == 1 and len(first["items"]) == 1
    second = tool.get_file_structure("auth.py", limit=1, offset=1, root=str(root))
    assert second["items"][0]["symbol"] == "AuthService.login"
    assert tool.get_file_structure("auth.py", limit=0, root=str(root))["status"] == "error"
    assert tool.get_task_context("", root=str(root))["status"] == "error"


def test_pending_buffer_changes_are_indexed_without_overwriting_disk(context_tool):
    tool, root = context_tool
    result = tool.embed_codebase(root)
    buf = result["buffer_id"]
    assert tool.write_code(buf, "routes.py", 2, ["def renamed_login():", "    return AuthService()"], end_line=3)["status"] == "ok"
    assert tool.get_symbol("renamed_login", buffer_id=buf)["status"] == "ok"
    assert "def login" in (root / "routes.py").read_text()
    assert tool.get_symbol("login", file="routes.py", buffer_id=buf)["status"] == "error"


def git(root, *args, check=True):
    return subprocess.run(["git", "-C", str(root), *args], check=check, capture_output=True,
                          text=True, timeout=10)


@pytest.fixture
def git_context(context_tool):
    tool, root = context_tool
    git(root, "init", "--quiet")
    # Use the environment's configured identity; never set/override it.
    if git(root, "var", "GIT_AUTHOR_IDENT", check=False).returncode:
        pytest.skip("Git author identity is not configured in this environment.")
    git(root, "add", ".")
    assert set(git(root, "diff", "--cached", "--name-only").stdout.splitlines()) == {
        "auth.py", "routes.py", "tests/test_auth.py",
    }
    assert all(line.startswith("A ") for line in git(root, "status", "--porcelain").stdout.splitlines())
    git(root, "commit", "--quiet", "-m",
        "Add context test fixtures\n\nCo-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>")
    return tool, root


def test_git_status_diff_history_and_blame(git_context):
    tool, root = git_context
    (root / "auth.py").write_text((root / "auth.py").read_text().replace("return user", "return bool(user)"))
    status = tool.git_status(root=str(root))
    assert status["status"] == "ok" and status["modified"] == ["auth.py"]
    assert "return bool(user)" in tool.git_diff(file="auth.py", root=str(root))["diff"]
    assert tool.git_diff(file="auth.py", staged=True, root=str(root))["diff"] == ""
    assert tool.recent_commits(root=str(root))["items"][0]["subject"] == "Add context test fixtures"
    assert tool.file_history("auth.py", root=str(root))["items"]
    provenance = tool.blame("auth.py", line=1, limit=1, root=str(root))
    assert len(provenance["items"]) == 1 and "author" not in provenance["items"][0]


def test_changed_symbols_distinguishes_staged_worktree_and_deletions(git_context):
    tool, root = git_context
    path = root / "auth.py"
    path.write_text(path.read_text().replace("return user", "return bool(user)"))
    git(root, "add", "auth.py")
    (root / "new.py").write_text("def added():\n    return True\n")
    (root / "routes.py").unlink()
    staged = tool.changed_symbols(staged=True, root=str(root))
    assert staged["status"] == "ok"
    assert {r["after"]["symbol"] for r in staged["items"] if r["after"]} == {"AuthService", "AuthService.login"}
    result = tool.changed_symbols(root=str(root))
    assert any(r["change"] == "added" and r["after"]["symbol"] == "added" for r in result["items"])
    assert any(r["change"] == "deleted" and r["before"]["symbol"] == "login" for r in result["items"])
    assert tool.git_diff(against="--output=outside", root=str(root))["status"] == "error"


def test_git_status_preserves_spaces_and_scopes_nested_roots(git_context):
    tool, root = git_context
    child = root / "nested"
    child.mkdir()
    (child / "space name.py").write_text("value = 1\n")
    (root / "outside.py").write_text("value = 2\n")
    result = tool.git_status(root=str(child))
    assert result["untracked"] == ["space name.py"]
    assert result["items"][0]["file"] == "space name.py"


def test_graph_cycles_are_bounded(context_tool):
    tool, root = context_tool
    (root / "one.py").write_text("import two\n")
    (root / "two.py").write_text("import one\n")
    result = tool.dependency_graph("one.py", limit=1, root=str(root))
    assert result["nodes"] == ["one.py"] and result["truncated"]


def test_renamed_git_symbols_report_both_sides(git_context):
    tool, root = git_context
    git(root, "mv", "routes.py", "renamed routes.py")
    status = tool.git_status(root=str(root))["items"][0]
    assert status["file"] == "renamed routes.py" and status["previous_file"] == "routes.py"
    changes = tool.changed_symbols(root=str(root))
    assert any(r["change"] == "deleted" and r["before"]["file"] == "routes.py" for r in changes["items"])
    assert any(r["change"] == "added" and r["after"]["file"] == "renamed routes.py" for r in changes["items"])


def test_nested_git_diff_scopes_repo_relative_paths(git_context):
    from gigacode.context_tools import GitContext

    _, root = git_context
    nested = root / "tests"
    path = nested / "test_auth.py"
    path.write_text(path.read_text() + "changed = True\n")
    git_context_reader = GitContext(nested)
    assert "changed = True" in git_context_reader.diff("test_auth.py")["diff"]
    assert git_context_reader.commits(file="test_auth.py")["items"]


def test_all_context_methods_are_discoverable_in_read_only_profile(context_tool):
    from gigacode.navigation_schema import NAVIGATION_TOOL_NAMES
    from gigacode.tool_schema import get_profile_tool_names

    tool, _ = context_tool
    assert NAVIGATION_TOOL_NAMES <= get_profile_tool_names("read_only")
    names = {s["name"] for s in tool.get_exposed_tool_schemas()}
    assert NAVIGATION_TOOL_NAMES <= names
