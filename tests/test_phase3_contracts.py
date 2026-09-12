"""Regression tests for Phase 3 contract fixes.

Covers:
- Documented top-level ``from gigacode import CodeEmbeddingTool`` import.
- TestRunner never reports success when tests did not run.
- MCP tool results carry ``isError``.
- Server authentication wiring (FastAPI app and CLI plumbing).
"""

import types

try:
    import sklearn

    if getattr(sklearn, "__spec__", None) is None:
        sklearn.__spec__ = types.ModuleSpec("sklearn", getattr(sklearn, "__file__", None))
except Exception:
    pass

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gigacode.test_runner import TestRunner as _TestRunner


class TestTopLevelImport:
    def test_code_embedding_tool_importable(self):
        from gigacode import CodeEmbeddingTool

        assert CodeEmbeddingTool.__name__ == "CodeEmbeddingTool"


class TestTestRunnerStatus:
    def test_failing_test_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "test_sample.py").write_text(
                "def test_ok():\n    assert True\n\ndef test_bad():\n    assert False\n"
            )
            runner = _TestRunner([], root, "python")
            summary = runner.run_tests(["test_sample.py"])
            assert summary.status == "ok"
            assert summary.failed >= 1
            assert summary.total >= 2

    def test_no_tests_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "test_empty.py").write_text("# no tests here\n")
            runner = _TestRunner([], root, "python")
            summary = runner.run_tests(["test_empty.py"])
            assert summary.status in {"no_tests", "error"}
            assert summary.status != "ok"
            assert summary.total == 0

    def test_missing_interpreter_is_not_success(self, monkeypatch):
        import subprocess

        def boom(*args, **kwargs):
            raise FileNotFoundError("python not found")

        monkeypatch.setattr(subprocess, "run", boom)
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = _TestRunner([], Path(tmpdir), "python")
            summary = runner.run_tests(["test_missing.py"])
            assert summary.status == "skipped"
            assert summary.total == 0


class TestMcpResults:
    def test_success_and_failure_results(self):
        mcp_server = pytest.importorskip("gigacode.mcp_server")
        if not mcp_server._HAS_MCP:
            pytest.skip("MCP SDK not installed")

        ok = mcp_server._success_result({"status": "ok", "value": 1})
        assert ok.isError is False
        assert ok.structuredContent == {"status": "ok", "value": 1}

        domain_error = mcp_server._success_result({"status": "error", "message": "nope"})
        assert domain_error.isError is True

        failure = mcp_server._failure_result("boom")
        assert failure.isError is True
        assert failure.content[0].text
        assert failure.structuredContent["status"] == "error"


class TestServerAuth:
    def test_run_server_forwards_api_key(self, monkeypatch):
        import gigacode.gigacode_server as server_mod

        captured = {}

        def fake_fastapi(tool, host, port, api_key=None):
            captured["api_key"] = api_key

        monkeypatch.setattr(server_mod, "_run_fastapi", fake_fastapi)
        server_mod.run_server(object(), use_fastapi=True, api_key="secret")
        assert captured["api_key"] == "secret"

    def test_production_app_requires_key(self):
        fastapi_testclient = pytest.importorskip("fastapi.testclient")
        from gigacode.gigacode_api import create_production_app

        class DummyTool:
            def get_tool_schemas(self):
                return [{"name": "ping"}]

        app = create_production_app(DummyTool(), api_key="secret")
        client = fastapi_testclient.TestClient(app)

        assert client.get("/health").status_code == 200
        assert client.get("/schemas").status_code == 401
        assert client.get("/schemas", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/schemas", headers={"X-API-Key": "secret"}).status_code == 200
