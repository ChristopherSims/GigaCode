"""Runs the documented examples so docs cannot drift from the code.

Covers the README quick start, the multiple advertised import paths, schema
export examples, CLI help, and installed-package configuration examples.
"""

import types

try:
    import sklearn

    if getattr(sklearn, "__spec__", None) is None:
        sklearn.__spec__ = types.ModuleSpec("sklearn", getattr(sklearn, "__file__", None))
except Exception:
    pass

import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gigacode import CodeEmbeddingTool


class TestDocumentedImports:
    def test_top_level_import(self):
        assert CodeEmbeddingTool is not None

    def test_module_import(self):
        from gigacode.gigacode_tool import CodeEmbeddingTool as FromModule

        assert FromModule is CodeEmbeddingTool


class TestReadmeQuickStart:
    def test_quickstart_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src"
            src.mkdir()
            (src / "main.py").write_text(
                "def handler():\n    return 1\n\ndef middleware(request):\n    return request\n"
            )

            with CodeEmbeddingTool(work_dir=str(root / "buffers"), device="cpu") as tool:
                result = tool.embed_codebase(str(src), pattern="*.py")
                assert result["status"] == "ok", result
                buf_id = result["buffer_id"]

                search = tool.semantic_search(buf_id, "middleware", top_k=5)
                assert search["status"] == "ok", search

                write = tool.write_code(
                    buf_id, file="main.py", start_line=1, new_lines=["def handler():\n"]
                )
                assert write["status"] == "ok", write

                diff = tool.diff(buf_id)
                assert diff["status"] in {"ok", "conflict"}

                commit = tool.commit(buf_id, check_impact=False)
                assert commit["status"] == "ok", commit


class TestSchemaExportExamples:
    def test_all_documented_formats(self):
        from gigacode.tool_schema import export_schemas

        for fmt in ("openai", "anthropic", "mcp", "ollama"):
            tools = export_schemas(fmt)
            assert tools, fmt

        read_only = export_schemas("anthropic", category="security", read_only_only=True)
        assert isinstance(read_only, list)

    def test_example_config_ships_with_package(self):
        example = resources.files("gigacode").joinpath("gigacode.toml.example")
        assert example.is_file()


class TestCliHelp:
    def test_server_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "gigacode.gigacode_server", "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "--api-key" in proc.stdout
        assert "--tool-profile" in proc.stdout

    def test_mcp_help(self):
        pytest.importorskip("mcp")
        proc = subprocess.run(
            [sys.executable, "-m", "gigacode.mcp_server", "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "--transport" in proc.stdout
        assert "streamable-http" in proc.stdout
