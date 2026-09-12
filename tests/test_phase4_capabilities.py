"""Phase 4 tests: capability surface honesty and working profile tools."""

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

from gigacode.gigacode_tool import CodeEmbeddingTool


def _embed_tool(tmpdir: str) -> tuple[CodeEmbeddingTool, str]:
    work_dir = Path(tmpdir)
    code_dir = work_dir / "code"
    code_dir.mkdir()
    lines = ["def add(a, b):", "    return a + b", ""]
    lines.append("class Calculator:")
    for i in range(40):
        lines.append(f"    def method_{i}(self, value):")
        lines.append(f"        result = value + {i}")
        lines.append("        return result")
    (code_dir / "module.py").write_text("\n".join(lines) + "\n")
    tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False, tool_profile="full")
    resp = tool.embed_codebase(str(code_dir))
    assert resp["status"] == "ok", resp
    return tool, resp["buffer_id"]


class TestCapabilitySurface:
    def test_cluster_code_reports_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool, buffer_id = _embed_tool(tmpdir)
            result = tool.cluster_code(buffer_id)
            assert result["status"] == "unavailable", result
            assert result.get("message")
            tool.close()

    def test_agent_profile_tools_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool, buffer_id = _embed_tool(tmpdir)

            set_result = tool.set_agent_profile(buffer_id, "debugger")
            assert set_result["status"] == "ok", set_result

            chunk_result = tool.chunk_with_profile(buffer_id, "debugger")
            assert chunk_result["status"] == "ok", chunk_result
            assert chunk_result["total_chunks"] >= 1

            adapt_result = tool.adapt_search(buffer_id, "auth", "debugger")
            assert adapt_result["status"] == "ok", adapt_result
            assert adapt_result["enhanced_query"]
            tool.close()

    def test_solve_constructs_without_type_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool, buffer_id = _embed_tool(tmpdir)
            result = tool.solve(buffer_id, "rename add to sum", max_iterations=1)
            # Must return a structured result, never raise.
            assert isinstance(result, dict)
            assert "status" in result
            tool.close()
