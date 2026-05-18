"""Focused integration tests for undo/redo tracking."""

from __future__ import annotations

import tempfile
from pathlib import Path

from gigacode.gigacode_tool import CodeEmbeddingTool


def test_write_code_records_operation_and_supports_undo_redo() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        code_dir = work_dir / "code"
        code_dir.mkdir()
        (code_dir / "module.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

        tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
        embed_result = tool.embed_codebase(str(code_dir))
        assert embed_result.get("status") == "ok"
        buffer_id = embed_result["buffer_id"]

        write_result = tool.write_code(
            buffer_id,
            "module.py",
            "def add(a, b):\n    return a - b\n",
        )
        assert write_result.get("status") == "ok"
        assert write_result.get("operation_id")

        changed = tool.read_code(buffer_id, "module.py")
        assert changed.get("status") == "ok"
        assert "return a - b" in "\n".join(changed.get("lines", []))

        undo_result = tool.undo(buffer_id, steps=1)
        assert undo_result.get("status") == "ok"
        assert undo_result.get("steps_undone") == 1

        reverted = tool.read_code(buffer_id, "module.py")
        assert reverted.get("status") == "ok"
        assert "return a + b" in "\n".join(reverted.get("lines", []))

        redo_result = tool.redo(buffer_id, steps=1)
        assert redo_result.get("status") == "ok"
        assert redo_result.get("steps_redone") == 1

        redone = tool.read_code(buffer_id, "module.py")
        assert redone.get("status") == "ok"
        assert "return a - b" in "\n".join(redone.get("lines", []))

        tool.close()