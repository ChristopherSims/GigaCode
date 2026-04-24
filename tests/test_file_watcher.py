"""Tests for src.file_watcher."""

import tempfile
from pathlib import Path
from src.file_watcher import BufferWatcher


class FakeTool:
    def __init__(self, root):
        self._root = root
        self.calls = []

    def _get_buffer_info(self, buffer_id):
        return {"root": str(self._root), "pattern": "*.py"}

    def reload_codebase(self, buffer_id):
        self.calls.append(buffer_id)


def test_polling_watcher():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "test.py").write_text("x = 1")
        tool = FakeTool(root)
        watcher = BufferWatcher(tool)
        watcher.watch_buffer("buf1")
        # Modify file
        (root / "test.py").write_text("x = 2")
        import time
        time.sleep(3)
        watcher.stop_all()
        assert "buf1" in tool.calls
