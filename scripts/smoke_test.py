"""Functional smoke test for an installed GigaCode distribution.

Exercises a complete core workflow without the heavy embedding stack by
supplying a tiny deterministic embedder, so it can run against a core-only
wheel in CI:

    python scripts/smoke_test.py

Exits non-zero on any failure.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

# Prefer an installed distribution (CI runs against a built wheel).  Fall back
# to the source tree for local development.
if importlib.util.find_spec("gigacode") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _HashEmbedder:
    """Dependency-free deterministic embedding provider for smoke tests."""

    embedding_dim = 16

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            rows.append(rng.random(self.embedding_dim, dtype=np.float32))
        return np.vstack(rows) if rows else np.zeros((0, self.embedding_dim), np.float32)

    def embed(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def main() -> int:
    from gigacode import CodeEmbeddingTool

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        code = root / "code"
        code.mkdir()
        (code / "sample.py").write_text("def add(a, b):\n    return a + b\n")

        work_dir = root / "buffers"
        tool = CodeEmbeddingTool(work_dir, use_gpu=False, embedder=_HashEmbedder())
        resp = tool.embed_codebase(str(code))
        assert resp["status"] == "ok", resp
        buffer_id = resp["buffer_id"]

        found = tool.search_for(buffer_id, "add")
        assert found.get("total", 0) >= 1, found

        write = tool.write_code(buffer_id, "sample.py", 1, ["def add(a, b):\n"], end_line=1)
        assert write["status"] == "ok", write

        assert tool.diff(buffer_id)["status"] in {"ok", "conflict"}
        commit = tool.commit(buffer_id, check_impact=False)
        assert commit["status"] == "ok", commit
        tool.close()

        # Buffer registration must survive a restart.
        reopened = CodeEmbeddingTool(work_dir, use_gpu=False, embedder=_HashEmbedder())
        ids = [b["buffer_id"] for b in reopened.list_buffers()["buffers"]]
        assert buffer_id in ids, ids
        reopened.close()

    print("smoke_test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
