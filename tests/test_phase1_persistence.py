"""Regression tests for Phase 1 persistence-safety fixes.

Covers:
- Content-hash conflict detection (external edits that preserve line count).
- Registry survives commit and restart (single-owner persistence).
- Session alias traversal validation.
- Line-ending / final-newline preservation on commit.
- Supported SentenceTransformers dimension API.
"""

import types

try:
    import sklearn

    if getattr(sklearn, "__spec__", None) is None:
        sklearn.__spec__ = types.ModuleSpec("sklearn", getattr(sklearn, "__file__", None))
except Exception:
    pass

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from gigacode.gigacode_tool import CodeEmbeddingTool


def _make_tool(work_dir: Path) -> CodeEmbeddingTool:
    return CodeEmbeddingTool(work_dir / "tool", use_gpu=False)


class TestConflictDetection:
    def test_external_edit_same_line_count_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            target = code_dir / "module.py"
            target.write_text("a = 1\nb = 2\n")

            tool = _make_tool(work_dir)
            resp = tool.embed_codebase(str(code_dir))
            buffer_id = resp["buffer_id"]

            # Buffer edit: replace the second line.
            write = tool.write_code(buffer_id, "module.py", 2, ["b = 3\n"], end_line=2)
            assert write["status"] == "ok"

            # External edit with the same number of lines but different content.
            target.write_text("a = 9\nb = 8\n")

            commit = tool.commit(buffer_id, check_impact=False)
            assert commit["status"] == "conflict", commit
            assert any(c["file"] == "module.py" for c in commit["conflict_files"])

            # The external edit must not have been overwritten.
            on_disk = target.read_text()
            assert "a = 9" in on_disk
            assert "b = 8" in on_disk
            tool.close()

    def test_matching_content_still_commits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            target = code_dir / "module.py"
            target.write_text("a = 1\nb = 2\n")

            tool = _make_tool(work_dir)
            buffer_id = tool.embed_codebase(str(code_dir))["buffer_id"]
            tool.write_code(buffer_id, "module.py", 1, ["a = 10\n"], end_line=1)

            commit = tool.commit(buffer_id, check_impact=False)
            assert commit["status"] == "ok", commit
            assert "a = 10" in target.read_text()
            tool.close()


class TestRegistryPersistence:
    def test_registry_survives_commit_and_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def add(a, b):\n    return a + b\n")

            tool = _make_tool(work_dir)
            buffer_id = tool.embed_codebase(str(code_dir))["buffer_id"]
            tool.write_code(buffer_id, "module.py", 1, ["# header\n"], end_line=0)
            commit = tool.commit(buffer_id, check_impact=False)
            assert commit["status"] == "ok", commit
            tool.close()

            registry_path = work_dir / "tool" / "registry.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            assert buffer_id in registry, "registry was overwritten by a stale copy"

            reopened = _make_tool(work_dir)
            ids = [b["buffer_id"] for b in reopened.list_buffers()["buffers"]]
            assert buffer_id in ids

            # The index must be searchable again after a restart.
            search = reopened.semantic_search(buffer_id, "header", top_k=2)
            assert search["status"] == "ok", search
            reopened.close()


class TestSessionAliasValidation:
    def test_traversal_alias_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("x = 1\n")

            tool = _make_tool(work_dir)
            buffer_id = tool.embed_codebase(str(code_dir))["buffer_id"]
            registry_path = work_dir / "tool" / "registry.json"
            before = registry_path.read_text(encoding="utf-8")

            for alias in ("../registry", "..\\registry", "a/b", "a\\b", "..", ""):
                save = tool.save_session(alias, [buffer_id])
                assert save["status"] == "error", alias
                load = tool.load_session(alias)
                assert load["status"] == "error", alias

            assert registry_path.read_text(encoding="utf-8") == before
            tool.close()

    def test_valid_alias_round_trips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            tool = _make_tool(work_dir)

            saved = tool.save_session("my-session", ["buf-1"])
            assert saved["status"] == "ok"
            loaded = tool.load_session("my-session")
            assert loaded["status"] == "ok"
            assert loaded["buffer_ids"] == ["buf-1"]
            tool.close()


class TestLineEndingPreservation:
    @pytest.mark.parametrize(
        "raw,expect_crlf,final_newline",
        [
            (b"a = 1\nb = 2\n", False, True),
            (b"a = 1\r\nb = 2\r\n", True, True),
            (b"a = 1\nb = 2", False, False),
        ],
    )
    def test_commit_preserves_line_endings(self, raw, expect_crlf, final_newline):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            target = code_dir / "module.py"
            target.write_bytes(raw)

            tool = _make_tool(work_dir)
            buffer_id = tool.embed_codebase(str(code_dir))["buffer_id"]
            write = tool.write_code(buffer_id, "module.py", 1, ["a = 10\n"], end_line=1)
            assert write["status"] == "ok"

            commit = tool.commit(buffer_id, check_impact=False)
            assert commit["status"] == "ok", commit

            result = target.read_bytes()
            assert (b"\r\n" in result) is expect_crlf, result
            assert result.endswith(b"\n") is final_newline, result
            assert b"a = 10" in result
            tool.close()


class TestPostEditIndexing:
    def test_search_reflects_committed_edit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def old_func():\n    return 1\n")

            tool = _make_tool(work_dir)
            buffer_id = tool.embed_codebase(str(code_dir))["buffer_id"]

            tool.write_code(buffer_id, "module.py", 1, ["def new_func():\n"], end_line=1)
            commit = tool.commit(buffer_id, check_impact=False)
            assert commit["status"] == "ok", commit

            new = tool.search_for(buffer_id, "new_func")
            assert new["status"] == "ok"
            assert new.get("total", 0) >= 1

            old = tool.search_for(buffer_id, "old_func")
            assert old.get("total", 0) == 0

            semantic = tool.semantic_search(buffer_id, "new_func", top_k=5)
            assert semantic["status"] == "ok"

            hybrid = tool.hybrid_search(buffer_id, "new_func", top_k=2)
            assert hybrid["status"] == "ok", hybrid
            tool.close()

    def test_reload_updates_indexed_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            target = code_dir / "module.py"
            target.write_text("def first_func():\n    return 1\n")

            tool = _make_tool(work_dir)
            buffer_id = tool.embed_codebase(str(code_dir))["buffer_id"]

            target.write_text("def second_func():\n    return 2\n")
            reload_result = tool.reload_codebase(buffer_id)
            assert reload_result["status"] == "ok", reload_result

            second = tool.search_for(buffer_id, "second_func")
            assert second.get("total", 0) >= 1
            first = tool.search_for(buffer_id, "first_func")
            assert first.get("total", 0) == 0
            tool.close()


class TestEmbedderInterface:
    def test_uses_supported_dimension_api(self, monkeypatch):
        import sentence_transformers

        import gigacode.embedder as embedder_mod

        class StrictModel:
            def __init__(self, name, device=None):
                self.name = name

            def get_sentence_embedding_dimension(self):
                return 768

            def encode(
                self,
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            ):
                return np.ones((len(texts), 768), dtype=np.float32)

        def factory(*args, **kwargs):
            return StrictModel(*args, **kwargs)

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", factory)

        emb = embedder_mod.Embedder(model_name="strict-test-model")
        assert emb.embedding_dim == 768
        out = emb.encode(["hello"])
        assert out.shape == (1, 768)
