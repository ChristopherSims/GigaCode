"""Regression tests for Phase 2 embedding-provider work.

Covers:
- Caller-supplied embedding providers (2.2).
- Lazy / offline embedding construction (2.1).
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

import numpy as np
import pytest

from gigacode.embedder import Embedder
from gigacode.embedding_provider import EmbeddingProvider, validate_embedding_provider
from gigacode.gigacode_tool import CodeEmbeddingTool


class _KeywordEmbedder:
    """A tiny deterministic provider that needs no ML dependencies."""

    embedding_dim = 8

    def encode(self, texts, batch_size=32):
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            rows.append(rng.random(self.embedding_dim, dtype=np.float32))
        return np.vstack(rows) if rows else np.zeros((0, self.embedding_dim), np.float32)


class TestCallerSuppliedEmbedder:
    def test_provider_satisfies_protocol(self):
        provider = _KeywordEmbedder()
        assert isinstance(provider, EmbeddingProvider)
        validate_embedding_provider(provider)

    def test_invalid_providers_rejected(self):
        with pytest.raises(TypeError):
            validate_embedding_provider(object())

        class NoDim:
            def encode(self, texts):  # pragma: no cover - never called
                return np.zeros((0, 0), np.float32)

        with pytest.raises(TypeError):
            validate_embedding_provider(NoDim())

        class BadDim:
            embedding_dim = 0

            def encode(self, texts):  # pragma: no cover - never called
                return np.zeros((0, 0), np.float32)

        with pytest.raises(TypeError):
            validate_embedding_provider(BadDim())

    def test_tool_uses_supplied_embedder_without_loading_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def alpha():\n    return 1\n")

            provider = _KeywordEmbedder()
            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False, embedder=provider)

            # Dimension is known immediately; no bundled model was loaded.
            assert tool._embedding_dim == 8
            assert tool._supplied_embedder is True

            resp = tool.embed_codebase(str(code_dir))
            assert resp["status"] == "ok", resp

            found = tool.search_for(resp["buffer_id"], "alpha")
            assert found["status"] == "ok"
            assert found.get("total", 0) >= 1

            semantic = tool.semantic_search(resp["buffer_id"], "alpha", top_k=1)
            assert semantic["status"] == "ok", semantic
            tool.close()

    def test_dimension_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def gamma():\n    return 3\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False, embedder=_KeywordEmbedder())
            resp = tool.embed_codebase(str(code_dir))
            assert resp["status"] == "ok", resp
            tool.close()

            class OtherEmbedder:
                embedding_dim = 16

                def encode(self, texts, batch_size=32):
                    return np.ones((len(texts), 16), np.float32)

            with pytest.raises(ValueError):
                CodeEmbeddingTool(work_dir / "tool", use_gpu=False, embedder=OtherEmbedder())


class TestRealDependencyInterface:
    def test_real_sentence_transformers_exposes_used_api(self):
        """When the real dependency is present, assert the API we call exists.

        This guards against test doubles drifting from the production
        interface (skips cleanly when the optional dependency is absent).
        """
        import sentence_transformers

        version = getattr(sentence_transformers, "__version__", "")
        # The conftest shim is a bare ModuleType without a version string.
        if not version:
            pytest.skip("real sentence-transformers not installed")
        from sentence_transformers import SentenceTransformer

        assert hasattr(SentenceTransformer, "get_sentence_embedding_dimension")
        assert hasattr(SentenceTransformer, "encode")


class TestLazyOfflineEmbedder:
    def test_lazy_defers_load_until_encode(self):
        emb = Embedder(lazy=True)
        assert emb.is_loaded is False
        assert emb.embedding_dim == 0
        # Module shim / real model loads on first encode.
        vectors = emb.encode(["hello"])
        assert emb.is_loaded is True
        assert vectors.shape == (1, emb.embedding_dim)
        assert emb.embedding_dim > 0

    def test_offline_options_forwarded(self, monkeypatch):
        import sentence_transformers

        captured = {}

        class FakeModel:
            def __init__(self, name, **kwargs):
                captured["name"] = name
                captured.update(kwargs)

            def get_sentence_embedding_dimension(self):
                return 16

            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 16), dtype=np.float32)

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)

        emb = Embedder(
            model_name="local-model",
            lazy=True,
            local_files_only=True,
            cache_folder="/tmp/model-cache",
        )
        emb.encode(["x"])
        assert captured["local_files_only"] is True
        assert captured["cache_folder"] == "/tmp/model-cache"
        assert emb.embedding_dim == 16

    def test_tool_constructs_lazily_and_loads_on_embed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            code_dir = work_dir / "code"
            code_dir.mkdir()
            (code_dir / "module.py").write_text("def beta():\n    return 2\n")

            tool = CodeEmbeddingTool(work_dir / "tool", use_gpu=False)
            assert tool._embedder.is_loaded is False

            resp = tool.embed_codebase(str(code_dir))
            assert resp["status"] == "ok", resp
            assert tool._embedder.is_loaded is True
            assert tool._embedding_dim > 0
            tool.close()
