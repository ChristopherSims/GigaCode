"""Pytest configuration for GigaCode tests."""

# CRITICAL: Suppress sklearn.__spec__ errors in torch._dynamo by catching them early
import importlib.util
import sys
import types
import warnings
from typing import Optional

import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r"builtin type (SwigPyPacked|SwigPyObject|swigvarlink) has no __module__ attribute",
    category=DeprecationWarning,
)

# Pre-import sklearn first
try:
    import sklearn
except Exception:
    sklearn = None


def _ensure_module(name: str) -> types.ModuleType:
    """Install a lightweight module shim into sys.modules if missing."""
    mod = sys.modules.get(name)
    if isinstance(mod, types.ModuleType):
        return mod

    shim = types.ModuleType(name)
    shim.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    sys.modules[name] = shim
    return shim


_ensure_module("sklearn")
metrics_mod = _ensure_module("sklearn.metrics")
pairwise_mod = _ensure_module("sklearn.metrics.pairwise")
_ensure_module("sklearn.decomposition")
_ensure_module("sklearn.feature_extraction")
_ensure_module("sklearn.feature_extraction.text")


def _pairwise_distances(x, y=None, metric: str = "euclidean"):
    """Small test shim for sklearn.metrics.pairwise_distances."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = x_arr if y is None else np.asarray(y, dtype=float)

    if metric == "cosine":
        x_norm = np.linalg.norm(x_arr, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y_arr, axis=1, keepdims=True)
        x_safe = np.divide(x_arr, np.maximum(x_norm, 1e-12))
        y_safe = np.divide(y_arr, np.maximum(y_norm, 1e-12))
        similarity = x_safe @ y_safe.T
        return 1.0 - similarity

    diff = x_arr[:, None, :] - y_arr[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _cosine_similarity(x, y=None):
    """Small test shim for sklearn.metrics.pairwise.cosine_similarity."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = x_arr if y is None else np.asarray(y, dtype=float)
    x_norm = np.linalg.norm(x_arr, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y_arr, axis=1, keepdims=True)
    x_safe = np.divide(x_arr, np.maximum(x_norm, 1e-12))
    y_safe = np.divide(y_arr, np.maximum(y_norm, 1e-12))
    return x_safe @ y_safe.T


metrics_mod.pairwise_distances = _pairwise_distances
pairwise_mod.pairwise_distances = _pairwise_distances
pairwise_mod.cosine_similarity = _cosine_similarity


def _install_sentence_transformers_shim() -> None:
    """Install a tiny SentenceTransformer shim when the real package is unavailable."""
    module_name = "sentence_transformers"
    shim = types.ModuleType(module_name)
    shim.__spec__ = importlib.util.spec_from_loader(module_name, loader=None)

    class SentenceTransformer:
        def __init__(self, model_name: str, device: Optional[str] = None):
            self.model_name = model_name
            self.device = device
            self._embedding_dim = 384

        def get_embedding_dimension(self) -> int:
            return self._embedding_dim

        def encode(
            self,
            texts,
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
        ):
            rows = []
            for text in texts:
                seed = abs(hash(text)) % (2**32)
                rng = np.random.default_rng(seed)
                rows.append(rng.random(self._embedding_dim, dtype=np.float32))
            result = (
                np.vstack(rows) if rows else np.zeros((0, self._embedding_dim), dtype=np.float32)
            )
            return result if convert_to_numpy else result.tolist()

    shim.SentenceTransformer = SentenceTransformer
    sys.modules[module_name] = shim


try:
    import sentence_transformers  # noqa: F401
except Exception:
    _install_sentence_transformers_shim()

# Add parent directory to path
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Session fixture
import pytest

# Patch importlib.util.find_spec to catch ValueError when __spec__ is None
_real_find_spec = importlib.util.find_spec


def _find_spec_wrapper(name, package=None):
    """Wrapper that suppresses sklearn.__spec__ errors."""
    # If the module is in sys.modules with __spec__=None, fix it before calling original
    full_name = name
    mod = sys.modules.get(full_name)
    if mod is not None and getattr(mod, "__spec__", None) is None:
        try:
            file_ = getattr(mod, "__file__", None)
            path_ = getattr(mod, "__path__", None)
            if file_:
                spec = importlib.util.spec_from_file_location(
                    full_name,
                    file_,
                    submodule_search_locations=list(path_) if path_ is not None else None,
                )
                if spec is not None:
                    mod.__spec__ = spec
        except Exception:
            pass
    try:
        return _real_find_spec(name, package)
    except ValueError as _e:
        if "__spec__" in str(_e):
            return None  # Suppress: treat as "not found"
        raise


# Apply patch
importlib.util.find_spec = _find_spec_wrapper

# Also patch the actual check that raises the error
try:
    import torch._dynamo.trace_rules

    # Store the original
    _original_module_dict_lookup = getattr(torch._dynamo.trace_rules, "_module_dict_lookup", None)

    # Create a wrapper that doesn't raise ValueError for sklearn
    if _original_module_dict_lookup:

        def _safe_module_dict_lookup(*args, **kwargs):
            try:
                return _original_module_dict_lookup(*args, **kwargs)
            except ValueError as e:
                if "sklearn.__spec__" in str(e):
                    return None  # Return None instead of raising
                raise

        torch._dynamo.trace_rules._module_dict_lookup = _safe_module_dict_lookup
except Exception:
    pass  # torch may not be imported yet


def pytest_configure(config):
    """Hook that runs before test collection."""
    import os

    os.environ["TORCH_COMPILE"] = "0"


@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """Setup test environment."""
    try:
        import torch

        torch.set_grad_enabled(False)
        torch.compiler.disable()
    except Exception:
        pass
    yield
