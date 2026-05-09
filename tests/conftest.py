"""Pytest configuration for GigaCode tests."""

# CRITICAL: Suppress sklearn.__spec__ errors in torch._dynamo by catching them early
import sys
import importlib.util

# Pre-import sklearn first
try:
    import sklearn
except Exception:
    sklearn = None

# Patch importlib.util.find_spec to catch ValueError when __spec__ is None
_real_find_spec = importlib.util.find_spec

def _find_spec_wrapper(name, package=None):
    """Wrapper that suppresses sklearn.__spec__ errors."""
    # If the module is in sys.modules with __spec__=None, fix it before calling original
    full_name = name
    mod = sys.modules.get(full_name)
    if mod is not None and getattr(mod, '__spec__', None) is None:
        try:
            file_ = getattr(mod, '__file__', None)
            path_ = getattr(mod, '__path__', None)
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
        if '__spec__' in str(_e):
            return None  # Suppress: treat as "not found"
        raise

# Apply patch
importlib.util.find_spec = _find_spec_wrapper

# Also patch the actual check that raises the error
try:
    import torch._dynamo.trace_rules
    
    # Store the original
    _original_module_dict_lookup = getattr(torch._dynamo.trace_rules, '_module_dict_lookup', None)
    
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


# Add parent directory to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Session fixture
import pytest

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
