"""GigaCode embedding backend modules."""

import importlib.util
import sys

# Ensure sklearn.__spec__ is set using spec_from_file_location (real loader, no broken None-loader spec)
try:
    import sklearn as _sk

    if _sk.__spec__ is None:
        _spec = importlib.util.spec_from_file_location(
            "sklearn",
            _sk.__file__,
            submodule_search_locations=list(_sk.__path__),
        )
        if _spec is not None:
            _sk.__spec__ = _spec
            sys.modules["sklearn"].__spec__ = _spec
    del _sk
except Exception:
    pass

import os

os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCH_COMPILE_DEBUG"] = "0"

__version__ = "0.6.0"
__all__ = [
    "tokenizer",
    "embedder",
    "flatten",
    "diff_engine",
    "size_guard",
    "metadata_store",
]
