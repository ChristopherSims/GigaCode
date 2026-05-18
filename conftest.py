"""Repository-root pytest configuration.

This file exists to install warning filters as early as possible in pytest's
startup sequence, before test modules import optional third-party dependencies
like FAISS.
"""

from __future__ import annotations

import warnings


warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPyPacked has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPyObject has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)


try:
    import faiss  # noqa: F401
except Exception:
    pass