"""Repository-root pytest configuration.

This file exists to install warning filters as early as possible in pytest's
startup sequence, before test modules import optional third-party dependencies
like FAISS.
"""

from __future__ import annotations

import os
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

# Unit tests never want the session auto-embed: it would index the repo the
# test process happens to run in. Individual auto-embed tests re-enable it
# explicitly (monkeypatch env around pytest-marked fixtures).
os.environ.setdefault("GIGACODE_AUTO_EMBED", "off")

try:
    import faiss  # noqa: F401
except Exception:
    pass
