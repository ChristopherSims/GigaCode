"""Early process-wide warning filters for local dev and test runs.

This repository depends on FAISS, which currently emits a few SWIG-generated
DeprecationWarning messages during import on newer Python versions. They come
from third-party extension bootstrap code rather than from GigaCode itself.

`sitecustomize` is imported by Python during interpreter startup when this
repository root is on `sys.path`, which makes it the earliest reliable place
to suppress those specific warnings before pytest or test modules import FAISS.
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