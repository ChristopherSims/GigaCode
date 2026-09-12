"""Synchronize the package version across VERSION, pyproject.toml, and __init__.py.

`VERSION` is the single source of truth. Run this before tagging a release:

    python scripts/sync_version.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SEMVER = re.compile(r"\d+\.\d+\.\d+(?:[.-][0-9A-Za-z.]+)?")


def main() -> int:
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    if not _SEMVER.fullmatch(version):
        print(f"ERROR: invalid version in VERSION: {version!r}", file=sys.stderr)
        return 1

    pyproject = ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    updated = re.sub(r'(?m)^version\s*=\s*"[^"]*"', f'version = "{version}"', text, count=1)
    if updated != text:
        pyproject.write_text(updated, encoding="utf-8")

    init = ROOT / "gigacode" / "__init__.py"
    itext = init.read_text(encoding="utf-8")
    iupdated = re.sub(
        r'(?m)^__version__\s*=\s*"[^"]*"', f'__version__ = "{version}"', itext, count=1
    )
    if iupdated != itext:
        init.write_text(iupdated, encoding="utf-8")

    print(f"Version synchronized to {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
