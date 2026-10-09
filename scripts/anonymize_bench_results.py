"""Anonymize benchmark results in place for publishing.

Redacts personally identifying strings from *.json, *.jsonl, *.txt, *.log
files under a results directory:

- any ``X:\\Users\\<name>`` home path       -> ``~`` (subpaths preserved)
- the bench temp sandbox root              -> ``<sandbox>``
- the opencode CLI install path            -> ``<opencode>``
- opencode's own temp workspace            -> ``<tmpdir>``
- repo working directory                   -> ``<repo>``
- opencode session ids (``ses_*``)         -> ``ses_<hash8>`` (stable)

Subdirectory tails (e.g. ``\\project\\rich\\text.py``) are preserved so file
coordinates stay readable; only identity-bearing prefixes are removed.
Idempotent: running it again changes nothing.

Usage:
    python scripts/anonymize_bench_results.py bench_results [more_dirs...]
    python scripts/anonymize_bench_results.py --check bench_results   # report only
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

SEP = r"[\\\\/]+"  # raw or JSON-escaped separators (one or repeated)

REDACTIONS: list[tuple[str, str]] = [
    # absolute windows user dir -> ~ (drive letter optional: titles can strip it)
    (rf"(?:[A-Za-z]?:?{SEP})?Users{SEP}[A-Za-z0-9._-]+", "~"),
    # opencode CLI install -> <opencode>
    (
        rf"~{SEP}AppData{SEP}Roaming{SEP}npm{SEP}node_modules{SEP}opencode-ai.*?"
        r"opencode\.exe",
        "<opencode>",
    ),
    # GitHub/Python install roots sometimes appear in tool outputs
    (rf"~{SEP}AppData{SEP}Local{SEP}Temp{SEP}opencode", "<tmpdir>"),
    # bench sandbox root (any suite) -> <sandbox>
    (
        rf"~{SEP}AppData{SEP}Local{SEP}Temp{SEP}gigacode_token_bench_[A-Za-z_0-9]+"
        rf"{SEP}[A-Za-z_0-9]+",
        "<sandbox>",
    ),
    # repository working dir -> <repo>
    (rf"[A-Za-z]:{SEP}Droid{SEP}GigaCode", "<repo>"),
]

SESSION_RE = re.compile(r"ses_[A-Za-z0-9]+")


def redact_text(text: str) -> str:
    """Apply all path redactions, then hash session ids."""
    for pattern, replacement in REDACTIONS:
        text = re.sub(pattern, replacement, text)
    text = SESSION_RE.sub(
        lambda m: "ses_" + hashlib.md5(m.group(0).encode()).hexdigest()[:8], text
    )
    return text


_FILE_SUFFIXES = {".json", ".jsonl", ".txt", ".log"}


def anonymize_dir(root: Path, check_only: bool) -> tuple[int, int]:
    files_scanned = files_changed = 0
    if not root.is_dir():
        print(f"skip (missing): {root}", file=sys.stderr)
        return 0, 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _FILE_SUFFIXES:
            continue
        files_scanned += 1
        original = path.read_text(encoding="utf-8", errors="replace")
        redacted = redact_text(original)
        if redacted != original:
            files_changed += 1
            if not check_only:
                path.write_text(redacted, encoding="utf-8")
            diff_preview = "changed" if not check_only else "would change"
            print(f"{diff_preview}: {path.relative_to(root.parent)}")
    return files_scanned, files_changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Anonymize bench result artifacts")
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="report changes without writing")
    args = parser.parse_args()
    total_scanned = total_changed = 0
    for d in args.dirs:
        scanned, changed = anonymize_dir(d, args.check)
        total_scanned += scanned
        total_changed += changed
        print(f"{d}: scanned={scanned} changed={changed} (check_only={args.check})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
