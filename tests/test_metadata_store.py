"""Tests for src.metadata_store."""

from __future__ import annotations

# CRITICAL: Initialize sklearn FIRST before any gigacode imports
import types
try:
    import sklearn
    if getattr(sklearn, "__spec__", None) is None:
        sklearn.__spec__ = types.ModuleSpec("sklearn", getattr(sklearn, "__file__", None))
except Exception:
    pass

from pathlib import Path

from gigacode.metadata_store import load_metadata, save_metadata


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    data = [
        {"line_num": 1, "file": "a.py"},
        {"line_num": 2, "file": "a.py"},
    ]
    save_metadata(path, data)
    loaded = load_metadata(path)
    assert loaded == data

