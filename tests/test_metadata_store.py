"""Tests for src.metadata_store."""

from __future__ import annotations

from pathlib import Path

from src.metadata_store import load_metadata, save_metadata


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    data = [
        {"line_num": 1, "file": "a.py"},
        {"line_num": 2, "file": "a.py"},
    ]
    save_metadata(path, data)
    loaded = load_metadata(path)
    assert loaded == data
