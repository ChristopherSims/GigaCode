"""Tests for src.diff_engine."""

from __future__ import annotations

from src.diff_engine import compute_diff, hash_lines


def test_hash_lines() -> None:
    hashes = hash_lines(["hello", "world"])
    assert len(hashes) == 2
    assert hashes[0] != hashes[1]


def test_compute_diff() -> None:
    old = hash_lines(["a", "b", "c"])
    new = hash_lines(["a", "B", "c", "d"])
    changed = compute_diff(old, new)
    assert changed == [1, 3]


def test_compute_diff_no_change() -> None:
    old = hash_lines(["a", "b"])
    new = hash_lines(["a", "b"])
    assert compute_diff(old, new) == []
