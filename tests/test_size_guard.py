"""Tests for gigacode.size_guard."""

from __future__ import annotations

# CRITICAL: Initialize sklearn FIRST before any gigacode imports
import types

try:
    import sklearn

    if getattr(sklearn, "__spec__", None) is None:
        sklearn.__spec__ = types.ModuleSpec("sklearn", getattr(sklearn, "__file__", None))
except Exception:
    pass

from gigacode.size_guard import check_size


def test_check_size_ok() -> None:
    result = check_size(token_count=1000, embedding_dim=384, threshold_mb=500)
    assert result["status"] == "ok"


def test_check_size_exceeds() -> None:
    result = check_size(token_count=10_000_000, embedding_dim=768, threshold_mb=100)
    assert result["status"] == "exceeds_threshold"
