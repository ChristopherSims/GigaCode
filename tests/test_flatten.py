"""Tests for src.flatten."""

from __future__ import annotations

import numpy as np

from src.flatten import flatten_embeddings


def test_flatten_embeddings() -> None:
    lines = [
        {"line_num": 1, "embedding": [1.0, 0.0, 0.0], "text": "a"},
        {"line_num": 2, "embedding": [0.0, 1.0, 0.0], "text": "b"},
    ]
    data_bytes, offsets_bytes, metadata = flatten_embeddings(lines, embedding_dim=3)
    assert len(data_bytes) == 2 * 3 * 4  # 2 lines * 3 floats * 4 bytes
    assert len(offsets_bytes) == 2 * 8  # 2 lines * uint64
    assert len(metadata) == 2
    assert metadata[0]["line_num"] == 1


def test_flatten_empty() -> None:
    data_bytes, offsets_bytes, metadata = flatten_embeddings([], embedding_dim=3)
    assert data_bytes == b""
    assert metadata == []
