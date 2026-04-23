"""Flatten variable-length per-line embeddings into compact buffers."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np


class FlattenedBuffer:
    """Container for a flattened embedding buffer.

    Attributes:
        data_bytes: Raw float32 bytes for all embeddings concatenated.
        offsets_bytes: Raw uint64 bytes with the byte offset of each line.
        metadata_list: Per-line metadata (e.g. line numbers, token counts).
    """

    def __init__(
        self,
        data_bytes: bytes,
        offsets_bytes: bytes,
        metadata_list: list[dict[str, Any]],
    ) -> None:
        self.data_bytes = data_bytes
        self.offsets_bytes = offsets_bytes
        self.metadata_list = metadata_list

    @property
    def line_count(self) -> int:
        return len(self.metadata_list)

    @property
    def data_size(self) -> int:
        return len(self.data_bytes)

    @property
    def offsets_size(self) -> int:
        return len(self.offsets_bytes)


def flatten_embeddings(
    lines_data: list[dict[str, Any]],
    embedding_dim: int,
) -> tuple[bytes, bytes, list[dict[str, Any]]]:
    """Flatten per-line embeddings into compact byte buffers.

    Each element in *lines_data* is expected to be a dict with at least:
    - ``"embedding"``: a 1-D sequence of floats of length *embedding_dim*.
    - ``"line_num"`` (optional): the original source line number.
    - any other keys are preserved in the returned metadata.

    The function concatenates all embeddings into a single dense float32
    buffer and builds a parallel uint64 offset table so that the embedding
    for line *i* starts at ``offset[i]`` bytes.

    Args:
        lines_data: List of line records containing embeddings.
        embedding_dim: Expected dimension of each embedding vector.

    Returns:
        A tuple of ``(data_bytes, offsets_bytes, metadata_list)``.

    Raises:
        ValueError: If an embedding length does not match *embedding_dim*.
    """
    if not lines_data:
        return b"", b"", []

    # Vectorised: stack all embeddings in one shot
    embeddings = np.stack([rec["embedding"] for rec in lines_data], dtype=np.float32)
    if embeddings.shape != (len(lines_data), embedding_dim):
        raise ValueError(
            f"Embedding shape {embeddings.shape} does not match expected "
            f"({len(lines_data)}, {embedding_dim})"
        )

    metadata_list = [{k: v for k, v in rec.items() if k != "embedding"} for rec in lines_data]
    data_bytes = embeddings.tobytes()

    # Uniform stride: every embedding is the same size
    offsets = np.arange(len(lines_data), dtype=np.uint64) * (embedding_dim * 4)
    offsets_bytes = offsets.tobytes()
    return data_bytes, offsets_bytes, metadata_list
