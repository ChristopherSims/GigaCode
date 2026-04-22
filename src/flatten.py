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
        empty_offsets = struct.pack(f"<{0}Q", *[])  # type: ignore[arg-type]
        return b"", empty_offsets, []

    float32_size = 4
    bytes_per_line = embedding_dim * float32_size

    offsets: list[int] = []
    metadata_list: list[dict[str, Any]] = []
    data_chunks: list[bytes] = []

    current_offset = 0
    for record in lines_data:
        emb = record["embedding"]
        arr = np.asarray(emb, dtype=np.float32)
        if arr.shape != (embedding_dim,):
            raise ValueError(
                f"Embedding shape {arr.shape} does not match expected ({embedding_dim},)"
            )
        offsets.append(current_offset)
        data_chunks.append(arr.tobytes())
        current_offset += bytes_per_line

        # Build metadata (copy everything except the raw embedding to save memory)
        meta = {k: v for k, v in record.items() if k != "embedding"}
        metadata_list.append(meta)

    data_bytes = b"".join(data_chunks)
    offsets_bytes = struct.pack(f"<{len(offsets)}Q", *offsets)
    return data_bytes, offsets_bytes, metadata_list
