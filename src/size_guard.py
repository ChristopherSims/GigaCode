"""Guard against oversized embedding buffers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_size(
    token_count: int,
    embedding_dim: int,
    threshold_mb: float = 500.0,
) -> dict[str, object]:
    """Check whether the estimated buffer size exceeds a threshold.

    Estimation accounts for:
    - float32 embeddings: ``token_count * embedding_dim * 4`` bytes.
    - uint64 offsets: ``token_count * 8`` bytes.
    - metadata overhead: ~256 bytes per line (JSON string overhead).

    Args:
        token_count: Total number of lines / tokens to embed.
        embedding_dim: Dimension of each embedding vector.
        threshold_mb: Maximum allowed size in megabytes.

    Returns:
        Dict with keys:
        - ``"status"`` (str): ``"ok"`` or ``"exceeds_threshold"``.
        - ``"estimated_bytes"`` (int): Raw byte estimate.
        - ``"estimated_mb"`` (float): Estimate in megabytes.
        - ``"threshold_mb"`` (float): The threshold used.
    """
    float32_size = 4
    uint64_size = 8
    metadata_overhead_per_line = 256

    estimated_bytes = (
        token_count * embedding_dim * float32_size
        + token_count * uint64_size
        + token_count * metadata_overhead_per_line
    )
    estimated_mb = estimated_bytes / (1024 * 1024)

    status = "ok" if estimated_mb <= threshold_mb else "exceeds_threshold"
    if status == "exceeds_threshold":
        logger.warning(
            "Estimated buffer size %.2f MB exceeds threshold %.2f MB",
            estimated_mb,
            threshold_mb,
        )

    return {
        "status": status,
        "estimated_bytes": estimated_bytes,
        "estimated_mb": round(estimated_mb, 4),
        "threshold_mb": threshold_mb,
    }
