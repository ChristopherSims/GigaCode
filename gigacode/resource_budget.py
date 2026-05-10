"""Resource budgeting and confidence scoring for AI agents.

Provides:
- Pre-embed cost estimation (RAM, time, chunks)
- Memory usage tracking with hard caps
- Confidence scoring for search results
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ResourceEstimate",
    "MemoryBudget",
    "ConfidenceScorer",
    "estimate_budget",
    "score_confidence",
]


@dataclass
class ResourceEstimate:
    """Estimated resource requirements for embedding a codebase."""

    estimated_ram_mb: float
    estimated_embed_time_s: float
    estimated_search_latency_ms: float
    num_files: int
    num_lines: int
    estimated_chunks: int
    embedding_size_mb: float
    recommended_device: str  # "cpu" or "cuda"
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryBudget:
    """Tracks memory usage and enforces limits."""

    max_memory_mb: float
    current_usage_mb: float = 0.0
    evicted_count: int = 0

    def can_fit(self, size_mb: float) -> bool:
        """Check if adding `size_mb` would exceed the budget."""
        return (self.current_usage_mb + size_mb) <= self.max_memory_mb

    def add(self, size_mb: float) -> bool:
        """Add usage. Returns False if it would exceed budget."""
        if not self.can_fit(size_mb):
            return False
        self.current_usage_mb += size_mb
        return True

    def remove(self, size_mb: float) -> None:
        """Remove usage."""
        self.current_usage_mb = max(0.0, self.current_usage_mb - size_mb)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfidenceScorer:
    """Score confidence of search results based on score distribution."""

    @staticmethod
    def classify(score: float, all_scores: list[float]) -> str:
        """Classify confidence as high/medium/low.

        Args:
            score: The score of the result being classified.
            all_scores: All scores from the search (for gap analysis).

        Returns:
            "high", "medium", or "low".
        """
        if not all_scores:
            return "low"

        # Absolute thresholds
        if score < 0.3:
            return "low"
        if score > 0.9:
            return "high"

        # Gap analysis
        sorted_scores = sorted(all_scores, reverse=True)
        if len(sorted_scores) >= 2:
            gap = sorted_scores[0] - sorted_scores[1]
            if gap > 0.2 and abs(score - sorted_scores[0]) < 0.01:
                return "high"
            if gap < 0.03:
                return "low"

        # Distribution-based
        mean_score = float(np.mean(all_scores))
        std_score = float(np.std(all_scores)) if len(all_scores) > 1 else 0.0

        if score > mean_score + 2 * std_score:
            return "high"
        if score > mean_score + 0.5 * std_score:
            return "medium"
        if score < mean_score - 0.5 * std_score:
            return "low"

        return "medium" if score > 0.5 else "low"

    @staticmethod
    def explain(score: float, confidence: str, all_scores: list[float]) -> str:
        """Generate human-readable explanation of confidence."""
        sorted_scores = sorted(all_scores, reverse=True) if all_scores else []

        if confidence == "high":
            if score > 0.9:
                return f"Very strong match (score={score:.3f}). High confidence."
            if len(sorted_scores) >= 2:
                gap = sorted_scores[0] - sorted_scores[1]
                return f"Top result stands out from rest (gap={gap:.3f}, score={score:.3f})."
            return f"Strong signal (score={score:.3f})."

        if confidence == "low":
            if score < 0.3:
                return f"Weak signal (score={score:.3f}). May be noise."
            if len(sorted_scores) >= 2:
                gap = sorted_scores[0] - sorted_scores[1]
                if gap < 0.03:
                    return (
                        f"Results are tightly clustered (gap={gap:.3f}). Low discriminative power."
                    )
            return f"Below average signal (score={score:.3f})."

        # Medium
        return f"Moderate signal (score={score:.3f}). Reasonable but not definitive."


def estimate_budget(
    path: str | Path,
    pattern: str = "*.py",
    embedding_dim: int = 384,
    device: str | None = None,
    avg_lines_per_chunk: int = 25,
    embedding_bytes: int = 4,  # float32
) -> dict[str, Any]:
    """Estimate resource requirements for embedding a codebase.

    Args:
        path: Path to codebase directory.
        pattern: File glob pattern.
        embedding_dim: Dimension of embeddings.
        device: Target device (cpu/cuda). If None, recommends based on size.
        avg_lines_per_chunk: Average lines per AST chunk.
        embedding_bytes: Bytes per float (4 for float32).

    Returns:
        ResourceEstimate as dict with warnings.
    """
    root = Path(path)
    files = sorted(root.rglob(pattern)) if root.is_dir() else [root]

    num_files = len(files)
    total_lines = 0
    total_size_bytes = 0

    for f in files:
        try:
            stat = f.stat()
            total_size_bytes += stat.st_size
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                total_lines += sum(1 for _ in fh)
        except (OSError, PermissionError):
            pass

    # Estimate chunks
    estimated_chunks = max(1, total_lines // avg_lines_per_chunk)

    # Embedding memory: chunks * dim * bytes_per_float
    embedding_size_bytes = estimated_chunks * embedding_dim * embedding_bytes
    embedding_size_mb = embedding_size_bytes / (1024 * 1024)

    # FAISS index overhead: ~1.5x for FlatIP + metadata
    index_overhead_mb = embedding_size_mb * 0.5

    # Total estimated RAM: embeddings + index + text snapshots (~20%)
    total_ram_mb = embedding_size_mb + index_overhead_mb + (total_size_bytes / (1024 * 1024)) * 0.2

    # Embed time: ~50ms per chunk on CPU, ~5ms on GPU (very rough)
    if device == "cuda":
        embed_time_s = estimated_chunks * 0.005
        search_latency_ms = 1.0
        recommended_device = "cuda"
    elif device == "cpu":
        embed_time_s = estimated_chunks * 0.05
        search_latency_ms = 20.0
        recommended_device = "cpu"
    else:
        # Auto-recommend
        if total_ram_mb > 4000:
            embed_time_s = estimated_chunks * 0.005
            search_latency_ms = 1.0
            recommended_device = "cuda"
        else:
            embed_time_s = estimated_chunks * 0.05
            search_latency_ms = 20.0
            recommended_device = "cpu"

    warnings: list[str] = []
    if total_ram_mb > 8000:
        warnings.append(
            f"Large codebase: estimated {total_ram_mb:.0f}MB RAM. Consider splitting into multiple buffers."
        )
    if num_files > 10000:
        warnings.append(f"Many files ({num_files}). File discovery may take time.")
    if estimated_chunks > 100000:
        warnings.append(
            f"Very large index ({estimated_chunks} chunks). Consider using GPU or quantized index."
        )

    estimate = ResourceEstimate(
        estimated_ram_mb=round(total_ram_mb, 1),
        estimated_embed_time_s=round(embed_time_s, 1),
        estimated_search_latency_ms=round(search_latency_ms, 1),
        num_files=num_files,
        num_lines=total_lines,
        estimated_chunks=estimated_chunks,
        embedding_size_mb=round(embedding_size_mb, 1),
        recommended_device=recommended_device,
        warnings=warnings,
    )

    return {
        "status": "ok",
        **estimate.to_dict(),
    }


def score_confidence(
    score: float,
    all_scores: list[float],
) -> dict[str, Any]:
    """Score confidence of a search result.

    Args:
        score: The result's score.
        all_scores: All scores from the same search.

    Returns:
        Dict with confidence level and explanation.
    """
    scorer = ConfidenceScorer()
    confidence = scorer.classify(score, all_scores)
    explanation = scorer.explain(score, confidence, all_scores)

    # Distribution stats
    sorted_scores = sorted(all_scores, reverse=True) if all_scores else []
    stats = {
        "mean": round(float(np.mean(all_scores)), 3) if all_scores else 0.0,
        "std": round(float(np.std(all_scores)), 3) if len(all_scores) > 1 else 0.0,
        "max": round(sorted_scores[0], 3) if sorted_scores else 0.0,
        "gap_to_second": (
            round(sorted_scores[0] - sorted_scores[1], 3) if len(sorted_scores) >= 2 else 0.0
        ),
    }

    return {
        "status": "ok",
        "score": round(score, 3),
        "confidence": confidence,
        "explanation": explanation,
        "distribution": stats,
    }
