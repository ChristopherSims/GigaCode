"""Basic metrics collection for GigaCode operations.

Tracks timing, cache hit rates, and operation counts in-memory.
Can be exported to monitoring systems (Prometheus, StatsD, etc.) in future versions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


__all__ = [
    "MetricType",
    "HistogramStats",
    "MetricsCollector",
    "get_metrics",
]


class MetricType(str, Enum):
    """Types of metrics we track."""

    COUNTER = "counter"  # Monotonically increasing count
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values (latencies, sizes)


@dataclass
class HistogramStats:
    """Statistics for a histogram metric."""

    count: int = 0
    sum_: float = 0.0
    min_: float = float("inf")
    max_: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    @property
    def mean(self) -> float:
        return self.sum_ / self.count if self.count > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "sum": self.sum_,
            "mean": self.mean,
            "min": self.min_,
            "max": self.max_,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


class MetricsCollector:
    """In-memory metrics collector for GigaCode operations."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._cache_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"hits": 0, "misses": 0})

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self._counters[name] += value
        logger.debug("counter %s = %d", name, self._counters[name])

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric to a specific value."""
        self._gauges[name] = value
        logger.debug("gauge %s = %.2f", name, value)

    def record_histogram(self, name: str, value: float) -> None:
        """Record a value in a histogram metric (for latency, size, etc.)."""
        self._histograms[name].append(value)
        logger.debug("histogram %s += %.3f (count=%d)", name, value, len(self._histograms[name]))

    def record_cache_hit(self, cache_type: str) -> None:
        """Record a cache hit."""
        self._cache_stats[cache_type]["hits"] += 1

    def record_cache_miss(self, cache_type: str) -> None:
        """Record a cache miss."""
        self._cache_stats[cache_type]["misses"] += 1

    def get_cache_hit_rate(self, cache_type: str) -> float:
        """Get cache hit rate (0.0-1.0) for a cache type."""
        stats = self._cache_stats.get(cache_type, {"hits": 0, "misses": 0})
        total = stats["hits"] + stats["misses"]
        return stats["hits"] / total if total > 0 else 0.0

    def get_histogram_stats(self, name: str) -> HistogramStats:
        """Compute statistics for a histogram."""
        values = sorted(self._histograms.get(name, []))
        if not values:
            return HistogramStats()
        stats = HistogramStats(
            count=len(values),
            sum_=sum(values),
            min_=min(values),
            max_=max(values),
        )
        # Compute percentiles
        if len(values) >= 2:
            stats.p50 = values[int(len(values) * 0.50)]
            stats.p95 = values[int(len(values) * 0.95)]
            stats.p99 = values[int(len(values) * 0.99)]
        return stats

    def dump_metrics(self) -> dict[str, Any]:
        """Export all metrics as a dict for monitoring integration."""
        result: dict[str, Any] = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {},
            "cache_stats": {},
        }
        for name in self._histograms:
            result["histograms"][name] = self.get_histogram_stats(name).to_dict()
        for name, stats in self._cache_stats.items():
            result["cache_stats"][name] = {
                "hits": stats["hits"],
                "misses": stats["misses"],
                "hit_rate": self.get_cache_hit_rate(name),
            }
        return result

    def reset(self) -> None:
        """Clear all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._cache_stats.clear()
        logger.debug("Metrics collector reset")


# Global metrics instance
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics
