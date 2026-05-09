"""Progress streaming for long-running operations."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["ProgressReporter", "ProgressEvent"]


@dataclass
class ProgressEvent:
    phase: str
    current: int
    total: int
    message: str = ""
    percent: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgressReporter:
    """Simple progress reporter that yields events."""

    def __init__(self, phases: list[str]) -> None:
        self.phases = phases
        self._current_phase = 0
        self._callbacks: list[callable] = []

    def add_callback(self, callback: callable) -> None:
        self._callbacks.append(callback)

    def report(self, current: int, total: int, message: str = "") -> ProgressEvent:
        phase = (
            self.phases[self._current_phase]
            if self._current_phase < len(self.phases)
            else "unknown"
        )
        percent = (current / max(total, 1)) * 100 if total > 0 else 0
        event = ProgressEvent(
            phase=phase, current=current, total=total, message=message, percent=round(percent, 1)
        )
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception:
                pass
        return event

    def next_phase(self) -> None:
        self._current_phase += 1

    def finish(self) -> ProgressEvent:
        return ProgressEvent(
            phase="complete", current=100, total=100, message="Done", percent=100.0
        )
