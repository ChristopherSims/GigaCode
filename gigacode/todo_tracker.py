"""TODO/FIXME comment extraction and tracking."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["TodoTracker", "TodoItem"]

_TODO_PATTERNS = [
    re.compile(r"(?i)(TODO|FIXME|HACK|XXX|NOTE)\s*(?:\(([^)]+)\))?\s*[\:\-]?\s*(.*)"),
    re.compile(r"(?i)#\s*(TODO|FIXME|HACK|XXX|NOTE)\s*(?:\(([^)]+)\))?\s*[\:\-]?\s*(.*)"),
    re.compile(r"(?i)//\s*(TODO|FIXME|HACK|XXX|NOTE)\s*(?:\(([^)]+)\))?\s*[\:\-]?\s*(.*)"),
]

_PRIORITY_MAP = {
    "FIXME": "high",
    "HACK": "medium",
    "XXX": "high",
    "TODO": "low",
    "NOTE": "low",
}

@dataclass
class TodoItem:
    file: str
    line: int
    text: str
    tag: str  # TODO, FIXME, etc.
    priority: str
    assignee: str | None
    age_days: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class TodoTracker:
    def extract_todos(self, chunks: list[Any]) -> list[TodoItem]:
        results: list[TodoItem] = []
        for ch in chunks:
            for i, line in enumerate(ch.text.splitlines(), start=ch.start_line):
                for pattern in _TODO_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        tag = match.group(1).upper()
                        assignee = match.group(2)
                        text = match.group(3).strip()
                        if text:
                            results.append(TodoItem(
                                file=ch.file,
                                line=i,
                                text=text,
                                tag=tag,
                                priority=_PRIORITY_MAP.get(tag, "low"),
                                assignee=assignee,
                                age_days=None,  # Would need git blame
                            ))
                        break
        return results

def extract_todos(chunks: list[Any]) -> list[dict[str, Any]]:
    tracker = TodoTracker()
    return [t.to_dict() for t in tracker.extract_todos(chunks)]
