"""Multi-turn conversation memory for agents."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["ConversationMemory", "MemoryEntry"]

@dataclass
class MemoryEntry:
    key: str
    value: str
    tags: list[str]
    created_at: str
    id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class ConversationMemory:
    def __init__(self, memory_file: Path) -> None:
        self.memory_file = Path(memory_file)
        self._memories: list[MemoryEntry] = []
        self._load()

    def _load(self) -> None:
        if self.memory_file.exists():
            try:
                data = json.loads(self.memory_file.read_text(encoding="utf-8"))
                self._memories = [MemoryEntry(**m) for m in data]
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to load memories: {e}")
                self._memories = []

    def _save(self) -> None:
        try:
            data = [m.to_dict() for m in self._memories]
            self.memory_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, TypeError) as e:
            logger.warning(f"Failed to save memories: {e}")

    def remember(self, key: str, value: str, tags: list[str] | None = None) -> dict[str, Any]:
        entry = MemoryEntry(
            key=key,
            value=value,
            tags=tags or [],
            created_at=datetime.now(timezone.utc).isoformat(),
            id=str(uuid.uuid4()),
        )
        # Remove existing entry with same key
        self._memories = [m for m in self._memories if m.key != key]
        self._memories.append(entry)
        self._save()
        return {"status": "ok", "key": key, "id": entry.id}

    def recall(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Simple text matching recall."""
        query_lower = query.lower()
        scored: list[tuple[MemoryEntry, float]] = []
        for m in self._memories:
            score = 0.0
            if query_lower in m.key.lower():
                score += 0.5
            if query_lower in m.value.lower():
                score += 0.3
            for tag in m.tags:
                if query_lower in tag.lower():
                    score += 0.2
            if score > 0:
                scored.append((m, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_k]]

    def list_memories(self, tag: str | None = None) -> list[MemoryEntry]:
        if tag:
            return [m for m in self._memories if tag in m.tags]
        return self._memories[:]

    def forget(self, key: str) -> dict[str, Any]:
        original_count = len(self._memories)
        self._memories = [m for m in self._memories if m.key != key]
        if len(self._memories) < original_count:
            self._save()
            return {"status": "ok", "message": f"Forgot memory '{key}'"}
        return {"status": "warning", "message": f"Memory '{key}' not found"}
