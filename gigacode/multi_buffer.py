"""Multi-buffer orchestration for monorepos and large projects.

Provides:
- Virtual buffers (aggregate multiple sub-buffers into one logical view)
- Cross-buffer search
- Buffer aliases (human-readable names for buffer IDs)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "VirtualBuffer",
    "BufferAliasRegistry",
    "MultiBufferManager",
]


@dataclass
class VirtualBuffer:
    """A virtual buffer composed of multiple sub-buffers."""

    alias: str
    buffer_ids: list[str]
    description: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BufferAliasRegistry:
    """Registry for buffer aliases.

    Maps human-readable names (e.g., "main-project") to buffer UUIDs.
    Stored in work_dir/aliases.json.
    """

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = Path(work_dir)
        self._aliases_path = self.work_dir / "aliases.json"
        self._aliases: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load aliases from disk."""
        if self._aliases_path.exists():
            try:
                data = json.loads(self._aliases_path.read_text(encoding="utf-8"))
                self._aliases = {k: v for k, v in data.items() if isinstance(v, str)}
            except (OSError, json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to load aliases: {e}")
                self._aliases = {}

    def _save(self) -> None:
        """Save aliases to disk."""
        try:
            self._aliases_path.write_text(
                json.dumps(self._aliases, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except (OSError, TypeError) as e:
            logger.warning(f"Failed to save aliases: {e}")

    def create_alias(self, alias: str, buffer_id: str) -> dict[str, Any]:
        """Create or update an alias."""
        self._aliases[alias] = buffer_id
        self._save()
        return {
            "status": "ok",
            "alias": alias,
            "buffer_id": buffer_id,
        }

    def resolve(self, alias: str) -> str | None:
        """Resolve an alias to a buffer ID."""
        return self._aliases.get(alias)

    def remove_alias(self, alias: str) -> dict[str, Any]:
        """Remove an alias."""
        if alias in self._aliases:
            del self._aliases[alias]
            self._save()
            return {"status": "ok", "alias": alias, "message": "Alias removed"}
        return {"status": "warning", "alias": alias, "message": "Alias not found"}

    def list_aliases(self) -> dict[str, Any]:
        """List all aliases."""
        return {
            "status": "ok",
            "aliases": [{"alias": k, "buffer_id": v} for k, v in self._aliases.items()],
        }


class MultiBufferManager:
    """Manages virtual buffers and cross-buffer operations."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = Path(work_dir)
        self.alias_registry = BufferAliasRegistry(work_dir)
        self._virtual_path = self.work_dir / "virtual_buffers.json"
        self._virtual: dict[str, VirtualBuffer] = {}
        self._load_virtual()

    def _load_virtual(self) -> None:
        """Load virtual buffer definitions."""
        if self._virtual_path.exists():
            try:
                data = json.loads(self._virtual_path.read_text(encoding="utf-8"))
                for alias, vb_data in data.items():
                    self._virtual[alias] = VirtualBuffer(
                        alias=alias,
                        buffer_ids=vb_data.get("buffer_ids", []),
                        description=vb_data.get("description", ""),
                        created_at=vb_data.get("created_at", ""),
                    )
            except (OSError, json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to load virtual buffers: {e}")
                self._virtual = {}

    def _save_virtual(self) -> None:
        """Save virtual buffer definitions."""
        try:
            data = {
                alias: {
                    "buffer_ids": vb.buffer_ids,
                    "description": vb.description,
                    "created_at": vb.created_at,
                }
                for alias, vb in self._virtual.items()
            }
            self._virtual_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except (OSError, TypeError) as e:
            logger.warning(f"Failed to save virtual buffers: {e}")

    def create_virtual_buffer(
        self,
        alias: str,
        buffer_ids: list[str],
        description: str = "",
    ) -> dict[str, Any]:
        """Create a virtual buffer from multiple sub-buffers.

        Args:
            alias: Human-readable name (e.g., "monorepo").
            buffer_ids: List of existing buffer IDs to aggregate.
            description: Optional description.

        Returns:
            Dict with status and virtual buffer info.
        """
        from datetime import datetime, timezone

        vb = VirtualBuffer(
            alias=alias,
            buffer_ids=buffer_ids,
            description=description,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._virtual[alias] = vb
        self._save_virtual()

        return {
            "status": "ok",
            "alias": alias,
            "buffer_ids": buffer_ids,
            "description": description,
        }

    def get_virtual_buffer(self, alias: str) -> VirtualBuffer | None:
        """Get a virtual buffer by alias."""
        return self._virtual.get(alias)

    def list_virtual_buffers(self) -> dict[str, Any]:
        """List all virtual buffers."""
        return {
            "status": "ok",
            "virtual_buffers": [
                {
                    "alias": vb.alias,
                    "buffer_ids": vb.buffer_ids,
                    "description": vb.description,
                    "created_at": vb.created_at,
                }
                for vb in self._virtual.values()
            ],
        }

    def delete_virtual_buffer(self, alias: str) -> dict[str, Any]:
        """Delete a virtual buffer."""
        if alias in self._virtual:
            del self._virtual[alias]
            self._save_virtual()
            return {"status": "ok", "alias": alias, "message": "Virtual buffer deleted"}
        return {"status": "warning", "alias": alias, "message": "Virtual buffer not found"}

    def resolve_buffer_id(self, handle: str) -> str | list[str] | None:
        """Resolve a handle (alias, virtual buffer name, or raw buffer ID).

        Returns:
            - Single buffer ID (str) for aliases and raw IDs
            - List of buffer IDs (list[str]) for virtual buffers
            - None if not found
        """
        # Check aliases first
        resolved = self.alias_registry.resolve(handle)
        if resolved:
            return resolved

        # Check virtual buffers
        vb = self._virtual.get(handle)
        if vb:
            return vb.buffer_ids

        # Assume it's already a buffer ID
        return handle
