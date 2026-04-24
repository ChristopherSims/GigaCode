"""File watcher for auto-reloading GigaCode buffers when source files change.

Uses ``watchdog`` when available; falls back to a lightweight polling thread.

Usage:
    from src.file_watcher import BufferWatcher
    watcher = BufferWatcher(tool)
    watcher.watch_buffer(buffer_id)
    # ... later ...
    watcher.unwatch_buffer(buffer_id)
    watcher.stop_all()
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Grace period (seconds) to batch rapid successive changes
_DEBOUNCE_SECONDS = 1.0


try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    _HAS_WATCHDOG = True
except ImportError:
    _HAS_WATCHDOG = False
    Observer = None  # type: ignore
    FileSystemEventHandler = None  # type: ignore


class _PollingWatcher:
    """Fallback polling-based file watcher."""

    def __init__(self, callback: Callable[[str], None], interval: float = 2.0) -> None:
        self._callback = callback
        self._interval = interval
        self._watches: dict[str, dict[str, str]] = {}  # buffer_id -> {rel_path -> hash}
        self._roots: dict[str, Path] = {}  # buffer_id -> root path
        self._patterns: dict[str, str] = {}  # buffer_id -> glob pattern
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def add_buffer(self, buffer_id: str, root: Path, pattern: str) -> None:
        self._roots[buffer_id] = root
        self._patterns[buffer_id] = pattern
        self._watches[buffer_id] = {}
        self._scan(buffer_id)
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def remove_buffer(self, buffer_id: str) -> None:
        self._roots.pop(buffer_id, None)
        self._patterns.pop(buffer_id, None)
        self._watches.pop(buffer_id, None)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _scan(self, buffer_id: str) -> None:
        root = self._roots.get(buffer_id)
        pattern = self._patterns.get(buffer_id, "*.py")
        if root is None:
            return
        files = [root] if root.is_file() else sorted(root.rglob(pattern))
        current: dict[str, str] = {}
        for f in files:
            rel = str(f.relative_to(root))
            try:
                raw = f.read_text(encoding="utf-8", errors="replace")
                normalized = "\n".join(raw.splitlines())
                current[rel] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            except Exception:
                continue
        old = self._watches.get(buffer_id, {})
        changed = [rel for rel in current if old.get(rel) != current[rel]]
        self._watches[buffer_id] = current
        if changed:
            for rel in changed:
                logger.info("[poll] Detected change in %s", rel)
            self._callback(buffer_id)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            for buffer_id in list(self._roots.keys()):
                self._scan(buffer_id)
            time.sleep(self._interval)


class BufferWatcher:
    """Manages file watchers for one or more GigaCode buffers."""

    def __init__(self, tool: Any) -> None:
        self._tool = tool
        self._observers: dict[str, Any] = {}  # buffer_id -> watchdog Observer
        self._polling = _PollingWatcher(self._on_change)
        self._last_rebuild: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def watch_buffer(self, buffer_id: str) -> dict[str, Any]:
        """Start watching the source directory of *buffer_id*."""
        info = self._tool._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        root = Path(info["root"])
        pattern = info.get("pattern", "*.py")

        if _HAS_WATCHDOG:
            self._watchdog_start(buffer_id, root, pattern)
        else:
            logger.warning("watchdog not installed; using polling fallback.")
            self._polling.add_buffer(buffer_id, root, pattern)

        return {"status": "ok", "message": f"Watching {root} for buffer {buffer_id}"}

    def unwatch_buffer(self, buffer_id: str) -> dict[str, Any]:
        """Stop watching *buffer_id*."""
        obs = self._observers.pop(buffer_id, None)
        if obs:
            obs.stop()
            obs.join(timeout=2.0)
        self._polling.remove_buffer(buffer_id)
        return {"status": "ok", "message": f"Stopped watching buffer {buffer_id}"}

    def stop_all(self) -> None:
        """Stop all active watchers."""
        for obs in self._observers.values():
            obs.stop()
        for obs in self._observers.values():
            obs.join(timeout=2.0)
        self._observers.clear()
        self._polling.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _on_change(self, buffer_id: str) -> None:
        now = time.time()
        last = self._last_rebuild.get(buffer_id, 0)
        if now - last < _DEBOUNCE_SECONDS:
            return
        self._last_rebuild[buffer_id] = now
        logger.info("Auto-rebuilding buffer %s", buffer_id)
        try:
            self._tool.reload_codebase(buffer_id)
        except Exception as exc:
            logger.error("Auto-rebuild failed for %s: %s", buffer_id, exc)

    def _watchdog_start(self, buffer_id: str, root: Path, pattern: str) -> None:
        if buffer_id in self._observers:
            return

        suffixes = {pattern.lstrip("*.")} if pattern.startswith("*.") else set()

        class _Handler(FileSystemEventHandler):  # type: ignore
            def __init__(self, outer: BufferWatcher, bid: str, sfx: set[str]) -> None:
                self.outer = outer
                self.bid = bid
                self.suffixes = sfx

            def on_modified(self, event) -> None:
                if event.is_directory:
                    return
                if self.suffixes and not any(str(event.src_path).endswith(s) for s in self.suffixes):
                    return
                self.outer._on_change(self.bid)

            def on_created(self, event) -> None:
                self.on_modified(event)

            def on_deleted(self, event) -> None:
                self.on_modified(event)

            def on_moved(self, event) -> None:
                self.on_modified(event)

        handler = _Handler(self, buffer_id, suffixes)
        observer = Observer()
        observer.schedule(handler, str(root), recursive=True)
        observer.start()
        self._observers[buffer_id] = observer
        logger.info("watchdog observer started for %s", root)
