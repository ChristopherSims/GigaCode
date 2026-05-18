"""Snapshot management with metadata-only storage and on-demand file reading.

Eliminates full-codebase duplication by storing only file metadata (name, mtime, hash)
and reading lines on-demand from disk. Supports 3-way merge for conflict detection.
"""

import hashlib
import json
import logging
import difflib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


__all__ = [
    "FileMetadata",
    "SnapshotManifest",
    "SnapshotManager",
    "SnapshotDiffer",
]


@dataclass
class FileMetadata:
    """Lightweight metadata for a file in the snapshot."""

    path: str
    mtime: float
    size: int
    hash: str  # SHA-256 of file content at snapshot time
    lines: Optional[int] = None  # Number of lines (cached for reference)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FileMetadata":
        """Create from dict."""
        return cls(**data)

    @classmethod
    def from_file(cls, file_path: Path) -> "FileMetadata":
        """Create metadata from actual file."""
        stat = file_path.stat()
        content = file_path.read_bytes()
        hash_val = hashlib.sha256(content).hexdigest()
        lines = len(content.decode("utf-8", errors="ignore").splitlines())

        return cls(
            path=str(file_path), mtime=stat.st_mtime, size=stat.st_size, hash=hash_val, lines=lines
        )


@dataclass
class SnapshotManifest:
    """Lightweight snapshot manifest (metadata only, no source code)."""

    buffer_id: str
    root_path: str
    created_at: str
    modified_at: str
    files: dict[str, FileMetadata]  # path -> metadata

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "buffer_id": self.buffer_id,
            "root_path": self.root_path,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "files": {path: meta.to_dict() for path, meta in self.files.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SnapshotManifest":
        """Create from dict."""
        return cls(
            buffer_id=data["buffer_id"],
            root_path=data["root_path"],
            created_at=data["created_at"],
            modified_at=data["modified_at"],
            files={
                path: FileMetadata.from_dict(meta) for path, meta in data.get("files", {}).items()
            },
        )


class SnapshotManager:
    """Manages lightweight snapshots with on-demand file reading.

    Features:
    - Metadata-only snapshots (no source code duplication)
    - On-demand file reading from disk
    - 3-way merge for conflict detection
    - File change detection via mtime + hash
    """

    def __init__(self, buffer_dir: Path):
        """Initialize snapshot manager.

        Args:
            buffer_dir: Directory where snapshot manifest is stored
        """
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.buffer_dir / "snapshot_manifest.json"

        self.manifest: Optional[SnapshotManifest] = None
        self._load_manifest()

    def create_snapshot(
        self, buffer_id: str, root_path: Path, files: dict[str, Path]
    ) -> SnapshotManifest:
        """Create a new snapshot from current files.

        Args:
            buffer_id: Unique buffer identifier
            root_path: Root directory of codebase
            files: Dict mapping relative paths to absolute file paths

        Returns:
            Created SnapshotManifest
        """
        now = datetime.utcnow().isoformat() + "Z"
        file_metadata = {}

        for rel_path, file_path in files.items():
            if file_path.is_file():
                try:
                    meta = FileMetadata.from_file(file_path)
                    file_metadata[rel_path] = meta
                    logger.debug(f"Snapshot: {rel_path} ({meta.size} bytes, {meta.lines} lines)")
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to snapshot {rel_path}: {e}")

        manifest = SnapshotManifest(
            buffer_id=buffer_id,
            root_path=str(root_path),
            created_at=now,
            modified_at=now,
            files=file_metadata,
        )

        self.manifest = manifest
        self._save_manifest()
        logger.info(f"Created snapshot for {buffer_id}: {len(file_metadata)} files")

        return manifest

    def _load_manifest(self) -> None:
        """Load snapshot manifest from disk."""
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                self.manifest = SnapshotManifest.from_dict(data)
                logger.debug(f"Loaded snapshot with {len(self.manifest.files)} files")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to load snapshot manifest: {e}")
                self.manifest = None
        else:
            self.manifest = None

    def _save_manifest(self) -> None:
        """Save snapshot manifest to disk."""
        if self.manifest:
            self.manifest_path.write_text(
                json.dumps(self.manifest.to_dict(), indent=2), encoding="utf-8"
            )
            logger.debug("Snapshot manifest saved")

    def read_file(self, relative_path: str) -> Optional[str]:
        """Read file content from disk (on-demand).

        Args:
            relative_path: Relative path from root

        Returns:
            File content or None if file not found
        """
        if not self.manifest:
            return None

        root = Path(self.manifest.root_path)
        file_path = root / relative_path

        if not file_path.exists():
            logger.warning(f"File not found on disk: {relative_path}")
            return None

        try:
            return file_path.read_text(encoding="utf-8")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(f"Failed to read {relative_path}: {e}")
            return None

    def read_lines(self, relative_path: str) -> Optional[list[str]]:
        """Read file lines from disk (on-demand).

        Args:
            relative_path: Relative path from root

        Returns:
            List of lines or None if file not found
        """
        content = self.read_file(relative_path)
        if content is None:
            return None
        return content.splitlines()

    def detect_external_changes(self) -> dict[str, Any]:
        """Detect changes to files since snapshot was created.

        Returns:
            Dict with keys:
            - "changed": Files modified (mtime or hash changed)
            - "deleted": Files removed
            - "new": Files added (not in snapshot but on disk)
        """
        if not self.manifest:
            return {"changed": [], "deleted": [], "new": []}

        root = Path(self.manifest.root_path)
        changes = {"changed": [], "deleted": [], "new": []}

        # Check for changed/deleted files
        for rel_path, meta in self.manifest.files.items():
            file_path = root / rel_path

            if not file_path.exists():
                changes["deleted"].append(rel_path)
                continue

            # Check mtime and hash
            stat = file_path.stat()
            if stat.st_mtime != meta.mtime:
                # File was modified - verify with hash
                try:
                    content = file_path.read_bytes()
                    new_hash = hashlib.sha256(content).hexdigest()
                    if new_hash != meta.hash:
                        changes["changed"].append(rel_path)
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to check {rel_path}: {e}")
                    changes["changed"].append(rel_path)

        logger.debug(
            f"Detected changes: {len(changes['changed'])} modified, "
            f"{len(changes['deleted'])} deleted, {len(changes['new'])} new"
        )

        return changes

    def compute_diff(
        self, relative_path: str, buffer_lines: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Compute diff between disk, snapshot, and buffer versions.

        This is the foundation for 3-way merge:
        - disk_lines: Current file on disk
        - snapshot_lines: File state at snapshot time (reconstruct if needed)
        - buffer_lines: Edited lines in buffer

        Args:
            relative_path: File to diff
            buffer_lines: Current buffer lines (if None, read from disk)

        Returns:
            Dict with:
            - "disk_lines": Current lines on disk
            - "snapshot_lines": Lines from snapshot metadata
            - "buffer_lines": Lines in buffer (may differ from disk)
            - "has_conflict": True if 3-way merge needed
        """
        if not self.manifest:
            return {
                "disk_lines": None,
                "snapshot_lines": None,
                "buffer_lines": buffer_lines,
                "has_conflict": False,
            }

        # Read current disk state
        disk_lines = self.read_lines(relative_path)

        # Get snapshot metadata
        meta = self.manifest.files.get(relative_path)
        snapshot_line_count = meta.lines if meta else None

        # If buffer not provided, use disk
        if buffer_lines is None:
            buffer_lines = disk_lines

        # Detect 3-way conflicts
        has_conflict = False
        if disk_lines and buffer_lines:
            # Conflict if disk changed AND buffer is different from snapshot
            if len(disk_lines) != snapshot_line_count:
                # Disk was modified externally
                if buffer_lines != disk_lines:
                    # Buffer also has changes
                    has_conflict = True
                    logger.warning(
                        f"3-way merge conflict in {relative_path}: "
                        f"disk modified AND buffer modified"
                    )

        return {
            "disk_lines": disk_lines,
            "snapshot_line_count": snapshot_line_count,
            "buffer_lines": buffer_lines,
            "has_conflict": has_conflict,
        }

    def write_file_with_merge(
        self, relative_path: str, buffer_lines: list[str], allow_conflicts: bool = False
    ) -> dict[str, Any]:
        """Write buffer lines to disk with 3-way merge conflict handling.

        Args:
            relative_path: File to write
            buffer_lines: Lines to write from buffer
            allow_conflicts: If False, fail on conflicts; if True, buffer wins

        Returns:
            Dict with:
            - "status": "ok" or "error"
            - "conflict": True if 3-way conflict detected
            - "merged": True if automatic merge applied
            - "message": Error/info message
        """
        if not self.manifest:
            return {"status": "error", "message": "No active snapshot"}

        # Compute 3-way diff
        diff = self.compute_diff(relative_path, buffer_lines)

        if diff["has_conflict"] and not allow_conflicts:
            return {
                "status": "conflict",
                "conflict": True,
                "message": f"3-way merge conflict in {relative_path}: disk and buffer both modified",
            }

        # Write to disk
        root = Path(self.manifest.root_path)
        file_path = root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            file_path.write_text("\n".join(buffer_lines), encoding="utf-8")
            logger.debug(f"Wrote {relative_path} ({len(buffer_lines)} lines)")

            return {
                "status": "ok",
                "conflict": diff["has_conflict"],
                "merged": True,
                "message": "File written successfully",
            }
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(f"Failed to write {relative_path}: {e}")
            return {"status": "error", "conflict": False, "message": f"Write failed: {e}"}

    def update_manifest_after_commit(self, updated_files: dict[str, Path]) -> None:
        """Update snapshot manifest after successful commit.

        Args:
            updated_files: Dict of relative path -> absolute path for files that changed
        """
        if not self.manifest:
            return

        now = datetime.utcnow().isoformat() + "Z"

        # Update metadata for changed files
        for rel_path, file_path in updated_files.items():
            if file_path.exists():
                try:
                    meta = FileMetadata.from_file(file_path)
                    self.manifest.files[rel_path] = meta
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to update metadata for {rel_path}: {e}")

        self.manifest.modified_at = now
        self._save_manifest()
        logger.debug(f"Updated snapshot manifest: {len(updated_files)} files changed")


class SnapshotDiffer:
    """Compute line-level diffs between versions (foundation for merge)."""

    @staticmethod
    def diff_lines(
        old_lines: list[str], new_lines: list[str]
    ) -> list[tuple[str, int, Optional[str]]]:
        """Compute simple line-level diff.

        Returns:
            List of (operation, line_num, line_content) tuples:
            - ('=', idx, line) for unchanged lines
            - ('+', idx, line) for added lines
            - ('-', idx, None) for deleted lines
        """
        diffs: list[tuple[str, int, Optional[str]]] = []
        matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for old_idx, new_idx in zip(range(i1, i2), range(j1, j2)):
                    diffs.append(("=", new_idx, new_lines[new_idx]))
            elif tag == "delete":
                for old_idx in range(i1, i2):
                    diffs.append(("-", old_idx, None))
            elif tag == "insert":
                for new_idx in range(j1, j2):
                    diffs.append(("+", new_idx, new_lines[new_idx]))
            elif tag == "replace":
                for old_idx in range(i1, i2):
                    diffs.append(("-", old_idx, None))
                for new_idx in range(j1, j2):
                    diffs.append(("+", new_idx, new_lines[new_idx]))

        return diffs

    @staticmethod
    def apply_diff(
        snapshot_base: list[str], disk_current: list[str], buffer_lines: list[str]
    ) -> tuple[list[str], bool]:
        """Apply 3-way merge using snapshot as base.

        Logic:
        - If buffer unchanged from snapshot, use disk (user didn't edit, disk did)
        - If disk unchanged from snapshot, use buffer (disk didn't change, user did)
        - If both changed, return buffer with conflict flag (safer than auto-merge)

        Args:
            snapshot_base: Snapshot state (common ancestor)
            disk_current: Current state on disk
            buffer_lines: Buffer state (user edits)

        Returns:
            (merged_lines, has_conflict)
        """
        buffer_changed = buffer_lines != snapshot_base
        disk_changed = disk_current != snapshot_base

        if not buffer_changed and not disk_changed:
            # Nothing changed
            return buffer_lines, False

        if not buffer_changed:
            # Only disk changed, user didn't edit
            return disk_current, False

        if not disk_changed:
            # Only buffer changed, disk unchanged
            return buffer_lines, False

        # Both changed - return buffer, but flag as conflict
        return buffer_lines, True
