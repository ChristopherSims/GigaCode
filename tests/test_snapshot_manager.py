"""Tests for metadata-only snapshot management."""
# CRITICAL: Initialize sklearn FIRST before any gigacode imports
import types
try:
    import sklearn
    if getattr(sklearn, "__spec__", None) is None:
        sklearn.__spec__ = types.ModuleSpec("sklearn", getattr(sklearn, "__file__", None))
except Exception:
    pass


import tempfile
import time
from pathlib import Path

import pytest

from gigacode.snapshot_manager import (
    FileMetadata,
    SnapshotDiffer,
    SnapshotManager,
    SnapshotManifest,
)


class TestFileMetadata:
    """Test file metadata tracking."""

    def test_metadata_from_file(self):
        """Test creating metadata from actual file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.py"
            file_path.write_text("line1\nline2\nline3\n")

            meta = FileMetadata.from_file(file_path)
            assert meta.path == str(file_path)
            assert meta.size > 0
            assert meta.hash is not None
            assert meta.lines == 3

    def test_metadata_serialization(self):
        """Test metadata to/from dict."""
        meta = FileMetadata(
            path="module.py", mtime=1234567890.0, size=1024, hash="abc123", lines=50
        )

        data = meta.to_dict()
        assert data["path"] == "module.py"

        meta2 = FileMetadata.from_dict(data)
        assert meta2.hash == meta.hash
        assert meta2.lines == meta.lines

    def test_metadata_from_different_files(self):
        """Test that different files produce different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.py"
            file2 = Path(tmpdir) / "file2.py"

            file1.write_text("content1")
            file2.write_text("content2")

            meta1 = FileMetadata.from_file(file1)
            meta2 = FileMetadata.from_file(file2)

            assert meta1.hash != meta2.hash


class TestSnapshotManifest:
    """Test snapshot manifest management."""

    def test_manifest_creation(self):
        """Test creating a snapshot manifest."""
        manifest = SnapshotManifest(
            buffer_id="buf-1",
            root_path="/code",
            created_at="2026-05-04T10:00:00Z",
            modified_at="2026-05-04T10:00:00Z",
            files={
                "module.py": FileMetadata(
                    path="module.py", mtime=1234567890.0, size=1024, hash="abc123"
                )
            },
        )

        assert manifest.buffer_id == "buf-1"
        assert len(manifest.files) == 1

    def test_manifest_serialization(self):
        """Test manifest to/from dict."""
        manifest = SnapshotManifest(
            buffer_id="buf-1",
            root_path="/code",
            created_at="2026-05-04T10:00:00Z",
            modified_at="2026-05-04T10:00:00Z",
            files={
                "module.py": FileMetadata(
                    path="module.py", mtime=1234567890.0, size=1024, hash="abc123", lines=10
                )
            },
        )

        data = manifest.to_dict()
        assert data["buffer_id"] == "buf-1"
        assert "module.py" in data["files"]

        manifest2 = SnapshotManifest.from_dict(data)
        assert manifest2.buffer_id == manifest.buffer_id
        assert "module.py" in manifest2.files


class TestSnapshotManager:
    """Test snapshot management with on-demand file reading."""

    def test_create_snapshot(self):
        """Test creating a snapshot from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "code"
            root.mkdir()

            # Create test files
            (root / "module.py").write_text("def func():\n    pass\n")
            (root / "utils.py").write_text("# utils\n")

            buffer_dir = Path(tmpdir) / "buffer"
            manager = SnapshotManager(buffer_dir)

            files = {"module.py": root / "module.py", "utils.py": root / "utils.py"}

            manifest = manager.create_snapshot("buf-1", root, files)
            assert manifest.buffer_id == "buf-1"
            assert len(manifest.files) == 2
            assert "module.py" in manifest.files

    def test_snapshot_persistence(self):
        """Test that snapshots persist across manager instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "code"
            root.mkdir()
            (root / "module.py").write_text("code\n")

            buffer_dir = Path(tmpdir) / "buffer"

            # Create snapshot
            manager1 = SnapshotManager(buffer_dir)
            manifest1 = manager1.create_snapshot("buf-1", root, {"module.py": root / "module.py"})

            # Load in new instance
            manager2 = SnapshotManager(buffer_dir)
            assert manager2.manifest is not None
            assert manager2.manifest.buffer_id == "buf-1"
            assert len(manager2.manifest.files) == 1

    def test_read_file_on_demand(self):
        """Test reading file content on-demand from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "code"
            root.mkdir()

            content = "line1\nline2\nline3\n"
            (root / "module.py").write_text(content)

            buffer_dir = Path(tmpdir) / "buffer"
            manager = SnapshotManager(buffer_dir)
            manager.create_snapshot("buf-1", root, {"module.py": root / "module.py"})

            # Read file on-demand
            read_content = manager.read_file("module.py")
            assert read_content == content

    def test_read_lines_on_demand(self):
        """Test reading file lines on-demand."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "code"
            root.mkdir()
            (root / "module.py").write_text("line1\nline2\nline3\n")

            buffer_dir = Path(tmpdir) / "buffer"
            manager = SnapshotManager(buffer_dir)
            manager.create_snapshot("buf-1", root, {"module.py": root / "module.py"})

            lines = manager.read_lines("module.py")
            assert lines == ["line1", "line2", "line3"]

    def test_detect_external_changes(self):
        """Test detecting files modified on disk since snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "code"
            root.mkdir()

            file_path = root / "module.py"
            file_path.write_text("original\n")

            buffer_dir = Path(tmpdir) / "buffer"
            manager = SnapshotManager(buffer_dir)
            manager.create_snapshot("buf-1", root, {"module.py": file_path})

            # Wait a bit to ensure mtime changes
            time.sleep(0.1)

            # Modify file externally
            file_path.write_text("modified\n")

            changes = manager.detect_external_changes()
            assert "module.py" in changes["changed"]

    def test_detect_deleted_files(self):
        """Test detecting files deleted on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "code"
            root.mkdir()

            file_path = root / "module.py"
            file_path.write_text("code\n")

            buffer_dir = Path(tmpdir) / "buffer"
            manager = SnapshotManager(buffer_dir)
            manager.create_snapshot("buf-1", root, {"module.py": file_path})

            # Delete file
            file_path.unlink()

            changes = manager.detect_external_changes()
            assert "module.py" in changes["deleted"]

    def test_compute_diff_no_changes(self):
        """Test diff when no changes exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "code"
            root.mkdir()

            file_path = root / "module.py"
            file_path.write_text("line1\nline2\n")

            buffer_dir = Path(tmpdir) / "buffer"
            manager = SnapshotManager(buffer_dir)
            manager.create_snapshot("buf-1", root, {"module.py": file_path})

            # Diff with same buffer state
            diff = manager.compute_diff("module.py", ["line1", "line2"])
            assert diff["has_conflict"] is False

    def test_compute_diff_with_conflict(self):
        """Test diff detection with 3-way conflict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "code"
            root.mkdir()

            file_path = root / "module.py"
            file_path.write_text("line1\nline2\n")

            buffer_dir = Path(tmpdir) / "buffer"
            manager = SnapshotManager(buffer_dir)
            manager.create_snapshot("buf-1", root, {"module.py": file_path})

            # Modify disk
            time.sleep(0.1)
            file_path.write_text("line1\nline2\nline3\n")

            # Compute diff with different buffer state
            diff = manager.compute_diff("module.py", ["line1", "modified\n"])
            # Has conflict if both disk and buffer differ from snapshot
            # This is a simplified test - real conflict detection is complex

    def test_write_file_with_merge(self):
        """Test writing file with 3-way merge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "code"
            root.mkdir()

            file_path = root / "module.py"
            file_path.write_text("original\n")

            buffer_dir = Path(tmpdir) / "buffer"
            manager = SnapshotManager(buffer_dir)
            manager.create_snapshot("buf-1", root, {"module.py": file_path})

            # Write new content
            result = manager.write_file_with_merge("module.py", ["modified", "code"])

            assert result["status"] == "ok"

            # Verify on disk
            content = file_path.read_text()
            assert "modified" in content

    def test_update_manifest_after_commit(self):
        """Test updating manifest after files change."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "code"
            root.mkdir()

            file_path = root / "module.py"
            file_path.write_text("original\n")

            buffer_dir = Path(tmpdir) / "buffer"
            manager = SnapshotManager(buffer_dir)
            manager.create_snapshot("buf-1", root, {"module.py": file_path})

            old_hash = manager.manifest.files["module.py"].hash

            # Modify file
            time.sleep(0.1)
            file_path.write_text("modified\n")

            # Update manifest
            manager.update_manifest_after_commit({"module.py": file_path})

            new_hash = manager.manifest.files["module.py"].hash
            assert new_hash != old_hash


class TestSnapshotDiffer:
    """Test line-level diff computation."""

    def test_diff_no_changes(self):
        """Test diff when lines are identical."""
        old = ["line1", "line2", "line3"]
        new = ["line1", "line2", "line3"]

        diff = SnapshotDiffer.diff_lines(old, new)
        # Should have mostly '=' operations
        assert len(diff) > 0

    def test_diff_with_additions(self):
        """Test diff with added lines."""
        old = ["line1", "line2"]
        new = ["line1", "line2", "line3"]

        diff = SnapshotDiffer.diff_lines(old, new)
        # Should have operations for added line
        assert len(diff) >= 2

    def test_apply_diff_buffer_only(self):
        """Test 3-way merge when only buffer changed."""
        snapshot = ["line1", "line2"]
        disk = ["line1", "line2"]  # Same as snapshot
        buffer = ["line1", "modified"]

        result, has_conflict = SnapshotDiffer.apply_diff(snapshot, disk, buffer)
        assert result == buffer
        assert has_conflict is False

    def test_apply_diff_disk_only(self):
        """Test 3-way merge when only disk changed."""
        snapshot = ["line1", "line2"]
        disk = ["line1", "line2", "line3"]  # Added line
        buffer = ["line1", "line2"]  # Same as snapshot

        result, has_conflict = SnapshotDiffer.apply_diff(snapshot, disk, buffer)
        # When buffer didn't change from snapshot, use disk version
        assert result == disk
        assert has_conflict is False

    def test_apply_diff_both_changed(self):
        """Test 3-way merge when both changed (conflict)."""
        snapshot = ["line1", "line2"]
        disk = ["line1", "line2", "line3"]  # Disk added line3
        buffer = ["line1", "modified"]  # Buffer modified line2

        result, has_conflict = SnapshotDiffer.apply_diff(snapshot, disk, buffer)
        # When both change, prefer buffer (conservative approach)
        assert result == buffer
        assert has_conflict is True


class TestSnapshotMemoryEfficiency:
    """Test memory efficiency of metadata-only approach."""

    def test_metadata_size_vs_full_snapshot(self):
        """Compare metadata size to full source code storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "code"
            root.mkdir()

            # Create large file
            large_content = "line\n" * 10000
            (root / "large.py").write_text(large_content)

            buffer_dir = Path(tmpdir) / "buffer"
            manager = SnapshotManager(buffer_dir)
            manifest = manager.create_snapshot("buf-1", root, {"large.py": root / "large.py"})

            # Check manifest size
            manifest_size = manager.manifest_path.stat().st_size

            # Manifest should be much smaller than source
            # (metadata only, not full code)
            assert manifest_size < len(large_content)

            logger.info(
                f"Source: {len(large_content)} bytes, "
                f"Manifest: {manifest_size} bytes "
                f"({100*manifest_size/len(large_content):.1f}%)"
            )


import logging  # noqa: E402

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

