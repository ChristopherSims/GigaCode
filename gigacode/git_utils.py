"""Git integration for buffer-aware version control operations.

Provides git status, diff, blame, and log operations scoped to embedded buffers,
helping AI agents understand what changed in git vs what they changed in the buffer.

Uses GitPython when available, falls back to subprocess-based git commands.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "GitInfo",
    "GitStatus",
    "GitBlameLine",
    "GitUtils",
    "run_git",
]


@dataclass
class GitInfo:
    """Basic git repository information."""

    is_git_repo: bool
    repo_root: str | None
    branch: str | None
    head_commit: str | None
    remote_url: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GitStatus:
    """Git working tree status."""

    branch: str
    ahead: int
    behind: int
    modified: list[str]
    staged: list[str]
    untracked: list[str]
    conflicted: list[str]
    clean: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GitBlameLine:
    """Single line blame information."""

    line: int
    commit: str
    author: str
    date: str
    message: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GitUtils:
    """Git operations scoped to a buffer's source directory."""

    def __init__(self, source_dir: str | Path) -> None:
        self.source_dir = Path(source_dir)
        self._repo_root: Path | None = None
        self._repo = None
        self._has_gitpython = False
        self._git_binary = "git"
        self._init_repo()

    def _init_repo(self) -> None:
        """Detect git repository and initialize backend."""
        # Find git root
        current = self.source_dir.resolve()
        while current != current.parent:
            if (current / ".git").exists():
                self._repo_root = current
                break
            current = current.parent

        if self._repo_root is None:
            return

        # Try GitPython
        try:
            import git

            self._repo = git.Repo(str(self._repo_root))
            self._has_gitpython = True
        except (ImportError, git.InvalidGitRepositoryError, git.NoSuchPathError) as e:
            logger.debug(f"GitPython not available for {self._repo_root}: {e}")
            self._has_gitpython = False

        # Check for git binary
        try:
            result = subprocess.run(
                [self._git_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                self._git_binary = None
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            self._git_binary = None

    def is_repo(self) -> bool:
        """Check if source directory is inside a git repository."""
        return self._repo_root is not None

    def get_info(self) -> GitInfo:
        """Get basic git repository information."""
        if not self._repo_root:
            return GitInfo(
                is_git_repo=False,
                repo_root=None,
                branch=None,
                head_commit=None,
                remote_url=None,
            )

        try:
            if self._has_gitpython and self._repo:
                branch = self._repo.active_branch.name
                head = self._repo.head.commit.hexsha[:8]
                remotes = list(self._repo.remotes)
                remote_url = remotes[0].url if remotes else None
            else:
                # Fallback to subprocess
                branch = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
                head = self._run_git(["rev-parse", "--short", "HEAD"]).strip()
                remote_url = self._run_git(["remote", "get-url", "origin"]).strip()
                if "fatal" in remote_url:
                    remote_url = None

            return GitInfo(
                is_git_repo=True,
                repo_root=str(self._repo_root),
                branch=branch,
                head_commit=head,
                remote_url=remote_url,
            )
        except (subprocess.CalledProcessError, RuntimeError, ValueError) as e:
            return GitInfo(
                is_git_repo=True,
                repo_root=str(self._repo_root),
                error=str(e),
            )

    def get_status(self) -> GitStatus | dict[str, Any]:
        """Get git working tree status."""
        if not self._repo_root:
            return {"status": "error", "message": "Not a git repository"}

        try:
            if self._has_gitpython and self._repo:
                return self._status_gitpython()
            else:
                return self._status_subprocess()
        except (subprocess.CalledProcessError, RuntimeError, ValueError) as e:
            return {"status": "error", "message": f"Git status failed: {e}"}

    def _status_gitpython(self) -> GitStatus:
        """Get status using GitPython."""
        repo = self._repo
        branch = repo.active_branch.name

        # Ahead/behind
        ahead = behind = 0
        try:
            tracking = repo.active_branch.tracking_branch()
            if tracking:
                commits_behind = list(repo.iter_commits(f"{tracking.name}..{branch}"))
                commits_ahead = list(repo.iter_commits(f"{branch}..{tracking.name}"))
                ahead = len(commits_behind)
                behind = len(commits_ahead)
        except (RuntimeError, ValueError):
            pass

        modified = []
        staged = []
        untracked = []
        conflicted = []

        for item in repo.index.diff(None):
            if item.change_type in ("M", "T"):
                modified.append(item.a_path)
            elif item.change_type == "D":
                modified.append(item.a_path)

        for item in repo.index.diff(repo.head.commit):
            staged.append(item.a_path)

        untracked = repo.untracked_files

        return GitStatus(
            branch=branch,
            ahead=ahead,
            behind=behind,
            modified=modified,
            staged=staged,
            untracked=untracked,
            conflicted=conflicted,
            clean=(not modified and not staged and not untracked),
        )

    def _status_subprocess(self) -> GitStatus:
        """Get status using git subprocess."""
        # Get branch
        branch = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()

        # Ahead/behind
        ahead = behind = 0
        tracking = self._run_git(["rev-parse", "--abbrev-ref", "@{upstream}"]).strip()
        if "fatal" not in tracking and "No upstream" not in tracking:
            count = self._run_git(
                ["rev-list", "--left-right", "--count", f"HEAD...{tracking}"]
            ).strip()
            parts = count.split("\t")
            if len(parts) == 2:
                ahead = int(parts[0])
                behind = int(parts[1])

        # Parse porcelain status
        status_output = self._run_git(["status", "--porcelain", "-uall"])

        modified = []
        staged = []
        untracked = []
        conflicted = []

        for line in status_output.splitlines():
            if len(line) < 2:
                continue
            x, y = line[0], line[1]
            filepath = line[3:].strip()

            if x == "U" or y == "U" or (x == "D" and y == "D"):
                conflicted.append(filepath)
            elif x != " " and x != "?":
                staged.append(filepath)
            elif y == "M" or y == "D":
                modified.append(filepath)
            elif x == "?" and y == "?":
                untracked.append(filepath)

        return GitStatus(
            branch=branch,
            ahead=ahead,
            behind=behind,
            modified=modified,
            staged=staged,
            untracked=untracked,
            conflicted=conflicted,
            clean=(not modified and not staged and not untracked and not conflicted),
        )

    def get_diff(
        self,
        file_path: str | None = None,
        against: str = "HEAD",
    ) -> dict[str, Any]:
        """Get diff of file(s) against a reference.

        Args:
            file_path: Specific file, or None for all files.
            against: Reference to diff against ("HEAD", "STAGED", commit hash).

        Returns:
            Dict with diff text and status.
        """
        if not self._repo_root:
            return {"status": "error", "message": "Not a git repository"}

        try:
            if against == "STAGED":
                ref = "--staged"
            else:
                ref = f"{against}..."

            cmd = ["diff", ref]
            if file_path:
                cmd.extend(["--", file_path])

            diff_text = self._run_git(cmd)
            return {
                "status": "ok",
                "against": against,
                "file": file_path,
                "diff": diff_text,
            }
        except (subprocess.CalledProcessError, RuntimeError, ValueError) as e:
            return {"status": "error", "message": f"Git diff failed: {e}"}

    def blame(
        self,
        file_path: str,
        line: int | None = None,
        num_lines: int = 1,
    ) -> dict[str, Any]:
        """Get blame information for a file or specific line.

        Args:
            file_path: File to blame.
            line: Specific line (1-based), or None for full file.
            num_lines: Number of lines to show context.

        Returns:
            Dict with blame entries.
        """
        if not self._repo_root:
            return {"status": "error", "message": "Not a git repository"}

        try:
            cmd = ["blame", "-l", "--date=iso", "--porcelain"]
            if line is not None:
                # Git blame uses 1-based line numbers
                cmd.extend(["-L", f"{line},{line + num_lines - 1}"])
            cmd.extend(["--", file_path])

            blame_output = self._run_git(cmd)

            entries: list[GitBlameLine] = []
            current_entry: dict[str, Any] | None = None
            current_line_content = ""
            line_number = 0

            for output_line in blame_output.splitlines():
                if not output_line:
                    continue

                # First line of each entry: <sha1> <sourceline> <resultline> <num_lines>
                parts = output_line.split()
                if len(parts) >= 3 and len(parts[0]) == 40:
                    # Save previous entry
                    if current_entry:
                        entries.append(
                            GitBlameLine(
                                line=current_entry.get("line", 0),
                                commit=current_entry.get("commit", "")[:8],
                                author=current_entry.get("author", ""),
                                date=current_entry.get("author-time", ""),
                                message=current_entry.get("summary", ""),
                                content=current_line_content,
                            )
                        )

                    current_entry = {
                        "commit": parts[0],
                        "line": int(parts[2]),
                    }
                    current_line_content = ""
                elif current_entry and output_line.startswith("\t"):
                    # Line content
                    current_line_content = output_line[1:]
                elif current_entry:
                    # Metadata
                    if " " in output_line:
                        key, value = output_line.split(" ", 1)
                        current_entry[key] = value

            # Save last entry
            if current_entry:
                entries.append(
                    GitBlameLine(
                        line=current_entry.get("line", 0),
                        commit=current_entry.get("commit", "")[:8],
                        author=current_entry.get("author", ""),
                        date=current_entry.get("author-time", ""),
                        message=current_entry.get("summary", ""),
                        content=current_line_content,
                    )
                )

            return {
                "status": "ok",
                "file": file_path,
                "line": line,
                "entries": [e.to_dict() for e in entries],
            }
        except (subprocess.CalledProcessError, RuntimeError, ValueError) as e:
            return {"status": "error", "message": f"Git blame failed: {e}"}

    def show_file_at_commit(
        self,
        file_path: str,
        commit: str = "HEAD",
    ) -> dict[str, Any]:
        """Show file content at a specific commit.

        Args:
            file_path: File path relative to repo root.
            commit: Commit hash or reference.

        Returns:
            Dict with file content.
        """
        if not self._repo_root:
            return {"status": "error", "message": "Not a git repository"}

        try:
            content = self._run_git(["show", f"{commit}:{file_path}"])
            return {
                "status": "ok",
                "file": file_path,
                "commit": commit,
                "content": content,
            }
        except (subprocess.CalledProcessError, RuntimeError, ValueError) as e:
            return {"status": "error", "message": f"Git show failed: {e}"}

    def _run_git(self, args: list[str]) -> str:
        """Run a git command in the repo root."""
        if not self._repo_root:
            raise RuntimeError("Not a git repository")

        if self._has_gitpython and self._repo:
            # For simple read-only operations, use subprocess for consistency
            pass

        if not self._git_binary:
            raise RuntimeError("Git binary not found")

        cmd = [self._git_binary, "-C", str(self._repo_root)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0 and result.stderr:
            # Some git commands return non-zero with no error (e.g., diff with no changes)
            if "not" in result.stderr.lower() or "fatal" in result.stderr.lower():
                raise RuntimeError(result.stderr.strip())

        return result.stdout


def get_git_info(source_dir: str | Path) -> dict[str, Any]:
    """Convenience function: get git info for a directory."""
    utils = GitUtils(source_dir)
    return utils.get_info().to_dict()


def get_git_status(source_dir: str | Path) -> dict[str, Any]:
    """Convenience function: get git status for a directory."""
    utils = GitUtils(source_dir)
    status = utils.get_status()
    if isinstance(status, dict):
        return status
    return status.to_dict()


def run_git(args: list[str], cwd: str | Path | None = None) -> str:
    """Run a git command and return stdout.

    Module-level convenience for one-off git calls without
    constructing a GitUtils instance.

    Args:
        args: Git subcommand and arguments, e.g. ["log", "--oneline", "-10"].
        cwd: Working directory for the git command.

    Returns:
        Git command stdout as a string.

    Raises:
        RuntimeError: If the git command exits with a non-zero code.
    """
    cmd = ["git"]
    if cwd:
        cmd += ["-C", str(cwd)]
    cmd += args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0 and result.stderr:
        raise RuntimeError(result.stderr.strip())
    return result.stdout
