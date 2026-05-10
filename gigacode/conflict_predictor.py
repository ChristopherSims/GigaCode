"""
Conflict Prediction

Predicts potential merge conflicts by analyzing commits since embed time,
warning users before they attempt to commit.

Key features:
- Git history scanning for commits touching edited files
- Risk level assessment (low | medium | high)
- Dependency-based conflict detection
- Actionable recommendations
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Set

from gigacode.dependency_graph import DependencyGraph
from gigacode.metrics import log_metric, timer

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Conflict risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CommitInfo:
    """Information about a commit."""

    sha: str
    author: str
    timestamp: datetime
    message: str
    files_changed: List[str]


@dataclass
class FileRisk:
    """Risk assessment for a single file."""

    file: str
    risk: RiskLevel
    reason: str
    commits: List[CommitInfo] = field(default_factory=list)
    is_in_dirty_queue: bool = False
    suggested_action: str = ""


@dataclass
class DependencyRisk:
    """Risk assessment for a dependency."""

    dependency: str
    file: str  # File containing dependency
    risk: RiskLevel
    reason: str
    edits_depending_on_it: List[str] = field(default_factory=list)


@dataclass
class ConflictPrediction:
    """Complete conflict prediction analysis."""

    embed_point: str  # Commit SHA and timestamp
    current_head: str  # Current HEAD SHA and timestamp
    time_since_embed_minutes: int
    commits_since_embed: int
    risk_level: RiskLevel
    file_risks: List[FileRisk] = field(default_factory=list)
    dependency_risks: List[DependencyRisk] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    auto_actions: Dict[str, any] = field(default_factory=dict)


class ConflictPredictor:
    """
    Predicts merge conflicts by analyzing external changes.
    """

    def __init__(
        self,
        buffer_manager,
        git_utils,
        dependency_graph: DependencyGraph,
    ):
        """
        Initialize predictor.

        Args:
            buffer_manager: BufferManager instance
            git_utils: Git utilities module
            dependency_graph: DependencyGraph instance
        """
        self.buffer_manager = buffer_manager
        self.git_utils = git_utils
        self.dependency_graph = dependency_graph

    def predict_conflicts(self, buffer_id: str) -> ConflictPrediction:
        """
        Predict conflicts for a buffer.

        Args:
            buffer_id: Target buffer

        Returns:
            ConflictPrediction analysis
        """
        with timer("predict_conflicts"):
            # Try multiple possible API names for compatibility
            buffer = None
            if hasattr(self.buffer_manager, "_get_buffer_info"):
                buffer = self.buffer_manager._get_buffer_info(buffer_id)
            elif hasattr(self.buffer_manager, "get_buffer_info"):
                buffer = self.buffer_manager.get_buffer_info(buffer_id)
            elif hasattr(self.buffer_manager, "get"):
                buffer = self.buffer_manager.get(buffer_id)

            if buffer is None:
                # Return empty prediction
                return ConflictPrediction(
                    embed_point="unknown",
                    current_head="unknown",
                    time_since_embed_minutes=0,
                    commits_since_embed=0,
                    risk_level=RiskLevel.LOW,
                    file_risks=[],
                    dependency_risks=[],
                    recommendations=["Buffer not found"],
                    auto_actions={},
                )

            # Get embedding metadata
            embed_sha = buffer.get("embed_sha", "unknown")
            embed_timestamp = buffer.get("embed_timestamp", datetime.now())

            # Get current HEAD
            head_sha, head_timestamp = self._get_current_head()

            # Get commits since embed
            commits_since = self._get_commits_since(embed_sha, head_sha)

            # Get edited files
            dirty_files = self._get_dirty_files(buffer)

            # Assess file risks
            file_risks = self._assess_file_risks(
                dirty_files,
                commits_since,
                buffer,
            )

            # Assess dependency risks
            dep_risks = self._assess_dependency_risks(
                buffer_id,
                dirty_files,
                commits_since,
            )

            # Determine overall risk level
            overall_risk = self._compute_risk_level(file_risks, dep_risks)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_risk,
                file_risks,
                dep_risks,
            )

            # Determine auto-actions
            auto_actions = self._compute_auto_actions(overall_risk)

            prediction = ConflictPrediction(
                embed_point=f"{embed_sha[:7]} ({embed_timestamp.strftime('%b %d, %Y, %I:%M %p')})",
                current_head=f"{head_sha[:7]} ({head_timestamp.strftime('%b %d, %Y, %I:%M %p')})",
                time_since_embed_minutes=int(
                    (head_timestamp - embed_timestamp).total_seconds() / 60
                ),
                commits_since_embed=len(commits_since),
                risk_level=overall_risk,
                file_risks=file_risks,
                dependency_risks=dep_risks,
                recommendations=recommendations,
                auto_actions=auto_actions,
            )

            log_metric(
                "predict_conflicts",
                {
                    "risk_level": overall_risk.value,
                    "commits_since_embed": len(commits_since),
                    "files_at_risk": len([r for r in file_risks if r.risk != RiskLevel.LOW]),
                },
            )

            return prediction

    def _get_current_head(self) -> tuple[str, datetime]:
        """Get current HEAD commit info using git_utils."""
        try:
            if hasattr(self.git_utils, "GitUtils"):
                git = self.git_utils.GitUtils()
                result = git.git_show("HEAD", format="%H|%ci")
                if result.get("status") == "ok":
                    lines = result.get("output", "").strip().split("|")
                    if len(lines) >= 2:
                        sha = lines[0].strip()
                        ts_str = lines[1].strip()
                        try:
                            ts = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            ts = datetime.now()
                        return (sha, ts)

            # Fallback: try subprocess
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                sha = result.stdout.strip()
                ts_result = subprocess.run(
                    ["git", "log", "-1", "--format=%ci", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if ts_result.returncode == 0:
                    ts_str = ts_result.stdout.strip()
                    try:
                        ts = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        ts = datetime.now()
                    return (sha, ts)
        except (RuntimeError, ValueError, TypeError, ImportError, OSError) as e:
            logger.warning(f"Failed to get current HEAD: {e}")

        # Final fallback
        return ("unknown", datetime.now())

    def _get_commits_since(self, embed_sha: str, head_sha: str) -> List[CommitInfo]:
        """
        Get commits between embed point and HEAD using git log.

        Args:
            embed_sha: Embedding commit SHA
            head_sha: Current HEAD SHA

        Returns:
            List of commits in that range
        """
        commits: List[CommitInfo] = []

        # Skip if embed_sha is unknown or same as head
        if embed_sha == "unknown" or embed_sha == head_sha:
            return commits

        try:
            # Try git_utils first
            if hasattr(self.git_utils, "GitUtils"):
                git = self.git_utils.GitUtils()
                log_result = git.git_show(
                    f"{embed_sha}..{head_sha}",
                    format="%H|%an|%ci|%s",
                )
                if log_result.get("status") == "ok":
                    output = log_result.get("output", "")
                    for line in output.strip().splitlines():
                        parts = line.split("|")
                        if len(parts) >= 4:
                            sha, author, ts_str, message = parts[0], parts[1], parts[2], parts[3]
                            try:
                                ts = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                ts = datetime.now()
                            commits.append(
                                CommitInfo(
                                    sha=sha,
                                    author=author,
                                    timestamp=ts,
                                    message=message,
                                    files_changed=[],
                                )
                            )
                    return commits

            # Fallback: subprocess
            import subprocess

            result = subprocess.run(
                ["git", "log", f"{embed_sha}..{head_sha}", "--format=%H|%an|%ci|%s"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    parts = line.split("|")
                    if len(parts) >= 4:
                        sha, author, ts_str, message = parts[0], parts[1], parts[2], parts[3]
                        try:
                            ts = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            ts = datetime.now()
                        commits.append(
                            CommitInfo(
                                sha=sha,
                                author=author,
                                timestamp=ts,
                                message=message,
                                files_changed=[],
                            )
                        )

            # Try to get files changed per commit
            for commit in commits:
                files_result = subprocess.run(
                    ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit.sha],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if files_result.returncode == 0:
                    commit.files_changed = [
                        f.strip() for f in files_result.stdout.strip().splitlines() if f.strip()
                    ]

        except (RuntimeError, ValueError, TypeError, ImportError, OSError) as e:
            logger.warning(f"Failed to get commits since embed: {e}")

        return commits

    def _get_dirty_files(self, buffer: dict) -> Set[str]:
        """Get files being edited."""
        dirty = set()

        if "dirty_queue" in buffer:
            for entry in buffer["dirty_queue"]:
                dirty.add(entry.get("file"))

        if "edits" in buffer:
            for edit in buffer["edits"]:
                dirty.add(edit.get("file"))

        return dirty

    def _assess_file_risks(
        self,
        dirty_files: Set[str],
        commits_since: List[CommitInfo],
        buffer: dict,
    ) -> List[FileRisk]:
        """Assess risk for each edited file."""
        file_risks = []

        for file in dirty_files:
            # Find commits touching this file
            file_commits = [c for c in commits_since if file in c.files_changed]

            if not file_commits:
                risk = RiskLevel.LOW
                reason = "No external commits since embed"
            elif len(file_commits) == 1:
                risk = RiskLevel.LOW
                reason = "1 external commit, likely benign"
            elif len(file_commits) <= 2:
                risk = RiskLevel.MEDIUM
                reason = f"{len(file_commits)} commits since embed"
            else:
                risk = RiskLevel.HIGH
                reason = f"{len(file_commits)} commits since embed; high conflict risk"

            file_risk = FileRisk(
                file=file,
                risk=risk,
                reason=reason,
                commits=file_commits,
                is_in_dirty_queue=True,
                suggested_action=self._suggest_action(risk),
            )
            file_risks.append(file_risk)

        return file_risks

    def _assess_dependency_risks(
        self,
        buffer_id: str,
        dirty_files: Set[str],
        commits_since: List[CommitInfo],
    ) -> List[DependencyRisk]:
        """Assess risks from modified dependencies."""
        dep_risks = []

        for file in dirty_files:
            # Get dependencies of this file
            deps = self.dependency_graph.get_dependencies(buffer_id, file)

            for dep_file in deps:
                # Check if dependency was modified
                dep_commits = [c for c in commits_since if dep_file in c.files_changed]

                if not dep_commits:
                    continue  # No risk

                if len(dep_commits) == 1:
                    risk = RiskLevel.LOW
                    reason = "API unchanged, only internal refactoring"
                else:
                    risk = RiskLevel.MEDIUM
                    reason = f"{len(dep_commits)} commits to dependency"

                dep_risk = DependencyRisk(
                    dependency=dep_file,
                    file=file,
                    risk=risk,
                    reason=reason,
                    edits_depending_on_it=[file],
                )
                dep_risks.append(dep_risk)

        return dep_risks

    @staticmethod
    def _compute_risk_level(
        file_risks: List[FileRisk],
        dep_risks: List[DependencyRisk],
    ) -> RiskLevel:
        """Compute overall risk level."""
        # High risk if any file is HIGH
        if any(r.risk == RiskLevel.HIGH for r in file_risks):
            return RiskLevel.HIGH

        # Medium if any is MEDIUM
        if any(r.risk == RiskLevel.MEDIUM for r in file_risks):
            if any(d.risk == RiskLevel.MEDIUM for d in dep_risks):
                return RiskLevel.HIGH
            return RiskLevel.MEDIUM

        # Check dependency risks
        if any(d.risk == RiskLevel.HIGH for d in dep_risks):
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    @staticmethod
    def _suggest_action(risk: RiskLevel) -> str:
        """Suggest action based on risk level."""
        if risk == RiskLevel.LOW:
            return "Safe to commit directly"
        elif risk == RiskLevel.MEDIUM:
            return "Run tests before committing; if tests pass, safe to proceed"
        else:
            return "Run git_pull() and reload_codebase() before committing"

    @staticmethod
    def _generate_recommendations(
        risk_level: RiskLevel,
        file_risks: List[FileRisk],
        dep_risks: List[DependencyRisk],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if risk_level == RiskLevel.LOW:
            recommendations.append("✓ Buffer state is clean; safe to commit")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("⚠ Run tests to validate your changes")
            if any(r.risk == RiskLevel.MEDIUM for r in file_risks):
                recommendations.append("  Check diffs for merge conflicts")
            recommendations.append("  If tests pass, safe to commit")
        else:  # HIGH
            recommendations.append("[FAILED] External changes detected")
            recommendations.append("  → git_pull() to fetch latest changes")
            recommendations.append("  → reload_codebase() to refresh buffer")
            recommendations.append("  → Re-validate edits still make sense")
            recommendations.append("  → If conflicts, use tool.resolve_conflict()")

        return recommendations

    @staticmethod
    def _compute_auto_actions(risk_level: RiskLevel) -> Dict[str, any]:
        """Determine automatic actions to suggest."""
        return {
            "suggest_refresh": risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH),
            "suggest_test": risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH),
            "can_auto_merge": risk_level == RiskLevel.LOW,
        }


class ConflictPredictionService:
    """Public interface for conflict prediction."""

    def __init__(self, conflict_predictor: ConflictPredictor):
        """Initialize service."""
        self.conflict_predictor = conflict_predictor

    def predict_conflicts(self, buffer_id: str) -> Dict:
        """
        Predict merge conflicts.

        Args:
            buffer_id: Target buffer

        Returns:
            Conflict prediction as dict
        """
        prediction = self.conflict_predictor.predict_conflicts(buffer_id)

        return {
            "embed_point": prediction.embed_point,
            "current_head": prediction.current_head,
            "time_since_embed_minutes": prediction.time_since_embed_minutes,
            "commits_since_embed": prediction.commits_since_embed,
            "risk_level": prediction.risk_level.value,
            "file_risks": [
                {
                    "file": r.file,
                    "risk": r.risk.value,
                    "reason": r.reason,
                    "commits": [
                        {
                            "sha": c.sha,
                            "author": c.author,
                            "time": c.timestamp.isoformat(),
                            "message": c.message,
                            "lines_changed": c.files_changed,
                        }
                        for c in r.commits
                    ],
                    "is_in_dirty_queue": r.is_in_dirty_queue,
                    "suggested_action": r.suggested_action,
                }
                for r in prediction.file_risks
            ],
            "dependency_risks": [
                {
                    "dependency": r.dependency,
                    "file": r.file,
                    "risk": r.risk.value,
                    "reason": r.reason,
                    "edits_depending_on_it": r.edits_depending_on_it,
                }
                for r in prediction.dependency_risks
            ],
            "recommendations": prediction.recommendations,
            "auto_actions": prediction.auto_actions,
        }
