"""
Phase 4: Intent-Based Action Router

Analyzes user intent and recommends optimal sequence of actions
to minimize token usage and iterations.

Key features:
- Intent classification (feature, bug, refactor, docs, etc.)
- Action planning with estimated costs
- Cache state awareness
- Conflict prediction integration
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from gigacode.metrics import log_metric, timer


class IntentCategory(Enum):
    """Classified intent types."""

    FEATURE_ADDITION = "feature_addition"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    OPTIMIZATION = "optimization"
    CLEANUP = "cleanup"
    UNKNOWN = "unknown"


class ActionType(Enum):
    """Types of actions the router can recommend."""

    SEMANTIC_SEARCH = "semantic_search"
    LEXICAL_SEARCH = "lexical_search"
    SYMBOL_SEARCH = "symbol_search"
    READ_CODE = "read_code"
    DIFF_AGAINST_HEAD = "diff_against_head"
    DIFF_AGAINST_BRANCH = "diff_against_branch"
    PREDICT_CONFLICTS = "predict_conflicts"
    RUN_TESTS = "run_tests"
    WRITE_CODE = "write_code"
    COMMIT = "commit"
    PACK_CONTEXT = "pack_context"


@dataclass
class SearchAction:
    """Recommended search action."""

    action: ActionType = ActionType.SEMANTIC_SEARCH
    query: str = ""
    query_variants: List[str] = field(default_factory=list)
    estimated_tokens: int = 300
    estimated_duration_ms: int = 50
    priority: int = 1
    reason: str = ""
    fallback_search_type: Optional[ActionType] = None


@dataclass
class ReadAction:
    """Recommended file read action."""

    action: ActionType = ActionType.READ_CODE
    file: str = ""
    lines: Optional[tuple] = None  # (start, end)
    estimated_tokens: int = 200
    estimated_duration_ms: int = 20
    priority: int = 2
    reason: str = ""
    context: str = ""  # Why this file matters


@dataclass
class DiffAction:
    """Recommended diff action."""

    action: ActionType = ActionType.DIFF_AGAINST_HEAD
    file: str = ""
    against: str = "HEAD"  # HEAD or branch name
    estimated_tokens: int = 150
    estimated_duration_ms: int = 30
    priority: int = 3
    reason: str = ""


@dataclass
class ConflictCheckAction:
    """Check for conflicts before proceeding."""

    action: ActionType = ActionType.PREDICT_CONFLICTS
    estimated_tokens: int = 100
    estimated_duration_ms: int = 200
    priority: int = 1
    reason: str = "File has recent external commits"


@dataclass
class ActionPlan:
    """Complete action plan for a user intent."""

    intent: str
    intent_category: IntentCategory
    confidence: float  # 0.0-1.0
    recommended_actions: List[Any] = field(default_factory=list)
    total_estimated_tokens: int = 0
    total_estimated_duration_ms: int = 0
    warnings: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "intent": self.intent,
            "intent_category": self.intent_category.value,
            "confidence": self.confidence,
            "recommended_actions": [
                asdict(action) if hasattr(action, "__dataclass_fields__") else action
                for action in self.recommended_actions
            ],
            "total_estimated_tokens": self.total_estimated_tokens,
            "total_estimated_duration_ms": self.total_estimated_duration_ms,
            "warnings": self.warnings,
        }


class IntentClassifier:
    """Classify user intent from natural language."""

    KEYWORDS_BY_CATEGORY = {
        IntentCategory.FEATURE_ADDITION: [
            "add",
            "implement",
            "create",
            "new feature",
            "support",
            "enable",
            "introduce",
            "allow",
            "build",
            "develop",
        ],
        IntentCategory.BUG_FIX: [
            "fix",
            "bug",
            "error",
            "crash",
            "break",
            "issue",
            "fail",
            "broken",
            "not working",
            "problem",
            "wrong",
            "incorrect",
        ],
        IntentCategory.REFACTORING: [
            "refactor",
            "cleanup",
            "reorganize",
            "simplify",
            "improve",
            "restructure",
            "rename",
            "extract",
            "consolidate",
            "optimize",
            "performance",
        ],
        IntentCategory.DOCUMENTATION: [
            "document",
            "docstring",
            "comment",
            "example",
            "explain",
            "readme",
            "guide",
            "tutorial",
            "api",
            "help",
        ],
        IntentCategory.TESTING: [
            "test",
            "unittest",
            "coverage",
            "mock",
            "verify",
            "assert",
            "validate",
        ],
        IntentCategory.OPTIMIZATION: [
            "optimize",
            "fast",
            "speed",
            "performance",
            "efficient",
            "memory",
            "cache",
            "reduce latency",
            "throughput",
        ],
        IntentCategory.CLEANUP: [
            "cleanup",
            "remove",
            "delete",
            "dead code",
            "unused",
            "deprecate",
            "retire",
            "decommission",
        ],
    }

    @staticmethod
    def classify(intent: str) -> tuple[IntentCategory, float]:
        """
        Classify intent and return category with confidence.

        Args:
            intent: User's intent description

        Returns:
            Tuple of (IntentCategory, confidence score 0.0-1.0)
        """
        intent_lower = intent.lower()

        # Score each category
        scores = {}
        for category, keywords in IntentClassifier.KEYWORDS_BY_CATEGORY.items():
            score = sum(1 for kw in keywords if kw in intent_lower)
            scores[category] = score

        # Get top category
        if max(scores.values()) == 0:
            return IntentCategory.UNKNOWN, 0.3

        top_category = max(scores, key=scores.get)
        confidence = min(0.95, 0.5 + (scores[top_category] / 5.0))

        return top_category, confidence


class ActionPlanner:
    """Plan optimal action sequence for an intent."""

    def __init__(self, buffer_manager, search_service, diff_engine):
        """
        Initialize planner.

        Args:
            buffer_manager: BufferManager instance
            search_service: SearchService instance
            diff_engine: DiffEngine instance
        """
        self.buffer_manager = buffer_manager
        self.search_service = search_service
        self.diff_engine = diff_engine

    def plan(
        self,
        buffer_id: str,
        intent: str,
        budget: int = 4000,
    ) -> ActionPlan:
        """
        Plan action sequence for intent.

        Args:
            buffer_id: Target buffer
            intent: User's intent description
            budget: Max tokens to use for planning

        Returns:
            ActionPlan with recommended sequence
        """
        with timer("plan_actions"):
            # Classify intent
            category, confidence = IntentClassifier.classify(intent)

            # Get buffer state
            # Try multiple possible API names for compatibility
            buffer = None
            if hasattr(self.buffer_manager, "_get_buffer_info"):
                buffer = self.buffer_manager._get_buffer_info(buffer_id)
            elif hasattr(self.buffer_manager, "get_buffer_info"):
                buffer = self.buffer_manager.get_buffer_info(buffer_id)
            elif hasattr(self.buffer_manager, "get"):
                buffer = self.buffer_manager.get(buffer_id)

            # Build action sequence based on category
            actions = []
            total_tokens = 0

            if category == IntentCategory.BUG_FIX:
                actions, total_tokens = self._plan_bug_fix(intent, buffer, budget)
            elif category == IntentCategory.FEATURE_ADDITION:
                actions, total_tokens = self._plan_feature(intent, buffer, budget)
            elif category == IntentCategory.REFACTORING:
                actions, total_tokens = self._plan_refactoring(intent, buffer, budget)
            elif category == IntentCategory.OPTIMIZATION:
                actions, total_tokens = self._plan_optimization(intent, buffer, budget)
            else:
                actions, total_tokens = self._plan_generic(intent, buffer, budget)

            # Check for cache staleness
            warnings = self._check_warnings(buffer, actions)

            # Compute total duration
            total_duration = sum(getattr(a, "estimated_duration_ms", 0) for a in actions)

            plan = ActionPlan(
                intent=intent,
                intent_category=category,
                confidence=confidence,
                recommended_actions=actions,
                total_estimated_tokens=total_tokens,
                total_estimated_duration_ms=total_duration,
                warnings=warnings,
            )

            log_metric(
                "intent_router.plan",
                {
                    "category": category.value,
                    "confidence": confidence,
                    "num_actions": len(actions),
                    "estimated_tokens": total_tokens,
                },
            )

            return plan

    def _plan_bug_fix(
        self,
        intent: str,
        buffer: dict,
        budget: int,
    ) -> tuple[List[Any], int]:
        """Plan actions for bug fixing."""
        actions = []
        tokens = 0

        # 1. Search for error/issue
        search_keywords = self._extract_error_keywords(intent)
        search_action = SearchAction(
            action=ActionType.SEMANTIC_SEARCH,
            query=f"error in {search_keywords} exception handler stack trace",
            query_variants=[
                f"bug in {search_keywords}",
                f"why does {search_keywords} fail",
                f"{search_keywords} traceback",
            ],
            estimated_tokens=320,
            priority=1,
            reason="Search for error location and related code",
            fallback_search_type=ActionType.LEXICAL_SEARCH,
        )
        actions.append(search_action)
        tokens += search_action.estimated_tokens

        # 2. Read identified files
        if tokens < budget * 0.5:
            read_action = ReadAction(
                action=ActionType.READ_CODE,
                file="",  # Will be populated from search results
                estimated_tokens=400,
                priority=2,
                reason="Understand error context and call stack",
                context="Error handling and recovery logic",
            )
            actions.append(read_action)
            tokens += read_action.estimated_tokens

        # 3. Run tests to identify failure
        if tokens < budget * 0.7:
            test_action = {
                "action": "run_tests",
                "estimated_tokens": 600,
                "priority": 3,
                "reason": "Identify failing test cases to narrow search",
            }
            actions.append(test_action)
            tokens += 600

        return actions, tokens

    def _plan_feature(
        self,
        intent: str,
        buffer: dict,
        budget: int,
    ) -> tuple[List[Any], int]:
        """Plan actions for feature addition."""
        actions = []
        tokens = 0

        # 1. Search for similar patterns
        search_action = SearchAction(
            action=ActionType.SEMANTIC_SEARCH,
            query=intent,
            estimated_tokens=300,
            priority=1,
            reason="Find existing patterns and similar implementations",
            fallback_search_type=ActionType.LEXICAL_SEARCH,
        )
        actions.append(search_action)
        tokens += 300

        # 2. Read related code
        if tokens < budget * 0.5:
            read_action = ReadAction(
                action=ActionType.READ_CODE,
                estimated_tokens=350,
                priority=2,
                reason="Understand architecture and dependencies for new feature",
            )
            actions.append(read_action)
            tokens += 350

        # 3. Check for conflicts
        if tokens < budget * 0.65:
            conflict_action = ConflictCheckAction(
                action=ActionType.PREDICT_CONFLICTS,
                priority=3,
                reason="Ensure no recent external changes that affect your edits",
            )
            actions.append(conflict_action)
            tokens += 100

        return actions, tokens

    def _plan_refactoring(
        self,
        intent: str,
        buffer: dict,
        budget: int,
    ) -> tuple[List[Any], int]:
        """Plan actions for refactoring."""
        actions = []
        tokens = 0

        # 1. Find all usages
        search_action = SearchAction(
            action=ActionType.SEMANTIC_SEARCH,
            query=f"where is {intent} used called referenced",
            estimated_tokens=350,
            priority=1,
            reason="Find all usages to ensure safe refactoring",
        )
        actions.append(search_action)
        tokens += 350

        # 2. Read related code
        if tokens < budget * 0.5:
            read_action = ReadAction(
                action=ActionType.READ_CODE,
                estimated_tokens=400,
                priority=2,
                reason="Understand current implementation and all call sites",
            )
            actions.append(read_action)
            tokens += 400

        # 3. Run comprehensive tests
        if tokens < budget * 0.7:
            test_action = {
                "action": "run_tests",
                "estimated_tokens": 800,
                "priority": 3,
                "reason": "Verify refactoring doesn't break functionality",
            }
            actions.append(test_action)
            tokens += 800

        return actions, tokens

    def _plan_optimization(
        self,
        intent: str,
        buffer: dict,
        budget: int,
    ) -> tuple[List[Any], int]:
        """Plan actions for performance optimization."""
        actions = []
        tokens = 0

        # 1. Profile and search for bottlenecks
        search_action = SearchAction(
            action=ActionType.SEMANTIC_SEARCH,
            query=f"slow performance bottleneck in {intent}",
            estimated_tokens=300,
            priority=1,
            reason="Identify performance bottlenecks and hot paths",
        )
        actions.append(search_action)
        tokens += 300

        # 2. Read implementation details
        if tokens < budget * 0.5:
            read_action = ReadAction(
                action=ActionType.READ_CODE,
                estimated_tokens=450,
                priority=2,
                reason="Analyze algorithm complexity and resource usage",
            )
            actions.append(read_action)
            tokens += 450

        return actions, tokens

    def _plan_generic(
        self,
        intent: str,
        buffer: dict,
        budget: int,
    ) -> tuple[List[Any], int]:
        """Plan actions for generic/unknown intent."""
        actions = []
        tokens = 0

        # Start with broad search
        search_action = SearchAction(
            action=ActionType.SEMANTIC_SEARCH,
            query=intent,
            estimated_tokens=350,
            priority=1,
            reason="Gather context on user's request",
        )
        actions.append(search_action)
        tokens += 350

        # Then read relevant code
        if tokens < budget * 0.6:
            read_action = ReadAction(
                action=ActionType.READ_CODE,
                estimated_tokens=350,
                priority=2,
                reason="Understand context and available options",
            )
            actions.append(read_action)
            tokens += 350

        return actions, tokens

    @staticmethod
    def _extract_error_keywords(intent: str) -> str:
        """Extract error-related keywords from intent."""
        words = intent.lower().split()
        error_words = [
            w
            for w in words
            if len(w) > 3 and w not in ["error", "bug", "fix", "what", "does", "break"]
        ]
        return " ".join(error_words[:3]) if error_words else "error"

    @staticmethod
    def _check_warnings(buffer: dict, actions: List[Any]) -> List[Dict[str, str]]:
        """Check for warnings about buffer state."""
        warnings = []

        # Check if cache is stale
        if "last_edit_count" in buffer and buffer["last_edit_count"] > 5:
            warnings.append(
                {
                    "type": "cache_stale",
                    "file": "multiple files",
                    "message": f"Buffer has {buffer['last_edit_count']} edits; context may be outdated",
                }
            )

        # Check if buffer is dirty
        if buffer.get("state") == "DIRTY":
            warnings.append(
                {
                    "type": "buffer_dirty",
                    "file": "",
                    "message": "Buffer has uncommitted changes; consider committing before major refactoring",
                }
            )

        return warnings


class IntentRouter:
    """Main router orchestrating intent classification and action planning."""

    def __init__(self, buffer_manager, search_service, diff_engine):
        """Initialize router."""
        self.classifier = IntentClassifier()
        self.planner = ActionPlanner(buffer_manager, search_service, diff_engine)

    def plan_actions(
        self,
        buffer_id: str,
        intent: str,
        budget: int = 4000,
    ) -> ActionPlan:
        """
        Plan optimal action sequence for user intent.

        Args:
            buffer_id: Target buffer
            intent: User's intent description
            budget: Max tokens for planning phase

        Returns:
            ActionPlan with recommended actions and estimates
        """
        return self.planner.plan(buffer_id, intent, budget)
