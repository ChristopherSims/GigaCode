"""
Integration of Phases 4-10 into CodeEmbeddingTool

This module provides mixin classes and factory functions to integrate
all the new phases (intent router, solver, diff-aware search, etc.)
into the main CodeEmbeddingTool.

Usage:
    tool = CodeEmbeddingTool(...)
    tool.setup_phases_4_10()  # Initialize all phases

    # Phase 4: Intent-based action planning
    plan = tool.plan_actions(buffer_id, intent, budget=4000)

    # Phase 5: Automated solving
    result = tool.solve(buffer_id, task, max_iterations=5)

    # Phase 6: Diff-aware search
    results = tool.search_since_last_edit(buffer_id, query, scope="changes+deps")

    # Phase 7: Profile-based embedding
    tool.embed_codebase(path, agent_profile="reviewer")

    # Phase 8: Conflict prediction
    risks = tool.predict_conflicts(buffer_id)

    # Phase 9: Why annotations
    annotated = tool.annotate_search_results(buffer_id, results, query)

    # Phase 10: Undo/redo with branching
    tool.undo(buffer_id, steps=1)
    tool.branch(buffer_id, "experiment-oauth")
    tool.checkout(buffer_id, "experiment-oauth")
"""

from typing import Any, Dict, List, Optional

from gigacode.agent_profile import AdaptiveChunker, AgentProfileService, ProfileAdapter
from gigacode.conflict_predictor import ConflictPredictionService
from gigacode.diff_aware_search import DiffAwareSearch, DiffAwareSearchService

# Phase 4-10 module imports
from gigacode.intent_router import IntentRouter
from gigacode.solver import Solver
from gigacode.undo_redo import BranchedBufferManager, UndoRedoService
from gigacode.why_annotator import AnnotationService, RelevanceExplainer, WhyAnnotator


class Phase4Mixin:
    """Phase 4: Intent-Based Action Router"""

    def _setup_phase4(self):
        """Initialize Phase 4 components."""
        self._intent_router = IntentRouter(
            self._buffer_manager,
            self._search_service,
            getattr(self, "_diff_engine", None),
        )

    def plan_actions(
        self,
        buffer_id: str,
        intent: str,
        budget: int = 4000,
    ) -> Dict[str, Any]:
        """
        Plan optimal action sequence for intent.

        Args:
            buffer_id: Target buffer
            intent: User's intent description
            budget: Max tokens for planning

        Returns:
            Action plan as dict
        """
        plan = self._intent_router.plan_actions(buffer_id, intent, budget)
        return plan.to_dict()


class Phase5Mixin:
    """Phase 5: solve() Unified Loop"""

    def _setup_phase5(self):
        """Initialize Phase 5 components."""
        # Import here to avoid circular dependencies
        from gigacode.solver import SolveExecutor

        executor = SolveExecutor(
            self._buffer_manager,
            self._search_service,
            getattr(self, "_diff_engine", None),
            self._intent_router,
        )
        self._solver = Solver(executor)

    def solve(
        self,
        buffer_id: str,
        task: str,
        max_tokens_per_turn: int = 4000,
        max_iterations: int = 5,
        auto_commit: bool = False,
        test_before_commit: bool = True,
        search_depth: str = "medium",
    ) -> Dict[str, Any]:
        """
        Automatically solve a coding task.

        Args:
            buffer_id: Target buffer
            task: Task description
            max_tokens_per_turn: Max tokens per iteration
            max_iterations: Max loop iterations
            auto_commit: Auto-commit if tests pass?
            test_before_commit: Run tests before commit?
            search_depth: Search breadth (quick|medium|thorough)

        Returns:
            Solve result as dict
        """
        result = self._solver.solve(
            buffer_id,
            task,
            max_tokens_per_turn,
            max_iterations,
            auto_commit,
            test_before_commit,
            search_depth,
        )
        return result.to_dict()


class Phase6Mixin:
    """Phase 6: Diff-Aware Search"""

    def _setup_phase6(self):
        """Initialize Phase 6 components."""
        self._diff_aware_search = DiffAwareSearchService(
            DiffAwareSearch(
                self._search_service,
                getattr(self, "_dependency_graph", None),
                self._buffer_manager,
            )
        )

    def search_since_last_edit(
        self,
        buffer_id: str,
        query: str,
        scope: str = "changes+deps",
        include_deleted: bool = False,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Search only modified files and dependencies.

        Args:
            buffer_id: Target buffer
            query: Search query
            scope: Search scope (changes|changes+deps|all)
            include_deleted: Include deleted files?
            top_k: Max results

        Returns:
            Diff-aware search results
        """
        return self._diff_aware_search.search_since_last_edit(
            buffer_id,
            query,
            scope,
            include_deleted,
            top_k,
        )


class Phase7Mixin:
    """Phase 7: Auto-Chunking by Agent Profile"""

    def _setup_phase7(self):
        """Initialize Phase 7 components."""
        # Assuming base_chunker exists in the tool
        base_chunker = getattr(self, "_chunker", None)

        if base_chunker is None:
            from gigacode.chunker import chunk_file

            base_chunker = type("BaseChunker", (), {"chunk": chunk_file})()

        self._adaptive_chunker = AdaptiveChunker(base_chunker)
        self._profile_adapter = ProfileAdapter(self._adaptive_chunker)
        self._agent_profile_service = AgentProfileService(
            self._adaptive_chunker,
            self._profile_adapter,
        )

    def embed_codebase(
        self,
        codebase_path: str,
        agent_profile: str = "generic",
    ) -> Dict[str, Any]:
        """
        Embed codebase with profile-specific chunking.

        Args:
            codebase_path: Path to codebase
            agent_profile: Agent profile (reviewer|debugger|architect|documenter)

        Returns:
            Embedding metadata
        """
        return self._agent_profile_service.embed_codebase(codebase_path, agent_profile)

    def set_agent_profile(self, profile: str) -> None:
        """Set agent profile for session."""
        self._agent_profile_service.set_agent_profile(profile)


class Phase8Mixin:
    """Phase 8: Conflict Prediction"""

    def _setup_phase8(self):
        """Initialize Phase 8 components."""
        from gigacode import git_utils
        from gigacode.conflict_predictor import ConflictPredictor

        predictor = ConflictPredictor(
            self._buffer_manager,
            git_utils,
            getattr(self, "_dependency_graph", None),
        )
        self._conflict_prediction = ConflictPredictionService(predictor)

    def predict_conflicts(self, buffer_id: str) -> Dict[str, Any]:
        """
        Predict merge conflicts for a buffer.

        Args:
            buffer_id: Target buffer

        Returns:
            Conflict prediction analysis
        """
        return self._conflict_prediction.predict_conflicts(buffer_id)


class Phase9Mixin:
    """Phase 9: "Why This Matters" Annotations"""

    def _setup_phase9(self):
        """Initialize Phase 9 components."""
        explainer = RelevanceExplainer(
            getattr(self, "_dependency_graph", None),
            self._buffer_manager,
            getattr(self, "_symbol_index", None),
        )
        annotation_service = AnnotationService(explainer)
        self._why_annotator = WhyAnnotator(annotation_service)

    def annotate_search_results(
        self,
        buffer_id: str,
        results: List[Dict],
        query: str,
        edit_context: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Annotate search results with "why this matters" explanations.

        Args:
            buffer_id: Target buffer
            results: Search results
            query: Original search query
            edit_context: Files being edited

        Returns:
            Annotated results with explanations
        """
        return self._why_annotator.annotate_search_results(
            buffer_id,
            results,
            query,
            edit_context,
        )


class Phase10Mixin:
    """Phase 10: Undo Stack with Branching"""

    def _setup_phase10(self):
        """Initialize Phase 10 components."""
        branched_manager = BranchedBufferManager(self._buffer_manager)
        self._undo_redo = UndoRedoService(branched_manager)

    def undo(self, buffer_id: str, steps: int = 1) -> None:
        """
        Undo last N operations.

        Args:
            buffer_id: Target buffer
            steps: Number of steps to undo
        """
        self._undo_redo.undo(buffer_id, steps)

    def redo(self, buffer_id: str, steps: int = 1) -> None:
        """Redo last N undone operations."""
        self._undo_redo.redo(buffer_id, steps)

    def branch(
        self,
        buffer_id: str,
        name: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Create new branch.

        Args:
            buffer_id: Target buffer
            name: Branch name
            description: Optional description

        Returns:
            Branch info
        """
        return self._undo_redo.branch(buffer_id, name, description)

    def checkout(self, buffer_id: str, branch: str) -> None:
        """Switch to branch."""
        self._undo_redo.checkout(buffer_id, branch)

    def list_branches(self, buffer_id: str) -> List[Dict[str, Any]]:
        """List all branches for buffer."""
        return self._undo_redo.list_branches(buffer_id)

    def delete_branch(self, buffer_id: str, branch: str) -> None:
        """Delete a branch."""
        self._undo_redo.delete_branch(buffer_id, branch)

    def history(self, buffer_id: str) -> List[Dict[str, Any]]:
        """Get edit history for current branch."""
        return self._undo_redo.history(buffer_id)


class PhasesIntegrationMixin(
    Phase4Mixin,
    Phase5Mixin,
    Phase6Mixin,
    Phase7Mixin,
    Phase8Mixin,
    Phase9Mixin,
    Phase10Mixin,
):
    """
    Combined mixin for all phases 4-10.

    Integrate into CodeEmbeddingTool by adding this as a base class:

        class CodeEmbeddingTool(PhasesIntegrationMixin, ...):
            ...

    Then call setup_phases_4_10() in __init__ after all managers are initialized.
    """

    def setup_phases_4_10(self) -> None:
        """
        Initialize all phase 4-10 components.

        Must be called after:
        - self._buffer_manager
        - self._search_service
        - self._diff_engine (optional)
        - self._dependency_graph (optional)
        - self._symbol_index (optional)

        are all initialized.
        """
        try:
            # Initialize phases in dependency order
            self._setup_phase4()  # Intent router (depends on managers)
            self._setup_phase5()  # Solver (depends on phase 4)
            self._setup_phase6()  # Diff-aware search
            self._setup_phase7()  # Profile-based chunking
            self._setup_phase8()  # Conflict prediction
            self._setup_phase9()  # Why annotator
            self._setup_phase10()  # Undo/redo with branching

            print("✓ All phases 4-10 initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing phases 4-10: {e}")
            raise


def create_enhanced_tool_class():
    """
    Factory function to create CodeEmbeddingTool with all phases integrated.

    Usage:
        EnhancedTool = create_enhanced_tool_class()
        tool = EnhancedTool(work_dir="/path/to/buffer")
        tool.setup_phases_4_10()
    """
    # Import here to avoid circular dependencies
    from gigacode.gigacode_tool import CodeEmbeddingTool

    class EnhancedCodeEmbeddingTool(PhasesIntegrationMixin, CodeEmbeddingTool):
        """CodeEmbeddingTool with all phases 4-10 integrated."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Auto-initialize phases after parent init
            try:
                self.setup_phases_4_10()
            except Exception as e:
                # Log but don't fail; some optional components may not be available
                import logging

                logging.warning(f"Phases 4-10 setup incomplete: {e}")

    return EnhancedCodeEmbeddingTool
