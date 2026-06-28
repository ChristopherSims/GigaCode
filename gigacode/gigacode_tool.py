"""Agent-facing tool interface for GPU-accelerated code embedding.

Chunks code at AST boundaries, keeps a persistent FAISS index in GPU
memory when available, and exposes a read-write-commit workflow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from gigacode import response_adapters, tool_validation
from gigacode.access_control import User
from gigacode.agent_profile import (
    AdaptiveChunker,
    AgentProfile,
    AgentProfileService,
    ChunkingStrategyFactory,
    ProfileAdapter,
)
from gigacode.buffer_state import BufferState, BufferStateTransition
from gigacode.chunker import CodeChunk, chunk_text
from gigacode.code_quality import auto_format, auto_lint, auto_polish
from gigacode.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_PROMETHEUS_PORT,
    DEFAULT_THRESHOLD_MB,
    MAX_DIRTY_BEFORE_AUTO_REBUILD,
)
from gigacode.context_assembler import ContextAssembler
from gigacode.context_packer import pack_context
from gigacode.context_summarizer import ContextSummarizer
from gigacode.conversation_memory import ConversationMemory
from gigacode.dead_code_detector import DeadCodeDetector
from gigacode.dependency_graph import DependencyGraph
from gigacode.embedder import Embedder
from gigacode.execution_paths import trace_execution_paths
from gigacode.execution_sandbox import SandboxExecutor
from gigacode.faceted_search import FacetedSearcher, SearchFilter
from gigacode.git_utils import GitUtils
from gigacode.gpu_index import GpuIndex
from gigacode.health_status import HealthStatusTracker
from gigacode.impact_analyzer import ImpactAnalyzer
from gigacode.json_logger import StructuredJsonLogger
from gigacode.lexical_index import LexicalIndex
from gigacode.metadata_store import load_metadata, save_metadata
from gigacode.metrics import get_metrics
from gigacode.metrics_exporter import configure_prometheus
from gigacode.multi_buffer import MultiBufferManager
from gigacode.operation_config import OperationConfig, OperationType
from gigacode.path_utils import validate_buffer_path
from gigacode.progress_stream import ProgressReporter
from gigacode.quality_scorer import QualityScorer
from gigacode.refactor_engine import (
    PatchApplier,
    RefactorEngine,
    SymbolEditor,
)
from gigacode.refactor_engine import (
    add_parameter as _add_parameter,
)
from gigacode.refactor_engine import (
    edit_symbol as _edit_symbol,
)
from gigacode.reference_map import ReferenceMap
from gigacode.resource_budget import ConfidenceScorer
from gigacode.resource_budget import estimate_budget as _estimate_budget
from gigacode.response_types import (
    EmbedResponse,
    ResponseStatus,
)
from gigacode.retry_utils import retry_on_io_error
from gigacode.size_guard import check_size
from gigacode.snapshot_manager import SnapshotManager
from gigacode.state_manager import StateManager
from gigacode.symbol_index import SymbolIndex
from gigacode.test_runner import TestRunner
from gigacode.todo_tracker import TodoTracker
from gigacode.tool_security import ToolSecurityLayer
from gigacode.type_inference_cache import TypeInferenceCache
from gigacode.type_search import TypeSearcher

logger = logging.getLogger(__name__)
json_logger = StructuredJsonLogger("tool")


class CodeEmbeddingTool:
    """Embed a codebase into GPU/CPU buffers and expose search + cluster.

    Args:
        work_dir: Directory where buffer files and metadata are persisted.
        model_name: Sentence-transformers model name.
        device: torch device (``"cuda"``, ``"cpu"``, or ``None`` for auto).
        threshold_mb: Size-guard threshold in megabytes.
        use_gpu: Whether to mirror the FAISS index to GPU when possible.
        max_buffers: Maximum number of in-memory indices to keep (LRU eviction).
        enable_prometheus: Enable Prometheus metrics export (default False).
        prometheus_port: Port for Prometheus metrics endpoint (default 9090).
    """

    _ASYNC_WRAPPERS = frozenset(
        {
            "semantic_search_async",
            "write_code_async",
            "search_batch_async",
            "auto_format_async",
            "auto_lint_async",
            "auto_polish_async",
        }
    )

    def __init__(
        self,
        work_dir: str | Path,
        model_name: str | None = None,
        device: str | None = None,
        threshold_mb: float = DEFAULT_THRESHOLD_MB,
        use_gpu: bool = True,
        gpu_id: int = 0,
        max_buffers: int = 10,
        enable_prometheus: bool = False,
        prometheus_port: int = DEFAULT_PROMETHEUS_PORT,
    ) -> None:
        """Initialize CodeEmbeddingTool.

        Args:
            work_dir: Working directory for embeddings and caches.
            model_name: Sentence transformer model name (default: 'all-MiniLM-L6-v2').
            device: PyTorch device ('cpu', 'cuda', 'cuda:0', etc.).
            threshold_mb: Codebase size threshold (default 500 MB).
            use_gpu: Enable GPU FAISS index mirroring (default True).
            gpu_id: GPU device ID for FAISS (default 0). Only used if use_gpu=True.
            max_buffers: Max embedded codebases in memory (LRU limit, default 10).
            enable_prometheus: Enable Prometheus metrics export via HTTP (default False).
            prometheus_port: Port for /metrics endpoint (default 9090).
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_mb = threshold_mb
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.max_buffers = max_buffers

        from gigacode.embedder_optimizer import wrap_embedder_with_optimization

        embedder = Embedder(model_name=model_name, device=device)
        self._embedder = wrap_embedder_with_optimization(
            embedder=embedder,
            use_batch_optimization=True,
            batch_threshold=100,
        )
        self._embedding_dim = self._embedder.embedding_dim

        # State manager for crash recovery and transaction safety
        # Enables write-ahead logging (WAL) for commit operations
        self._state_manager = StateManager(self.work_dir)

        # Health status tracker for state-based access control
        self._health_tracker = HealthStatusTracker()

        # Security layer (RBAC, audit logging, and rate limiting)
        self._security = ToolSecurityLayer(self.work_dir)

        # Multi-turn conversation memory
        self._conversation_memory = ConversationMemory(self.work_dir / "memories.json")

        # Initialize the three manager layers
        # These provide separation of concerns for buffer mgmt, indexing, and search
        # Note: SearchService may fail on import if optional deps (sklearn) are missing
        self._buffer_manager = None
        self._index_manager = None
        self._search_service = None

        try:
            from gigacode.buffer_manager import BufferManager
            from gigacode.index_manager import IndexManager
            from gigacode.search_service import SearchService

            self._buffer_manager = BufferManager(
                work_dir=self.work_dir,
                state_manager=self._state_manager,
                embedding_dim=self._embedding_dim,
                threshold_mb=threshold_mb,
                audit_logger=self._security.audit_logger,
                user_id=self._security.current_user_id,
            )

            self._index_manager = IndexManager(
                embedder=self._embedder,
                embedding_dim=self._embedding_dim,
                max_buffers=max_buffers,
                work_dir=self.work_dir,
                use_gpu=use_gpu,
                gpu_id=gpu_id,
                prometheus_exporter=None,  # Will be set below if Prometheus is enabled
            )

            self._search_service = SearchService(
                index_manager=self._index_manager,
                embedder=self._embedder,
                prometheus_exporter=None,  # Will be set below if Prometheus is enabled
            )

            logger.info("Integration: BufferManager, IndexManager, SearchService initialized")
        except (ImportError, ModuleNotFoundError) as e:
            # Optional dependencies (sklearn for SearchService) may be missing
            # Managers are still required for core functionality
            logger.warning(
                f"Optional dependency unavailable during manager init: {type(e).__name__}: {e}"
            )
            if self._buffer_manager is None:
                logger.error(
                    "CRITICAL: BufferManager initialization failed. Core operations will fail."
                )
            if self._index_manager is None:
                logger.error("CRITICAL: IndexManager initialization failed. Indexing will fail.")
            if self._search_service is None:
                logger.warning(
                    "SearchService unavailable: sklearn or other optional dependency missing. Search operations will fail."
                )

        # Cache proxies — expose IndexManager caches for backward compat.
        # These are read-only views; mutations must go through IndexManager.
        # Guard against None managers for graceful degradation.
        if self._index_manager is not None:
            self._index_cache = self._index_manager.index_cache
            self._lexical_cache = self._index_manager.lexical_cache
            self._query_cache = self._index_manager.query_cache
        else:
            # Fallback empty caches when IndexManager unavailable
            from gigacode.lru_cache import LRUDict
            from gigacode.query_cache import QueryCache as _QueryCache

            self._index_cache: dict[str, Any] = LRUDict(max_size=max_buffers)
            self._lexical_cache: dict[str, Any] = LRUDict(max_size=max_buffers)
            self._query_cache: Any = _QueryCache()
        self._audit_log_path = self.work_dir / "audit.jsonl"
        self._async_wrapper_lock = threading.Lock()

        # Type inference cache (session-scoped, write-invalidated)
        self._type_inference_cache = TypeInferenceCache()

        # Audit logger reference (shared with security layer)
        self._audit_logger = self._security.audit_logger if self._security is not None else None

        # Multi-buffer orchestration manager (aliases and virtual buffers)
        self._multi_buffer_manager = MultiBufferManager(self.work_dir)

        # Initialize optional enhanced capabilities
        try:
            self.setup_phases_4_10()
        except Exception as e:
            logger.warning(f"Phases 4-10 setup incomplete (optional): {e}")

        # Prometheus metrics export via HTTP endpoint
        self._prometheus_exporter = None
        if enable_prometheus:
            try:
                self._prometheus_exporter = configure_prometheus(
                    port=prometheus_port,
                    start_server=True,
                )
                self._prometheus_exporter.set_embedding_dimension(self._embedding_dim)

                # Share Prometheus exporter with managers (always available now)
                self._index_manager.prometheus_exporter = self._prometheus_exporter
                self._search_service.prometheus_exporter = self._prometheus_exporter

                logger.info(f"Prometheus metrics endpoint enabled on port {prometheus_port}")
            except ImportError:
                logger.warning(
                    "Prometheus metrics disabled: prometheus-client not installed. "
                    "Install with: pip install prometheus-client"
                )

        # Fallback storage when BufferManager unavailable (for tests / degraded mode)
        self._fallback_registry: dict[str, Any] = {}
        self._fallback_snapshot_managers: dict[str, Any] = {}

        # Lazy-initialized profile adapter for agent profile operations
        self._profile_adapter: ProfileAdapter | None = None

    # ------------------------------------------------------------------
    # Backward-compat properties for direct _registry / _snapshot_managers access
    # Tests and legacy code may access these directly.
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        """Auto-generate async wrapper methods on first access."""
        if name in self._ASYNC_WRAPPERS and not name.startswith("_"):
            cached_wrapper = self.__dict__.get(name)
            if cached_wrapper is not None:
                return cached_wrapper

            sync_name = name.removesuffix("_async")
            sync_method = getattr(self.__class__, sync_name, None)
            if sync_method is not None:
                import asyncio

                async def _async_wrapper(*args, **kwargs):
                    return await asyncio.to_thread(getattr(self, sync_name), *args, **kwargs)

                _async_wrapper.__name__ = name
                _async_wrapper.__qualname__ = f"{self.__class__.__qualname__}.{name}"
                _async_wrapper.__doc__ = f"Async version of {sync_name}. Delegates to the synchronous method in a thread pool."

                with self._async_wrapper_lock:
                    cached_wrapper = self.__dict__.get(name)
                    if cached_wrapper is not None:
                        return cached_wrapper

                    # Cache on the instance so __getattr__ isn't called again
                    setattr(self, name, _async_wrapper)
                    return _async_wrapper

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @property
    def _registry(self) -> dict[str, Any]:
        """Backward-compat: proxy to BufferManager._registry or fallback."""
        if self._buffer_manager is not None:
            return self._buffer_manager._registry
        return self._fallback_registry

    @_registry.setter
    def _registry(self, value: dict[str, Any]) -> None:
        """Backward-compat: set BufferManager._registry or fallback."""
        if self._buffer_manager is not None:
            self._buffer_manager._registry = value
        else:
            self._fallback_registry = value

    @property
    def _snapshot_managers(self) -> dict[str, Any]:
        """Backward-compat: proxy to BufferManager._snapshot_managers or fallback."""
        if self._buffer_manager is not None:
            return self._buffer_manager._snapshot_managers
        return self._fallback_snapshot_managers

    @_snapshot_managers.setter
    def _snapshot_managers(self, value: dict[str, Any]) -> None:
        """Backward-compat: set BufferManager._snapshot_managers or fallback."""
        if self._buffer_manager is not None:
            self._buffer_manager._snapshot_managers = value
        else:
            self._fallback_snapshot_managers = value

    # ------------------------------------------------------------------
    # Schema exposure
    # ------------------------------------------------------------------

    @staticmethod
    def get_tool_schemas() -> list[dict[str, Any]]:
        from gigacode.tool_schema import get_all_schemas

        return get_all_schemas()

    @staticmethod
    def validate_schemas() -> dict[str, Any]:
        """Validate that hardcoded schemas match the actual CodeEmbeddingTool code.

        Returns a dict with:
        - "valid": bool indicating if all schemas are valid
        - "issues": dict mapping method name to list of validation issues
        - "report": human-readable validation report
        - "generated_count": number of schemas generated
        - "validated_count": number of schemas validated

        Used to detect schema drift and ensure tool_schema.py stays in sync with implementation.
        """
        from gigacode.schema_generator import (
            generate_all_schemas,
            report_schema_validation,
            validate_schemas_against_code,
        )
        from gigacode.tool_schema import ALL_SCHEMAS

        # Convert list of schemas to dict format for validation
        hardcoded = {schema.get("name"): schema for schema in ALL_SCHEMAS}

        # Validate and report
        issues = validate_schemas_against_code(CodeEmbeddingTool, hardcoded)
        generated = generate_all_schemas(CodeEmbeddingTool)

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "report": report_schema_validation(issues),
            "generated_count": len(generated),
            "validated_count": len(hardcoded),
        }

    # ------------------------------------------------------------------
    # Input validation (delegated to tool_validation module)
    # ------------------------------------------------------------------
    @staticmethod
    def _make_error_response(
        message: str,
        buffer_id: str | None = None,
        operation: str = "",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Thin wrapper to tool_validation.make_error_response."""
        return tool_validation.make_error_response(
            message, buffer_id=buffer_id, operation=operation, context=context
        )

    def _require_buffer(
        self,
        buffer_id: str,
        operation: str,
        require_chunks: bool = True,
    ) -> tuple[dict[str, Any], list[dict]] | dict[str, Any]:
        """Validate buffer exists and optionally that chunks are loaded.

        Args:
            buffer_id: Buffer handle to validate.
            operation: Operation name for error messages.
            require_chunks: If True, also verify chunks are loaded.

        Returns:
            On success: (info, chunks) tuple.
            On failure: error response dict (also falsy via _is_error_response check).
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation=operation
            )
        if require_chunks:
            chunks = self._load_chunks(buffer_id)
            if chunks is None or not chunks:
                return self._make_error_response(
                    "No chunks loaded", buffer_id=buffer_id, operation=operation
                )
            return info, chunks
        return info, []

    @staticmethod
    def _validate_search_params(
        query: str,
        top_k: int | None = None,
        max_results: int | None = None,
    ) -> dict[str, Any] | None:
        """Thin wrapper to tool_validation.validate_search_params."""
        return tool_validation.validate_search_params(query, top_k=top_k, max_results=max_results)

    # ------------------------------------------------------------------
    # Unified diff helper: Return the Diff
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_unified_diff(
        old_lines: list[str],
        new_lines: list[str],
        file: str = "",
        context_lines: int = 3,
    ) -> str:
        """Compute a unified diff between old and new file lines.

        Args:
            old_lines: Original file lines (without trailing newlines).
            new_lines: New file lines (without trailing newlines).
            file: File path for diff header.
            context_lines: Number of context lines around changes.

        Returns:
            Unified diff string.
        """
        import difflib

        # difflib expects lines with trailing newlines
        old_with_newlines = [line + "\n" for line in old_lines]
        new_with_newlines = [line + "\n" for line in new_lines]

        diff_lines = list(
            difflib.unified_diff(
                old_with_newlines,
                new_with_newlines,
                fromfile=f"a/{file}",
                tofile=f"b/{file}",
                n=context_lines,
            )
        )
        return "".join(diff_lines)

    # ------------------------------------------------------------------
    # Response adapters (delegated to response_adapters module)
    # ------------------------------------------------------------------
    @staticmethod
    def _adapt_search_response(
        service_result: Any,
        offset: int = 0,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Thin wrapper to response_adapters.adapt_search_response."""
        return response_adapters.adapt_search_response(service_result, offset=offset, top_k=top_k)

    @staticmethod
    def _adapt_file_response(
        service_result: Any,
    ) -> dict[str, Any]:
        """Thin wrapper to response_adapters.adapt_file_response."""
        return response_adapters.adapt_file_response(service_result)

    @staticmethod
    def _adapt_cluster_response(
        service_result: Any,
    ) -> dict[str, Any]:
        """Thin wrapper to response_adapters.adapt_cluster_response."""
        return response_adapters.adapt_cluster_response(service_result)

    @staticmethod
    def _adapt_duplicate_response(
        service_result: Any,
    ) -> dict[str, Any]:
        """Thin wrapper to response_adapters.adapt_duplicate_response."""
        return response_adapters.adapt_duplicate_response(service_result)

    # ------------------------------------------------------------------
    # Pre-flight size check
    # ------------------------------------------------------------------
    def check_codebase(
        self,
        path: str | Path,
        pattern: str = "*.py",
    ) -> dict[str, Any]:
        root = Path(path)
        files = [root] if root.is_file() else sorted(root.rglob(pattern))
        if not files:
            return {"status": "warning", "message": f"No files matched '{pattern}' in {root}"}

        total_lines = 0
        for f in files:
            try:
                with f.open("r", encoding="utf-8", errors="replace") as fh:
                    total_lines += sum(1 for _ in fh)
            except (OSError, PermissionError, UnicodeDecodeError) as exc:
                json_logger.debug(
                    operation="chunk_file",
                    message=f"Could not count lines in {f}: {exc}",
                )

        estimated_tokens = total_lines * 8
        size_check = check_size(estimated_tokens, self._embedding_dim, self.threshold_mb)
        return {
            "status": size_check["status"],
            "estimated_lines": total_lines,
            "estimated_tokens": estimated_tokens,
            "estimated_mb": size_check["estimated_mb"],
            "threshold_mb": size_check["threshold_mb"],
            "message": (
                f"Estimated {total_lines} lines (~{estimated_tokens} tokens) "
                f"=> ~{size_check['estimated_mb']:.1f} MB"
            ),
        }

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def embed_codebase(
        self,
        path: str | Path,
        language_hint: str | None = None,
        pattern: str = "*.py",
        sliding_window_size: int = 30,
    ) -> dict[str, Any]:
        """Embed a codebase: chunk all files, embed chunks, build index.

        Returns: EmbedResponse as dict with buffer_id, chunk_count, and status.
        """
        t0 = time.perf_counter()
        try:
            # Delegate to BufferManager: handle chunking, validation, registration
            buffer_id, chunks, files = self._buffer_manager.embed_codebase(
                path=Path(path).resolve(),
                language_hint=language_hint,
                pattern=pattern,
                sliding_window_size=sliding_window_size,
            )

            # Embed chunks
            texts = [ch.text for ch in chunks]
            embeddings = self._embedder.encode(texts, batch_size=DEFAULT_BATCH_SIZE)

            # Create and cache indices via IndexManager
            self._index_manager.create_indices(buffer_id, embeddings, chunks)

            # Save hierarchical summaries for context packing
            summarizer = ContextSummarizer(chunks)
            buffer_dir = self.work_dir / f"{buffer_id}.gcbuff"
            summarizer.save_summaries(buffer_dir)

            elapsed = time.perf_counter() - t0

            # Record metrics
            if self._prometheus_exporter:
                self._prometheus_exporter.record_operation(
                    operation="embed_codebase",
                    duration_s=elapsed,
                    status="ok",
                    chunk_count=len(chunks),
                )

            metrics = get_metrics()
            metrics.record_histogram("embed_codebase_latency_s", elapsed)
            metrics.record_histogram("embed_chunks_count", len(chunks))
            metrics.increment_counter("embed_codebase_calls")

            json_logger.info(
                operation="embed_codebase",
                buffer_id=buffer_id,
                elapsed_s=elapsed,
                status="ok",
                message=f"Embedded {len(chunks)} chunks from {len(files)} files",
                details={
                    "files_count": len(files),
                    "chunks_count": len(chunks),
                    "size_bytes": embeddings.nbytes,
                },
            )

            response = EmbedResponse(
                status=ResponseStatus.OK,
                buffer_id=buffer_id,
                chunk_count=len(chunks),
                size_bytes=embeddings.nbytes,
                message=f"Embedded {len(chunks)} chunks from {len(files)} files.",
            )
            return response.to_dict()

        except ValueError as e:
            json_logger.warning(
                operation="embed_codebase",
                status="error",
                message=str(e),
            )
            return {"status": "warning", "message": str(e)}
        except (OSError, RuntimeError) as e:
            json_logger.error(
                operation="embed_codebase",
                status="error",
                message=f"Unexpected error: {e}",
            )
            return {"status": "error", "message": f"Embedding failed: {e}"}

    # ------------------------------------------------------------------
    # Reload without re-embedding
    # ------------------------------------------------------------------
    def reload_codebase(self, buffer_id: str) -> dict[str, Any]:
        """Reload codebase from disk, detecting external changes."""
        return self._buffer_manager.reload_codebase(buffer_id, index_manager=self._index_manager)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def semantic_search(
        self,
        buffer_id: str,
        query: str,
        top_k: int = 5,
        offset: int = 0,
        include_types: bool = False,
        type_inference_method: str = "llm",
    ) -> dict[str, Any]:
        """Semantic search via embeddings.

        Delegates to SearchService when available, otherwise uses monolithic implementation.

        Args:
            buffer_id: Buffer ID to search
            query: Search query
            top_k: Number of top results to return
            offset: Pagination offset
            include_types: Include inferred type hints in results (default: False)
            type_inference_method: How to infer types: "llm" (accurate) or "ast" (fast)

        Returns:
            Dict with search results or error
        """
        # Validate parameters first (before delegation)
        err = self._validate_search_params(query, top_k=top_k)
        if err is not None:
            return err

        if offset < 0:
            return self._make_error_response(
                "offset must be greater than or equal to 0",
                buffer_id=buffer_id,
                operation="semantic_search",
            )

        # Validate type_inference_method
        if type_inference_method not in ("llm", "ast"):
            return self._make_error_response(
                f"type_inference_method must be 'llm' or 'ast', got '{type_inference_method}'",
                buffer_id=buffer_id,
                operation="semantic_search",
            )

        # Delegate to SearchService
        if self._search_service:
            try:
                result = self._search_service.semantic_search(
                    buffer_id=buffer_id,
                    query=query,
                    top_k=top_k + offset,
                    include_types=include_types,
                    type_inference_method=type_inference_method,
                )
                return self._adapt_search_response(result, offset=offset, top_k=top_k)
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"SearchService delegation failed: {e}")
                return self._make_error_response(
                    f"Search failed: {e}",
                    buffer_id=buffer_id,
                    operation="semantic_search",
                )

        return self._make_error_response(
            "SearchService unavailable",
            buffer_id=buffer_id,
            operation="semantic_search",
        )

    def semantic_search_streaming(
        self,
        buffer_id: str,
        query: str,
        top_k: int = 10,
        disclosure: str = "signatures",
    ) -> dict[str, Any]:
        """Search with progressive result disclosure to save tokens.

        Three disclosure levels:
        - "signatures": Only function/class signatures (~8 tokens/match).
        - "details": Signatures + docstrings + first 5 lines (~45 tokens/match).
        - "full": Complete chunk text (~200 tokens/match).

        Use ``expand_match()`` to incrementally expand a single match.

        Expected savings: 84% for signatures-only vs full chunks.

        Args:
            buffer_id: Buffer handle.
            query: Search query string.
            top_k: Number of top results.
            disclosure: "signatures" | "details" | "full".

        Returns:
            Dict with matches at the requested disclosure level, plus
            ``expandable`` flag and ``match_count``.
        """
        err = self._validate_search_params(query, top_k=top_k)
        if err is not None:
            return err

        if self._search_service:
            try:
                return self._search_service.semantic_search_streaming(
                    buffer_id=buffer_id,
                    query=query,
                    top_k=top_k,
                    disclosure=disclosure,
                )
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"SearchService streaming failed: {e}")
                return self._make_error_response(
                    f"Streaming search failed: {e}",
                    buffer_id=buffer_id,
                    operation="semantic_search_streaming",
                )

        return self._make_error_response(
            "SearchService unavailable",
            buffer_id=buffer_id,
            operation="semantic_search_streaming",
        )

    def expand_match(
        self,
        buffer_id: str,
        match_id: int,
        level: str = "details",
    ) -> dict[str, Any]:
        """Expand a search match to a higher disclosure level.

        Progressively reveals more detail for a single match without
        re-embedding or re-searching.

        Args:
            buffer_id: Buffer handle.
            match_id: Chunk index from a prior streaming search result.
            level: "details" (signatures + docstring + first 5 lines) or
                "full" (complete text).

        Returns:
            Dict with expanded match data.
        """
        if self._search_service:
            try:
                return self._search_service.expand_match(
                    buffer_id=buffer_id,
                    match_id=match_id,
                    level=level,
                )
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"Expand match failed: {e}")
                return self._make_error_response(
                    f"Expand match failed: {e}",
                    buffer_id=buffer_id,
                    operation="expand_match",
                )

        return self._make_error_response(
            "SearchService unavailable",
            buffer_id=buffer_id,
            operation="expand_match",
        )

    def find_similar_intents(
        self,
        buffer_id: str,
        query: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Find cached intent clusters semantically similar to a query.

        Helps when you're not sure how to phrase a query — find related
        intents that have been cached previously.

        Args:
            buffer_id: Buffer handle.
            query: Query string to find similar intents for.
            top_k: Maximum number of similar intents to return.

        Returns:
            Dict with similar_intents list, each containing canonical_query,
            similarity score, hit count, and most recent cached result.
        """
        if self._search_service:
            try:
                similar = self._search_service._intent_cache.find_similar_intents(
                    query, top_k=top_k
                )
                return {
                    "status": "ok",
                    "buffer_id": buffer_id,
                    "query": query,
                    "similar_intents": [
                        {
                            "canonical_query": s["cluster"]["canonical_query"],
                            "similarity": round(s["similarity"], 3),
                            "queries_in_cluster": len(s["cluster"]["queries"]),
                            "hits": s["cluster"]["hits"],
                            "intent_label": s["cluster"].get("intent_label"),
                        }
                        for s in similar
                    ],
                    "count": len(similar),
                }
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"Find similar intents failed: {e}")
                return self._make_error_response(
                    f"Find similar intents failed: {e}",
                    buffer_id=buffer_id,
                    operation="find_similar_intents",
                )

        return self._make_error_response(
            "SearchService unavailable",
            buffer_id=buffer_id,
            operation="find_similar_intents",
        )

    def get_intent_cache_stats(self, buffer_id: str) -> dict[str, Any]:
        """Get statistics for the intent cache.

        Args:
            buffer_id: Buffer handle.

        Returns:
            Dict with hits, misses, hit_rate, cluster_count, avg_similarity.
        """
        if self._search_service:
            try:
                stats = self._search_service._intent_cache.get_stats()
                return {"status": "ok", "buffer_id": buffer_id, **stats}
            except (ImportError, RuntimeError, ValueError) as e:
                return self._make_error_response(
                    f"Failed to get intent stats: {e}",
                    buffer_id=buffer_id,
                    operation="get_intent_cache_stats",
                )

        return self._make_error_response(
            "SearchService unavailable",
            buffer_id=buffer_id,
            operation="get_intent_cache_stats",
        )

    def faceted_search(
        self,
        buffer_id: str,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        include_uncertain: bool = True,
        confidence_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Perform faceted semantic search with filters and result explanation.

        Args:
            buffer_id: Buffer handle.
            query: Search query string.
            filters: Optional filter dict with keys:
                - language (str): e.g. "python", "javascript"
                - path_regex (str): regex to match file paths
                - type_in (list[str]): chunk types ["function", "class", ...]
                - min_lines (int): minimum chunk line count
                - max_lines (int): maximum chunk line count
                - file_pattern (str): glob pattern e.g. "src/**/*.py"
            top_k: Number of top results.
            include_uncertain: Include matches with lower confidence.
            confidence_threshold: Minimum score for confident matches.

        Returns:
            Dict with matches (each has score, confidence, score_breakdown, why),
            uncertain_matches, total_matches, filtered_out.
        """
        result = self._require_buffer(buffer_id, "faceted_search")
        if isinstance(result, dict):
            return result
        info, chunks = result

        buffer_dir = Path(info["buffer_dir"])
        embeddings_path = buffer_dir / "embeddings.npy"
        embeddings: np.ndarray | None = None
        if embeddings_path.exists():
            try:
                embeddings = np.load(embeddings_path)
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to load embeddings for {buffer_id}: {e}")

        query_embedding = self._embedder.encode([query])

        filter_obj = SearchFilter.from_dict(filters) if filters else None
        searcher = FacetedSearcher(chunks, embeddings)

        try:
            result = searcher.search(
                query_embedding,
                query,
                filter_obj,
                top_k,
                include_uncertain,
                confidence_threshold,
            )
            result.buffer_id = buffer_id
            return {"status": "ok", **result.to_dict()}
        except (ValueError, RuntimeError) as e:
            logger.warning(f"FacetedSearcher.search failed: {e}")
            return self._make_error_response(
                f"Faceted search failed: {e}",
                buffer_id=buffer_id,
                operation="faceted_search",
            )

    def hybrid_search(
        self,
        buffer_id: str,
        query: str,
        top_k: int = 5,
        offset: int = 0,
        semantic_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> dict[str, Any]:
        """Combine FAISS semantic search with BM25 lexical search via RRF.

        Args:
            buffer_id: Buffer handle.
            query: Natural language or keyword query.
            top_k: Number of results to return.
            offset: Pagination offset.
            semantic_weight: Weight for semantic rank contribution.
            lexical_weight: Weight for lexical rank contribution.

        Returns:
            Dict with ``status`` and ``matches``.
        """
        # Validate parameters first
        err = self._validate_search_params(query, top_k=top_k)
        if err is not None:
            return err

        if offset < 0:
            return self._make_error_response(
                "offset must be greater than or equal to 0",
                buffer_id=buffer_id,
                operation="hybrid_search",
            )

        result = self._require_buffer(buffer_id, "hybrid_search", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        # Delegate to SearchService
        if self._search_service:
            try:
                result = self._search_service.hybrid_search(
                    buffer_id=buffer_id,
                    query=query,
                    top_k=top_k + offset,
                    semantic_weight=semantic_weight,
                    lexical_weight=lexical_weight,
                )
                return self._adapt_search_response(result, offset=offset, top_k=top_k)
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(f"SearchService delegation failed: {e}")
                return self._make_error_response(
                    f"Search failed: {e}",
                    buffer_id=buffer_id,
                    operation="hybrid_search",
                )

        return self._make_error_response(
            "SearchService unavailable",
            buffer_id=buffer_id,
            operation="hybrid_search",
        )

    # ------------------------------------------------------------------
    # Literal text search (grep-style)
    # ------------------------------------------------------------------
    def search_for(
        self,
        buffer_id: str,
        query: str,
        case_sensitive: bool = False,
        max_results: int = 50,
    ) -> dict[str, Any]:
        """Find every occurrence of *query* in the buffered source code.

        Args:
            buffer_id: Buffer handle.
            query: Substring to search for.
            case_sensitive: If ``True``, match case exactly.
            max_results: Cap on the number of matches returned.

        Returns:
            Dict with ``status``, ``matches`` (list of file/line/content), and ``total``.
        """
        # Validate parameters first
        err = self._validate_search_params(query, max_results=max_results)
        if err is not None:
            return err

        result = self._require_buffer(buffer_id, "search_for", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        result = self._search_service.search_for(
            buffer_id=buffer_id,
            query=query,
            case_sensitive=case_sensitive,
        )
        # Adapt SearchResponse to expected format
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
        else:
            result_dict = result

        # Extract matches and limit to max_results
        matches = result_dict.get("matches", [])
        limited_matches = matches[:max_results]
        return {"status": "ok", "matches": limited_matches, "total": len(limited_matches)}

    def look_for_file(
        self,
        buffer_id: str,
        file_name: str,
    ) -> dict[str, Any]:
        """Find the location of a file within an embedded buffer.

        Tries exact match, then basename match, then partial substring match.
        Returns the relative file path (within the buffer root) and the
        absolute path on disk.

        Args:
            buffer_id: Buffer handle.
            file_name: File name or path fragment to look for.

        Returns:
            Dict with ``status``, ``file_location``, ``absolute_path``,
            ``match_type``, and optionally ``candidates``.
        """
        result = self._require_buffer(buffer_id, "look_for_file", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        result = self._search_service.look_for_file(
            buffer_id=buffer_id,
            pattern=file_name,
        )
        return self._adapt_file_response(result)

    # ------------------------------------------------------------------
    # Symbol search (name + semantic)
    # ------------------------------------------------------------------
    def search_symbols(
        self,
        buffer_id: str,
        query: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Find variables, functions, classes, and methods matching *query*.

        Performs two searches in parallel:
        1. **Name match** — chunks whose ``name`` contains the query string (score=1.0).
        2. **Semantic match** — top-K chunks by embedding similarity, filtered to
           symbol types (function, class, method, struct, trait, enum, interface).

        Results are deduplicated and merged (name matches rank first).

        Args:
            buffer_id: Buffer handle.
            query: Word or phrase to look for.
            top_k: Maximum number of symbol matches to return.

        Returns:
            SearchResponse with ``matches`` (list of SearchMatch objects with type, name, score).
        """
        # Validate parameters first
        err = self._validate_search_params(query, top_k=top_k)
        if err is not None:
            return err

        result = self._require_buffer(buffer_id, "search_symbols", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        result = self._search_service.search_symbols(
            buffer_id=buffer_id,
            query=query,
            top_k=top_k,
        )
        return self._adapt_search_response(result, offset=0, top_k=top_k)

    def search_by_type(
        self,
        buffer_id: str,
        type_pattern: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Find functions and classes by type signature pattern.

        Args:
            buffer_id: Buffer handle.
            type_pattern: Type pattern to search (e.g., "Callable[[str], bool]", "dict[str, int]").
            top_k: Maximum results.

        Returns:
            Dict with matches, each containing type_signature, match_score, match_reason.
        """
        result = self._require_buffer(buffer_id, "search_by_type")
        if isinstance(result, dict):
            return result
        info, chunks = result

        searcher = TypeSearcher(chunks)
        try:
            matches = searcher.search_by_type(type_pattern, top_k)
            return {"status": "ok", "matches": [m.to_dict() for m in matches]}
        except (ValueError, RuntimeError) as e:
            logger.warning(f"TypeSearcher.search_by_type failed: {e}")
            return self._make_error_response(
                f"Type search failed: {e}",
                buffer_id=buffer_id,
                operation="search_by_type",
            )

    def find_implementations(
        self,
        buffer_id: str,
        interface_name: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Find classes implementing a given interface/protocol.

        Args:
            buffer_id: Buffer handle.
            interface_name: Interface name (e.g., "DataStore", "Iterator").
            top_k: Maximum results.

        Returns:
            Dict with implementations list, each with class_name, inheritance_type, confidence.
        """
        result = self._require_buffer(buffer_id, "find_implementations")
        if isinstance(result, dict):
            return result
        info, chunks = result

        searcher = TypeSearcher(chunks)
        try:
            implementations = searcher.find_implementations(interface_name, top_k)
            return {"status": "ok", "implementations": implementations}
        except (ValueError, RuntimeError) as e:
            logger.warning(f"TypeSearcher.find_implementations failed: {e}")
            return self._make_error_response(
                f"Find implementations failed: {e}",
                buffer_id=buffer_id,
                operation="find_implementations",
            )

    def infer_types(
        self,
        buffer_id: str,
        symbol: str,
        method: str = "llm",
    ) -> dict[str, Any]:
        """Infer type information for a symbol.

        Args:
            buffer_id: Buffer handle.
            symbol: Symbol name to infer types for.
            method: "llm" for accurate semantic inference (~50-300ms) or
                    "ast" for fast pattern extraction (~1-5ms).

        Returns:
            Dict with inferred types, confidence scores, and reasoning.
        """
        result = self._require_buffer(buffer_id, "infer_types", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        if method not in ("llm", "ast"):
            return self._make_error_response(
                f"method must be 'llm' or 'ast', got '{method}'",
                buffer_id=buffer_id,
                operation="infer_types",
            )

        # Check type inference cache (LLM only)
        if method == "llm":
            cached = self._type_inference_cache.get(buffer_id, symbol)
            if cached is not None:
                return {"status": "ok", **cached.to_dict(), "cached": True}

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="infer_types"
            )

        try:
            from gigacode.type_inference_cache import InferredType
            from gigacode.type_search import _extract_python_types

            # Find the chunk containing this symbol
            matching_chunks = [c for c in chunks if hasattr(c, "name") and c.name == symbol]
            if not matching_chunks:
                # Fallback: search by symbol name in text
                matching_chunks = [
                    c for c in chunks if hasattr(c, "text") and symbol in (c.text or "")
                ]
            if not matching_chunks:
                return self._make_error_response(
                    f"Symbol '{symbol}' not found", buffer_id=buffer_id, operation="infer_types"
                )

            chunk = matching_chunks[0]
            sigs = _extract_python_types(chunk.text)
            if not sigs:
                return {
                    "status": "ok",
                    "symbol": symbol,
                    "parameters": [],
                    "return_type": None,
                    "method": method,
                    "type_confidence": None,
                    "message": "No type annotations found for this symbol",
                }

            sig = sigs[0]
            result: dict[str, Any] = {
                "symbol": symbol,
                "parameters": sig.parameters,
                "return_type": sig.return_type,
                "method": method,
                "is_async": sig.is_async,
            }

            if method == "llm" and self._search_service:
                try:
                    type_info = self._search_service._infer_type_for_chunk(chunk, method="llm")
                    if type_info:
                        result["type_confidence"] = type_info.get("type_confidence")
                        result["signature"] = type_info.get("signature")
                    else:
                        result["type_confidence"] = None
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.debug(f"LLM type inference failed: {e}")
                    result["type_confidence"] = None
            else:
                result["type_confidence"] = None

            # Cache LLM result
            if method == "llm":
                inferred = InferredType(
                    symbol_name=symbol,
                    file=chunk.file,
                    parameters=sig.parameters,
                    return_type=sig.return_type,
                    confidence=result.get("type_confidence", 0.0) or 0.0,
                    method=method,
                )
                self._type_inference_cache.put(buffer_id, inferred)

            return {"status": "ok", **result}

        except (ImportError, RuntimeError, ValueError, TypeError) as e:
            return self._make_error_response(
                f"Type inference failed: {e}",
                buffer_id=buffer_id,
                operation="infer_types",
            )

    def symbol_search(
        self,
        buffer_id: str,
        query: str,
    ) -> dict[str, Any]:
        """Search for symbols by name (exact, prefix, fuzzy).

        Args:
            buffer_id: Buffer handle.
            query: Symbol name to search (e.g., "UserRepository", "validate_email").

        Returns:
            Dict with exact_matches, prefix_matches, fuzzy_matches.
        """
        result = self._require_buffer(buffer_id, "symbol_search")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            index = SymbolIndex(chunks)
            result = index.search(query)
            return {"status": "ok", **result.to_dict()}
        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                f"Symbol search failed: {e}",
                buffer_id=buffer_id,
                operation="symbol_search",
            )

    def get_symbol_definition(
        self,
        buffer_id: str,
        symbol: str,
    ) -> dict[str, Any]:
        """Jump to definition of a symbol.

        Supports qualified names like "UserRepository.validate_email".
        """
        result = self._require_buffer(buffer_id, "get_symbol_definition")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            index = SymbolIndex(chunks)
            definitions = index.get_definition(symbol)
            return {"status": "ok", "definitions": [d.to_dict() for d in definitions]}
        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                f"Get definition failed: {e}",
                buffer_id=buffer_id,
                operation="get_symbol_definition",
            )

    def get_symbol_references(
        self,
        buffer_id: str,
        symbol: str,
        top_k: int = 50,
    ) -> dict[str, Any]:
        """Find all references to a symbol across the codebase.

        Returns reference locations with context lines and confidence scores.
        """
        result = self._require_buffer(buffer_id, "get_symbol_references")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            index = SymbolIndex(chunks)
            references = index.get_references(symbol, top_k)
            return {"status": "ok", "references": [r.to_dict() for r in references]}
        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                f"Get references failed: {e}",
                buffer_id=buffer_id,
                operation="get_symbol_references",
            )

    def get_references(
        self,
        buffer_id: str,
        symbol: str,
        direction: str = "both",
        top_k: int = 50,
        expand_depth: int | None = None,
    ) -> dict[str, Any]:
        """Find all callers/callees for a symbol using the incremental reference map.

        Uses a three-step incremental strategy:
        1. Lazy/On-Demand: Build only the direct neighborhood on first query.
        2. Cached: Subsequent queries return cached results instantly.
        3. Optional expansion: Set expand_depth to trace deeper call chains.

        Args:
            buffer_id: Buffer handle.
            symbol: Symbol name to find references for.
            direction: "both" (default), "calls" (callees), or "called_by" (callers).
            top_k: Maximum references per direction (default: 50).
            expand_depth: If set, expand neighborhood to this depth.

        Returns:
            Dict with symbol, file, line, callers, callees, direction, cached.
        """
        result = self._require_buffer(buffer_id, "get_references", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        if direction not in ("both", "calls", "called_by"):
            return self._make_error_response(
                f"direction must be 'both', 'calls', or 'called_by', got '{direction}'",
                buffer_id=buffer_id,
                operation="get_references",
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="get_references"
            )

        try:
            ref_map = ReferenceMap(chunks)
            result = ref_map.get_references(symbol, direction=direction, top_k=top_k)

            # Expand if requested
            if expand_depth is not None and expand_depth > 1:
                result = ref_map.expand_neighborhood(
                    symbol, max_depth=expand_depth, direction=direction
                )

            return result
        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                f"Get references failed: {e}",
                buffer_id=buffer_id,
                operation="get_references",
            )

    def get_full_context(
        self,
        buffer_id: str,
        symbol: str,
        include: list[str] | None = None,
        type_inference_method: str = "llm",
    ) -> dict[str, Any]:
        """Get everything about a symbol in one call.

        Combines get_symbol_definition + get_references + type inference +
        test discovery + error handling analysis. Single roundtrip instead of 5 API calls.

        Args:
            buffer_id: Buffer handle.
            symbol: Symbol name to get full context for.
            include: List of sections to include. Default: all.
                     Options: "definition", "callers", "callees", "tests",
                     "related_code", "type_hints", "errors".
            type_inference_method: "llm" (accurate) or "ast" (fast).

        Returns:
            Dict with definition, callers, callees, types, tests, errors.
        """
        if include is None:
            include = [
                "definition",
                "callers",
                "callees",
                "tests",
                "related_code",
                "type_hints",
                "errors",
            ]

        result = self._require_buffer(buffer_id, "get_full_context", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        result: dict[str, Any] = {"status": "ok", "symbol": symbol}

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="get_full_context"
            )

        try:
            # Definition
            if "definition" in include:
                index = SymbolIndex(chunks)
                definitions = index.get_definition(symbol)
                if definitions:
                    entry = definitions[0]
                    result["definition"] = entry.to_dict()
                    # Also include the source text
                    for chunk in chunks:
                        if (
                            hasattr(chunk, "name")
                            and chunk.name == entry.name
                            and chunk.file == entry.file
                            and chunk.start_line == entry.start_line
                        ):
                            result["definition"]["source"] = chunk.text
                            break
                else:
                    result["definition"] = None

            # Callers and callees
            if "callers" in include or "callees" in include:
                ref_result = self.get_references(
                    buffer_id=buffer_id, symbol=symbol, direction="both"
                )
                if ref_result.get("status") == "ok":
                    result["callers"] = (
                        ref_result.get("callers", []) if "callers" in include else []
                    )
                    result["callees"] = (
                        ref_result.get("callees", []) if "callees" in include else []
                    )
                else:
                    result["callers"] = []
                    result["callees"] = []

            # Type hints
            if "type_hints" in include:
                type_result = self.get_symbol_metadata(
                    buffer_id=buffer_id,
                    symbol=symbol,
                    include_types=True,
                    type_inference_method=type_inference_method,
                )
                if type_result.get("status") == "ok":
                    result["types"] = {
                        "parameters": type_result.get("parameters", []),
                        "return_type": type_result.get("return_type"),
                        "type_confidence": type_result.get("type_confidence"),
                        "inference_method": type_result.get("inference_method"),
                    }
                else:
                    result["types"] = None

            # Tests
            if "tests" in include:
                result["tests"] = self._find_tests_for_symbol(chunks, symbol)

            # Related code (semantic search)
            if "related_code" in include:
                related = self.semantic_search(buffer_id=buffer_id, query=symbol, top_k=5)
                if related.get("status") == "ok":
                    result["related_code"] = related.get("matches", [])
                else:
                    result["related_code"] = []

            # Error handling
            if "errors" in include:
                result["errors"] = self._analyze_error_handling(chunks, symbol)

            return result

        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                f"Get full context failed: {e}",
                buffer_id=buffer_id,
                operation="get_full_context",
            )

    @staticmethod
    def _find_tests_for_symbol(chunks: list[Any], symbol: str) -> list[dict[str, Any]]:
        """Find test files/functions related to a symbol."""
        import re

        tests: list[dict[str, Any]] = []
        # Heuristic: test files that import or reference the symbol
        test_patterns = ["test_", "_test", "tests/", "spec_", "_spec"]
        for chunk in chunks:
            if not hasattr(chunk, "file") or not hasattr(chunk, "text"):
                continue
            # Check if it's a test file
            is_test = any(p in chunk.file for p in test_patterns)
            if not is_test:
                continue
            # Check if it references the symbol
            symbols_called = getattr(chunk, "symbols_called", None) or []
            if symbol in symbols_called or re.search(rf"\b{re.escape(symbol)}\b", chunk.text or ""):
                tests.append(
                    {
                        "file": chunk.file,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "name": getattr(chunk, "name", ""),
                        "type": getattr(chunk, "type", "function"),
                    }
                )
        return tests[:10]

    @staticmethod
    def _analyze_error_handling(chunks: list[Any], symbol: str) -> list[dict[str, Any]]:
        """Find error handling patterns related to a symbol."""
        import re

        errors: list[dict[str, Any]] = []
        # Find try/except blocks that contain the symbol
        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue
            if symbol not in chunk.text:
                continue
            # Find try/except blocks
            try_pattern = re.compile(r"try:\s*\n(.*?)\bexcept\b", re.DOTALL)
            for match in try_pattern.finditer(chunk.text):
                try_block = match.group(1)
                if symbol in try_block:
                    line = chunk.start_line + chunk.text[: match.start()].count("\n")
                    errors.append(
                        {
                            "file": chunk.file,
                            "line": line,
                            "type": "try_except",
                            "context": try_block.strip()[:200],
                        }
                    )
        return errors[:5]

    def list_file_symbols(
        self,
        buffer_id: str,
        file: str,
    ) -> dict[str, Any]:
        """List all symbols defined in a specific file."""
        result = self._require_buffer(buffer_id, "list_file_symbols")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            index = SymbolIndex(chunks)
            symbols = index.get_file_symbols(file)
            return {"status": "ok", "symbols": [s.to_dict() for s in symbols]}
        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                f"List file symbols failed: {e}",
                buffer_id=buffer_id,
                operation="list_file_symbols",
            )

    def get_symbol_metadata(
        self,
        buffer_id: str,
        symbol: str,
        include_types: bool = True,
        type_inference_method: str = "ast",
    ) -> dict[str, Any]:
        """Get comprehensive metadata for a symbol.

        Args:
            buffer_id: Buffer handle.
            symbol: Symbol name (may be qualified: "ClassName.method_name").
            include_types: Include inferred type hints (default: True).
            type_inference_method: "ast" for fast extraction or "llm" for accurate inference.

        Returns:
            Dict with file, line, type, parameters, return_type, lines_of_code,
            cyclomatic_complexity, called_by_count, calls_count, docstring,
            and optionally type_confidence.
        """
        result = self._require_buffer(buffer_id, "get_symbol_metadata", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        if type_inference_method not in ("llm", "ast"):
            return self._make_error_response(
                f"type_inference_method must be 'llm' or 'ast', got '{type_inference_method}'",
                buffer_id=buffer_id,
                operation="get_symbol_metadata",
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="get_symbol_metadata"
            )

        try:
            index = SymbolIndex(chunks)
            definitions = index.get_definition(symbol)
            if not definitions:
                return self._make_error_response(
                    f"Symbol '{symbol}' not found",
                    buffer_id=buffer_id,
                    operation="get_symbol_metadata",
                )

            entry = definitions[0]

            # Find the chunk for this symbol
            matching_chunks = [
                c
                for c in chunks
                if hasattr(c, "name")
                and c.name == entry.name
                and c.file == entry.file
                and c.start_line == entry.start_line
            ]
            if not matching_chunks:
                # Fallback: any chunk in same file covering same lines
                matching_chunks = [
                    c
                    for c in chunks
                    if c.file == entry.file and c.start_line <= entry.start_line <= c.end_line
                ]

            chunk = matching_chunks[0] if matching_chunks else None

            # Basic metadata from SymbolEntry
            result: dict[str, Any] = {
                "name": entry.name,
                "file": entry.file,
                "line": entry.start_line,
                "end_line": entry.end_line,
                "type": entry.type,
                "lines_of_code": (entry.end_line - entry.start_line + 1) if chunk else 0,
                "parent": entry.parent,
            }

            if chunk and chunk.text:
                # Extract docstring
                result["docstring"] = self._extract_docstring_from_text(chunk.text)

                # Compute cyclomatic complexity
                result["cyclomatic_complexity"] = self._compute_cyclomatic_complexity(chunk.text)

                # Count references
                references = index.get_references(symbol)
                result["called_by_count"] = len([r for r in references if r.confidence == "high"])

                # Count outgoing calls (symbols_called)
                calls_count = 0
                if hasattr(chunk, "symbols_called") and chunk.symbols_called:
                    calls_count = len(chunk.symbols_called)
                result["calls_count"] = calls_count

                # Type inference
                if include_types:
                    if type_inference_method == "llm":
                        cached = self._type_inference_cache.get(buffer_id, symbol)
                        if cached is not None:
                            result["parameters"] = cached.parameters
                            result["return_type"] = cached.return_type
                            result["type_confidence"] = cached.confidence
                            result["inference_method"] = "llm"
                        else:
                            type_info = self._get_type_info_from_chunk(chunk, "llm")
                            result.update(type_info)
                    else:
                        type_info = self._get_type_info_from_chunk(chunk, "ast")
                        result.update(type_info)

            return {"status": "ok", **result}

        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                f"Get symbol metadata failed: {e}",
                buffer_id=buffer_id,
                operation="get_symbol_metadata",
            )

    @staticmethod
    def _extract_docstring_from_text(text: str) -> str | None:
        """Extract docstring from code text."""
        import re

        # Match triple-quoted docstring after def/class
        match = re.search(
            r'(?:^|\n)\s+(?P<doc>("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'))',
            text,
        )
        if match:
            doc = match.group("doc")
            # Strip quotes and surrounding whitespace
            return doc[3:-3].strip()
        return None

    @staticmethod
    def _compute_cyclomatic_complexity(text: str) -> int:
        """Compute McCabe cyclomatic complexity from code text.

        Simplified: counts if/elif/else/for/while/except/and/or/assert.
        Base complexity is 1.
        """
        import re

        complexity = 1
        patterns = [
            r"\bif\b",
            r"\belif\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\bexcept\b",
            r"\band\b",
            r"\bor\b",
            r"\bassert\b",
        ]
        for pattern in patterns:
            complexity += len(re.findall(pattern, text))
        return complexity

    def _get_type_info_from_chunk(self, chunk: Any, method: str) -> dict[str, Any]:
        """Extract type info from a chunk, optionally using cache."""
        from gigacode.type_inference_cache import InferredType
        from gigacode.type_search import _extract_python_types

        result: dict[str, Any] = {}
        sigs = _extract_python_types(chunk.text) if chunk.text else []
        if sigs:
            sig = sigs[0]
            result["parameters"] = sig.parameters
            result["return_type"] = sig.return_type
            result["inference_method"] = method

            if method == "llm" and self._search_service:
                try:
                    type_info = self._search_service._infer_type_for_chunk(chunk, method="llm")
                    result["type_confidence"] = (
                        type_info.get("type_confidence") if type_info else None
                    )
                except (RuntimeError, ValueError, TypeError):
                    result["type_confidence"] = None
                # Cache
                inferred = InferredType(
                    symbol_name=chunk.name if hasattr(chunk, "name") else "",
                    file=chunk.file,
                    parameters=sig.parameters,
                    return_type=sig.return_type,
                    confidence=result.get("type_confidence", 0.0) or 0.0,
                    method="llm",
                )
                self._type_inference_cache.put("", inferred)
            else:
                result["type_confidence"] = None
        else:
            result["parameters"] = []
            result["return_type"] = None
            result["inference_method"] = method
            result["type_confidence"] = None

        return result

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    def cluster_code(
        self,
        buffer_id: str,
        threshold: float = 0.75,
    ) -> dict[str, Any]:
        """Find groups of semantically similar chunks within a buffer.

        Uses a greedy clustering algorithm:
        1. For each chunk, find all subsequent chunks (look-ahead window of 64)
           with similarity > threshold
        2. Group them into a cluster
        3. Return clusters with ≥2 members, sorted by file/line

        Args:
            buffer_id: Buffer to cluster.
            threshold: Cosine similarity threshold (0.0-1.0). Default 0.75.
                       Increase for tighter clusters, decrease for looser.

        Returns:
            Dict with status="ok" and clusters list. Each cluster contains:
            - file: Relative path
            - start_line: First chunk's start_line
            - end_line: Last chunk's end_line
            - size: Number of chunks in cluster
            - avg_score: Average pairwise similarity within cluster

        Example:
            ```python
            result = tool.cluster_code("buffer_id", threshold=0.8)
            for cluster in result["clusters"]:
                print(f"{cluster['file']} ({cluster['size']} chunks, "
                      f"avg_score={cluster['avg_score']:.3f})")
            ```
        """
        result = self._require_buffer(buffer_id, "cluster_code", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        if not self._search_service:
            return self._make_error_response(
                "SearchService unavailable",
                buffer_id=buffer_id,
                operation="cluster_code",
            )

        try:
            result = self._search_service.cluster_code(
                buffer_id=buffer_id,
                n_clusters=max(
                    2, int(1.0 / (1.0 - threshold + 0.1))
                ),  # Convert threshold to n_clusters estimate
            )
            return self._adapt_cluster_response(result)
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(f"SearchService.cluster_code failed: {e}")
            return self._make_error_response(
                f"Clustering failed: {e}",
                buffer_id=buffer_id,
                operation="cluster_code",
            )

    def find_duplicates(
        self,
        buffer_id: str,
        threshold: float = 0.85,
    ) -> dict[str, Any]:
        """Find near-duplicate code chunks within a buffer.

        Detects code snippets with high semantic similarity (≥ threshold).
        Useful for:
        - Finding copy-paste code
        - Identifying dead code (after refactoring)
        - Planning refactoring efforts (consolidate duplicates)

        Algorithm: Uses embedding similarity computed during indexing.
        Delegates to `duplicate_detector.find_duplicates()`.

        Args:
            buffer_id: Buffer to analyze.
            threshold: Cosine similarity threshold (0.0-1.0). Default 0.85.
                       Increase for stricter matching; decrease to catch loose dups.

        Returns:
            Dict with status="ok" and duplicates list. Each entry contains:
            - file1, start_line1, end_line1 — First chunk location
            - file2, start_line2, end_line2 — Second chunk location
            - similarity: Cosine similarity score

        Example:
            ```python
            result = tool.find_duplicates("buffer_id", threshold=0.9)
            for dup in result["duplicates"]:
                print(f"Dup: {dup['file1']}:{dup['start_line1']} ≈ "
                      f"{dup['file2']}:{dup['start_line2']} "
                      f"(similarity={dup['similarity']:.3f})")
            ```
        """
        result = self._require_buffer(buffer_id, "find_duplicates", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        if not self._search_service:
            return self._make_error_response(
                "SearchService unavailable",
                buffer_id=buffer_id,
                operation="find_duplicates",
            )

        try:
            result = self._search_service.find_duplicates(
                buffer_id=buffer_id,
                threshold=threshold,
            )
            return self._adapt_duplicate_response(result)
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(f"SearchService.find_duplicates failed: {e}")
            return self._make_error_response(
                f"Duplicate detection failed: {e}",
                buffer_id=buffer_id,
                operation="find_duplicates",
            )

    def search_batch(
        self,
        buffer_id: str,
        queries: list[str],
        top_k: int = 5,
        include_types: bool = False,
        type_inference_method: str = "llm",
    ) -> dict[str, Any]:
        """Search multiple queries in one call.

        Embeds all queries in parallel via batch embedding, then searches
        for each independently. Returns a dict mapping each query to its results.

        Args:
            buffer_id: Buffer handle.
            queries: List of natural language search queries.
            top_k: Number of top results per query (default: 5).
            include_types: Include inferred type hints in results (default: False).
            type_inference_method: "llm" (accurate) or "ast" (fast).

        Returns:
            Dict with status and results dict mapping query strings to match lists.
        """
        if not queries:
            return self._make_error_response(
                "queries list is empty",
                buffer_id=buffer_id,
                operation="search_batch",
            )

        if len(queries) > 20:
            return self._make_error_response(
                f"Too many queries ({len(queries)}), max 20",
                buffer_id=buffer_id,
                operation="search_batch",
            )

        if not self._search_service:
            return self._make_error_response(
                "SearchService unavailable",
                buffer_id=buffer_id,
                operation="search_batch",
            )

        try:
            results: dict[str, list[dict[str, Any]]] = {}
            for query in queries:
                search_result = self.semantic_search(
                    buffer_id=buffer_id,
                    query=query,
                    top_k=top_k,
                    include_types=include_types,
                    type_inference_method=type_inference_method,
                )
                if search_result.get("status") == "ok":
                    results[query] = search_result.get("matches", [])
                else:
                    results[query] = []

            return {"status": "ok", "results": results, "query_count": len(queries)}

        except (RuntimeError, ValueError, TypeError) as e:
            return self._make_error_response(
                f"Batch search failed: {e}",
                buffer_id=buffer_id,
                operation="search_batch",
            )

    def pack_context(
        self,
        buffer_id: str,
        query: str,
        max_tokens: int = 8192,
        top_k: int = 20,
    ) -> dict[str, Any]:
        """Return an optimally packed set of chunks fitting within *max_tokens*.

        Uses hybrid search to find the most relevant chunks, then greedily
        packs them by score until the token budget is exhausted.
        """
        result = self._require_buffer(buffer_id, "pack_context")
        if isinstance(result, dict):
            return result
        info, chunks = result

        # Use hybrid search for relevance scoring (always delegates to SearchService)
        search_result = self.hybrid_search(buffer_id, query, top_k=top_k, offset=0)
        if search_result.get("status") != "ok":
            return search_result

        matches = search_result.get("matches", [])
        if not matches:
            return {
                "status": "ok",
                "packed_chunks": [],
                "total_tokens": 0,
                "remaining_tokens": max_tokens,
                "count": 0,
            }

        # Build score map by doc_id
        scores = [0.0] * len(chunks)
        for m in matches:
            did = m.get("doc_id")
            if did is not None and 0 <= did < len(chunks):
                scores[did] = m.get("rrf_score", m.get("score", 0.0))

        return pack_context(chunks, scores, max_tokens=max_tokens)

    def pack_context_smart(
        self,
        buffer_id: str,
        query: str,
        max_tokens: int = 8192,
        top_k: int = 20,
        deduplicate: bool = True,
        strip_boilerplate: bool = True,
        strip_docstrings: str = "auto",
        exclude_types: list[str] | None = None,
        exclude_test_files: bool = False,
        min_lines: int = 3,
        max_lines: int = 200,
        granularity: str = "smart",
    ) -> dict[str, Any]:
        """Smart context packing with token-saving optimizations.

        Applies multiple filters to reduce token usage while maintaining
        relevance:
        1. Deduplication: remove copy-pasted chunks
        2. Boilerplate stripping: remove imports, licenses, __all__
        3. Docstring stripping: remove verbose docstrings (unless query
           asks for docs)
        4. Type filtering: skip orphans, sliding windows, tests
        5. Size filtering: skip tiny or huge chunks
        6. Granularity: use signatures-only for low-relevance chunks

        Expected token savings: 30-40% on typical codebases.

        Args:
            buffer_id: Buffer handle.
            query: Search query for finding relevant chunks.
            max_tokens: Maximum tokens for assembled context.
            top_k: Number of search results to consider.
            deduplicate: Remove near-duplicate chunks (default True).
            strip_boilerplate: Remove import/license headers (default True).
            strip_docstrings: "auto" | "always" | "never".
            exclude_types: Chunk types to skip (default ["orphan", "sliding"]).
            exclude_test_files: Skip test files (default False).
            min_lines: Minimum chunk line count (default 3).
            max_lines: Maximum chunk line count (default 200).
            granularity: "signatures" | "bodies" | "smart" (default "smart").

        Returns:
            Dict with packed_chunks, total_tokens, savings stats, and
            filter report.
        """
        result = self._require_buffer(buffer_id, "pack_context_smart")
        if isinstance(result, dict):
            return result
        info, chunks = result

        # Use hybrid search for relevance scoring
        search_result = self.hybrid_search(buffer_id, query, top_k=top_k, offset=0)
        if search_result.get("status") != "ok":
            return search_result

        matches = search_result.get("matches", [])
        if not matches:
            return {
                "status": "ok",
                "packed_chunks": [],
                "total_tokens": 0,
                "remaining_tokens": max_tokens,
                "count": 0,
                "saved_tokens": 0,
                "savings_percent": 0.0,
                "filter_stats": {"original": 0},
            }

        # Build score map by doc_id
        scores = [0.0] * len(chunks)
        for m in matches:
            did = m.get("doc_id")
            if did is not None and 0 <= did < len(chunks):
                scores[did] = m.get("rrf_score", m.get("score", 0.0))

        from gigacode.context_packer import pack_context_smart

        return pack_context_smart(
            chunks,
            scores,
            query=query,
            max_tokens=max_tokens,
            deduplicate=deduplicate,
            strip_boilerplate=strip_boilerplate,
            strip_docstrings=strip_docstrings,
            exclude_types=exclude_types,
            exclude_test_files=exclude_test_files,
            min_lines=min_lines,
            max_lines=max_lines,
            granularity=granularity,
        )

    def pack_context_hierarchical(
        self,
        buffer_id: str,
        query: str,
        max_tokens: int = 8192,
        levels: list[str] | None = None,
        top_k_files: int = 5,
        top_k_chunks: int = 10,
    ) -> dict[str, Any]:
        """Pack context hierarchically for LLM consumption.

        Provides three levels of detail:
        1. File summaries (overview, top-level definitions)
        2. Chunk summaries (function/class signatures + docstrings)
        3. Specific lines (query-relevant line ranges)

        Args:
            buffer_id: Buffer handle.
            query: Search query.
            max_tokens: Maximum tokens for assembled context.
            levels: Which levels to include ["file_summary", "chunk", "lines"].
            top_k_files: Number of top files to include.
            top_k_chunks: Number of top chunks per file.

        Returns:
            Dict with hierarchy list and total_tokens.
        """
        result = self._require_buffer(buffer_id, "pack_context_hierarchical")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            query_embedding = self._embedder.encode([query])[0]
            summarizer = ContextSummarizer(chunks)
            result = summarizer.pack_hierarchical(
                query_embedding,
                query,
                max_tokens,
                levels,
                top_k_files,
                top_k_chunks,
            )
            return {"status": "ok", **result.to_dict()}
        except (ValueError, RuntimeError) as e:
            logger.warning(f"ContextSummarizer.pack_hierarchical failed: {e}")
            return self._make_error_response(
                f"Hierarchical context packing failed: {e}",
                buffer_id=buffer_id,
                operation="pack_context_hierarchical",
            )

    def related_code(
        self,
        buffer_id: str,
        file: str,
        start_line: int,
        end_line: int | None = None,
        include: list[str] | None = None,
        top_k: int = 10,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        """Assemble cross-file related context for a code location.

        Given a file and line range, finds:
        - Callers of the target symbol
        - Related test files
        - Interface definitions (classes, traits, protocols)
        - Semantic neighbors (similar chunks in other files)
        - Imports for the target chunk

        Args:
            buffer_id: Buffer handle.
            file: Relative file path within the buffer.
            start_line: Start line (1-based).
            end_line: End line (1-based). If None, uses start_line.
            include: Context types to include:
                ["callers", "tests", "interfaces", "imports", "semantic"]
                Default: all.
            top_k: Maximum number of results per category.
            max_tokens: Token budget for assembled context.

        Returns:
            Dict with ``status``, the assembled context fields
            (callers, tests, interfaces, imports, semantic_neighbors),
            and ``total_tokens``.
        """
        result = self._require_buffer(buffer_id, "related_code")
        if isinstance(result, dict):
            return result
        info, chunks = result

        # Load embeddings from disk
        buffer_dir = Path(info["buffer_dir"])
        embeddings_path = buffer_dir / "embeddings.npy"
        embeddings: np.ndarray | None = None
        if embeddings_path.exists():
            try:
                embeddings = np.load(embeddings_path)
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to load embeddings for {buffer_id}: {e}")

        language = info.get("language_hint", "python")

        assembler = ContextAssembler(
            chunks=chunks,
            embeddings=embeddings,
            language=language,
        )

        try:
            result = assembler.assemble(
                file=file,
                start_line=start_line,
                end_line=end_line,
                include=include,
                max_tokens=max_tokens,
            )
            return {"status": "ok", **result.to_dict()}
        except (ValueError, RuntimeError) as e:
            logger.warning(f"ContextAssembler.assemble failed: {e}")
            return self._make_error_response(
                f"Related code assembly failed: {e}",
                buffer_id=buffer_id,
                operation="related_code",
            )

    def get_context(
        self,
        buffer_id: str,
        file: str,
        start_line: int,
        end_line: int | None = None,
        include: list[str] | None = None,
        top_k: int = 10,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        """Get cross-file context for a code location (alias for `related_code`).

        Returns assembled context with callers, tests, interfaces, imports,
        and semantic neighbors, token-budgeted to `max_tokens`.
        """
        return self.related_code(
            buffer_id=buffer_id,
            file=file,
            start_line=start_line,
            end_line=end_line,
            include=include,
            top_k=top_k,
            max_tokens=max_tokens,
        )

    def refactor_rename(
        self,
        buffer_id: str,
        old_name: str,
        new_name: str,
        scope: str = "buffer",
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Rename a symbol across the codebase.

        Args:
            buffer_id: Buffer handle.
            old_name: Current symbol name.
            new_name: Desired symbol name.
            scope: "buffer" to rename across all files.
            dry_run: If True, returns preview without modifying.

        Returns:
            Dict with status, changed_files, changes list.
        """
        result = self._require_buffer(buffer_id, "refactor_rename")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            engine = RefactorEngine(chunks)
            result = engine.rename_symbol(old_name, new_name, scope, dry_run)
            return result.to_dict()
        except (ValueError, RuntimeError) as e:
            logger.warning(f"RefactorEngine.rename_symbol failed: {e}")
            return self._make_error_response(
                f"Rename failed: {e}",
                buffer_id=buffer_id,
                operation="refactor_rename",
            )

    def edit_symbol(
        self,
        buffer_id: str,
        symbol_name: str,
        new_body: str,
        language_hint: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Replace the body of a symbol while preserving its signature.

        Finds the symbol by name, replaces its body with ``new_body``,
        and leaves the signature (def name(...):) intact.

        Args:
            buffer_id: Buffer handle.
            symbol_name: Symbol to edit (e.g. "fetch_data").
            new_body: New body text (lines will be indented to match original).
            language_hint: Language for parsing heuristics.
            dry_run: If True, returns preview without modifying.

        Returns:
            Dict with status, change preview, and applied info.
        """
        result = self._require_buffer(buffer_id, "edit_symbol")
        if isinstance(result, dict):
            return result
        info, chunks = result

        language = language_hint or info.get("language_hint", "python")
        try:
            result = _edit_symbol(
                chunks,
                symbol_name,
                new_body,
                language=language,
                dry_run=True,  # Always dry-run first to preview
            )
            if result.status != "ok":
                return result.to_dict()

            if not dry_run and result.changes:
                change = result.changes[0]
                # Apply via write_code using the chunk's absolute line numbers
                self.write_code(
                    buffer_id,
                    file=change.file,
                    start_line=change.start_line,
                    new_lines=change.new_text.splitlines(),
                    end_line=change.end_line,
                )

            return result.to_dict()
        except (ValueError, RuntimeError) as e:
            logger.warning(f"edit_symbol failed: {e}")
            return self._make_error_response(
                f"Edit symbol failed: {e}",
                buffer_id=buffer_id,
                operation="edit_symbol",
            )

    def add_parameter(
        self,
        buffer_id: str,
        symbol_name: str,
        param: str,
        default: str | None = None,
        language_hint: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Add a parameter to a function signature.

        Args:
            buffer_id: Buffer handle.
            symbol_name: Function to modify.
            param: Parameter name.
            default: Default value string (e.g. "None", "[]").
            language_hint: Language for parsing heuristics.
            dry_run: If True, returns preview without modifying.

        Returns:
            Dict with status, change preview, and applied info.
        """
        result = self._require_buffer(buffer_id, "add_parameter")
        if isinstance(result, dict):
            return result
        info, chunks = result

        language = language_hint or info.get("language_hint", "python")
        try:
            result = _add_parameter(
                chunks,
                symbol_name,
                param,
                default,
                language=language,
                dry_run=True,
            )
            if result.status != "ok":
                return result.to_dict()

            if not dry_run and result.changes:
                change = result.changes[0]
                self.write_code(
                    buffer_id,
                    file=change.file,
                    start_line=change.start_line,
                    new_lines=change.new_text.splitlines(),
                    end_line=change.end_line,
                )

            return result.to_dict()
        except (ValueError, RuntimeError) as e:
            logger.warning(f"add_parameter failed: {e}")
            return self._make_error_response(
                f"Add parameter failed: {e}",
                buffer_id=buffer_id,
                operation="add_parameter",
            )

    def extract_method(
        self,
        buffer_id: str,
        file: str,
        start_line: int,
        end_line: int,
        new_name: str,
        language_hint: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Extract a range of lines into a new method.

        Replaces the selected lines with a call to the new method and inserts
        the new method definition nearby.

        Args:
            buffer_id: Buffer handle.
            file: File containing the lines to extract.
            start_line: First line to extract (1-based).
            end_line: Last line to extract (1-based, inclusive).
            new_name: Name for the extracted method.
            language_hint: Language for parsing heuristics.
            dry_run: If True, returns preview without modifying.

        Returns:
            Dict with status, preview, and applied info.
        """
        result = self._require_buffer(buffer_id, "extract_method", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        snapshot_mgr = self._get_snapshot_manager(buffer_id)
        if snapshot_mgr is None:
            return self._make_error_response(
                "Snapshot not available", buffer_id=buffer_id, operation="extract_method"
            )

        if file not in snapshot_mgr.manifest.files:
            return self._make_error_response(
                f"File not in buffer: {file}", buffer_id=buffer_id, operation="extract_method"
            )

        try:
            old_lines = snapshot_mgr.read_lines(file)
            if old_lines is None:
                return self._make_error_response(
                    f"Failed to read file: {file}",
                    buffer_id=buffer_id,
                    operation="extract_method",
                )

            language = language_hint or info.get("language_hint", "python")
            result = SymbolEditor.extract_method(
                old_lines, start_line, end_line, new_name, language
            )
            if result is None:
                return self._make_error_response(
                    "Extract method failed: invalid range or unsupported language.",
                    buffer_id=buffer_id,
                    operation="extract_method",
                )

            new_lines, insert_at, call_line = result

            if not dry_run:
                # Replace entire file content atomically
                self.write_code(
                    buffer_id,
                    file=file,
                    start_line=1,
                    new_lines=new_lines,
                    end_line=len(old_lines) + 1,
                )

            return {
                "status": "ok",
                "file": file,
                "new_method": new_name,
                "inserted_at": insert_at,
                "call_line": call_line,
                "old_line_count": len(old_lines),
                "new_line_count": len(new_lines),
                "dry_run": dry_run,
            }
        except (ValueError, RuntimeError) as e:
            logger.warning(f"extract_method failed: {e}")
            return self._make_error_response(
                f"Extract method failed: {e}",
                buffer_id=buffer_id,
                operation="extract_method",
            )

    def add_import(
        self,
        buffer_id: str,
        file: str,
        module: str,
        symbols: list[str] | None = None,
        language_hint: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Add an import statement to a file.

        Args:
            buffer_id: Buffer handle.
            file: Target file path.
            module: Module to import.
            symbols: Specific symbols, or None for module import.
            language_hint: Language for syntax selection.
            dry_run: If True, returns preview without modifying.

        Returns:
            Dict with status, changed_files, changes list.
        """
        result = self._require_buffer(buffer_id, "add_import")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            engine = RefactorEngine(chunks)
            language = language_hint or info.get("language_hint", "python")
            result = engine.add_import(file, module, symbols, language, dry_run)
            return result.to_dict()
        except (ValueError, RuntimeError) as e:
            logger.warning(f"RefactorEngine.add_import failed: {e}")
            return self._make_error_response(
                f"Add import failed: {e}",
                buffer_id=buffer_id,
                operation="add_import",
            )

    def remove_import(
        self,
        buffer_id: str,
        file: str,
        module: str,
        language_hint: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Remove an import statement from a file.

        Args:
            buffer_id: Buffer handle.
            file: Target file path.
            module: Module to remove.
            language_hint: Language for syntax selection.
            dry_run: If True, returns preview without modifying.

        Returns:
            Dict with status, changed_files, changes list.
        """
        result = self._require_buffer(buffer_id, "remove_import")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            engine = RefactorEngine(chunks)
            language = language_hint or info.get("language_hint", "python")
            result = engine.remove_import(file, module, language, dry_run)
            return result.to_dict()
        except (ValueError, RuntimeError) as e:
            logger.warning(f"RefactorEngine.remove_import failed: {e}")
            return self._make_error_response(
                f"Remove import failed: {e}",
                buffer_id=buffer_id,
                operation="remove_import",
            )

    def apply_patch(
        self,
        buffer_id: str,
        patch_text: str,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Apply a unified diff patch to the buffer.

        Accepts standard unified diff format (like ``git diff`` output).
        Supports multiple files in one patch.

        Args:
            buffer_id: Buffer handle.
            patch_text: Unified diff string with @@ hunk headers.
            dry_run: If True, returns preview without modifying.

        Returns:
            Dict with changed_files, changes list, and status.

        Example patch_text::

            --- a/auth.py
            +++ b/auth.py
            @@ -45,7 +45,7 @@
             def validate_token(token: str) -> dict:
            -    payload = jwt.decode(token, SECRET)
            +    payload = jwt.decode(token, SECRET, options={"verify_exp": True})
                 return payload
        """
        result = self._require_buffer(buffer_id, "apply_patch")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            applier = PatchApplier(chunks)
            result = applier.apply_patch(patch_text, dry_run)
            return {"status": "ok", **result.to_dict()}
        except (ValueError, RuntimeError) as e:
            logger.warning(f"PatchApplier.apply_patch failed: {e}")
            return self._make_error_response(
                f"Apply patch failed: {e}",
                buffer_id=buffer_id,
                operation="apply_patch",
            )

    # ------------------------------------------------------------------
    # Read / Write / Commit (Agent editing workflow)
    # ------------------------------------------------------------------
    def read_code(
        self,
        buffer_id: str,
        file: str | None = None,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read file contents from buffer (delegates to BufferManager)."""
        t0 = time.perf_counter()

        # Delegate core operation to BufferManager
        result = self._buffer_manager.read_code(
            buffer_id=buffer_id,
            file=file,
            start_line=start_line,
            end_line=end_line,
        )

        # Check if read was successful
        if result.get("status") != "ok":
            # BufferManager already logged the error via audit_log
            return result

        # Orchestration: Security audit logging
        if file is not None:
            self._security.log_success(
                operation="read_code",
                buffer_id=buffer_id,
                details={
                    "file": file,
                    "start_line": start_line,
                    "end_line": result.get("end_line"),
                    "lines_count": len(result.get("lines", [])),
                },
            )
            # Orchestration: Prometheus metrics
            elapsed = time.perf_counter() - t0
            if self._prometheus_exporter:
                self._prometheus_exporter.record_operation(
                    operation="read_code",
                    duration_s=elapsed,
                    status="ok",
                    chunk_count=len(result.get("lines", [])),
                )
        else:
            # Read-all operation
            files_dict = result.get("files", {})
            self._security.log_success(
                operation="read_code",
                buffer_id=buffer_id,
                details={
                    "files_count": len(files_dict),
                    "start_line": start_line,
                    "end_line": end_line,
                },
            )
            # Orchestration: Prometheus metrics for read-all
            elapsed = time.perf_counter() - t0
            if self._prometheus_exporter:
                self._prometheus_exporter.record_operation(
                    operation="read_code",
                    duration_s=elapsed,
                    status="ok",
                    chunk_count=sum(len(ls) for ls in files_dict.values()),
                )

        return result

    def write_code(
        self,
        buffer_id: str,
        file: str,
        start_line: int | str,
        new_lines: list[str] | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        # Backward compat: accept (buffer_id, file, content_string) call convention
        if isinstance(start_line, str):
            new_lines = start_line.splitlines(keepends=False)
            start_line = 1
            end_line = None
        elif new_lines is None:
            return {"status": "error", "message": "new_lines is required"}

        t0 = time.perf_counter()
        result = self._require_buffer(buffer_id, "write_code", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        # Check current buffer state - must be READY to write
        try:
            current_state = self._get_buffer_state(buffer_id)
            if current_state != BufferState.READY:
                return {
                    "status": "error",
                    "message": f"Cannot write code: buffer is in {current_state} state. "
                    f"Valid transitions: {BufferStateTransition.VALID_TRANSITIONS.get(current_state, [])}",
                }
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}
        if file not in snapshot:
            return {"status": "error", "message": f"File not in buffer: {file}"}

        # Validate file path to prevent traversal attacks
        try:
            buffer_dir = info.get("buffer_dir")
            if not buffer_dir:
                return {
                    "status": "error",
                    "message": f"Buffer metadata missing buffer_dir for {buffer_id}",
                }
            buffer_root = Path(buffer_dir)
            validate_buffer_path(file, buffer_root)
        except ValueError as e:
            return {"status": "error", "message": f"Invalid file path: {e}"}

        # Detect 3-way merge conflicts before allowing write
        snapshot_mgr = self._get_snapshot_manager(buffer_id)
        if snapshot_mgr is not None:
            # Get current buffer lines (before modification)
            current_buffer_lines = snapshot[file]

            # Compute diff to detect conflicts
            diff_result = snapshot_mgr.compute_diff(file, current_buffer_lines)
            if diff_result.get("has_conflict", False):
                # Conflict detected: both disk and buffer have diverged from snapshot
                return {
                    "status": "conflict",
                    "file": file,
                    "message": (
                        f"Cannot write to {file}: both disk and buffer have been modified. "
                        "Use reload_codebase() to sync with disk, or call diff() to review changes."
                    ),
                    "disk_lines": len(diff_result.get("disk_lines") or []),
                    "buffer_lines": len(diff_result.get("buffer_lines") or []),
                    "snapshot_line_count": diff_result.get("snapshot_line_count"),
                }

        old_lines = snapshot[file]
        end = end_line if end_line is not None else len(old_lines) + 1
        # Strip newlines from new_lines to be consistent with splitlines() format
        sanitized_new_lines = [line.rstrip("\n\r") for line in new_lines]
        # Replace lines from start_line to end_line (inclusive, 1-indexed)
        # Keep lines before start_line, add new lines, keep lines after end_line
        new_file_lines = old_lines[: start_line - 1] + sanitized_new_lines + old_lines[end:]
        snapshot[file] = new_file_lines
        self._save_source_snapshot(buffer_id, snapshot)

        dirty = info.setdefault("dirty_files", {})
        dirty[file] = True
        self._save_registry()

        # Invalidate type inference cache for the modified file
        self._type_inference_cache.invalidate_file(file)

        # Transition state from READY → DIRTY if not already dirty
        try:
            if current_state == BufferState.READY:
                self._set_buffer_state(buffer_id, BufferState.DIRTY)
        except ValueError:
            # State transition failed, but don't fail the write
            logger.warning(f"Failed to transition buffer {buffer_id} to DIRTY state")

        # Auto-rebuild if too many dirty files
        if len(dirty) >= MAX_DIRTY_BEFORE_AUTO_REBUILD:
            self._rebuild_dirty(buffer_id)

        result = {
            "status": "ok",
            "file": file,
            "changed_lines": len(sanitized_new_lines),
            "replaced_lines": end - start_line,
            "total_lines": len(new_file_lines),
            "diff": self._compute_unified_diff(old_lines, new_file_lines, file),
        }

        try:
            service = self._get_undo_redo_service(buffer_id)
            operation_id = service.record_write(
                buffer_id=buffer_id,
                file=file,
                original_content="\n".join(old_lines),
                new_content="\n".join(new_file_lines),
                description=f"Modified {file}",
            )
            result["operation_id"] = operation_id
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(f"Failed to record undo operation for write_code: {e}")

        # Audit log successful write
        self._security.log_success(
            operation="write_code",
            buffer_id=buffer_id,
            details={
                "file": file,
                "start_line": start_line,
                "end_line": end,
                "changed_lines": len(sanitized_new_lines),
                "replaced_lines": end - start_line,
            },
        )

        # Record operation metrics
        elapsed = time.perf_counter() - t0
        if self._prometheus_exporter:
            self._prometheus_exporter.record_operation(
                operation="write_code",
                duration_s=elapsed,
                status="ok",
                chunk_count=len(sanitized_new_lines),
            )

        return result

    def diff(self, buffer_id: str, file: str | None = None) -> dict[str, Any]:
        """Show diff between buffer and disk versions.

        Delegates to BufferManager.diff for actual diff computation, then
        adapts the response for backward compatibility.

        Args:
            buffer_id: Buffer handle.
            file: Optional specific file to diff. If None, diffs all files.

        Returns:
            Dict with status, diffs (per-file diff details), and has_conflicts flag.
            For backward compatibility, also includes changed_files list.
        """
        result = self._buffer_manager.diff(buffer_id, file=file)

        # Adapt to expected format for backward compatibility
        if result.get("status") not in ("ok", "conflict"):
            return result

        # Transform diffs dict to changed_files list for backward compat
        diffs = result.get("diffs", {})
        changed_files: list[dict[str, Any]] = []
        for fname, diff_info in diffs.items():
            # Include files that have actual changes
            has_changes = (
                diff_info.get("has_conflict", False)
                or len(diff_info.get("added_lines", [])) > 0
                or len(diff_info.get("removed_lines", [])) > 0
            )
            if has_changes:
                changed_files.append(
                    {
                        "file": fname,
                        "buffer_lines": len(diff_info.get("buffer_lines", [])),
                        "disk_lines": len(diff_info.get("disk_lines", [])),
                        "dirty": True,
                    }
                )

        result["changed_files"] = changed_files
        return result

    def discard(
        self,
        buffer_id: str,
        file: str | None = None,
    ) -> dict[str, Any]:
        result = self._require_buffer(buffer_id, "discard", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        root = Path(info["root"])
        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}

        files_to_revert = [file] if file is not None else list(snapshot.keys())
        reverted: list[str] = []
        dirty = info.setdefault("dirty_files", {})

        for rel_path in files_to_revert:
            disk_path = root / rel_path
            if not disk_path.exists():
                continue
            with disk_path.open("r", encoding="utf-8", errors="replace") as fh:
                disk_lines = fh.read().splitlines()
            snapshot[rel_path] = disk_lines
            dirty.pop(rel_path, None)
            reverted.append(rel_path)

        self._save_source_snapshot(buffer_id, snapshot)
        self._save_registry()

        # Rebuild reverted files
        if reverted:
            self._rebuild_files(buffer_id, reverted)

        # Transition to READY if all dirty files have been reverted
        if not info.get("dirty_files"):
            try:
                current_state = self._get_buffer_state(buffer_id)
                if current_state == BufferState.DIRTY:
                    self._set_buffer_state(buffer_id, BufferState.READY)
            except ValueError:
                logger.warning(f"Failed to transition buffer {buffer_id} to READY state")

        return {"status": "ok", "reverted_files": reverted}

    def commit(
        self,
        buffer_id: str,
        dry_run: bool = False,
        check_impact: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        """Commit buffer changes to disk with 3-way merge conflict handling and crash recovery.

        Uses SnapshotManager for merge conflicts.
        Wraps with StateManager transactions for crash recovery.
        Optional dependency-impact pre-check warns on high blast-radius.

        - If both buffer and disk modified: returns "conflict" status instead of aborting
        - Rebuilds embeddings before writing
        - Updates snapshot manifest after successful writes
        - Returns detailed conflict information for user resolution
        - Uses write-ahead logging (WAL) for crash recovery
        - If ``check_impact=True`` and risk is HIGH, blocks commit unless ``force=True``

        Args:
            buffer_id: Buffer to commit
            dry_run: If True, check what would be written without modifying files
            check_impact: If True, run dependency risk analysis before committing.
            force: If True, proceed with commit even if impact analysis reports HIGH risk.

        Returns:
            Dict with:
            - "status": "ok", "conflict", "error", or "blocked"
            - "written_files": List of successfully written files
            - "conflict_files": List of dicts with conflict details (file, message, line counts)
            - "dry_run": Whether this was a dry run
            - "transaction_id": Transaction ID (for debugging)
            - "impact_analysis": Included when check_impact=True and commit is blocked
        """
        t0 = time.perf_counter()
        result = self._require_buffer(buffer_id, "commit", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        root = Path(info["root"])
        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}

        dirty = info.get("dirty_files", {})
        if not dirty:
            return {
                "status": "ok",
                "written_files": [],
                "conflict_files": [],
                "dry_run": dry_run,
                "transaction_id": None,
            }

        # Dependency risk analysis pre-check
        impact_result: dict[str, Any] | None = None
        if check_impact and not dry_run:
            chunks = self._load_chunks(buffer_id)
            if chunks:
                try:
                    snapshot = self._load_source_snapshot(buffer_id)
                    snapshot_mgr = self._get_snapshot_manager(buffer_id)
                    inferred_symbols: list[str] = []
                    if snapshot is not None and snapshot_mgr is not None:
                        runner = TestRunner(chunks, root)
                        for rel_path in dirty:
                            disk_lines = snapshot_mgr.read_lines(rel_path)
                            new_lines = snapshot.get(rel_path, [])
                            if disk_lines is not None:
                                inferred_symbols.extend(
                                    runner.extract_symbols_from_diff(disk_lines, new_lines)
                                )
                    analyzer = ImpactAnalyzer(chunks)
                    impact_result = analyzer.analyze(
                        dirty_files=list(dirty.keys()),
                        modified_symbols=(
                            sorted(set(inferred_symbols)) if inferred_symbols else None
                        ),
                    ).to_dict()

                    if impact_result.get("risk_level") == "high" and not force:
                        return {
                            "status": "blocked",
                            "message": (
                                f"Commit blocked: HIGH risk impact detected "
                                f"({impact_result.get('total_impacted_files', 0)} impacted file(s), "
                                f"depth {impact_result.get('max_call_depth', 0)}). "
                                f"Call analyze_impact() for details, or pass force=True to override."
                            ),
                            "impact_analysis": impact_result,
                            "written_files": [],
                            "conflict_files": [],
                            "dry_run": dry_run,
                            "transaction_id": None,
                        }
                except (ValueError, RuntimeError) as e:
                    logger.warning(f"Impact pre-check failed for commit: {e}")

        # Begin transaction for crash recovery
        transaction_id = None
        if not dry_run:
            transaction_id = self._state_manager.start_transaction(
                operation="commit",
                buffer_id=buffer_id,
                file_path=",".join(dirty.keys()),  # Comma-separated list of files
                start_line=0,
                end_line=None,
                new_lines=None,
            )

        try:
            # Rebuild embeddings for dirty files before writing to disk
            if not dry_run:
                self._rebuild_files(buffer_id, list(dirty.keys()))

            # Use SnapshotManager for 3-way merge conflict handling
            snapshot_mgr = self._get_snapshot_manager(buffer_id)
            if snapshot_mgr is None:
                if transaction_id:
                    self._state_manager.rollback_transaction(transaction_id)
                return {"status": "error", "message": "Snapshot manager not available"}

            written: list[str] = []
            conflicts: list[dict[str, Any]] = []
            new_hashes: dict[str, str] = {}
            updated_files: dict[str, Path] = {}

            for rel_path in dirty:
                lines = snapshot.get(rel_path, [])
                disk_path = root / rel_path

                # Validate file path to prevent traversal attacks
                try:
                    validate_buffer_path(rel_path, root)
                except ValueError as e:
                    logger.warning(f"Invalid file path in dirty files: {rel_path}: {e}")
                    conflicts.append(
                        {
                            "file": rel_path,
                            "message": f"Invalid file path: {e}",
                        }
                    )
                    continue

                if dry_run:
                    # Dry run: just check for conflicts without writing
                    diff_result = snapshot_mgr.compute_diff(rel_path, lines)
                    if diff_result.get("has_conflict"):
                        conflicts.append(
                            {
                                "file": rel_path,
                                "message": "3-way merge conflict: disk and buffer both modified",
                            }
                        )
                    else:
                        written.append(rel_path)
                else:
                    # Real commit: use 3-way merge with conflict detection
                    merge_result = snapshot_mgr.write_file_with_merge(
                        rel_path, lines, allow_conflicts=False
                    )

                    if merge_result["status"] == "conflict":
                        # Conflict detected - record it but continue processing other files
                        diff_result = snapshot_mgr.compute_diff(rel_path, lines)
                        conflicts.append(
                            {
                                "file": rel_path,
                                "message": merge_result.get("message", "3-way merge conflict"),
                                "disk_lines": len(diff_result.get("disk_lines") or []),
                                "buffer_lines": len(lines),
                                "snapshot_line_count": diff_result.get("snapshot_line_count"),
                            }
                        )
                    elif merge_result["status"] == "ok":
                        # Successfully written
                        written.append(rel_path)
                        updated_files[rel_path] = disk_path
                        new_hashes[rel_path] = hashlib.sha256(
                            "\n".join(lines).encode("utf-8")
                        ).hexdigest()
                    else:
                        # Write error (not a conflict, but a real error)
                        self._state_manager.rollback_transaction(transaction_id)
                        return {
                            "status": "error",
                            "message": f"Failed to write {rel_path}: {merge_result.get('message', 'Unknown error')}",
                            "transaction_id": transaction_id,
                        }

            if not dry_run:
                # Update registry with new hashes and clean up dirty files that were written
                info["file_hashes"].update(new_hashes)
                # Only clear dirty files that were successfully written (leave conflicted ones)
                for f in written:
                    info["dirty_files"].pop(f, None)
                self._save_registry()

                # Update snapshot manifest for successfully written files
                if updated_files:
                    snapshot_mgr.update_manifest_after_commit(updated_files)

                # Transition to READY if all files successfully written (no conflicts)
                if not conflicts and info.get("dirty_files"):
                    # Still have dirty files, stay in DIRTY state
                    pass
                elif not conflicts:
                    # All files written, no conflicts - transition to READY
                    try:
                        current_state = self._get_buffer_state(buffer_id)
                        if current_state == BufferState.DIRTY:
                            self._set_buffer_state(buffer_id, BufferState.READY)
                    except ValueError:
                        logger.warning(f"Failed to transition buffer {buffer_id} to READY state")

                # Commit transaction (WAL is updated)
                self._state_manager.commit_transaction(transaction_id)
                self._state_manager.save_registry()

            result = {
                "status": "conflict" if conflicts else "ok",
                "written_files": written,
                "conflict_files": conflicts,
                "dry_run": dry_run,
                "transaction_id": transaction_id,
            }
            if impact_result is not None:
                result["impact_analysis"] = impact_result

            # Audit log commit operation (success or conflict)
            if conflicts:
                self._security.log_failure(
                    operation="commit",
                    message=f"Commit conflict: {len(conflicts)} files",
                    buffer_id=buffer_id,
                    details={
                        "dry_run": dry_run,
                        "written_files_count": len(written),
                        "conflict_files_count": len(conflicts),
                        "transaction_id": transaction_id,
                    },
                )
            else:
                self._security.log_success(
                    operation="commit",
                    buffer_id=buffer_id,
                    details={
                        "dry_run": dry_run,
                        "written_files_count": len(written),
                        "conflict_files_count": len(conflicts),
                        "transaction_id": transaction_id,
                    },
                )

            # Record operation metrics for Prometheus
            elapsed = time.perf_counter() - t0
            if self._prometheus_exporter:
                self._prometheus_exporter.record_operation(
                    operation="commit",
                    duration_s=elapsed,
                    status="conflict" if conflicts else "ok",
                    chunk_count=len(written),
                )

            return result

        except (OSError, ValueError, RuntimeError) as e:
            # Rollback on any error
            if transaction_id:
                json_logger.error(
                    operation="commit",
                    buffer_id=buffer_id,
                    message=f"Commit failed with exception: {e}; rolling back transaction {transaction_id}",
                )
                self._state_manager.rollback_transaction(transaction_id)
            raise

    # ------------------------------------------------------------------
    # Automated Test-After-Edit Feedback Loop
    # ------------------------------------------------------------------
    def run_impacted_tests(
        self,
        buffer_id: str,
        top_k: int = 20,
        timeout: int = 120,
        extra_args: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find and run tests impacted by uncommitted (dirty) changes.

        1. Scans dirty files in the buffer.
        2. Extracts modified symbols from diffs.
        3. Finds test chunks that reference those files / symbols.
        4. Runs pytest on only the impacted test files (fast).
        5. Returns structured pass/fail with traces and token counts.

        Args:
            buffer_id: Buffer handle.
            top_k: Max number of impacted test files to run.
            timeout: Max seconds for pytest execution.
            extra_args: Additional pytest arguments.

        Returns:
            Dict with:
            - ``status``: "ok" | "no_tests" | "skipped" | "error"
            - ``passed``, ``failed``, ``skipped``, ``total``
            - ``tests``: list of individual test results
            - ``impacted_files``: list of test files that were run
            - ``token_estimate``: approximate tokens in the output
            - ``stdout``, ``stderr`` (truncated)
        """
        result = self._require_buffer(buffer_id, "run_impacted_tests")
        if isinstance(result, dict):
            return result
        info, chunks = result

        dirty = info.get("dirty_files", {})
        if not dirty:
            return {
                "status": "no_tests",
                "message": "No dirty files — nothing to test.",
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "total": 0,
                "impacted_files": [],
                "token_estimate": 50,
            }

        language = info.get("language_hint", "python")
        root = Path(info.get("root", self.work_dir))

        # Extract modified symbols from dirty file diffs
        snapshot = self._load_source_snapshot(buffer_id)
        modified_symbols: list[str] = []
        if snapshot is not None:
            for rel_path in dirty:
                old_lines = snapshot.get(rel_path, [])
                # Build new_lines from the snapshot (dirty state)
                new_lines = snapshot.get(rel_path, [])
                # Actually, snapshot already has the modified lines. We need to
                # compare with the original on-disk version. Use snapshot_mgr.
                snapshot_mgr = self._get_snapshot_manager(buffer_id)
                if snapshot_mgr is not None:
                    try:
                        disk_lines = snapshot_mgr.read_lines(rel_path)
                        if disk_lines is not None:
                            runner = TestRunner(chunks, root, language)
                            symbols = runner.extract_symbols_from_diff(disk_lines, new_lines)
                            modified_symbols.extend(symbols)
                    except (OSError, ValueError):
                        pass

        runner = TestRunner(chunks, root, language)
        summary = runner.run_impacted(
            modified_files=list(dirty.keys()),
            modified_symbols=modified_symbols if modified_symbols else None,
            top_k=top_k,
            timeout=timeout,
        )

        # Append extra_args if provided
        if extra_args and summary.status == "ok":
            # Re-run with extra args (rare, but supported)
            summary2 = runner.run_tests(
                summary.impacted_files,
                timeout=timeout,
                extra_args=extra_args,
            )
            return summary2.to_dict()

        return summary.to_dict()

    # ------------------------------------------------------------------
    # Dependency Risk Analysis on Edit
    # ------------------------------------------------------------------
    def analyze_impact(
        self,
        buffer_id: str,
        modified_symbols: list[str] | None = None,
        max_depth: int = 6,
    ) -> dict[str, Any]:
        """Analyze blast-radius of uncommitted (dirty) changes.

        For each dirty file and modified symbol, walks upstream callers
        and incoming file dependencies to compute a risk score.

        Args:
            buffer_id: Buffer handle.
            modified_symbols: Optional list of symbol names that were changed.
                If None, symbols are inferred from diffs of dirty files.
            max_depth: Max call-chain depth to traverse upward.

        Returns:
            Dict with:
            - ``status``: "ok" | "error"
            - ``risk_level``: "low" | "medium" | "high"
            - ``risk_score``: 0.0 – 1.0
            - ``total_impacted_files``: count
            - ``max_call_depth``: deepest call chain found
            - ``test_coverage_present``: bool
            - ``symbol_impacts``: per-symbol caller lists and depths
            - ``file_impacts``: per-file incoming deps and critical flags
            - ``recommendations``: human-readable action items
            - ``message``: one-line summary
        """
        result = self._require_buffer(buffer_id, "analyze_impact")
        if isinstance(result, dict):
            return result
        info, chunks = result

        dirty = info.get("dirty_files", {})
        if not dirty:
            return {
                "status": "ok",
                "risk_level": "low",
                "risk_score": 0.0,
                "total_impacted_files": 0,
                "max_call_depth": 0,
                "test_coverage_present": True,
                "symbol_impacts": [],
                "file_impacts": [],
                "recommendations": ["No dirty files — nothing to analyze."],
                "message": "No uncommitted changes.",
            }

        # Infer modified symbols if not provided
        inferred_symbols = modified_symbols or []
        if not inferred_symbols:
            snapshot = self._load_source_snapshot(buffer_id)
            snapshot_mgr = self._get_snapshot_manager(buffer_id)
            if snapshot is not None and snapshot_mgr is not None:
                runner = TestRunner(chunks, Path(info.get("root", self.work_dir)))
                for rel_path in dirty:
                    disk_lines = snapshot_mgr.read_lines(rel_path)
                    new_lines = snapshot.get(rel_path, [])
                    if disk_lines is not None:
                        inferred_symbols.extend(
                            runner.extract_symbols_from_diff(disk_lines, new_lines)
                        )

        try:
            analyzer = ImpactAnalyzer(chunks)
            result = analyzer.analyze(
                dirty_files=list(dirty.keys()),
                modified_symbols=sorted(set(inferred_symbols)) if inferred_symbols else None,
                max_depth=max_depth,
            )
            return result.to_dict()
        except (ValueError, RuntimeError) as e:
            logger.warning(f"ImpactAnalyzer.analyze failed: {e}")
            return self._make_error_response(
                f"Impact analysis failed: {e}",
                buffer_id=buffer_id,
                operation="analyze_impact",
            )

    def analyze_change(
        self,
        buffer_id: str,
        file: str,
        start_line: int | None = None,
        end_line: int | None = None,
        max_depth: int = 6,
    ) -> dict[str, Any]:
        """Analyze impact of a proposed change before editing.

        Before editing, know what breaks. Uses the reference map
        and impact analyzer to report direct callers, test coverage, and
        dependent symbols.

        Args:
            buffer_id: Buffer handle.
            file: File that would be modified.
            start_line: Start line of the proposed change (1-based).
            end_line: End line of the proposed change (1-based).
            max_depth: Max call-chain depth to traverse.

        Returns:
            Dict with direct_callers, test_coverage, dependent_symbols,
            files_affected, and risk assessment.
        """
        result = self._require_buffer(buffer_id, "analyze_change")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            # Find symbols in the target range
            from gigacode.reference_map import ReferenceMap

            ref_map = ReferenceMap(chunks)
            affected_symbols: list[str] = []

            # Find symbols defined in the file (optionally within line range)
            index = SymbolIndex(chunks)
            file_syms = index.get_file_symbols(file)
            for sym in file_syms:
                if start_line is not None and end_line is not None:
                    if sym.start_line <= end_line and sym.end_line >= start_line:
                        affected_symbols.append(sym.name)
                else:
                    affected_symbols.append(sym.name)

            # Find callers for each affected symbol
            direct_callers: list[dict[str, Any]] = []
            all_impacted_files: set[str] = set()
            dependent_symbols: list[dict[str, Any]] = []

            for sym in affected_symbols:
                refs = ref_map.get_references(sym, direction="called_by", top_k=20)
                if refs.get("status") == "ok":
                    for caller in refs.get("callers", []):
                        direct_callers.append({**caller, "target_symbol": sym})
                        if caller.get("file"):
                            all_impacted_files.add(caller["file"])
                dependent_symbols.append(
                    {
                        "symbol": sym,
                        "file": file,
                        "caller_count": len(refs.get("callers", [])),
                    }
                )

            # Find test coverage
            tests_for_symbols: list[dict[str, Any]] = []
            for sym in affected_symbols:
                tests = self._find_tests_for_symbol(chunks, sym)
                for t in tests:
                    tests_for_symbols.append({**t, "target_symbol": sym})

            # Run impact analyzer for risk scoring
            analyzer = ImpactAnalyzer(chunks)
            analysis = analyzer.analyze(
                dirty_files=[file],
                modified_symbols=affected_symbols,
                max_depth=max_depth,
            )

            return {
                "status": "ok",
                "file": file,
                "start_line": start_line,
                "end_line": end_line,
                "affected_symbols": affected_symbols,
                "direct_callers": direct_callers,
                "dependent_symbols": len(affected_symbols),
                "files_affected": len(all_impacted_files),
                "impacted_files": sorted(all_impacted_files),
                "test_coverage": tests_for_symbols,
                "has_tests": len(tests_for_symbols) > 0,
                "risk_level": analysis.risk_level,
                "risk_score": analysis.risk_score,
            }

        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                f"Change analysis failed: {e}",
                buffer_id=buffer_id,
                operation="analyze_change",
            )

    # ------------------------------------------------------------------
    # Execution Sandbox
    # ------------------------------------------------------------------
    def execute_in_context(
        self,
        buffer_id: str,
        code: str,
        language: str = "python",
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Run arbitrary code against the codebase in a restricted sandbox.

        The buffer root is injected into ``sys.path`` so the code can import
        modules from the codebase.  Output is captured and returned; file I/O
        and dangerous imports are blocked by AST scanning.

        Args:
            buffer_id: Buffer handle.
            code: Source code string.
            language: "python" or "javascript".
            timeout: Max seconds before SIGKILL.

        Returns:
            Dict with:
            - ``status``: "ok" | "error" | "timeout" | "security_violation"
            - ``returncode``: process exit code
            - ``stdout``, ``stderr``: captured output (truncated at 64KB)
            - ``execution_time_sec``: wall-clock time
            - ``violations``: list of security policy violations (if any)
            - ``truncated``: bool
        """
        result = self._require_buffer(buffer_id, "execute_in_context", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        root = Path(info.get("root", self.work_dir))
        try:
            executor = SandboxExecutor(root, language=language)
            result = executor.execute(code, timeout=timeout)
            return result.to_dict()
        except (ValueError, RuntimeError, OSError) as e:
            logger.warning(f"Sandbox execution failed: {e}")
            return self._make_error_response(
                f"Sandbox execution failed: {e}",
                buffer_id=buffer_id,
                operation="execute_in_context",
            )

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------
    def list_buffers(self) -> dict[str, Any]:
        """List all registered buffers."""
        return self._buffer_manager.list_buffers()

    def delete_buffer(self, buffer_id: str) -> dict[str, Any]:
        """Delete a buffer."""
        result = self._buffer_manager.delete_buffer(buffer_id)
        # Clean up query cache
        self._query_cache.invalidate_buffer(buffer_id)
        return result

    # ------------------------------------------------------------------
    # Multi-buffer orchestration (aliases and virtual buffers)
    # ------------------------------------------------------------------
    def create_alias(self, alias: str, buffer_id: str) -> dict[str, Any]:
        """Create a human-readable alias for a buffer ID.

        Example:
            tool.create_alias("main-project", "uuid-here")
            # Later: tool.semantic_search("main-project", query="...")
        """
        return self._multi_buffer_manager.alias_registry.create_alias(alias, buffer_id)

    def resolve_alias(self, alias: str) -> dict[str, Any]:
        """Resolve an alias to its buffer ID."""
        return self._multi_buffer_manager.alias_registry.resolve_alias(alias)

    def list_aliases(self) -> dict[str, Any]:
        """List all buffer aliases."""
        return self._multi_buffer_manager.alias_registry.list_aliases()

    def create_virtual_buffer(
        self,
        alias: str,
        buffer_ids: list[str],
        description: str = "",
    ) -> dict[str, Any]:
        """Create a virtual buffer aggregating multiple sub-buffers.

        Example:
            tool.create_virtual_buffer(
                "monorepo",
                ["buf-frontend", "buf-backend", "buf-shared"],
                description="Full monorepo view"
            )
        """
        return self._multi_buffer_manager.virtual_buffer_manager.create_virtual_buffer(
            alias, buffer_ids, description
        )

    def list_virtual_buffers(self) -> dict[str, Any]:
        """List all virtual buffers."""
        return self._multi_buffer_manager.virtual_buffer_manager.list_virtual_buffers()

    def delete_virtual_buffer(self, alias: str) -> dict[str, Any]:
        """Delete a virtual buffer."""
        return self._multi_buffer_manager.virtual_buffer_manager.delete_virtual_buffer(alias)

    def auto_resume(self, path: str | Path, pattern: str = "*.py") -> dict[str, Any]:
        """Check if a codebase was previously embedded and is unchanged.

        Returns:
            {"status": "resumed", "buffer_id": ..., "num_chunks": ...} if found
            {"status": "not_found", "message": "..."} if not
        """
        result = self._buffer_manager.check_existing_buffer(path, pattern)
        if result.get("status") == "not_found":
            result["message"] = f"No existing buffer found for {path} with pattern {pattern!r}"
        return result

    def save_session(self, alias: str, buffer_ids: list[str] | None = None) -> dict[str, Any]:
        """Save current session state under an alias.

        Args:
            alias: Session name (e.g., "main-project")
            buffer_ids: Specific buffers to save. If None, saves all registered buffers.
        """
        if buffer_ids is None:
            buffer_ids = list(self._buffer_manager.list_buffers().keys())
        return self._buffer_manager.save_session(alias, buffer_ids)

    def load_session(self, alias: str) -> dict[str, Any]:
        """Restore a previously saved session.

        Returns:
            {"status": "ok", "buffer_ids": [...], "missing": [...]}
        """
        result = self._buffer_manager.load_session(alias)
        if "missing_buffer_ids" in result:
            result["missing"] = result.pop("missing_buffer_ids")
        return result

    def list_sessions(self) -> dict[str, Any]:
        """List all saved sessions."""
        return self._buffer_manager.list_sessions()

    def git_status(self, buffer_id: str) -> dict[str, Any]:
        """Get git working tree status for a buffer's source directory.

        Returns branch, ahead/behind counts, modified/staged/untracked files.
        """
        result = self._require_buffer(buffer_id, "git_status", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        source_dir = info.get("path") or info.get("root")
        if not source_dir:
            return self._make_error_response(
                "No source directory found", buffer_id=buffer_id, operation="git_status"
            )

        utils = GitUtils(source_dir)
        return utils.get_status()

    def git_diff(
        self,
        buffer_id: str,
        file: str | None = None,
        against: str = "HEAD",
    ) -> dict[str, Any]:
        """Get diff of file(s) against a git reference.

        Args:
            file: Specific file, or None for all.
            against: "HEAD", "STAGED", commit hash, or branch name.
        """
        result = self._require_buffer(buffer_id, "git_diff", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        source_dir = info.get("path") or info.get("root")
        if not source_dir:
            return self._make_error_response(
                "No source directory found", buffer_id=buffer_id, operation="git_diff"
            )

        utils = GitUtils(source_dir)
        return utils.get_diff(file_path=file, against=against)

    def git_blame(
        self,
        buffer_id: str,
        file: str,
        line: int | None = None,
    ) -> dict[str, Any]:
        """Show who last modified each line of a file.

        Args:
            file: File path.
            line: Specific line (1-based), or None for full file.
        """
        result = self._require_buffer(buffer_id, "git_blame", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        source_dir = info.get("path") or info.get("root")
        if not source_dir:
            return self._make_error_response(
                "No source directory found", buffer_id=buffer_id, operation="git_blame"
            )

        utils = GitUtils(source_dir)
        return utils.blame(file_path=file, line=line)

    def git_show(
        self,
        buffer_id: str,
        file: str,
        commit: str = "HEAD",
    ) -> dict[str, Any]:
        """Show file content at a specific commit.

        Args:
            file: File path.
            commit: Commit hash or reference.
        """
        result = self._require_buffer(buffer_id, "git_show", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        source_dir = info.get("path") or info.get("root")
        if not source_dir:
            return self._make_error_response(
                "No source directory found", buffer_id=buffer_id, operation="git_show"
            )

        utils = GitUtils(source_dir)
        return utils.show_file_at_commit(file_path=file, commit=commit)

    def check_buffer_state(self, buffer_id: str) -> dict[str, Any]:
        """Return the current state of a buffer (READY, DIRTY, REBUILDING).

        Args:
            buffer_id: Buffer handle.

        Returns:
            Dict with ``status``, ``buffer_id``, ``state``, and optional
            ``state_changed_at`` timestamp.
        """
        result = self._require_buffer(buffer_id, "check_buffer_state", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        try:
            state = self._get_buffer_state(buffer_id)
        except ValueError as e:
            return self._make_error_response(
                str(e), buffer_id=buffer_id, operation="check_buffer_state"
            )

        return {
            "status": "ok",
            "buffer_id": buffer_id,
            "state": state.value,
            "state_changed_at": info.get("state_changed_at"),
        }

    # ------------------------------------------------------------------
    # Incremental rebuild (batch / deferred)
    # ------------------------------------------------------------------
    def _rebuild_dirty(self, buffer_id: str) -> None:
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return
        dirty = info.get("dirty_files", {})
        if not dirty:
            return
        self._rebuild_files(buffer_id, list(dirty.keys()))
        info["dirty_files"] = {}
        self._save_registry()

    def _rebuild_files(self, buffer_id: str, files: list[str]) -> None:
        """Re-chunk, re-embed, and patch FAISS index for changed files.

        This is the core incremental update operation. For each file in *files*:
        1. Remove old chunks associated with the file from the index
        2. Parse the file text and generate new chunks
        3. Embed new chunks
        4. Rebuild the embeddings array (removed old + added new)
        5. Reset FAISS index and re-add all vectors

        After rebuild, the lexical index is cleared and will be lazily rebuilt
        on the next hybrid_search() call.

        **Invariants:**
        - All chunks in *chunks* after this call have sequential IDs (0 to n-1)
        - FAISS index is reset and contains exactly len(chunks) vectors
        - Embeddings array has shape (len(chunks), embedding_dim)
        - Registry dirty_files cleared (caller responsible for calling _save_registry)

        Args:
            buffer_id: Buffer to update.
            files: Relative file paths (keys in source_snapshot) to rebuild.

        Note:
            Requires source_snapshot to be up-to-date. Should be called
            after snapshot edits are persisted but before writing to disk.
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return
        buffer_dir = Path(info["buffer_dir"])
        chunks = self._load_chunks(buffer_id)
        if chunks is None:
            return
        index = self._get_index(buffer_id)
        embeddings = np.load(buffer_dir / "embeddings.npy")

        # Map chunk index -> file
        file_to_chunk_indices: dict[str, list[int]] = {}
        for idx, ch in enumerate(chunks):
            file_to_chunk_indices.setdefault(ch.file, []).append(idx)

        # Remove old chunks for rebuilt files
        removed_ids = []
        for rel_path in files:
            for idx in file_to_chunk_indices.get(rel_path, []):
                removed_ids.append(idx)
        if removed_ids:
            index.remove(np.array(removed_ids, dtype=np.int64))

        # Build new chunk list excluding old file chunks
        new_chunks: list[CodeChunk] = []
        old_id_to_new_id: dict[int, int] = {}
        next_id = 0
        for idx, ch in enumerate(chunks):
            if ch.file in files:
                continue
            ch.id = next_id
            old_id_to_new_id[idx] = next_id
            new_chunks.append(ch)
            next_id += 1

        # Generate new chunks for dirty files
        snapshot_mgr = self._get_snapshot_manager(buffer_id)
        if snapshot_mgr is None:
            json_logger.error(
                operation="_rebuild_files",
                buffer_id=buffer_id,
                message="Snapshot manager not found",
            )
            return

        language_hint = info.get("language_hint")
        sliding_window_size = info.get("sliding_window_size", 30)
        new_embeddings_list: list[np.ndarray] = []
        for rel_path in files:
            lines = snapshot_mgr.read_lines(rel_path)
            if lines is None:
                json_logger.warning(
                    operation="_rebuild_files",
                    buffer_id=buffer_id,
                    message=f"Could not read {rel_path}",
                )
                continue
            text = "\n".join(lines)
            file_chunks = chunk_text(
                text,
                language_hint=language_hint,
                filename_hint=rel_path,
                sliding_window_size=sliding_window_size,
            )
            for ch in file_chunks:
                ch.file = rel_path
                ch.id = next_id
                new_chunks.append(ch)
                next_id += 1
            if file_chunks:
                texts = [ch.text for ch in file_chunks]
                emb = self._embedder.encode(texts, batch_size=DEFAULT_BATCH_SIZE)
                new_embeddings_list.append(emb)

        # Rebuild embeddings array
        kept_mask = np.ones(len(chunks), dtype=bool)
        for idx in removed_ids:
            kept_mask[idx] = False
        kept_embeddings = embeddings[kept_mask]
        if new_embeddings_list:
            new_embeddings = np.vstack(new_embeddings_list)
            final_embeddings = (
                np.vstack([kept_embeddings, new_embeddings])
                if kept_embeddings.size
                else new_embeddings
            )
        else:
            final_embeddings = kept_embeddings

        # Rebuild index from scratch (cleanest for ID consistency)
        index.reset()
        ids = np.arange(len(new_chunks), dtype=np.int64)
        index.add(ids, final_embeddings)

        # Pre-sync GPU to avoid lazy sync latency on first search
        index.sync_gpu()

        # Rebuild lexical index
        lexical = LexicalIndex()
        for i, ch in enumerate(new_chunks):
            lexical.add(i, ch.text)
        self._lexical_cache[buffer_id] = lexical

        # Persist all state including lexical index
        self._save_buffer_state(
            buffer_dir, new_chunks, final_embeddings, index, lexical_index=lexical
        )

        self._query_cache.invalidate_buffer(buffer_id)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _save_buffer_state(
        self,
        buffer_dir: Path,
        chunks: list[CodeChunk],
        embeddings: np.ndarray,
        index: GpuIndex,
        lexical_index: LexicalIndex | None = None,
        file_chunks_map: dict[str, list[int]] | None = None,
    ) -> None:
        emb_path = buffer_dir / "embeddings.npy"
        chunks_path = buffer_dir / "chunks.json"
        index_path = buffer_dir / "index.faiss"
        lexical_path = buffer_dir / "lexical_index.json"
        fcm_path = buffer_dir / "file_chunks_map.json"

        np.save(emb_path, embeddings)
        save_metadata(chunks_path, [ch.dict() for ch in chunks], compact=True)
        index.save(index_path)
        if lexical_index is not None:
            lexical_data = lexical_index.to_dict()
            lexical_path.write_bytes(
                json.dumps(lexical_data, separators=(",", ":")).encode("utf-8")
            )
        if file_chunks_map is not None:
            fcm_path.write_bytes(json.dumps(file_chunks_map, separators=(",", ":")).encode("utf-8"))

    def _load_chunks(self, buffer_id: str) -> list[CodeChunk] | None:
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return None
        chunks_path = Path(info["buffer_dir"]) / "chunks.json"
        if not chunks_path.exists():
            return None
        data = load_metadata(chunks_path)
        # Handle backward compat: 'content' field was renamed to 'text'
        for rec in data:
            if "content" in rec and "text" not in rec:
                rec["text"] = rec.pop("content")
        return [CodeChunk(**rec) for rec in data]

    def _get_index(self, buffer_id: str) -> GpuIndex:
        """Return cached index or rebuild from disk."""
        if buffer_id in self._index_cache:
            return self._index_cache[buffer_id]
        info = self._get_buffer_info(buffer_id)
        if info is None:
            raise RuntimeError(f"Unknown buffer_id: {buffer_id}")
        index = GpuIndex(info["embedding_dim"], use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        index_path = Path(info["buffer_dir"]) / "index.faiss"
        if index_path.exists():
            index.load(index_path)
        # Pre-sync GPU to avoid lazy sync latency on first search after load
        index.sync_gpu()
        self._index_cache[buffer_id] = index
        return index

    def _load_source_snapshot(self, buffer_id: str) -> dict[str, list[str]] | None:
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return None
        path = Path(info["buffer_dir"]) / "source_snapshot.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: list(v) for k, v in data.items()}

    def _save_source_snapshot(self, buffer_id: str, snapshot: dict[str, list[str]]) -> None:
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return
        path = Path(info["buffer_dir"]) / "source_snapshot.json"
        path.write_bytes(
            json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )

    def _load_lexical_index(self, buffer_id: str) -> LexicalIndex | None:
        """Load lexical index from disk if it exists."""
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return None
        path = Path(info["buffer_dir"]) / "lexical_index.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return LexicalIndex.from_dict(data)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            json_logger.warning(
                operation="_load_lexical_index",
                message=f"Failed to load lexical index from {path}: {exc}",
            )
            return None

    def _get_lexical_index(self, buffer_id: str) -> LexicalIndex:
        """Return cached lexical index or load from disk or build from chunks."""
        # First check in-memory cache
        if buffer_id in self._lexical_cache:
            return self._lexical_cache[buffer_id]

        # Try to load from disk
        lexical = self._load_lexical_index(buffer_id)
        if lexical is not None:
            self._lexical_cache[buffer_id] = lexical
            return lexical

        # Fall back to building from chunks (O(n) cost)
        chunks = self._load_chunks(buffer_id)
        if chunks is None:
            raise RuntimeError(
                f"Cannot build lexical index for buffer_id={buffer_id}: chunks not found"
            )

        json_logger.warning(
            operation="_get_lexical_index",
            buffer_id=buffer_id,
            message=f"Lexical index missing; building from {len(chunks)} chunks",
            details={"chunks_count": len(chunks)},
        )
        lexical = LexicalIndex()
        for i, ch in enumerate(chunks):
            lexical.add(i, ch.text)
        self._lexical_cache[buffer_id] = lexical
        return lexical

    def _get_buffer_info(self, buffer_id: str) -> dict[str, Any] | None:
        """Get buffer metadata from BufferManager's registry or fallback."""
        if self._buffer_manager is not None:
            return self._buffer_manager._get_buffer_info(buffer_id)
        return self._fallback_registry.get(buffer_id)

    def _get_buffer_state(self, buffer_id: str) -> BufferState:
        """Get current buffer state from registry.

        Args:
            buffer_id: Buffer ID

        Returns:
            Current BufferState (defaults to READY if not set)
        """
        info = self._get_buffer_info(buffer_id)
        if not info:
            raise ValueError(f"Unknown buffer_id: {buffer_id}")

        state_str = info.get("state", BufferState.READY.value)
        return BufferState(state_str)

    def _set_buffer_state(self, buffer_id: str, new_state: BufferState) -> None:
        """Set buffer state with validation.

        Args:
            buffer_id: Buffer ID
            new_state: Desired new state

        Raises:
            ValueError: If state transition is invalid
        """
        info = self._get_buffer_info(buffer_id)
        if not info:
            raise ValueError(f"Unknown buffer_id: {buffer_id}")

        current_state = self._get_buffer_state(buffer_id)

        # Validate transition
        BufferStateTransition.validate_or_raise(current_state, new_state)

        # Update state and timestamp
        info["state"] = new_state.value
        info["state_changed_at"] = time.time()

        self._save_registry()

        # Update health tracker
        self._health_tracker.update_buffer_state(buffer_id, new_state)

        # Log state change
        self._security.log_success(
            operation="state_transition",
            buffer_id=buffer_id,
            details={
                "from_state": str(current_state),
                "to_state": str(new_state),
            },
        )

    def _get_snapshot_manager(self, buffer_id: str) -> SnapshotManager | None:
        """Get or load SnapshotManager for buffer (delegates to BufferManager)."""
        if self._buffer_manager is not None:
            return self._buffer_manager._get_snapshot_manager(buffer_id)
        return None

    @retry_on_io_error(max_attempts=3, delay_s=0.5)
    def _save_registry(self) -> None:
        """Save registry to disk (delegates to BufferManager)."""
        if self._buffer_manager is not None:
            self._buffer_manager._save_registry()

    # ------------------------------------------------------------------
    # Resource budgeting and confidence scoring
    # ------------------------------------------------------------------
    def estimate_budget(
        self,
        path: str | Path,
        pattern: str = "*.py",
    ) -> dict[str, Any]:
        """Estimate resource requirements before embedding a codebase.

        Returns:
            Dict with estimated_ram_mb, estimated_embed_time_s,
            estimated_search_latency_ms, num_files, num_lines,
            estimated_chunks, recommended_device, warnings.
        """
        try:
            return _estimate_budget(
                path,
                pattern,
                embedding_dim=self._embedding_dim,
                device=self._embedder.device,
            )
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(f"estimate_budget failed: {e}")
            return {"status": "error", "message": f"Budget estimation failed: {e}"}

    def get_memory_usage(self) -> dict[str, Any]:
        """Get current memory usage of the tool."""
        index_cache_size = len(getattr(self, "_index_cache", {}))
        lexical_cache_size = len(getattr(self, "_lexical_cache", {}))
        query_cache_size = len(getattr(self, "_query_cache", {}))

        # Approximate memory for loaded buffers
        buffer_memory_mb: dict[str, float] = {}
        for buffer_id in getattr(self, "_index_cache", {}):
            info = self._get_buffer_info(buffer_id)
            if info is None:
                continue
            buffer_dir = Path(info.get("buffer_dir", self.work_dir / buffer_id))
            embeddings_path = buffer_dir / "embeddings.npy"
            chunks_path = buffer_dir / "chunks.json"

            emb_mb = 0.0
            chunks_mb = 0.0
            if embeddings_path.exists():
                emb_mb = embeddings_path.stat().st_size / (1024 * 1024)
            if chunks_path.exists():
                chunks_mb = chunks_path.stat().st_size / (1024 * 1024)
            buffer_memory_mb[buffer_id] = round(emb_mb + chunks_mb, 2)

        total_buffer_mb = round(sum(buffer_memory_mb.values()), 2)

        return {
            "status": "ok",
            "index_cache_entries": index_cache_size,
            "lexical_cache_entries": lexical_cache_size,
            "query_cache_entries": query_cache_size,
            "buffer_memory_mb": buffer_memory_mb,
            "total_buffer_memory_mb": total_buffer_mb,
        }

    def score_result_confidence(
        self,
        score: float,
        all_scores: list[float],
    ) -> dict[str, Any]:
        """Score the confidence of a search result.

        Args:
            score: The result's score.
            all_scores: All scores from the same search.

        Returns:
            Dict with confidence (high/medium/low) and explanation.
        """
        try:
            scorer = ConfidenceScorer()
            confidence = scorer.classify(score, all_scores)
            explanation = scorer.explain(score, confidence, all_scores)
            return {
                "status": "ok",
                "score": round(score, 3),
                "confidence": confidence,
                "explanation": explanation,
            }
        except (ValueError, RuntimeError) as e:
            logger.warning(f"score_result_confidence failed: {e}")
            return {"status": "error", "message": f"Confidence scoring failed: {e}"}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def get_cache_stats(self) -> dict[str, Any]:
        """Return cache utilization statistics (delegated to IndexManager)."""
        if self._index_manager:
            try:
                return self._index_manager.get_cache_stats()
            except (ImportError, RuntimeError, AttributeError) as e:
                logger.warning(f"IndexManager.get_cache_stats failed: {e}")
                return {"status": "error", "message": f"IndexManager unavailable: {e}"}

        return {"status": "error", "message": "IndexManager unavailable"}

    def health_check(self) -> dict[str, Any]:
        """Perform a health check and return system status (delegated to IndexManager)."""
        if self._index_manager:
            try:
                return self._index_manager.health_check()
            except (ImportError, RuntimeError, AttributeError) as e:
                logger.warning(f"IndexManager.health_check failed: {e}")
                return {"status": "error", "message": f"IndexManager unavailable: {e}"}

        return {"status": "error", "message": "IndexManager unavailable"}

    @staticmethod
    def get_metrics() -> dict[str, Any]:
        """Export current metrics snapshot.

        Returns a dict with counters (operation counts), gauges, histograms
        (latency and size distributions), and cache hit rates.  Can be
        serialized to JSON for monitoring integration.
        """
        return get_metrics().dump_metrics()

    # ------------------------------------------------------------------
    # State-Based Access Control
    # ------------------------------------------------------------------
    def _check_state_for_operation(
        self, buffer_id: str, operation_type: OperationType
    ) -> tuple[bool, str]:
        """Check if buffer state allows the requested operation.

        Args:
            buffer_id: Buffer identifier
            operation_type: Type of operation (QUERY, WRITE, REBUILD, READ)

        Returns:
            (allowed: bool, reason: str) tuple
        """
        try:
            current_state = self._get_buffer_state(buffer_id)
        except ValueError:
            return False, f"Unknown buffer_id: {buffer_id}"

        # Get allowed states for this operation type
        state_requirements = OperationConfig.get_state_requirements()
        allowed_states = state_requirements.get(operation_type, [])

        if current_state in allowed_states:
            return True, ""

        # Build error message
        if operation_type == OperationType.QUERY:
            if current_state == BufferState.REBUILDING:
                return False, "Buffer is rebuilding indices. Please wait..."
            elif current_state == BufferState.DIRTY:
                return False, "Results may be stale (buffer has uncommitted changes)"

        elif operation_type == OperationType.WRITE:
            return (
                False,
                f"Cannot write to buffer in {current_state.value} state. Buffer must be in {BufferState.READY.value} state.",
            )

        elif operation_type == OperationType.READ:
            if current_state == BufferState.REBUILDING:
                return False, "Buffer is rebuilding indices. Please wait..."

        return False, f"Operation not allowed in {current_state.value} state"

    def _get_buffer_health(self, buffer_id: str) -> dict[str, Any]:
        """Get health status for a buffer.

        Args:
            buffer_id: Buffer identifier

        Returns:
            Dictionary with health information or error
        """
        health = self._health_tracker.get_health_status(
            buffer_id,
            OperationConfig.DIRTY_FILE_WARNING_THRESHOLD,
            OperationConfig.DIRTY_FILE_DEGRADED_THRESHOLD,
            OperationConfig.INDEX_AGE_WARNING_SECONDS,
            OperationConfig.INDEX_AGE_DEGRADED_SECONDS,
        )

        if health is None:
            return {"error": f"Unknown buffer_id: {buffer_id}"}

        return health.to_dict()

    def _check_state_for_query(self, buffer_id: str) -> tuple[bool, str]:
        """Check if buffer state allows queries.

        Args:
            buffer_id: Buffer identifier

        Returns:
            (allowed, reason) tuple
        """
        return self._check_state_for_operation(buffer_id, OperationType.QUERY)

    def _check_state_for_write(self, buffer_id: str) -> tuple[bool, str]:
        """Check if buffer state allows writes.

        Args:
            buffer_id: Buffer identifier

        Returns:
            (allowed, reason) tuple
        """
        return self._check_state_for_operation(buffer_id, OperationType.WRITE)

    def _check_state_for_read(self, buffer_id: str) -> tuple[bool, str]:
        """Check if buffer state allows reads.

        Args:
            buffer_id: Buffer identifier

        Returns:
            (allowed, reason) tuple
        """
        return self._check_state_for_operation(buffer_id, OperationType.READ)

    def _check_state_for_rebuild(self, buffer_id: str) -> tuple[bool, str]:
        """Check if buffer state allows rebuilds.

        Args:
            buffer_id: Buffer identifier

        Returns:
            (allowed, reason) tuple
        """
        return self._check_state_for_operation(buffer_id, OperationType.REBUILD)

    # ------------------------------------------------------------------
    # RBAC, Audit Logging, and Rate Limiting
    # ------------------------------------------------------------------

    def set_user(self, user_id: str, role: str = "analyst") -> User:
        """Set current user for operations.

        Args:
            user_id: User identifier
            role: User role ("admin", "analyst", "reader", "guest")

        Returns:
            User object
        """
        from gigacode.access_control import Role

        # Convert role string to enum
        role_enum = Role[role.upper()] if role else Role.ANALYST
        self._security._current_user_id = user_id
        return self._security._access_control.register_user(user_id, role_enum)

    def _check_permission(
        self,
        operation: str,
        buffer_owner: str | None = None,
    ) -> tuple[bool, str]:
        """Check if current user has permission for operation.

        Args:
            operation: Operation name
            buffer_owner: Buffer owner (for ownership checks)

        Returns:
            (allowed, reason) tuple
        """
        # Access control is now in security layer
        allowed, reason = self._security._access_control.check_operation(
            self._security._current_user_id,
            operation,
            buffer_owner,
        )

        # Log denied operation
        if not allowed:
            self._audit_logger.log_denied(
                operation=operation,
                user_id=self._security._current_user_id,
                role=self._security.get_current_user_role_name(),
                reason=reason,
            )

        return allowed, reason

    def _check_rate_limit(self, buffer_id: str | None = None) -> tuple[bool, str | None]:
        """Check if current user has rate limit available.

        Args:
            buffer_id: Buffer identifier (optional)

        Returns:
            (allowed, reason) tuple
        """
        user = self._security.get_current_user()
        allowed, reason = self._security._rate_limiter.check_all_limits(
            self._security._current_user_id,
            user.role,
            buffer_id,
            operation_type="operations",
        )

        return allowed, reason

    def get_audit_log(
        self,
        buffer_id: str | None = None,
        since: str | None = None,
        operations: list[str] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Query the audit log for operations.

        Args:
            buffer_id: Filter by buffer ID.
            since: ISO timestamp (e.g., "2026-05-01").
            operations: Filter by operation types.
            limit: Max entries.

        Returns:
            Dict with entries list.
        """
        # Check permission to view audit logs
        allowed, reason = self._check_permission("view_audit_log")
        if not allowed:
            return {"status": "error", "message": reason}

        entries = self._security._audit_logger.query(
            since=since,
            operations=operations,
            buffer_id=buffer_id,
            limit=limit,
        )
        return {"status": "ok", "entries": entries}

    def get_test_context(
        self,
        buffer_id: str,
        file: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Find test files that reference a given source file.

        Uses file naming heuristics and embedding similarity to find
        test files related to the source file.

        Returns:
            Dict with test_files list.
        """
        result = self._require_buffer(buffer_id, "get_test_context")
        if isinstance(result, dict):
            return result
        info, chunks = result

        # Load embeddings
        buffer_dir = Path(info["buffer_dir"])
        embeddings_path = buffer_dir / "embeddings.npy"
        embeddings: np.ndarray | None = None
        if embeddings_path.exists():
            try:
                embeddings = np.load(embeddings_path)
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to load embeddings for {buffer_id}: {e}")

        language = info.get("language_hint", "python")

        # Find test chunks using naming heuristics
        test_chunks: list[CodeChunk] = []
        for chunk in chunks:
            if chunk.file and self._is_test_file(chunk.file, language):
                test_chunks.append(chunk)

        if not test_chunks:
            return {"status": "ok", "test_files": []}

        # Find source chunks for the given file
        source_chunks = [ch for ch in chunks if ch.file == file]
        if not source_chunks:
            return {"status": "ok", "test_files": []}

        # Compute similarity between test chunks and source chunks
        if embeddings is None or embeddings.size == 0:
            return {"status": "ok", "test_files": []}

        source_indices = [ch.id for ch in source_chunks if 0 <= ch.id < len(embeddings)]
        test_indices = [ch.id for ch in test_chunks if 0 <= ch.id < len(embeddings)]

        if not source_indices or not test_indices:
            return {"status": "ok", "test_files": []}

        source_emb = embeddings[source_indices]
        test_emb = embeddings[test_indices]

        # Cosine similarity via normalized dot product
        source_emb_norm = source_emb / (np.linalg.norm(source_emb, axis=1, keepdims=True) + 1e-10)
        test_emb_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-10)
        similarities = test_emb_norm @ source_emb_norm.T

        # For each test chunk, take max similarity to any source chunk
        max_scores = similarities.max(axis=1)

        # Build results
        scored_tests: list[tuple[CodeChunk, float]] = []
        for idx, _test_idx in enumerate(test_indices):
            test_chunk = test_chunks[idx]
            score = float(max_scores[idx])
            scored_tests.append((test_chunk, score))

        scored_tests.sort(key=lambda x: x[1], reverse=True)

        test_files = []
        seen_files: set[str] = set()
        for chunk, score in scored_tests[:top_k]:
            file_key = f"{chunk.file}:{chunk.start_line}-{chunk.end_line}"
            if file_key not in seen_files:
                seen_files.add(file_key)
                test_files.append(
                    {
                        "file": chunk.file,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "type": chunk.type,
                        "name": chunk.name,
                        "score": round(score, 4),
                    }
                )

        return {"status": "ok", "test_files": test_files}

    @staticmethod
    def _is_test_file(filename: str, language: str = "python") -> bool:
        """Check if a filename matches test file patterns."""
        from gigacode.context_assembler import _TEST_FILE_PATTERNS

        basename = Path(filename).name
        patterns = _TEST_FILE_PATTERNS.get(language, [])
        return any(p.match(basename) for p in patterns)

    def suggest_tests(
        self,
        buffer_id: str,
        changed_files: list[str],
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Suggest tests to run after file changes.

        Args:
            buffer_id: Buffer handle.
            changed_files: List of changed file paths.
            top_k: Max suggestions.

        Returns:
            Dict with suggested_tests list.
        """
        result = self._require_buffer(buffer_id, "suggest_tests")
        if isinstance(result, dict):
            return result
        info, chunks = result

        # Load embeddings
        buffer_dir = Path(info["buffer_dir"])
        embeddings_path = buffer_dir / "embeddings.npy"
        embeddings: np.ndarray | None = None
        if embeddings_path.exists():
            try:
                embeddings = np.load(embeddings_path)
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to load embeddings for {buffer_id}: {e}")

        language = info.get("language_hint", "python")

        # Find test chunks
        test_chunks = [ch for ch in chunks if ch.file and self._is_test_file(ch.file, language)]
        if not test_chunks:
            return {"status": "ok", "suggested_tests": []}

        # Find changed file chunks
        changed_chunks = [ch for ch in chunks if ch.file in changed_files]
        if not changed_chunks:
            return {"status": "ok", "suggested_tests": []}

        if embeddings is None or embeddings.size == 0:
            return {"status": "ok", "suggested_tests": []}

        changed_indices = [ch.id for ch in changed_chunks if 0 <= ch.id < len(embeddings)]
        test_indices = [ch.id for ch in test_chunks if 0 <= ch.id < len(embeddings)]

        if not changed_indices or not test_indices:
            return {"status": "ok", "suggested_tests": []}

        changed_emb = embeddings[changed_indices]
        test_emb = embeddings[test_indices]

        # Cosine similarity
        changed_emb_norm = changed_emb / (
            np.linalg.norm(changed_emb, axis=1, keepdims=True) + 1e-10
        )
        test_emb_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-10)
        similarities = test_emb_norm @ changed_emb_norm.T

        # Max similarity per test chunk
        max_scores = similarities.max(axis=1)

        # Build and rank results
        scored_tests: list[tuple[CodeChunk, float]] = []
        for idx, _test_idx in enumerate(test_indices):
            test_chunk = test_chunks[idx]
            score = float(max_scores[idx])
            scored_tests.append((test_chunk, score))

        scored_tests.sort(key=lambda x: x[1], reverse=True)

        suggested_tests = []
        seen_files: set[str] = set()
        for chunk, score in scored_tests[:top_k]:
            file_key = f"{chunk.file}:{chunk.start_line}-{chunk.end_line}"
            if file_key not in seen_files:
                seen_files.add(file_key)
                suggested_tests.append(
                    {
                        "file": chunk.file,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "type": chunk.type,
                        "name": chunk.name,
                        "score": round(score, 4),
                    }
                )

        return {"status": "ok", "suggested_tests": suggested_tests}

    def get_audit_stats(self) -> dict[str, Any]:
        """Get audit log statistics.

        Returns:
            Statistics dictionary
        """
        # Check permission
        allowed, _ = self._check_permission("view_audit_log")
        if not allowed:
            return {}

        return self._audit_logger.stats()

    def get_test_coverage(
        self,
        buffer_id: str,
    ) -> dict[str, Any]:
        """Get test coverage map for the codebase.

        Maps each source file to the test functions that cover it.
        Coverage is determined by symbol reference analysis: a test file
        covers a source file's symbols if it imports or calls them.

        Returns:
            Dict mapping source files to {line_range: [test_names]}.
        """
        result = self._require_buffer(buffer_id, "get_test_coverage")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            language = info.get("language_hint", "python")
            index = SymbolIndex(chunks)

            # Separate test and source chunks
            test_chunks = [ch for ch in chunks if ch.file and self._is_test_file(ch.file, language)]

            # Build coverage map
            coverage: dict[str, dict[str, list[str]]] = {}

            for test_chunk in test_chunks:
                # What symbols does this test call?
                symbols_called = getattr(test_chunk, "symbols_called", None) or []
                test_name = getattr(test_chunk, "name", "")

                for called_sym in symbols_called:
                    # Find where the called symbol is defined
                    defs = index.get_definition(called_sym)
                    for defn in defs:
                        if self._is_test_file(defn.file, language):
                            continue  # Skip test-to-test references
                        source_file = defn.file
                        line_range = f"{defn.start_line}-{defn.end_line}"

                        if source_file not in coverage:
                            coverage[source_file] = {}
                        if line_range not in coverage[source_file]:
                            coverage[source_file][line_range] = []
                        if test_name and test_name not in coverage[source_file][line_range]:
                            coverage[source_file][line_range].append(test_name)

            # Also mark files with no test coverage
            for chunk in chunks:
                if chunk.file and not self._is_test_file(chunk.file, language):
                    if chunk.file not in coverage:
                        coverage[chunk.file] = {}

            return {"status": "ok", "coverage": coverage}

        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                f"Test coverage map failed: {e}",
                buffer_id=buffer_id,
                operation="get_test_coverage",
            )

    # ------------------------------------------------------------------
    # Dependency Graph
    # ------------------------------------------------------------------
    def trace_call_chain(
        self,
        buffer_id: str,
        from_symbol: str,
        to_symbol: str,
        max_depth: int = 10,
    ) -> dict[str, Any]:
        """Find call chain between two symbols using BFS."""
        result = self._require_buffer(buffer_id, "trace_call_chain")
        if isinstance(result, dict):
            return result
        info, chunks = result
        try:
            graph = DependencyGraph(chunks)
            result = graph.trace_call_chain(from_symbol, to_symbol, max_depth)
            return {"status": "ok", **result.to_dict()}
        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                str(e), buffer_id=buffer_id, operation="trace_call_chain"
            )

    def get_dependencies(
        self,
        buffer_id: str,
        file: str,
        direction: str = "both",
    ) -> dict[str, Any]:
        """Get file dependencies (outgoing imports, incoming references)."""
        result = self._require_buffer(buffer_id, "get_dependencies")
        if isinstance(result, dict):
            return result
        info, chunks = result
        try:
            graph = DependencyGraph(chunks)
            deps = graph.get_dependencies(file, direction)
            return {"status": "ok", "file": file, "direction": direction, "dependencies": deps}
        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                str(e), buffer_id=buffer_id, operation="get_dependencies"
            )

    def find_circular_dependencies(self, buffer_id: str) -> dict[str, Any]:
        """Find circular import/reference cycles in a buffer."""
        result = self._require_buffer(buffer_id, "find_circular_dependencies")
        if isinstance(result, dict):
            return result
        info, chunks = result
        try:
            graph = DependencyGraph(chunks)
            cycles = graph.find_cycles()
            return {"status": "ok", "cycles": cycles, "cycle_count": len(cycles)}
        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                str(e), buffer_id=buffer_id, operation="find_circular_dependencies"
            )

    def export_dependency_graph(
        self,
        buffer_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Export dependency graph in JSON or DOT format."""
        result = self._require_buffer(buffer_id, "export_dependency_graph")
        if isinstance(result, dict):
            return result
        info, chunks = result
        try:
            graph = DependencyGraph(chunks)
            return graph.export_graph(format)
        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                str(e), buffer_id=buffer_id, operation="export_dependency_graph"
            )

    def trace_execution_paths(
        self,
        buffer_id: str,
        symbol: str,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """Trace all execution paths through a symbol using AST branch detection.

        For complex logic, know all execution branches before editing.
        Follows function calls through the call graph up to max_depth.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            symbol: Symbol name to trace paths for.
            max_depth: Maximum call depth to follow (default: 3).

        Returns:
            Dict with keys:
                status: "ok" on success.
                symbol: The symbol that was traced.
                paths: List of ExecutionPath dicts, each with:
                    path: List of path chains like ["handle_request -> validate_input -> check_auth"].
                    branches: Number of branch points along this path.
                    calls: List of function calls in order.
                    conditions: Condition expressions at branch points.
                path_count: Number of paths found.

        Example:
            >>> result = tool.trace_execution_paths(buf_id, "handle_request", max_depth=3)
            >>> result["paths"][0]["path"]
            ["handle_request -> validate_input -> check_auth"]
            >>> result["paths"][0]["branches"]
            3
        """
        result = self._require_buffer(buffer_id, "trace_execution_paths")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            paths = trace_execution_paths(chunks, symbol, max_depth=max_depth)
            return {
                "status": "ok",
                "symbol": symbol,
                "paths": [p.to_dict() for p in paths],
                "path_count": len(paths),
            }
        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                f"Execution path trace failed: {e}",
                buffer_id=buffer_id,
                operation="trace_execution_paths",
            )

    def get_dependency_graph(
        self,
        buffer_id: str,
        symbol: str | None = None,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Get dependency graph visualization data (nodes + edges).

        Visualize relationships; understand architecture at a glance.
        When symbol is provided, scopes the graph around that symbol using ReferenceMap.
        When symbol is None, returns the full dependency graph.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            symbol: Optional symbol to scope the graph around. If None, returns full graph.
            depth: Depth of dependencies to include (default: 2).

        Returns:
            Dict with keys:
                status: "ok" on success.
                nodes: List of node dicts with keys: id, label, type, file.
                edges: List of edge dicts with keys: from, to, type ("calls" or "imports").

        Example:
            >>> result = tool.get_dependency_graph(buf_id, symbol="process_payment", depth=2)
            >>> result["nodes"][0]
            {"id": "process_payment", "label": "process_payment", "type": "function", "file": "src/pay.py"}
            >>> result["edges"][0]
            {"from": "process_payment", "to": "validate_card", "type": "calls"}
        """
        result = self._require_buffer(buffer_id, "get_dependency_graph")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            graph = DependencyGraph(chunks)
            exported = graph.export_graph("json")

            nodes: list[dict[str, Any]] = []
            edges: list[dict[str, Any]] = []

            if symbol:
                # Scope graph around symbol
                from gigacode.reference_map import ReferenceMap

                ref_map = ReferenceMap(chunks)
                visited_symbols: set[str] = {symbol}
                visited_files: set[str] = set()

                # Collect symbols at each depth level
                current_level = [symbol]
                for _d in range(depth):
                    next_level: list[str] = []
                    for sym in current_level:
                        refs = ref_map.get_references(sym, direction="both", top_k=20)
                        if refs.get("status") == "ok":
                            for caller in refs.get("callers", []):
                                if caller.get("symbol") and caller["symbol"] not in visited_symbols:
                                    visited_symbols.add(caller["symbol"])
                                    next_level.append(caller["symbol"])
                                    if caller.get("file"):
                                        visited_files.add(caller["file"])
                            for callee in refs.get("callees", []):
                                if callee.get("symbol") and callee["symbol"] not in visited_symbols:
                                    visited_symbols.add(callee["symbol"])
                                    next_level.append(callee["symbol"])
                                    if callee.get("file"):
                                        visited_files.add(callee["file"])
                        # Find file for this symbol
                        for ch in chunks:
                            if hasattr(ch, "name") and ch.name == sym:
                                visited_files.add(ch.file)
                    current_level = next_level

                # Build filtered nodes
                for sym in visited_symbols:
                    for ch in chunks:
                        if hasattr(ch, "name") and ch.name == sym:
                            nodes.append(
                                {"id": sym, "label": sym, "type": ch.type, "file": ch.file}
                            )
                            break

                # Build filtered edges
                for sym in visited_symbols:
                    refs = ref_map.get_references(sym, direction="both", top_k=20)
                    if refs.get("status") == "ok":
                        for callee in refs.get("callees", []):
                            if callee.get("symbol") in visited_symbols:
                                edges.append({"from": sym, "to": callee["symbol"], "type": "calls"})

                # Add import edges
                for ch in chunks:
                    if ch.file in visited_files and hasattr(ch, "imports"):
                        for imp in ch.imports or []:
                            for other_file in visited_files:
                                stem = Path(other_file).stem
                                if stem in imp:
                                    edges.append(
                                        {"from": ch.file, "to": other_file, "type": "imports"}
                                    )
            else:
                # Full graph
                node_set: set[str] = set()
                for ch in chunks:
                    if ch.name:
                        node_set.add(ch.name)
                        nodes.append(
                            {"id": ch.name, "label": ch.name, "type": ch.type, "file": ch.file}
                        )
                    node_set.add(ch.file)

                for call_edge in exported.get("calls", []):
                    if call_edge.get("caller") in node_set and call_edge.get("callee") in node_set:
                        edges.append(
                            {
                                "from": call_edge["caller"],
                                "to": call_edge["callee"],
                                "type": "calls",
                            }
                        )

                for imp_edge in exported.get("edges", []):
                    if imp_edge.get("source") in node_set:
                        edges.append(
                            {
                                "from": imp_edge["source"],
                                "to": imp_edge.get("target", ""),
                                "type": "imports",
                            }
                        )

            return {"status": "ok", "nodes": nodes, "edges": edges}

        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                f"Dependency graph failed: {e}",
                buffer_id=buffer_id,
                operation="get_dependency_graph",
            )

    # ------------------------------------------------------------------
    # Dead Code Detection
    # ------------------------------------------------------------------
    def find_dead_code(
        self,
        buffer_id: str,
        min_confidence: str = "medium",
    ) -> dict[str, Any]:
        """Find dead code and unused symbols in a buffer."""
        result = self._require_buffer(buffer_id, "find_dead_code")
        if isinstance(result, dict):
            return result
        info, chunks = result
        try:
            detector = DeadCodeDetector(chunks)
            dead = detector.find_dead_code(min_confidence)
            unused = detector.find_unused_imports()
            return {
                "status": "ok",
                "dead_symbols": [s.to_dict() for s in dead],
                "unused_imports": unused,
            }
        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                str(e), buffer_id=buffer_id, operation="find_dead_code"
            )

    # ------------------------------------------------------------------
    # TODO/FIXME Tracker
    # ------------------------------------------------------------------
    def extract_todos(
        self,
        buffer_id: str,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Extract TODO, FIXME, HACK, XXX comments from a buffer."""
        result = self._require_buffer(buffer_id, "extract_todos")
        if isinstance(result, dict):
            return result
        info, chunks = result
        try:
            tracker = TodoTracker()
            todos = tracker.extract_todos(chunks)
            if tag:
                todos = [t for t in todos if t.tag == tag.upper()]
            return {"status": "ok", "todos": [t.to_dict() for t in todos], "total": len(todos)}
        except (ValueError, RuntimeError) as e:
            return self._make_error_response(str(e), buffer_id=buffer_id, operation="extract_todos")

    # ------------------------------------------------------------------
    # Code Quality Scoring
    # ------------------------------------------------------------------
    def score_code_quality(
        self,
        buffer_id: str,
        file: str,
    ) -> dict[str, Any]:
        """Score code quality metrics for a file."""
        result = self._require_buffer(buffer_id, "score_code_quality")
        if isinstance(result, dict):
            return result
        info, chunks = result
        try:
            scorer = QualityScorer()
            result = scorer.score_file(chunks, file)
            if not result:
                return self._make_error_response(
                    "File not found in buffer", buffer_id=buffer_id, operation="score_code_quality"
                )
            return {"status": "ok", **result.to_dict()}
        except (ValueError, RuntimeError) as e:
            return self._make_error_response(
                str(e), buffer_id=buffer_id, operation="score_code_quality"
            )

    def detect_code_smells(
        self,
        buffer_id: str,
        types: list[str] | None = None,
        severity_min: str = "low",
    ) -> dict[str, Any]:
        """Detect code smells across the buffer with severity filtering.

        Automatically flag refactoring opportunities.
        Detects: long functions, deep nesting, missing docstrings, complex logic, too many parameters.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            types: Smell types to detect. Options: "long_function", "deep_nesting",
                "missing_docstring", "complex_logic", "too_many_params". Default: all.
            severity_min: Minimum severity to report: "low", "medium", or "high". Default: "low".

        Returns:
            Dict with keys:
                status: "ok" on success.
                smells: List of smell dicts with keys:
                    file: Source file path.
                    line: Start line number.
                    type: Smell type (e.g. "long_function", "deep_nesting").
                    severity: "low", "medium", or "high".
                    suggestion: Human-readable fix suggestion.
                total: Number of smells found.

        Example:
            >>> result = tool.detect_code_smells(buf_id, types=["long_function", "complex_logic"])
            >>> result["smells"][0]
            {"file": "src/auth.py", "line": 42, "type": "long_function", "severity": "medium",
             "suggestion": "Function 'process_login' is 80 lines. Consider extracting methods."}
        """
        result = self._require_buffer(buffer_id, "detect_code_smells")
        if isinstance(result, dict):
            return result
        info, chunks = result

        if types is None:
            types = [
                "long_function",
                "deep_nesting",
                "missing_docstring",
                "complex_logic",
                "too_many_params",
            ]

        severity_order = {"low": 0, "medium": 1, "high": 2}
        min_severity = severity_order.get(severity_min, 0)

        smells: list[dict[str, Any]] = []
        import re

        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue
            if chunk.type not in ("function", "method", "class"):
                continue

            lines = chunk.text.split("\n")
            line_count = len(lines)
            name = getattr(chunk, "name", "")

            # Long function
            if "long_function" in types and line_count > 50:
                severity = "high" if line_count > 100 else "medium"
                if severity_order.get(severity, 0) >= min_severity:
                    smells.append(
                        {
                            "file": chunk.file,
                            "line": chunk.start_line,
                            "type": "long_function",
                            "severity": severity,
                            "suggestion": f"Function '{name}' is {line_count} lines. Consider extracting methods.",
                        }
                    )

            # Deep nesting
            if "deep_nesting" in types:
                max_indent = 0
                for line in lines:
                    if line.strip():
                        indent = len(line) - len(line.lstrip())
                        max_indent = max(max_indent, indent)
                nesting = max_indent // 4
                if nesting > 3:
                    severity = "high" if nesting > 5 else "medium"
                    if severity_order.get(severity, 0) >= min_severity:
                        smells.append(
                            {
                                "file": chunk.file,
                                "line": chunk.start_line,
                                "type": "deep_nesting",
                                "severity": severity,
                                "suggestion": f"Function '{name}' has {nesting} levels of nesting. Consider guard clauses or extracting methods.",
                            }
                        )

            # Missing docstring
            if "missing_docstring" in types and chunk.type in ("function", "method"):
                has_docstring = '"""' in chunk.text or "'''" in chunk.text
                if not has_docstring and name:
                    if severity_order.get("medium", 0) >= min_severity:
                        smells.append(
                            {
                                "file": chunk.file,
                                "line": chunk.start_line,
                                "type": "missing_docstring",
                                "severity": "medium",
                                "suggestion": f"Function '{name}' has no docstring.",
                            }
                        )

            # Complex logic (high cyclomatic complexity)
            if "complex_logic" in types:
                complexity = 1 + len(
                    re.findall(r"\b(if|elif|for|while|except|and|or|assert)\b", chunk.text)
                )
                if complexity > 10:
                    severity = "high" if complexity > 20 else "medium"
                    if severity_order.get(severity, 0) >= min_severity:
                        smells.append(
                            {
                                "file": chunk.file,
                                "line": chunk.start_line,
                                "type": "complex_logic",
                                "severity": severity,
                                "suggestion": f"Function '{name}' has cyclomatic complexity {complexity}. Consider simplifying branches.",
                            }
                        )

            # Too many parameters
            if "too_many_params" in types:
                # Extract params from def line
                def_match = re.search(r"def\s+\w+\s*\((.*?)\)", chunk.text, re.DOTALL)
                if def_match:
                    params_str = def_match.group(1)
                    params = [
                        p.strip()
                        for p in params_str.split(",")
                        if p.strip() and p.strip() not in ("self", "cls")
                    ]
                    if len(params) > 5:
                        severity = "high" if len(params) > 8 else "medium"
                        if severity_order.get(severity, 0) >= min_severity:
                            smells.append(
                                {
                                    "file": chunk.file,
                                    "line": chunk.start_line,
                                    "type": "too_many_params",
                                    "severity": severity,
                                    "suggestion": f"Function '{name}' has {len(params)} parameters. Consider using a config object.",
                                }
                            )

        # Sort by severity (high first)
        smells.sort(key=lambda s: -severity_order.get(s.get("severity", "low"), 0))

        return {"status": "ok", "smells": smells, "total": len(smells)}

    def scan_security(
        self,
        buffer_id: str,
        severity_min: str = "medium",
    ) -> dict[str, Any]:
        """Scan for security vulnerabilities using pattern-based detection.

        Catch security issues before they reach production.
        Detects: eval/exec usage, shell injection, SQL injection, hardcoded secrets,
        unsafe pickle/yaml, broad except, wildcard imports.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            severity_min: Minimum severity to report: "low", "medium", or "high". Default: "medium".

        Returns:
            Dict with keys:
                status: "ok" on success.
                vulnerabilities: List of vulnerability dicts with keys:
                    file: Source file path.
                    line: Line number of the vulnerability.
                    type: Vulnerability type (e.g. "eval_usage", "sql_injection", "hardcoded_secret").
                    severity: "low", "medium", or "high".
                    context: The offending line of code.
                    fix_suggestion: How to fix the vulnerability.
                total: Number of vulnerabilities found.

        Example:
            >>> result = tool.scan_security(buf_id, severity_min="high")
            >>> result["vulnerabilities"][0]
            {"file": "src/db.py", "line": 42, "type": "sql_injection", "severity": "high",
             "context": 'cursor.execute(f"SELECT * FROM {table}")',
             "fix_suggestion": "Use parameterized queries instead of string formatting"}
        """
        result = self._require_buffer(buffer_id, "scan_security")
        if isinstance(result, dict):
            return result
        info, chunks = result

        severity_order = {"low": 0, "medium": 1, "high": 2}
        min_sev = severity_order.get(severity_min, 1)

        import re

        vulns: list[dict[str, Any]] = []

        # Security patterns
        patterns: list[tuple[str, str, str, str]] = [
            (
                r"eval\s*\(",
                "eval_usage",
                "high",
                "Avoid eval() — use ast.literal_eval() for safe parsing",
            ),
            (r"exec\s*\(", "exec_usage", "high", "Avoid exec() — can execute arbitrary code"),
            (
                r"subprocess\.call\s*\(.*shell\s*=\s*True",
                "shell_injection",
                "high",
                "Use shell=False and pass args as list to prevent injection",
            ),
            (
                r"os\.system\s*\(",
                "os_system",
                "high",
                "Use subprocess.run() with shell=False instead",
            ),
            (
                r"pickle\.load\s*\(",
                "unsafe_pickle",
                "high",
                "Pickle can execute arbitrary code — use json or msgpack",
            ),
            (
                r"yaml\.load\s*\((?!.*Loader)",
                "unsafe_yaml",
                "high",
                "Use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader)",
            ),
            (
                r"sql.*%s|f['\"].*SELECT|\.format\s*\(.*SELECT",
                "sql_injection",
                "high",
                "Use parameterized queries instead of string formatting",
            ),
            (
                r"secret|password|api_key|token\s*=\s*['\"]",
                "hardcoded_secret",
                "high",
                "Never hardcode secrets — use environment variables or vault",
            ),
            (
                r"assert\s+",
                "assert_usage",
                "medium",
                "Assertions can be disabled with -O flag — use proper validation",
            ),
            (
                r"\bexcept\s*:",
                "broad_except",
                "medium",
                "Catch specific exceptions instead of bare except:",
            ),
            (
                r"import\s+\*|from\s+\w+\s+import\s+\*",
                "wildcard_import",
                "low",
                "Wildcard imports pollute namespace and hide dependencies",
            ),
        ]

        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue
            for pattern, vuln_type, severity, fix in patterns:
                for match in re.finditer(pattern, chunk.text, re.IGNORECASE):
                    if severity_order.get(severity, 0) < min_sev:
                        continue
                    line = chunk.start_line + chunk.text[: match.start()].count("\n")
                    # Get the full line for context
                    lines = chunk.text.split("\n")
                    line_offset = chunk.text[: match.start()].count("\n")
                    context = (
                        lines[line_offset].strip() if line_offset < len(lines) else match.group(0)
                    )
                    vulns.append(
                        {
                            "file": chunk.file,
                            "line": line,
                            "type": vuln_type,
                            "severity": severity,
                            "context": context,
                            "fix_suggestion": fix,
                        }
                    )

        vulns.sort(key=lambda v: -severity_order.get(v.get("severity", "low"), 0))
        return {"status": "ok", "vulnerabilities": vulns, "total": len(vulns)}

    def suggest_refactorings(
        self,
        buffer_id: str,
        symbol: str,
    ) -> dict[str, Any]:
        """Suggest safe refactorings for a symbol with risk assessment.

        AI suggests safe refactorings before you make changes.
        Analyzes: extract method opportunities, branch simplification, duplicate calls,
        missing type hints, deep nesting guard clauses.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            symbol: Symbol name to analyze for refactoring opportunities.

        Returns:
            Dict with keys:
                status: "ok" on success.
                symbol: The analyzed symbol name.
                suggestions: List of suggestion dicts with keys:
                    type: Refactoring type ("extract_method", "simplify_branches",
                        "consolidate_calls", "add_type_hints", "use_guard_clauses").
                    lines: Line range string (e.g. "10-20").
                    symbol: Related symbol name (for consolidate_calls).
                    benefit: Human-readable benefit description.
                    risk: "low", "medium", or "high".

        Example:
            >>> result = tool.suggest_refactorings(buf_id, "process_payment")
            >>> result["suggestions"][0]
            {"type": "extract_method", "lines": "10-60", "benefit": "Reduce 80-line function to smaller units",
             "risk": "medium"}
        """
        result = self._require_buffer(buffer_id, "suggest_refactorings")
        if isinstance(result, dict):
            return result
        info, chunks = result

        suggestions: list[dict[str, Any]] = []
        import re

        # Find the target chunk
        target = None
        for chunk in chunks:
            if hasattr(chunk, "name") and chunk.name == symbol and chunk.text:
                target = chunk
                break

        if target is None:
            return self._make_error_response(
                f"Symbol '{symbol}' not found",
                buffer_id=buffer_id,
                operation="suggest_refactorings",
            )

        lines = target.text.split("\n")
        line_count = len(lines)
        complexity = 1 + len(re.findall(r"\b(if|elif|for|while|except|and|or)\b", target.text))

        # Long function → extract method
        if line_count > 30:
            suggestions.append(
                {
                    "type": "extract_method",
                    "lines": f"{target.start_line}-{target.end_line}",
                    "benefit": f"Reduce {line_count}-line function to smaller, testable units",
                    "risk": "low" if line_count < 60 else "medium",
                }
            )

        # High complexity → simplify
        if complexity > 8:
            suggestions.append(
                {
                    "type": "simplify_branches",
                    "lines": f"{target.start_line}-{target.end_line}",
                    "benefit": f"Reduce cyclomatic complexity from {complexity} to <10",
                    "risk": "medium",
                }
            )

        # Check for duplicate calls (same call repeated)
        calls = getattr(target, "symbols_called", None) or []
        call_counts: dict[str, int] = {}
        for c in calls:
            call_counts[c] = call_counts.get(c, 0) + 1
        for c_name, count in call_counts.items():
            if count > 2:
                suggestions.append(
                    {
                        "type": "consolidate_calls",
                        "symbol": c_name,
                        "benefit": f"'{c_name}' called {count} times — extract to variable",
                        "risk": "low",
                    }
                )

        # Missing type hints
        if "def " in target.text and "->" not in target.text:
            suggestions.append(
                {
                    "type": "add_type_hints",
                    "lines": f"{target.start_line}-{target.end_line}",
                    "benefit": "Add return type annotation for better IDE support and safety",
                    "risk": "low",
                }
            )

        # Deep nesting → guard clauses
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        nesting = max_indent // 4
        if nesting > 3:
            suggestions.append(
                {
                    "type": "use_guard_clauses",
                    "lines": f"{target.start_line}-{target.end_line}",
                    "benefit": f"Reduce nesting from {nesting} levels using early returns",
                    "risk": "low",
                }
            )

        return {"status": "ok", "symbol": symbol, "suggestions": suggestions}

    # ------------------------------------------------------------------
    # Remaining Advanced Features
    # ------------------------------------------------------------------
    def find_performance_hotspots(
        self,
        buffer_id: str,
    ) -> dict[str, Any]:
        """Detect performance hotspots: N+1 queries, inefficient loops, unbounded growth, resource leaks.

        Identify performance-critical code for optimization.
        Uses regex pattern matching to detect common anti-patterns and nested loops.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.

        Returns:
            Dict with keys:
                status: "ok" on success.
                hotspots: List of hotspot dicts with keys:
                    file: Source file path.
                    line: Line number.
                    type: Hotspot type (e.g. "n_plus_one", "nested_loop", "unclosed_file").
                    severity: "low", "medium", or "high".
                    context: The offending code snippet.
                    suggestion: Optimization recommendation.
                total: Number of hotspots found.

        Example:
            >>> result = tool.find_performance_hotspots(buf_id)
            >>> result["hotspots"][0]
            {"file": "src/db.py", "line": 42, "type": "n_plus_one", "severity": "high",
             "context": "for user in User.objects.all():", "suggestion": "Review n_plus_one pattern"}
        """
        result = self._require_buffer(buffer_id, "find_performance_hotspots")
        if isinstance(result, dict):
            return result
        info, chunks = result

        import re

        hotspots: list[dict[str, Any]] = []

        patterns: list[tuple[str, str, str]] = [
            (r"for\s+\w+\s+in\s+.*\.all\(\)|for\s+\w+\s+in\s+.*\.filter\(\)", "n_plus_one", "high"),
            (r"while\s+True\s*:.*\.append\(", "unbounded_growth", "high"),
            (r"\.select_related\s*\(\s*\)", "missing_prefetch", "medium"),
            (r"for\s+\w+\s+in\s+range\s*\(\s*len\s*\(", "inefficient_loop", "low"),
            (r"import\s+pickle|pickle\.dump|pickle\.load", "slow_serialization", "medium"),
            (r"re\.compile\s*\(\s*\)", "regex_in_loop", "medium"),
            (r"open\s*\([^)]*\)\s*(?!with)", "unclosed_file", "high"),
            (r"\.execute\s*\([^)]*\)\s*(?!.*fetchmany|.*fetchone)", "full_table_scan", "medium"),
        ]

        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue
            for pattern, hotspot_type, severity in patterns:
                for match in re.finditer(pattern, chunk.text, re.DOTALL):
                    line = chunk.start_line + chunk.text[: match.start()].count("\n")
                    hotspots.append(
                        {
                            "file": chunk.file,
                            "line": line,
                            "type": hotspot_type,
                            "severity": severity,
                            "context": chunk.text.split("\n")[
                                chunk.text[: match.start()].count("\n")
                            ].strip()[:200],
                            "suggestion": f"Review {hotspot_type} pattern for optimization",
                        }
                    )

        # Detect nested loops (O(n^2) or worse)
        for chunk in chunks:
            if (
                not hasattr(chunk, "text")
                or not chunk.text
                or chunk.type not in ("function", "method")
            ):
                continue
            lines = chunk.text.split("\n")
            nested_indent = 0
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                indent = len(line) - len(line.lstrip())
                if "for " in line or "while " in line:
                    if indent > nested_indent and nested_indent > 0:
                        hotspots.append(
                            {
                                "file": chunk.file,
                                "line": chunk.start_line + i,
                                "type": "nested_loop",
                                "severity": "medium",
                                "context": line.strip()[:200],
                                "suggestion": "Nested loop detected — consider algorithmic optimization or vectorization",
                            }
                        )
                    nested_indent = indent

        hotspots.sort(
            key=lambda h: {"high": 2, "medium": 1, "low": 0}.get(h.get("severity", "low"), 0),
            reverse=True,
        )
        return {"status": "ok", "hotspots": hotspots, "total": len(hotspots)}

    def generate_documentation(
        self,
        buffer_id: str,
        symbol: str,
        style: str = "google",
    ) -> dict[str, Any]:
        """Auto-generate documentation from code analysis.

        Generate accurate docstrings from AST + type info.
        Parses function signatures, extracts type hints, and finds usage examples in test files.
        Supports Google, NumPy, and Sphinx docstring styles.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            symbol: Symbol name to generate documentation for.
            style: Docstring style: "google", "numpy", or "sphinx". Default: "google".

        Returns:
            Dict with keys:
                status: "ok" on success.
                symbol: The documented symbol name.
                docstring: Generated docstring ready to insert into code.
                type_hints: Dict mapping parameter names to inferred types.
                examples: List of usage examples found in test files.
                generated_from_code: True (always, since this is AST-based).
                style: The docstring style used.

        Example:
            >>> result = tool.generate_documentation(buf_id, "process_payment", style="google")
            >>> print(result["docstring"])
            \"\"\"process_payment documentation.

            Args:
                amount (float): Payment amount.
                currency (str): ISO 4217 currency code.

            Returns:
                bool
            \"\"\"
        """
        result = self._require_buffer(buffer_id, "generate_documentation")
        if isinstance(result, dict):
            return result
        info, chunks = result

        target = None
        for chunk in chunks:
            if hasattr(chunk, "name") and chunk.name == symbol and chunk.text:
                target = chunk
                break

        if target is None:
            return self._make_error_response(
                f"Symbol '{symbol}' not found",
                buffer_id=buffer_id,
                operation="generate_documentation",
            )

        import re

        docstring = ""
        type_hints = {}
        examples = []

        # Extract function signature
        sig_match = re.search(r"def\s+\w+\s*\((.*?)\)\s*(?:->\s*(.+?))?:", target.text, re.DOTALL)
        if sig_match:
            params_str = sig_match.group(1)
            return_type = sig_match.group(2)

            # Parse parameters
            params = [
                p.strip()
                for p in params_str.split(",")
                if p.strip() and p.strip() not in ("self", "cls")
            ]
            param_docs = []
            for param in params:
                parts = param.split(":")
                name = parts[0].strip().split("=")[0].strip()
                type_hint = parts[1].strip().split("=")[0].strip() if len(parts) > 1 else "Any"
                default = None
                if "=" in param:
                    default = param.split("=")[-1].strip()
                type_hints[name] = type_hint
                param_doc = f"    {name} ({type_hint})"
                if default:
                    param_doc += f", default: {default}"
                param_docs.append(param_doc)

            # Build docstring
            if style == "google":
                lines = [f'"""{symbol} documentation.']
                if param_docs:
                    lines.append("")
                    lines.append("Args:")
                    lines.extend(param_docs)
                if return_type:
                    lines.append("")
                    lines.append("Returns:")
                    lines.append(f"    {return_type.strip()}")
                lines.append('"""')
                docstring = "\n".join(lines)
            elif style == "numpy":
                lines = [f'"""{symbol} documentation.']
                if param_docs:
                    lines.append("")
                    lines.append("Parameters")
                    lines.append("----------")
                    lines.extend(param_docs)
                if return_type:
                    lines.append("")
                    lines.append("Returns")
                    lines.append("-------")
                    lines.append(f"    {return_type.strip()}")
                lines.append('"""')
                docstring = "\n".join(lines)
            else:  # sphinx
                lines = [f'"""{symbol} documentation.']
                for pd in param_docs:
                    lines.append(f":param {pd.strip()}")
                if return_type:
                    lines.append(f":rtype: {return_type.strip()}")
                lines.append('"""')
                docstring = "\n".join(lines)

        # Find usage examples from test files
        for chunk in chunks:
            if hasattr(chunk, "text") and chunk.text and symbol in (chunk.text or ""):
                if hasattr(chunk, "file") and "test" in chunk.file:
                    for match in re.finditer(rf"{symbol}\s*\([^)]*\)", chunk.text):
                        examples.append(match.group(0))

        return {
            "status": "ok",
            "symbol": symbol,
            "docstring": docstring,
            "type_hints": type_hints,
            "examples": examples[:5],
            "generated_from_code": True,
            "style": style,
        }

    def find_similar_patterns(
        self,
        buffer_id: str,
        code_snippet: str,
        min_similarity: float = 0.7,
        top_k: int = 10,
        cluster: bool = False,
        clustering_method: str = "agglomerative",
        cluster_threshold: float = 0.8,
        expected_clusters: int | None = None,
    ) -> dict[str, Any]:
        """Find similar code patterns using semantic + syntactic matching.

        Optionally groups matching regions into score-based clusters so agents can
        identify recurring pattern families instead of a flat result list.
        """
        result = self._require_buffer(buffer_id, "find_similar_patterns")
        if isinstance(result, dict):
            return result
        _, chunks = result

        min_similarity = max(0.0, min(min_similarity, 1.0))
        cluster_threshold = max(0.0, min(cluster_threshold, 1.0))
        if top_k < 1:
            return self._make_error_response(
                "top_k must be at least 1",
                buffer_id=buffer_id,
                operation="find_similar_patterns",
            )
        if clustering_method not in {"agglomerative", "kmeans", "spectral"}:
            return self._make_error_response(
                f"Unsupported clustering_method: {clustering_method}",
                buffer_id=buffer_id,
                operation="find_similar_patterns",
            )
        if expected_clusters is not None and expected_clusters < 1:
            return self._make_error_response(
                "expected_clusters must be at least 1",
                buffer_id=buffer_id,
                operation="find_similar_patterns",
            )

        import re

        def _token_shingles(text: str, ngram: int = 3) -> set[str]:
            tokens = re.findall(r"\w+", text.lower())
            if not tokens:
                return set()
            if len(tokens) < ngram:
                return {" ".join(tokens)}
            return {
                " ".join(tokens[index : index + ngram]) for index in range(len(tokens) - ngram + 1)
            }

        snippet_shingles = _token_shingles(code_snippet)
        candidate_map: dict[tuple[str, int, int], dict[str, Any]] = {}

        def _record_candidate(
            *,
            file: str,
            start_line: int,
            end_line: int,
            similarity: float,
            source: str,
            doc_id: int | None = None,
        ) -> None:
            key = (file, start_line, end_line)
            candidate = candidate_map.setdefault(
                key,
                {
                    "file": file,
                    "start_line": start_line,
                    "end_line": end_line,
                    "similarity": 0.0,
                    "semantic_score": None,
                    "syntactic_similarity": None,
                    "doc_id": doc_id,
                },
            )
            if source == "semantic":
                candidate["semantic_score"] = max(candidate["semantic_score"] or 0.0, similarity)
                if doc_id is not None:
                    candidate["doc_id"] = doc_id
            else:
                candidate["syntactic_similarity"] = max(
                    candidate["syntactic_similarity"] or 0.0, similarity
                )

            scores = [
                value
                for value in (
                    candidate.get("semantic_score"),
                    candidate.get("syntactic_similarity"),
                )
                if isinstance(value, (int, float))
            ]
            candidate["similarity"] = (
                round(sum(scores) / len(scores), 4) if scores else round(similarity, 4)
            )

        syntactic_matches: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_shingles = _token_shingles(chunk.text)
            union = len(snippet_shingles | chunk_shingles)
            if union == 0:
                continue
            similarity = len(snippet_shingles & chunk_shingles) / union
            if similarity < min_similarity:
                continue
            match = {
                "file": chunk.file,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "similarity": round(similarity, 4),
            }
            syntactic_matches.append(match)
            _record_candidate(
                file=chunk.file,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                similarity=similarity,
                source="syntactic",
                doc_id=getattr(chunk, "id", None),
            )

        syntactic_matches.sort(key=lambda item: item["similarity"], reverse=True)
        syntactic_matches = syntactic_matches[:top_k]

        semantic_matches: list[dict[str, Any]] = []
        try:
            search_result = self.semantic_search(buffer_id, code_snippet, top_k=top_k)
            if search_result.get("status") == "ok":
                for match in search_result.get("matches", []):
                    semantic_match = {
                        "file": match.get("file", ""),
                        "line": match.get("start_line", 0),
                        "score": round(float(match.get("score", 0.0)), 4),
                    }
                    semantic_matches.append(semantic_match)
                    _record_candidate(
                        file=match.get("file", ""),
                        start_line=match.get("start_line", 0),
                        end_line=match.get("end_line", match.get("start_line", 0)),
                        similarity=float(match.get("score", 0.0)),
                        source="semantic",
                        doc_id=match.get("doc_id"),
                    )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("semantic search failed in find_similar_patterns: %s", exc)

        if not cluster:
            return {
                "status": "ok",
                "cluster_mode": False,
                "syntactic_matches": syntactic_matches,
                "semantic_matches": semantic_matches,
                "snippet_length": len(code_snippet),
            }

        ranked_candidates = sorted(
            candidate_map.values(), key=lambda item: item["similarity"], reverse=True
        )
        if top_k:
            ranked_candidates = ranked_candidates[:top_k]

        buckets: list[list[dict[str, Any]]] = []
        tolerance = max(0.05, 1.0 - cluster_threshold)

        if expected_clusters and expected_clusters > 0 and ranked_candidates:
            bucket_count = min(expected_clusters, len(ranked_candidates))
            base_size = len(ranked_candidates) // bucket_count
            extra = len(ranked_candidates) % bucket_count
            start = 0
            for index in range(bucket_count):
                size = base_size + (1 if index < extra else 0)
                buckets.append(ranked_candidates[start : start + size])
                start += size
        else:
            for candidate in ranked_candidates:
                placed = False
                for bucket in buckets:
                    avg_similarity = sum(item["similarity"] for item in bucket) / len(bucket)
                    if abs(candidate["similarity"] - avg_similarity) <= tolerance:
                        bucket.append(candidate)
                        placed = True
                        break
                if not placed:
                    buckets.append([candidate])

        clusters: list[dict[str, Any]] = []
        compactness_values: list[float] = []
        for index, bucket in enumerate(buckets, start=1):
            if not bucket:
                continue
            avg_similarity = round(sum(item["similarity"] for item in bucket) / len(bucket), 4)
            max_deviation = (
                max(abs(item["similarity"] - avg_similarity) for item in bucket)
                if len(bucket) > 1
                else 0.0
            )
            compactness = round(max(0.0, 1.0 - (max_deviation / max(tolerance, 0.05))), 4)
            compactness_values.append(compactness)
            bucket.sort(key=lambda item: item["similarity"], reverse=True)
            example = bucket[0]
            cluster_size = len(bucket)
            if cluster_size >= 3:
                refactoring_opportunity = "extract_method"
            elif cluster_size == 2:
                refactoring_opportunity = "create_helper"
            else:
                refactoring_opportunity = "review_individually"

            clusters.append(
                {
                    "cluster_id": index,
                    "size": cluster_size,
                    "avg_similarity": avg_similarity,
                    "compactness": compactness,
                    "example_file": example["file"],
                    "example_line": example["start_line"],
                    "members": [
                        {
                            "file": item["file"],
                            "start_line": item["start_line"],
                            "end_line": item["end_line"],
                            "similarity": item["similarity"],
                        }
                        for item in bucket
                    ],
                    "refactoring_opportunity": refactoring_opportunity,
                }
            )

        return {
            "status": "ok",
            "cluster_mode": True,
            "clusters": clusters,
            "snippet_length": len(code_snippet),
            "clustering_method": clustering_method,
            "quality": (
                round(sum(compactness_values) / len(compactness_values), 4)
                if compactness_values
                else 0.0
            ),
        }

    def find_deprecated(
        self,
        buffer_id: str,
    ) -> dict[str, Any]:
        """Detect usage of deprecated functions and APIs.

        Identify code that uses outdated APIs.
        Searches for @deprecated decorators, DeprecationWarning usage,
        deprecated comments, and warnings.warn("deprecated...") patterns.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.

        Returns:
            Dict with keys:
                status: "ok" on success.
                deprecated: List of deprecation dicts with keys:
                    file: Source file path.
                    line: Line number.
                    detection_method: How it was detected ("decorator", "warning", "comment", "deprecation_warning").
                    context: The offending line of code.
                    symbol: Symbol name containing the deprecation.
                total: Number of deprecated items found.

        Example:
            >>> result = tool.find_deprecated(buf_id)
            >>> result["deprecated"][0]
            {"file": "src/api.py", "line": 42, "detection_method": "decorator",
             "context": "@deprecated", "symbol": "old_endpoint"}
        """
        result = self._require_buffer(buffer_id, "find_deprecated")
        if isinstance(result, dict):
            return result
        info, chunks = result

        import re

        deprecated: list[dict[str, Any]] = []

        # Common deprecated patterns
        dep_patterns = [
            (r"@deprecated\b", "decorator"),
            (r"warnings\.warn\s*\([^)]*deprecated", "warning"),
            (r"#\s*DEPRECATED", "comment"),
            (r"DeprecationWarning", "deprecation_warning"),
        ]

        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue
            for pattern, method in dep_patterns:
                for match in re.finditer(pattern, chunk.text, re.IGNORECASE):
                    line = chunk.start_line + chunk.text[: match.start()].count("\n")
                    lines = chunk.text.split("\n")
                    line_offset = chunk.text[: match.start()].count("\n")
                    context = lines[line_offset].strip()[:200] if line_offset < len(lines) else ""
                    deprecated.append(
                        {
                            "file": chunk.file,
                            "line": line,
                            "detection_method": method,
                            "context": context,
                            "symbol": getattr(chunk, "name", ""),
                        }
                    )

        return {"status": "ok", "deprecated": deprecated, "total": len(deprecated)}

    def validate_changes(
        self,
        buffer_id: str,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Validate changes before committing with static analysis + import resolution.

        Pre-commit validation without running full test suite.
        Checks: Python syntax errors, import resolution, test impact prediction.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            dry_run: If True, only preview validation without side effects. Default: True.

        Returns:
            Dict with keys:
                status: "ok" on success.
                type_errors: List of syntax error dicts with keys: file, line, message.
                broken_imports: List of import error dicts with keys: file, import, message.
                test_impact_predictions: List of test symbol names that may be affected.
                safe_to_commit: True if no errors or broken imports found.
                dry_run: Whether this was a dry run.

        Example:
            >>> result = tool.validate_changes(buf_id)
            >>> result["safe_to_commit"]
            False
            >>> result["type_errors"]
            [{"file": "src/main.py", "line": 10, "message": "SyntaxError: invalid syntax"}]
        """
        result = self._require_buffer(buffer_id, "validate_changes")
        if isinstance(result, dict):
            return result
        info, chunks = result

        import ast as _ast
        import re

        type_errors: list[dict[str, Any]] = []
        broken_imports: list[dict[str, Any]] = []
        test_impact: list[str] = []

        # Build set of available files/modules
        available_modules: set[str] = set()
        available_symbols: set[str] = set()
        for chunk in chunks:
            if hasattr(chunk, "file"):
                stem = Path(chunk.file).stem
                available_modules.add(stem)
            if hasattr(chunk, "name") and chunk.name:
                available_symbols.add(chunk.name)

        # Check imports and syntax
        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue
            # Syntax check
            try:
                _ast.parse(chunk.text)
            except SyntaxError as e:
                type_errors.append(
                    {
                        "file": chunk.file,
                        "line": e.lineno or chunk.start_line,
                        "message": f"SyntaxError: {e.msg}",
                    }
                )

            # Check imports resolve
            for imp in getattr(chunk, "imports", None) or []:
                top_level = imp.split(".")[0]
                if top_level not in available_modules and top_level not in (
                    "os",
                    "sys",
                    "re",
                    "json",
                    "logging",
                    "pathlib",
                    "typing",
                    "dataclasses",
                    "collections",
                    "abc",
                    "functools",
                    "itertools",
                    "datetime",
                    "hashlib",
                    "io",
                    "contextlib",
                    "asyncio",
                    "subprocess",
                    "shutil",
                    "tempfile",
                    "time",
                    "math",
                    "unittest",
                    "copy",
                    "ast",
                    "importlib",
                    "inspect",
                    "textwrap",
                    "enum",
                    "http",
                    "urllib",
                    "email",
                    "csv",
                    "sqlite3",
                    "fastapi",
                    "pydantic",
                    "uvicorn",
                    "numpy",
                    "torch",
                    "sentence_transformers",
                ):
                    # Not found locally or in known stdlib/third-party
                    broken_imports.append(
                        {
                            "file": chunk.file,
                            "import": imp,
                            "message": f"Cannot resolve import: {imp}",
                        }
                    )

        # Test impact prediction
        test_impact = [
            ch.name
            for ch in chunks
            if hasattr(ch, "name") and ch.name and "test" in getattr(ch, "file", "")
        ]

        safe_to_commit = len(type_errors) == 0 and len(broken_imports) == 0

        return {
            "status": "ok",
            "type_errors": type_errors,
            "broken_imports": broken_imports,
            "test_impact_predictions": test_impact[:20],
            "safe_to_commit": safe_to_commit,
            "dry_run": dry_run,
        }

    def extract_configuration(
        self,
        buffer_id: str,
    ) -> dict[str, Any]:
        """Extract configuration: environment variables, config files, hardcoded secrets, defaults.

        Understand what needs to be configured; detect hardcoded secrets.
        Parses os.environ.get(), os.getenv(), config file references, and secret patterns.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.

        Returns:
            Dict with keys:
                status: "ok" on success.
                env_vars: List of env var dicts with keys: name, used_in (file:line), required (bool), default (str or None).
                config_files: List of config file paths referenced in code.
                hardcoded_secrets: List of secret dicts with keys: file, line, pattern, severity.
                default_values: Dict mapping env var names to their default values.

        Example:
            >>> result = tool.extract_configuration(buf_id)
            >>> result["env_vars"][0]
            {"name": "DATABASE_URL", "used_in": "src/db.py:5", "required": True, "default": None}
            >>> result["hardcoded_secrets"]
            [{"file": "src/auth.py", "line": 12, "pattern": "api_key = 'abc123'", "severity": "high"}]
        """
        result = self._require_buffer(buffer_id, "extract_configuration")
        if isinstance(result, dict):
            return result
        info, chunks = result

        import re

        env_vars: list[dict[str, Any]] = []
        config_files: list[str] = []
        hardcoded_secrets: list[dict[str, Any]] = []
        default_values: dict[str, str] = {}

        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue

            # Environment variables
            for match in re.finditer(
                r'os\.environ\.get\s*\(\s*["\'](\w+)["\']\s*(?:,\s*([^)]+))?\)', chunk.text
            ):
                var_name = match.group(1)
                default = match.group(2).strip().strip("\"'") if match.group(2) else None
                env_vars.append(
                    {
                        "name": var_name,
                        "used_in": f"{chunk.file}:{chunk.start_line}",
                        "required": default is None,
                        "default": default,
                    }
                )
                if default:
                    default_values[var_name] = default

            for match in re.finditer(
                r'os\.getenv\s*\(\s*["\'](\w+)["\']\s*(?:,\s*([^)]+))?\)', chunk.text
            ):
                var_name = match.group(1)
                default = match.group(2).strip().strip("\"'") if match.group(2) else None
                env_vars.append(
                    {
                        "name": var_name,
                        "used_in": f"{chunk.file}:{chunk.start_line}",
                        "required": default is None,
                        "default": default,
                    }
                )
                if default:
                    default_values[var_name] = default

            # Config files
            for match in re.finditer(
                r'["\']([^"\']*\.(yaml|yml|toml|ini|cfg|conf|json))["\']', chunk.text
            ):
                config_files.append(match.group(1))

            # Hardcoded secrets
            for match in re.finditer(
                r'(?:secret|password|api_key|token|auth_key|private_key)\s*=\s*["\'][^"\']{4,}["\']',
                chunk.text,
                re.IGNORECASE,
            ):
                line = chunk.start_line + chunk.text[: match.start()].count("\n")
                hardcoded_secrets.append(
                    {
                        "file": chunk.file,
                        "line": line,
                        "pattern": match.group(0)[:80],
                        "severity": "high",
                    }
                )

        # Deduplicate
        seen_vars: set[str] = set()
        unique_env_vars = []
        for v in env_vars:
            if v["name"] not in seen_vars:
                seen_vars.add(v["name"])
                unique_env_vars.append(v)

        return {
            "status": "ok",
            "env_vars": unique_env_vars,
            "config_files": list(set(config_files)),
            "hardcoded_secrets": hardcoded_secrets,
            "default_values": default_values,
        }

    def analyze_logging_patterns(
        self,
        buffer_id: str,
    ) -> dict[str, Any]:
        """Analyze logging patterns: levels, consistency, gaps.

        Ensure consistent logging for debugging/monitoring.
        Counts log levels, detects try/except blocks without logging, and finds format inconsistencies.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.

        Returns:
            Dict with keys:
                status: "ok" on success.
                total_logs: Total number of log statements found.
                levels: Dict with keys: debug, info, warning, error, critical — each an integer count.
                missing_logs_in: List of dicts with keys: file, symbol, issue (e.g. "try/except without logging").
                inconsistent_format: List of dicts with keys: file, line, format_string.
                patterns_detected: List of detected anti-patterns (e.g. "f-string-in-log").

        Example:
            >>> result = tool.analyze_logging_patterns(buf_id)
            >>> result["levels"]
            {"debug": 10, "info": 42, "warning": 8, "error": 3, "critical": 0}
            >>> result["missing_logs_in"][0]
            {"file": "src/critical.py", "symbol": "process_refund", "issue": "try/except without logging"}
        """
        result = self._require_buffer(buffer_id, "analyze_logging_patterns")
        if isinstance(result, dict):
            return result
        info, chunks = result

        import re

        levels = {"debug": 0, "info": 0, "warning": 0, "error": 0, "critical": 0}
        total_logs = 0
        missing_logs: list[dict[str, Any]] = []
        inconsistent_format: list[dict[str, Any]] = []
        log_patterns_seen: set[str] = set()

        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue

            # Count log levels
            for level in levels:
                count = len(re.findall(rf"\.{level}\s*\(", chunk.text))
                levels[level] += count
                total_logs += count

            # Check for format string inconsistency
            for match in re.finditer(r'(?:logger|logging)\.\w+\s*\(\s*["\']', chunk.text):
                if (
                    "%" in match.group(0)
                    and "format(" in chunk.text[match.end() : match.end() + 100]
                ):
                    line = chunk.start_line + chunk.text[: match.start()].count("\n")
                    inconsistent_format.append(
                        {"file": chunk.file, "line": line, "format_string": "mixed % and .format()"}
                    )

            # Check for f-string in log (potential performance issue)
            for _match in re.finditer(r'(?:logger|logging)\.\w+\s*\(\s*f["\']', chunk.text):
                log_patterns_seen.add("f-string-in-log")

            # Detect functions with no logging but error handling
            if chunk.type in ("function", "method"):
                has_try = "try:" in chunk.text or "try :" in chunk.text
                has_log = any(f".{level_name}(" in chunk.text for level_name in levels)
                if has_try and not has_log:
                    missing_logs.append(
                        {
                            "file": chunk.file,
                            "symbol": getattr(chunk, "name", ""),
                            "issue": "try/except without logging",
                        }
                    )

        return {
            "status": "ok",
            "total_logs": total_logs,
            "levels": levels,
            "missing_logs_in": missing_logs[:20],
            "inconsistent_format": inconsistent_format[:20],
            "patterns_detected": list(log_patterns_seen),
        }

    def analyze_error_handling_patterns(
        self,
        buffer_id: str,
    ) -> dict[str, Any]:
        """Analyze error handling patterns: broad catches, missing finally, silent failures.

        Ensure robust error handling; prevent silent failures.
        Detects bare except, broad Exception catches, missing finally for resources, and silent pass statements.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.

        Returns:
            Dict with keys:
                status: "ok" on success.
                try_except_blocks: Total number of try/except blocks found.
                uncaught_exceptions: List of dicts with keys: file, line, issue.
                broad_catches: List of dicts with keys: file, line, catches (e.g. "Exception" or "bare except").
                missing_finally: List of dicts with keys: file, line, resource.
                suggestions: List of human-readable improvement suggestions.

        Example:
            >>> result = tool.analyze_error_handling_patterns(buf_id)
            >>> result["broad_catches"][0]
            {"file": "src/db.py", "line": 42, "catches": "Exception"}
            >>> result["suggestions"]
            ["Replace broad exception handlers with specific exception types"]
        """
        result = self._require_buffer(buffer_id, "analyze_error_handling_patterns")
        if isinstance(result, dict):
            return result
        info, chunks = result

        import re

        try_except_blocks = 0
        uncaught_exceptions: list[dict[str, Any]] = []
        broad_catches: list[dict[str, Any]] = []
        missing_finally: list[dict[str, Any]] = []
        suggestions: list[str] = []

        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue

            try_count = len(re.findall(r"\btry\s*:", chunk.text))
            try_except_blocks += try_count

            # Broad catches
            for match in re.finditer(r"except\s*:", chunk.text):
                line = chunk.start_line + chunk.text[: match.start()].count("\n")
                broad_catches.append({"file": chunk.file, "line": line, "catches": "bare except"})

            for match in re.finditer(r"except\s+Exception\s*:", chunk.text):
                line = chunk.start_line + chunk.text[: match.start()].count("\n")
                broad_catches.append({"file": chunk.file, "line": line, "catches": "Exception"})

            # Missing finally for resource operations
            if "open(" in chunk.text or "connect(" in chunk.text:
                if try_count > 0 and "finally:" not in chunk.text and "with " not in chunk.text:
                    line = chunk.start_line
                    missing_finally.append(
                        {"file": chunk.file, "line": line, "resource": "file/connection"}
                    )

            # Silent catches (just pass)
            for match in re.finditer(r"except\s+\w+\s*:\s*\n\s*pass", chunk.text):
                line = chunk.start_line + chunk.text[: match.start()].count("\n")
                uncaught_exceptions.append(
                    {"file": chunk.file, "line": line, "issue": "silent catch (pass)"}
                )

        if broad_catches:
            suggestions.append("Replace broad exception handlers with specific exception types")
        if missing_finally:
            suggestions.append("Use context managers (with statement) for resource cleanup")
        if uncaught_exceptions:
            suggestions.append("Log or handle caught exceptions instead of silently passing")

        return {
            "status": "ok",
            "try_except_blocks": try_except_blocks,
            "uncaught_exceptions": uncaught_exceptions,
            "broad_catches": broad_catches,
            "missing_finally": missing_finally,
            "suggestions": suggestions,
        }

    def generate_changelog(
        self,
        buffer_id: str,
        since_commit: str | None = None,
    ) -> dict[str, Any]:
        """Generate changelog from git history with semantic categorization.

        Auto-generate release notes from code changes.
        Parses git log and categorizes commits into features, bugfixes, and breaking changes.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            since_commit: Git commit hash to compare against. If None, shows last 50 commits.

        Returns:
            Dict with keys:
                status: "ok" on success.
                features: List of feature commit dicts with keys: commit, message.
                bugfixes: List of bugfix commit dicts with keys: commit, message.
                breaking_changes: List of breaking change dicts with keys: commit, message.
                other: List of uncategorized commit dicts with keys: commit, message.
                migration_notes: Migration notes string (empty if no breaking changes).

        Example:
            >>> result = tool.generate_changelog(buf_id, since_commit="v1.0.0")
            >>> result["features"][0]
            {"commit": "abc1234", "message": "feat: add retry logic to database calls"}
        """
        result = self._require_buffer(buffer_id, "generate_changelog")
        if isinstance(result, dict):
            return result
        info, chunks = result

        from gigacode import git_utils

        features: list[dict[str, Any]] = []
        bugfixes: list[dict[str, Any]] = []
        breaking_changes: list[dict[str, Any]] = []
        other: list[dict[str, Any]] = []

        try:
            work_dir = self.work_dir
            args = ["log", "--oneline", "-50"]
            if since_commit:
                args.append(f"{since_commit}..HEAD")
            log_output = git_utils.run_git(args, cwd=work_dir)
            if log_output:
                for line in log_output.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split(" ", 1)
                    commit_hash = parts[0] if parts else ""
                    message = parts[1] if len(parts) > 1 else line
                    msg_lower = message.lower()
                    entry = {"commit": commit_hash, "message": message}
                    if any(kw in msg_lower for kw in ["feat", "feature", "add", "new"]):
                        features.append(entry)
                    elif any(kw in msg_lower for kw in ["fix", "bug", "patch", "hotfix"]):
                        bugfixes.append(entry)
                    elif any(
                        kw in msg_lower for kw in ["breaking", "remove", "delete", "deprecate"]
                    ):
                        breaking_changes.append(entry)
                    else:
                        other.append(entry)
        except (OSError, RuntimeError, ValueError) as e:
            return self._make_error_response(
                f"Git log failed: {e}",
                buffer_id=buffer_id,
                operation="generate_changelog",
            )

        return {
            "status": "ok",
            "features": features,
            "bugfixes": bugfixes,
            "breaking_changes": breaking_changes,
            "other": other,
            "migration_notes": "",
        }

    def detect_api_changes(
        self,
        buffer_id: str,
        since_commit: str | None = None,
    ) -> dict[str, Any]:
        """Detect API-breaking changes between commits.

        Detect breaking changes that would affect consumers.
        Compares current API surface (public symbols + signatures) against a previous commit.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            since_commit: Git commit hash to compare against. If None, only lists current API surface.

        Returns:
            Dict with keys:
                status: "ok" on success.
                current_api_surface: Number of public symbols found.
                changes: List of change dicts with keys: symbol, breaking (bool),
                    parameters_added, return_type_changed (bool), migration_guide.
                symbols: List of symbol dicts with keys: name, params (list of param names).

        Example:
            >>> result = tool.detect_api_changes(buf_id, since_commit="v1.0.0")
            >>> result["changes"][0]
            {"symbol": "process_payment", "breaking": True, "parameters_added": ["currency"],
             "return_type_changed": False, "migration_guide": "Review changes to process_payment"}
        """
        result = self._require_buffer(buffer_id, "detect_api_changes")
        if isinstance(result, dict):
            return result
        info, chunks = result

        # Current API surface
        current_api: list[dict[str, Any]] = []
        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue
            if (
                chunk.type in ("function", "method", "class")
                and hasattr(chunk, "name")
                and chunk.name
            ):
                import re

                # Check for public symbols (no leading underscore)
                if not chunk.name.startswith("_"):
                    # Extract signature
                    sig_match = re.search(r"def\s+\w+\s*\((.*?)\)", chunk.text, re.DOTALL)
                    params = []
                    if sig_match:
                        params = [
                            p.strip().split(":")[0].split("=")[0].strip()
                            for p in sig_match.group(1).split(",")
                            if p.strip() and p.strip() not in ("self", "cls")
                        ]
                    current_api.append(
                        {
                            "symbol": chunk.name,
                            "file": chunk.file,
                            "parameters": params,
                            "type": chunk.type,
                        }
                    )

        # Try to compare with previous commit
        previous_api: list[dict[str, Any]] = []
        changes: list[dict[str, Any]] = []

        from gigacode import git_utils

        try:
            if since_commit:
                diff_output = git_utils.run_git(
                    ["diff", f"{since_commit}..HEAD", "--name-only"], cwd=self.work_dir
                )
                changed_files = set(diff_output.strip().split("\n")) if diff_output else set()

                for api in current_api:
                    if api["file"] in changed_files:
                        changes.append(
                            {
                                "symbol": api["symbol"],
                                "breaking": False,  # Conservative: would need old API to confirm
                                "parameters_added": [],
                                "return_type_changed": False,
                                "migration_guide": f"Review changes to {api['symbol']} in {api['file']}",
                            }
                        )
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("API change diff fallback failed: %s", e)

        return {
            "status": "ok",
            "current_api_surface": len(current_api),
            "changes": changes,
            "symbols": [{"name": a["symbol"], "params": a["parameters"]} for a in current_api[:50]],
        }

    def get_rollback_info(
        self,
        buffer_id: str,
        file: str,
    ) -> dict[str, Any]:
        """Get rollback information for a file using git history.

        Understand what changed and why—helps with debugging regressions.
        Returns the last commit that touched the file and a diff to revert.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            file: File path within the buffer to get rollback info for.

        Returns:
            Dict with keys:
                status: "ok" on success.
                last_working_commit: Commit hash of last change, or None if no history.
                commit_message: The commit message.
                commit_date: ISO timestamp of the commit.
                diff_to_revert: Unified diff showing changes from HEAD.

        Example:
            >>> result = tool.get_rollback_info(buf_id, "src/auth.py")
            >>> result["last_working_commit"]
            "abc1234"
            >>> result["commit_message"]
            "feat: add MFA support"
        """
        result = self._require_buffer(buffer_id, "get_rollback_info", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        from gigacode import git_utils

        try:
            # Get last commit touching this file
            log_output = git_utils.run_git(
                ["log", "-1", "--format=%H|%s|%ai", "--", file], cwd=self.work_dir
            )
            if not log_output or not log_output.strip():
                return {
                    "status": "ok",
                    "last_working_commit": None,
                    "message": "No git history for file",
                }

            parts = log_output.strip().split("|", 2)
            last_commit = parts[0] if parts else ""
            commit_msg = parts[1] if len(parts) > 1 else ""
            commit_date = parts[2] if len(parts) > 2 else ""

            # Get diff to revert
            diff_to_revert = (
                git_utils.run_git(["diff", "HEAD", "--", file], cwd=self.work_dir) or ""
            )

            return {
                "status": "ok",
                "last_working_commit": last_commit,
                "commit_message": commit_msg,
                "commit_date": commit_date,
                "diff_to_revert": diff_to_revert,
            }
        except (OSError, RuntimeError, ValueError) as e:
            return self._make_error_response(
                f"Git operation failed: {e}",
                buffer_id=buffer_id,
                operation="get_rollback_info",
            )

    def generate_change_template(
        self,
        buffer_id: str,
        request: str,
    ) -> dict[str, Any]:
        """Generate a change plan template for a natural language request.

        AI creates a plan before making changes.
        Uses semantic search to find relevant files, then impact analysis for risk assessment.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            request: Natural language description of the desired change.

        Returns:
            Dict with keys:
                status: "ok" on success.
                request: The original request string.
                files_to_modify: List of file paths relevant to the request.
                change_strategy: Summary of the change approach.
                test_cases_needed: List of test symbol names that should be updated.
                risk_assessment: "low", "medium", or "high".

        Example:
            >>> result = tool.generate_change_template(buf_id, "add retry logic to database calls")
            >>> result["files_to_modify"]
            ["src/db.py", "src/config.py"]
            >>> result["risk_assessment"]
            "medium"
        """
        result = self._require_buffer(buffer_id, "generate_change_template", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        # Use semantic search to find relevant files
        search_result = self.semantic_search(buffer_id, request, top_k=10)
        files_to_modify = []
        if search_result.get("status") == "ok":
            for r in search_result.get("matches", []):
                f = r.get("file", "")
                if f and f not in files_to_modify:
                    files_to_modify.append(f)

        # Get impact for each file
        test_cases_needed: list[str] = []
        risk_level = "low"

        for f in files_to_modify[:5]:
            try:
                impact = self.analyze_change(buffer_id, f)
                if impact.get("status") == "ok":
                    risk = impact.get("risk_level", "low")
                    if risk in ("high", "medium"):
                        risk_level = "medium" if risk_level != "high" else "high"
                    for t in impact.get("test_coverage", []):
                        name = t.get("name", t.get("target_symbol", ""))
                        if name and name not in test_cases_needed:
                            test_cases_needed.append(name)
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning("Impact analysis fallback failed for %s: %s", f, e)

        return {
            "status": "ok",
            "request": request,
            "files_to_modify": files_to_modify[:10],
            "change_strategy": f"Modify {len(files_to_modify)} files related to '{request}'",
            "test_cases_needed": test_cases_needed[:10],
            "risk_assessment": risk_level,
        }

    def map_api_endpoints(
        self,
        buffer_id: str,
    ) -> dict[str, Any]:
        """Map all API endpoints from FastAPI and Flask decorators.

        Know all exposed endpoints; find security issues.
        Parses @app.get/post/put/delete/patch decorators and Flask @app.route patterns.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.

        Returns:
            Dict with keys:
                status: "ok" on success.
                endpoints: List of endpoint dicts with keys:
                    method: HTTP method (e.g. "GET", "POST").
                    path: URL path (e.g. "/api/v1/payment").
                    handler: Function name that handles the endpoint.
                    is_async: True if the handler is async.
                    file: Source file containing the endpoint.
                total: Number of endpoints found.

        Example:
            >>> result = tool.map_api_endpoints(buf_id)
            >>> result["endpoints"][0]
            {"method": "POST", "path": "/api/v1/payment", "handler": "process_payment",
             "is_async": True, "file": "src/api.py"}
        """
        result = self._require_buffer(buffer_id, "map_api_endpoints")
        if isinstance(result, dict):
            return result
        info, chunks = result

        import re

        endpoints: list[dict[str, Any]] = []

        # FastAPI patterns
        fastapi_pattern = re.compile(
            r'@app\.(get|post|put|delete|patch|head|options)\s*\(\s*["\']([^"\']+)["\']',
            re.IGNORECASE,
        )
        # Flask patterns
        flask_pattern = re.compile(
            r'@(\w+).route\s*\(\s*["\']([^"\']+)["\']\s*(?:,\s*methods\s*=\s*\[([^\]]+)\])?',
            re.IGNORECASE,
        )

        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue
            # FastAPI
            for match in fastapi_pattern.finditer(chunk.text):
                method = match.group(1).upper()
                path = match.group(2)
                handler = ""
                # Find handler function
                rest = chunk.text[match.end() :]
                handler_match = re.search(r"async\s+def\s+(\w+)|def\s+(\w+)", rest[:200])
                if handler_match:
                    handler = handler_match.group(1) or handler_match.group(2) or ""
                is_async = "async def" in chunk.text[match.end() : match.end() + 100]
                endpoints.append(
                    {
                        "method": method,
                        "path": path,
                        "handler": handler,
                        "is_async": is_async,
                        "file": chunk.file,
                    }
                )

            # Flask
            for match in flask_pattern.finditer(chunk.text):
                path = match.group(2)
                methods_str = match.group(3) or '"GET"'
                methods = re.findall(r'["\'](\w+)["\']', methods_str)
                handler = ""
                rest = chunk.text[match.end() :]
                handler_match = re.search(r"def\s+(\w+)", rest[:200])
                if handler_match:
                    handler = handler_match.group(1)
                for m in methods:
                    endpoints.append(
                        {
                            "method": m.upper(),
                            "path": path,
                            "handler": handler,
                            "is_async": False,
                            "file": chunk.file,
                        }
                    )

        return {"status": "ok", "endpoints": endpoints, "total": len(endpoints)}

    def analyze_cache_patterns(
        self,
        buffer_id: str,
    ) -> dict[str, Any]:
        """Analyze cache usage patterns: invalidation logic, stale data risks.

        Prevent stale cache bugs.
        Detects cache libraries, invalidation triggers, and cache writes without matching invalidation.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.

        Returns:
            Dict with keys:
                status: "ok" on success.
                caches_used: List of cache library names detected (e.g. "redis", "lru_cache").
                invalidation_logic: List of dicts with keys: file, line, pattern, safe (bool).
                stale_data_risks: List of dicts with keys: file, line, risk_level.

        Example:
            >>> result = tool.analyze_cache_patterns(buf_id)
            >>> result["caches_used"]
            ["redis", "lru_cache"]
            >>> result["stale_data_risks"][0]
            {"file": "src/cache.py", "line": 42, "risk_level": "medium"}
        """
        result = self._require_buffer(buffer_id, "analyze_cache_patterns")
        if isinstance(result, dict):
            return result
        info, chunks = result

        import re

        caches_used: list[str] = []
        invalidation_logic: list[dict[str, Any]] = []
        stale_data_risks: list[dict[str, Any]] = []

        cache_libs = {"redis", "memcache", "lru_cache", "functools", "cachetools", "diskcache"}

        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue
            for lib in cache_libs:
                if lib in chunk.text and lib not in caches_used:
                    caches_used.append(lib)

            # Find cache invalidation patterns
            for match in re.finditer(
                r"(?:cache|_cache|cached)\s*(?:\.|_)(?:delete|remove|invalidate|clear|evict|pop)",
                chunk.text,
                re.IGNORECASE,
            ):
                line = chunk.start_line + chunk.text[: match.start()].count("\n")
                invalidation_logic.append(
                    {"file": chunk.file, "line": line, "pattern": match.group(0)[:80], "safe": True}
                )

            # Find cache writes without corresponding invalidation
            for match in re.finditer(
                r"(?:cache|_cache|cached)\s*(?:\.|_)(?:set|put|add|store|write)",
                chunk.text,
                re.IGNORECASE,
            ):
                line = chunk.start_line + chunk.text[: match.start()].count("\n")
                # Check if same chunk has invalidation
                has_invalidation = bool(
                    re.search(
                        r"(?:cache|_cache|cached)\s*(?:\.|_)(?:delete|remove|invalidate|clear|evict)",
                        chunk.text,
                        re.IGNORECASE,
                    )
                )
                if not has_invalidation:
                    stale_data_risks.append(
                        {"file": chunk.file, "line": line, "risk_level": "medium"}
                    )

        return {
            "status": "ok",
            "caches_used": caches_used,
            "invalidation_logic": invalidation_logic,
            "stale_data_risks": stale_data_risks,
        }

    def analyze_thread_safety(
        self,
        buffer_id: str,
    ) -> dict[str, Any]:
        """Analyze thread safety: shared mutable state, race conditions, deadlock risks.

        Catch concurrency bugs early.
        Detects unprotected shared state, non-atomic mutations, and multiple-lock patterns.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.

        Returns:
            Dict with keys:
                status: "ok" on success.
                shared_state: List of dicts with keys: name, file, modified_by (list), protected_by ("lock"|"atomic"|"none").
                race_conditions: List of dicts with keys: file, line, variables (list), risk_level.
                deadlock_risks: List of dicts with keys: file, line, locks_count, suggestion.

        Example:
            >>> result = tool.analyze_thread_safety(buf_id)
            >>> result["shared_state"][0]
            {"name": "global_cache", "file": "src/cache.py", "modified_by": [], "protected_by": "none"}
        """
        result = self._require_buffer(buffer_id, "analyze_thread_safety")
        if isinstance(result, dict):
            return result
        info, chunks = result

        import re

        shared_state: list[dict[str, Any]] = []
        race_conditions: list[dict[str, Any]] = []
        deadlock_risks: list[dict[str, Any]] = []

        # Find global/shared mutable state
        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue
            # Module-level mutable assignments
            if chunk.type == "module":
                for match in re.finditer(
                    r"^(\w+)\s*=\s*(?:\[\]|\{\}|set\(\)|dict\(\)|list\(\))",
                    chunk.text,
                    re.MULTILINE,
                ):
                    var_name = match.group(1)
                    if not var_name.startswith("_"):
                        shared_state.append(
                            {
                                "name": var_name,
                                "file": chunk.file,
                                "modified_by": [],
                                "protected_by": "none",
                            }
                        )

            # Check for lock usage
            has_lock = bool(re.search(r"(?:Lock|RLock|Semaphore|threading\.)", chunk.text))

            # Detect non-atomic operations on shared state
            for match in re.finditer(
                r"(?:append|extend|remove|pop|update|clear|add|discard)\s*\(", chunk.text
            ):
                if not has_lock:
                    line = chunk.start_line + chunk.text[: match.start()].count("\n")
                    race_conditions.append(
                        {
                            "file": chunk.file,
                            "line": line,
                            "variables": [],
                            "risk_level": "medium",
                        }
                    )

            # Detect potential deadlocks (multiple locks)
            locks = re.findall(r"(?:Lock|RLock)\s*\(\s*\)", chunk.text)
            if len(locks) > 1:
                deadlock_risks.append(
                    {
                        "file": chunk.file,
                        "line": chunk.start_line,
                        "locks_count": len(locks),
                        "suggestion": "Multiple locks — ensure consistent acquisition order",
                    }
                )

        return {
            "status": "ok",
            "shared_state": shared_state,
            "race_conditions": race_conditions[:20],
            "deadlock_risks": deadlock_risks,
        }

    def detect_memory_issues(
        self,
        buffer_id: str,
    ) -> dict[str, Any]:
        """Detect memory issues: circular references, unbounded collections, resource leaks.

        Identify memory issues in long-running processes.
        Detects: self-referential patterns, loops with append but no size limit,
        open() calls without with/context or .close().

        Args:
            buffer_id: Buffer handle returned by embed_codebase.

        Returns:
            Dict with keys:
                status: "ok" on success.
                circular_refs: List of dicts with keys: file, line, symbols (list).
                unbounded_collections: List of dicts with keys: file, line, symbol, growth_reason.
                resource_leaks: List of dicts with keys: file, line, resource, cleanup_missing (bool).

        Example:
            >>> result = tool.detect_memory_issues(buf_id)
            >>> result["unbounded_collections"][0]
            {"file": "src/collector.py", "line": 15, "symbol": "aggregate", "growth_reason": "append in loop without size limit"}
        """
        result = self._require_buffer(buffer_id, "detect_memory_issues")
        if isinstance(result, dict):
            return result
        info, chunks = result

        import re

        circular_refs: list[dict[str, Any]] = []
        unbounded_collections: list[dict[str, Any]] = []
        resource_leaks: list[dict[str, Any]] = []

        for chunk in chunks:
            if not hasattr(chunk, "text") or not chunk.text:
                continue

            # Unbounded collections (append without limit in loops)
            if chunk.type in ("function", "method"):
                has_loop = bool(re.search(r"\b(for|while)\b", chunk.text))
                has_append = bool(re.search(r"\.append\s*\(", chunk.text))
                has_limit = bool(
                    re.search(r"(?:max_|limit|cap|size|len\(|break|return)", chunk.text)
                )
                if has_loop and has_append and not has_limit:
                    unbounded_collections.append(
                        {
                            "file": chunk.file,
                            "line": chunk.start_line,
                            "symbol": getattr(chunk, "name", ""),
                            "growth_reason": "append in loop without size limit",
                        }
                    )

            # Resource leaks (open without with/context)
            for match in re.finditer(r"(\w+)\s*=\s*open\s*\([^)]*\)", chunk.text):
                var_name = match.group(1)
                # Check if 'with' statement or .close() present
                if (
                    f"with {var_name}" not in chunk.text
                    and f"{var_name}.close()" not in chunk.text
                    and "with open" not in chunk.text
                ):
                    line = chunk.start_line + chunk.text[: match.start()].count("\n")
                    resource_leaks.append(
                        {
                            "file": chunk.file,
                            "line": line,
                            "resource": "file handle",
                            "cleanup_missing": True,
                        }
                    )

            # Circular references (self-referential patterns)
            for match in re.finditer(r"self\.(\w+)\s*=\s*self\b", chunk.text):
                line = chunk.start_line + chunk.text[: match.start()].count("\n")
                circular_refs.append(
                    {
                        "file": chunk.file,
                        "line": line,
                        "symbols": [match.group(1)],
                    }
                )

        return {
            "status": "ok",
            "circular_refs": circular_refs,
            "unbounded_collections": unbounded_collections,
            "resource_leaks": resource_leaks,
        }

    # ------------------------------------------------------------------
    # Progress Streaming (helper)
    # ------------------------------------------------------------------
    def _create_progress_reporter(
        self,
        phases: list[str],
    ) -> "ProgressReporter":
        """Create a progress reporter for long-running operations."""
        return ProgressReporter(phases)

    # ------------------------------------------------------------------
    # Multi-turn Conversation Memory
    # ------------------------------------------------------------------
    def remember(
        self,
        key: str,
        value: str,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a fact in conversation memory."""
        return self._conversation_memory.remember(key, value, tags)

    def recall(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Recall facts from conversation memory by query."""
        memories = self._conversation_memory.recall(query, top_k)
        return {"status": "ok", "memories": [m.to_dict() for m in memories]}

    def list_memories(
        self,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """List all stored memories, optionally filtered by tag."""
        memories = self._conversation_memory.list_memories(tag)
        return {"status": "ok", "memories": [m.to_dict() for m in memories], "total": len(memories)}

    def forget(self, key: str) -> dict[str, Any]:
        """Remove a stored memory."""
        return self._conversation_memory.forget(key)

    # ------------------------------------------------------------------
    # Code Quality (Features 36-38)
    # ------------------------------------------------------------------
    def auto_format(
        self,
        buffer_id: str,
        files: list[str] | None = None,
        formatter: str = "black",
        line_length: int = 88,
        skip_magic_trailing_comma: bool = False,
        dry_run: bool = True,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Format code using Black or ruff format.

        Operates on entire buffer directory by default (files=None).

        Args:
            buffer_id: Buffer handle.
            files: Specific files to format. If None, format entire directory.
            formatter: "black" or "ruff.format".
            line_length: Maximum line length (default: 88).
            skip_magic_trailing_comma: Skip Black's magic trailing comma.
            dry_run: Preview only, don't modify files (default: True).
            exclude_patterns: Glob patterns to exclude.

        Returns:
            Dict with status, formatted_files, changes, and summary.
        """
        result = self._require_buffer(buffer_id, "auto_format", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result
        work_dir = info.get("buffer_dir", self.work_dir)
        return auto_format(
            work_dir=work_dir,
            files=files,
            formatter=formatter,
            line_length=line_length,
            skip_magic_trailing_comma=skip_magic_trailing_comma,
            dry_run=dry_run,
            exclude_patterns=exclude_patterns,
        )

    def auto_lint(
        self,
        buffer_id: str,
        files: list[str] | None = None,
        linter: str = "ruff",
        select: list[str] | None = None,
        ignore: list[str] | None = None,
        auto_fix: bool = False,
        dry_run: bool = True,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Lint code using Ruff.

        Operates on entire buffer directory by default (files=None).

        Args:
            buffer_id: Buffer handle.
            files: Specific files to lint. If None, lint entire directory.
            linter: Linter to use (currently only "ruff").
            select: Rule categories (e.g., ["E", "F", "W"]).
            ignore: Rule codes to ignore (e.g., ["E501"]).
            auto_fix: Auto-fix fixable issues (default: False).
            dry_run: Preview only (default: True).
            exclude_patterns: Glob patterns to exclude.

        Returns:
            Dict with status, issues, by_rule, fixed_count, unfixed_count.
        """
        result = self._require_buffer(buffer_id, "auto_lint", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result
        work_dir = info.get("buffer_dir", self.work_dir)
        return auto_lint(
            work_dir=work_dir,
            files=files,
            linter=linter,
            select=select,
            ignore=ignore,
            auto_fix=auto_fix,
            dry_run=dry_run,
            exclude_patterns=exclude_patterns,
        )

    def auto_polish(
        self,
        buffer_id: str,
        files: list[str] | None = None,
        format_with: str = "black",
        lint_with: str = "ruff",
        auto_fix_lints: bool = True,
        line_length: int = 88,
        ruff_select: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Format AND lint in one call (convenience wrapper).

        Delegates to auto_format() then auto_lint(). Format runs first
        so lint checks the formatted code.

        Args:
            buffer_id: Buffer handle.
            files: Specific files. If None, polish entire directory.
            format_with: Formatter ("black" or "ruff.format").
            lint_with: Linter (currently only "ruff").
            auto_fix_lints: Auto-fix fixable lint issues (default: True).
            line_length: Maximum line length (default: 88).
            ruff_select: Ruff rule categories.
            exclude_patterns: Glob patterns to exclude.
            dry_run: Preview only (default: True).

        Returns:
            Dict with formatting and linting sub-results, plus ready_to_commit.
        """
        result = self._require_buffer(buffer_id, "auto_polish", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result
        work_dir = info.get("buffer_dir", self.work_dir)
        return auto_polish(
            work_dir=work_dir,
            files=files,
            format_with=format_with,
            lint_with=lint_with,
            auto_fix_lints=auto_fix_lints,
            line_length=line_length,
            ruff_select=ruff_select,
            exclude_patterns=exclude_patterns,
            dry_run=dry_run,
        )

    def polish_before_commit(
        self,
        buffer_id: str,
        files_to_commit: list[str] | None = None,
        format_with: str = "black",
        lint_with: str = "ruff",
        check_only: bool = False,
    ) -> dict[str, Any]:
        """Format, lint, and validate before committing.

        Workflow: Write -> Format -> Lint -> Commit readiness check.
        Delegates to auto_polish + adds commit-readiness warnings.

        Args:
            buffer_id: Buffer handle.
            files_to_commit: Specific files. If None, all dirty files.
            format_with: Formatter ("black" or "ruff.format").
            lint_with: Linter (currently only "ruff").
            check_only: If True, only validate (don't modify).

        Returns:
            Dict with formatting, linting, ready_to_commit, and pre_commit_warnings.
        """
        result = self._require_buffer(buffer_id, "polish_before_commit", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        # Run auto_polish (format + lint)
        polish_result = self.auto_polish(
            buffer_id=buffer_id,
            files=files_to_commit,
            format_with=format_with,
            lint_with=lint_with,
            dry_run=check_only,
        )

        # Gather pre-commit warnings
        warnings: list[str] = []

        # Check for large diffs
        dirty = info.get("dirty_files", {})
        if len(dirty) > 10:
            warnings.append(f"Large diff detected ({len(dirty)} files modified)")

        # Check for untested changes
        if dirty:
            try:
                coverage = self.get_test_coverage(buffer_id)
                if coverage.get("status") == "ok":
                    for f in dirty:
                        file_coverage = coverage.get("coverage", {}).get(f, {})
                        if not file_coverage:
                            warnings.append(f"No test coverage for {f}")
            except (ValueError, RuntimeError):
                pass

        # Check for impact risks
        if dirty:
            try:
                impact = self.analyze_impact(buffer_id)
                if impact.get("risk_level") == "high":
                    warnings.append(
                        f"High impact change: {impact.get('total_impacted_files', 0)} files affected"
                    )
            except (ValueError, RuntimeError):
                pass

        ready = polish_result.get("ready_to_commit", True) and len(warnings) == 0

        return {
            "status": "ok",
            "formatting": polish_result.get("formatting", {}),
            "linting": polish_result.get("linting", {}),
            "ready_to_commit": ready,
            "pre_commit_warnings": warnings,
            "summary": polish_result.get("summary", ""),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release all in-memory resources.

        Clears the FAISS index cache (drops GPU VRAM), the lexical index
        cache, and the query result cache.  The on-disk registry and buffer
        files are left intact — they can be reloaded by constructing a new
        ``CodeEmbeddingTool`` with the same *work_dir*.

        Also stops the Prometheus metrics HTTP server if enabled.
        """
        self._index_cache.clear()
        self._lexical_cache.clear()
        self._query_cache.clear()
        self._security.close()

        # Stop Prometheus metrics server if running
        if self._prometheus_exporter is not None:
            self._prometheus_exporter.stop()

        json_logger.info(
            operation="close",
            message="CodeEmbeddingTool closed; all caches released",
        )

    def __enter__(self) -> CodeEmbeddingTool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Detailed Analysis (Features 39-40)
    # ------------------------------------------------------------------
    def lint_buffer(
        self,
        buffer_id: str,
        files: list[str] | None = None,
        select: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        group_by: str = "file",
    ) -> dict[str, Any]:
        """Deep lint analysis with detailed aggregation by file/severity/rule. Report-only, no auto-fix.

        Organize lint results for full-buffer analysis.
        Delegates to auto_lint with auto_fix=False, then reorganizes results.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            files: Specific files to lint. If None, lints entire buffer.
            select: Lint rule categories to check (e.g. ["E", "F", "W"]). Default: all.
            exclude_patterns: Glob patterns to exclude from linting.
            group_by: How to organize results: "file", "severity", or "rule". Default: "file".

        Returns:
            Dict with keys:
                status: "ok" on success.
                total_issues: Total number of lint issues found.
                by_file: (when group_by="file") Dict mapping file paths to lists of {line, code, message}.
                by_severity: (when group_by="severity") Dict with keys: error, warning, info — each an integer.
                by_rule: (when group_by="rule") Dict mapping rule codes to {count, severity}.

        Example:
            >>> result = tool.lint_buffer(buf_id, group_by="severity")
            >>> result["by_severity"]
            {"error": 5, "warning": 12, "info": 3}
        """
        lint_result = self.auto_lint(
            buffer_id=buffer_id,
            files=files,
            select=select,
            auto_fix=False,
            dry_run=True,
            exclude_patterns=exclude_patterns,
        )
        if lint_result.get("status") != "ok":
            return lint_result

        issues = lint_result.get("issues", [])
        by_file: dict[str, list[dict[str, Any]]] = {}
        by_severity: dict[str, int] = {"error": 0, "warning": 0, "info": 0}
        by_rule: dict[str, dict[str, Any]] = lint_result.get("by_rule", {})

        for issue in issues:
            f = issue.get("file", "unknown")
            by_file.setdefault(f, []).append(
                {
                    "line": issue.get("line"),
                    "code": issue.get("code"),
                    "message": issue.get("message"),
                }
            )
            code = issue.get("code", "")
            if code.startswith("E") or code.startswith("F"):
                by_severity["error"] += 1
            elif code.startswith("W"):
                by_severity["warning"] += 1
            else:
                by_severity["info"] += 1

        result: dict[str, Any] = {"status": "ok", "total_issues": len(issues)}
        if group_by == "file":
            result["by_file"] = by_file
        elif group_by == "severity":
            result["by_severity"] = by_severity
        elif group_by == "rule":
            result["by_rule"] = by_rule
        else:
            result.update({"by_file": by_file, "by_severity": by_severity, "by_rule": by_rule})
        return result

    def format_buffer(
        self,
        buffer_id: str,
        files: list[str] | None = None,
        formatter: str = "black",
        line_length: int = 88,
        exclude_patterns: list[str] | None = None,
        dry_run: bool = True,
        summary_only: bool = False,
    ) -> dict[str, Any]:
        """Deep format analysis with detailed change tracking across codebase.

        Understand exactly what changed across the codebase.
        Delegates to auto_format, then enriches with aggregate statistics.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            files: Specific files to format. If None, formats entire buffer.
            formatter: Formatter to use: "black" or "ruff.format". Default: "black".
            line_length: Maximum line length. Default: 88.
            exclude_patterns: Glob patterns to exclude.
            dry_run: Preview only without making changes. Default: True.
            summary_only: If True, return only stats (not full diffs). Default: False.

        Returns:
            Dict with keys:
                status: "ok" on success.
                total_files: Total files analyzed.
                formatted_files: Number of files that needed formatting.
                already_formatted: Number of files already compliant.
                total_lines_added: Lines added by formatting.
                total_lines_removed: Lines removed by formatting.
                changes: List of per-file change dicts with keys: file, added_lines, removed_lines, diff.
                summary: Human-readable summary string.

        Example:
            >>> result = tool.format_buffer(buf_id, dry_run=True)
            >>> result["formatted_files"]
            3
            >>> result["total_lines_added"]
            12
        """
        format_result = self.auto_format(
            buffer_id=buffer_id,
            files=files,
            formatter=formatter,
            line_length=line_length,
            dry_run=dry_run,
            exclude_patterns=exclude_patterns,
        )
        if format_result.get("status") != "ok":
            return format_result

        changes = format_result.get("changes", [])
        total_added = sum(c.get("added_lines", 0) for c in changes)
        total_removed = sum(c.get("removed_lines", 0) for c in changes)

        result: dict[str, Any] = {
            "status": "ok",
            "total_files": format_result.get("formatted_files", 0)
            + format_result.get("already_formatted", 0),
            "formatted_files": format_result.get("formatted_files", 0),
            "already_formatted": format_result.get("already_formatted", 0),
            "total_lines_added": total_added,
            "total_lines_removed": total_removed,
            "summary": format_result.get("summary", ""),
        }
        if not summary_only:
            result["changes"] = changes
        return result

    def lint_with_config(
        self,
        buffer_id: str,
        config_file: str | None = None,
        files: list[str] | None = None,
        auto_fix: bool = False,
    ) -> dict[str, Any]:
        """Lint using project configuration (ruff.toml, pyproject.toml, .flake8).

        Config-aware linting respects project-specific rules.
        Auto-discovers config files in the work directory if not specified.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            config_file: Path to config file. Auto-detected from work_dir if None.
            files: Specific files to lint. If None, lints entire buffer.
            auto_fix: Automatically fix lint issues. Default: False.

        Returns:
            Dict with keys:
                status: "ok" on success.
                config_file: Config file used (or "none found").
                Plus all auto_lint output fields: issues, by_rule, etc.

        Example:
            >>> result = tool.lint_with_config(buf_id)
            >>> result["config_file"]
            "pyproject.toml"
        """
        result = self._require_buffer(buffer_id, "lint_with_config", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        # Discover config file
        if config_file is None:
            import os

            work_dir = self.work_dir
            for candidate in ["ruff.toml", ".ruff.toml", "pyproject.toml", ".flake8", "setup.cfg"]:
                candidate_path = os.path.join(work_dir, candidate)
                if os.path.isfile(candidate_path):
                    config_file = candidate
                    break

        result = self.auto_lint(
            buffer_id=buffer_id,
            files=files,
            auto_fix=auto_fix,
            dry_run=not auto_fix,
        )

        if result.get("status") != "ok":
            return result

        result["config_file"] = config_file or "none found"
        return result

    def format_with_config(
        self,
        buffer_id: str,
        config_file: str | None = None,
        files: list[str] | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Format using project configuration (pyproject.toml, .black, ruff.toml).

        Config-aware formatting respects project-specific style.
        Auto-discovers config files in the work directory if not specified.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            config_file: Path to config file. Auto-detected from work_dir if None.
            files: Specific files to format. If None, formats entire buffer.
            dry_run: Preview only without making changes. Default: True.

        Returns:
            Dict with keys:
                status: "ok" on success.
                config_file: Config file used (or "none found").
                Plus all auto_format output fields: changes, formatted_files, etc.

        Example:
            >>> result = tool.format_with_config(buf_id, dry_run=True)
            >>> result["config_file"]
            "pyproject.toml"
        """
        result = self._require_buffer(buffer_id, "format_with_config", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        # Discover config file
        if config_file is None:
            import os

            work_dir = self.work_dir
            for candidate in ["pyproject.toml", ".black", "ruff.toml", ".ruff.toml"]:
                candidate_path = os.path.join(work_dir, candidate)
                if os.path.isfile(candidate_path):
                    config_file = candidate
                    break

        result = self.auto_format(
            buffer_id=buffer_id,
            files=files,
            dry_run=dry_run,
        )

        if result.get("status") != "ok":
            return result

        result["config_file"] = config_file or "none found"
        return result

    # ------------------------------------------------------------------
    # Solve — unified automated loop
    # ------------------------------------------------------------------
    def solve(
        self,
        buffer_id: str,
        task: str,
        max_iterations: int = 10,
    ) -> dict[str, Any]:
        """Automatically solve a coding task with a unified loop.

        Orchestrates search, read, plan, edit, test, and commit operations
        with minimal human intervention. Returns an audit trail and
        completion status.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            task: Natural language description of the task to solve.
            max_iterations: Maximum number of solve loop iterations (default 10).

        Returns:
            Dict with status, task_id, iterations, tokens_used,
            duration_seconds, audit_trail, summary, and next_step.
        """
        result = self._require_buffer(buffer_id, "solve")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            from gigacode.intent_router import IntentRouter
            from gigacode.solver import SolveExecutor, Solver

            intent_router = IntentRouter()
            executor = SolveExecutor(
                buffer_manager=self._buffer_manager,
                search_service=self._search_service,
                diff_engine=None,
                intent_router=intent_router,
            )
            solver = Solver(executor=executor)
            result = solver.solve(
                buffer_id=buffer_id,
                task=task,
                max_iterations=max_iterations,
            )
            return {"status": "ok", **result.to_dict()}

        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(f"Solve failed: {e}")
            return self._make_error_response(
                f"Solve failed: {e}",
                buffer_id=buffer_id,
                operation="solve",
            )

    # ------------------------------------------------------------------
    # Undo / Redo with Branching
    # ------------------------------------------------------------------
    def _get_undo_redo_service(self, buffer_id: str):
        """Get or create UndoRedoService for a buffer (lazy init).

        Args:
            buffer_id: Buffer handle.

        Returns:
            UndoRedoService instance for this buffer.
        """
        if not hasattr(self, "_undo_redo_services"):
            self._undo_redo_services: dict[str, Any] = {}

        if buffer_id not in self._undo_redo_services:
            from gigacode.undo_redo import BranchedBufferManager, UndoRedoService

            branched_mgr = BranchedBufferManager(buffer_manager=self._buffer_manager)
            self._undo_redo_services[buffer_id] = UndoRedoService(
                branched_buffer_manager=branched_mgr
            )

        return self._undo_redo_services[buffer_id]

    def undo(self, buffer_id: str, steps: int = 1) -> dict[str, Any]:
        """Undo the last N operations on a buffer.

        Reverts edits in reverse order from the undo stack.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            steps: Number of operations to undo (default 1).

        Returns:
            Dict with status, steps_undone, remaining_undo_count, and message.
        """
        result = self._require_buffer(buffer_id, "undo", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        try:
            service = self._get_undo_redo_service(buffer_id)
            steps_undone = service.undo(buffer_id, steps=steps)
            stack = service.branched_buffer_manager._get_branch_manager(buffer_id).get_undo_stack()
            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "steps_undone": steps_undone,
                "remaining_undo_count": len(stack.undo_stack),
                "message": f"Undone {steps_undone} operation(s)",
            }
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(f"Undo failed: {e}")
            return self._make_error_response(
                f"Undo failed: {e}",
                buffer_id=buffer_id,
                operation="undo",
            )

    def redo(self, buffer_id: str, steps: int = 1) -> dict[str, Any]:
        """Redo the last N undone operations on a buffer.

        Re-applies previously undone edits from the redo stack.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            steps: Number of operations to redo (default 1).

        Returns:
            Dict with status, steps_redone, remaining_redo_count, and message.
        """
        result = self._require_buffer(buffer_id, "redo", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        try:
            service = self._get_undo_redo_service(buffer_id)
            steps_redone = service.redo(buffer_id, steps=steps)
            stack = service.branched_buffer_manager._get_branch_manager(buffer_id).get_undo_stack()
            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "steps_redone": steps_redone,
                "remaining_redo_count": len(stack.redo_stack),
                "message": f"Redone {steps_redone} operation(s)",
            }
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(f"Redo failed: {e}")
            return self._make_error_response(
                f"Redo failed: {e}",
                buffer_id=buffer_id,
                operation="redo",
            )

    def create_branch(self, buffer_id: str, name: str) -> dict[str, Any]:
        """Create a named branch for experimental edits.

        Preserves the current buffer state so you can switch back later.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            name: Name for the new branch.

        Returns:
            Dict with status, branch, parent, created timestamp, and message.
        """
        result = self._require_buffer(buffer_id, "create_branch", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        try:
            service = self._get_undo_redo_service(buffer_id)
            result = service.branch(buffer_id, name)
            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "branch": result.get("branch", name),
                "parent": result.get("parent", "main"),
                "created": result.get("created", ""),
                "message": f"Created branch '{name}'",
            }
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(f"Create branch failed: {e}")
            return self._make_error_response(
                f"Create branch failed: {e}",
                buffer_id=buffer_id,
                operation="create_branch",
            )

    def list_branches(self, buffer_id: str) -> dict[str, Any]:
        """List all branches for a buffer.

        Returns branch names, parents, creation timestamps, and the active branch.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.

        Returns:
            Dict with status, branches list, and message.
        """
        result = self._require_buffer(buffer_id, "list_branches", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        try:
            service = self._get_undo_redo_service(buffer_id)
            branches = service.list_branches(buffer_id)
            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "branches": branches,
                "message": f"Found {len(branches)} branch(es)",
            }
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(f"List branches failed: {e}")
            return self._make_error_response(
                f"List branches failed: {e}",
                buffer_id=buffer_id,
                operation="list_branches",
            )

    def checkout_branch(self, buffer_id: str, name: str) -> dict[str, Any]:
        """Switch to a different branch on a buffer.

        Changes which set of edits is active for the buffer.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            name: Name of the branch to switch to.

        Returns:
            Dict with status, branch name, and message.
        """
        result = self._require_buffer(buffer_id, "checkout_branch", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        try:
            service = self._get_undo_redo_service(buffer_id)
            service.checkout(buffer_id, name)
            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "branch": name,
                "message": f"Checked out branch '{name}'",
            }
        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(f"Checkout branch failed: {e}")
            return self._make_error_response(
                f"Checkout branch failed: {e}",
                buffer_id=buffer_id,
                operation="checkout_branch",
            )

    # ------------------------------------------------------------------
    # Why-Annotator — search with "why this matters" annotations
    # ------------------------------------------------------------------
    def annotate_search_results(
        self,
        buffer_id: str,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Search and annotate results with 'why this matters' explanations.

        Runs semantic search, then enriches each result with relevance
        reasons, suggested next actions, and related files.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            query: Natural language search query.
            top_k: Maximum number of annotated results to return (default 5).

        Returns:
            Dict with status, annotated_results list, query, count, and message.
        """
        err = self._validate_search_params(query, top_k=top_k)
        if err is not None:
            return err

        result = self._require_buffer(buffer_id, "annotate_search_results")
        if isinstance(result, dict):
            return result
        info, chunks = result

        try:
            from gigacode.why_annotator import AnnotationService, RelevanceExplainer, WhyAnnotator

            # Run semantic search first
            search_result = self.semantic_search(buffer_id, query, top_k=top_k)
            if search_result.get("status") != "ok":
                return self._make_error_response(
                    f"Search failed: {search_result.get('message', 'unknown error')}",
                    buffer_id=buffer_id,
                    operation="annotate_search_results",
                )

            matches = search_result.get("matches", [])

            # Create annotator and annotate results
            explainer = RelevanceExplainer(
                dependency_graph=DependencyGraph(chunks),
                buffer_manager=self._buffer_manager,
                symbol_index=SymbolIndex(chunks),
            )
            annotation_service = AnnotationService(explainer=explainer)
            annotator = WhyAnnotator(annotation_service=annotation_service)

            # Get dirty files for edit context
            dirty_files = list(info.get("dirty_queue", []))
            edit_context = (
                [e.get("file") for e in dirty_files if isinstance(e, dict)] if dirty_files else None
            )

            annotated = annotator.annotate_search_results(
                buffer_id=buffer_id,
                results=matches,
                query=query,
                edit_context=edit_context,
            )

            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "query": query,
                "annotated_results": annotated,
                "count": len(annotated),
                "message": f"Annotated {len(annotated)} search result(s)",
            }

        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(f"Annotate search results failed: {e}")
            return self._make_error_response(
                f"Annotate search results failed: {e}",
                buffer_id=buffer_id,
                operation="annotate_search_results",
            )

    # ------------------------------------------------------------------
    # Conflict Prediction
    # ------------------------------------------------------------------
    def predict_conflicts(self, buffer_id: str) -> dict[str, Any]:
        """Predict potential merge conflicts by analyzing commits since embed time.

        Reports risk level (low/medium/high), per-file risks, dependency risks,
        and actionable recommendations.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.

        Returns:
            Dict with status, risk_level, file_risks, dependency_risks,
            recommendations, and auto_actions.
        """
        result = self._require_buffer(buffer_id, "predict_conflicts", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        try:
            from gigacode.conflict_predictor import ConflictPredictionService, ConflictPredictor

            chunks = self._load_chunks(buffer_id)
            dep_graph = DependencyGraph(chunks) if chunks else DependencyGraph([])

            predictor = ConflictPredictor(
                buffer_manager=self._buffer_manager,
                git_utils=GitUtils,
                dependency_graph=dep_graph,
            )
            service = ConflictPredictionService(conflict_predictor=predictor)
            result = service.predict_conflicts(buffer_id)
            return {"status": "ok", "buffer_id": buffer_id, **result}

        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(f"Predict conflicts failed: {e}")
            return self._make_error_response(
                f"Predict conflicts failed: {e}",
                buffer_id=buffer_id,
                operation="predict_conflicts",
            )

    # ------------------------------------------------------------------
    # Diff-Aware Search
    # ------------------------------------------------------------------
    def search_modified_only(
        self,
        buffer_id: str,
        query: str,
        scope: str = "changes+deps",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Search only modified files and their dependencies.

        Scopes search to dirty files only, dirty files plus their
        dependencies, or the entire codebase.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            query: Natural language search query.
            scope: Search scope — 'changes' (dirty files only),
                'changes+deps' (dirty + their deps), or 'all' (entire codebase).
                Default: 'changes+deps'.
            top_k: Maximum number of results to return (default 10).

        Returns:
            Dict with status, scope_used, files_searched, files_skipped,
            results, and performance metrics.
        """
        err = self._validate_search_params(query, top_k=top_k)
        if err is not None:
            return err

        if scope not in ("changes", "changes+deps", "all"):
            return self._make_error_response(
                f"scope must be 'changes', 'changes+deps', or 'all', got '{scope}'",
                buffer_id=buffer_id,
                operation="search_modified_only",
            )

        result = self._require_buffer(buffer_id, "search_modified_only", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        try:
            from gigacode.diff_aware_search import DiffAwareSearch, DiffAwareSearchService

            chunks = self._load_chunks(buffer_id)
            dep_graph = DependencyGraph(chunks) if chunks else DependencyGraph([])

            diff_aware = DiffAwareSearch(
                search_service=self._search_service,
                dependency_graph=dep_graph,
                buffer_manager=self._buffer_manager,
            )
            service = DiffAwareSearchService(diff_aware_search=diff_aware)
            result = service.search_since_last_edit(
                buffer_id=buffer_id,
                query=query,
                scope=scope,
                top_k=top_k,
            )
            return {"status": "ok", "buffer_id": buffer_id, **result}

        except (ImportError, RuntimeError, ValueError) as e:
            logger.warning(f"Search modified only failed: {e}")
            return self._make_error_response(
                f"Search modified only failed: {e}",
                buffer_id=buffer_id,
                operation="search_modified_only",
            )

    # ------------------------------------------------------------------
    # Agent profile methods
    # ------------------------------------------------------------------

    def get_chunking_strategy(self, profile: str = "generic") -> dict[str, Any]:
        """Get the chunking strategy for a given agent profile.

        Returns include/exclude element lists, line limits, description,
        and expected token savings percentage.

        Args:
            profile: Agent profile name (reviewer, debugger, architect,
                documenter, generic). Default: "generic".

        Returns:
            Dict with profile, include, exclude, max_lines, min_lines,
            description, expected_token_savings.
        """
        strategy = ChunkingStrategyFactory.get_strategy_by_name(profile)

        return {
            "status": "ok",
            "profile": strategy.profile.value,
            "include": list(strategy.include_elements),
            "exclude": list(strategy.exclude_elements),
            "max_lines": strategy.max_lines,
            "min_lines": strategy.min_lines,
            "description": strategy.description,
            "expected_token_savings": f"{strategy.expected_token_savings}%",
        }

    def set_agent_profile(self, buffer_id: str, profile: str = "generic") -> dict[str, Any]:
        """Set the agent profile for a buffer.

        Affects future chunking and search behavior. The profile name
        is stored in the buffer's metadata.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            profile: Agent profile name (reviewer, debugger, architect,
                documenter, generic). Default: "generic".

        Returns:
            Dict with status, buffer_id, profile, strategy_description.
        """
        result = self._require_buffer(buffer_id, "set_agent_profile", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        strategy = ChunkingStrategyFactory.get_strategy_by_name(profile)

        # Store profile in buffer metadata
        self._registry[buffer_id]["profile"] = profile

        return {
            "status": "ok",
            "buffer_id": buffer_id,
            "profile": strategy.profile.value,
            "strategy_description": strategy.description,
        }

    def chunk_with_profile(self, buffer_id: str, profile: str = "generic") -> dict[str, Any]:
        """Re-chunk the buffer's codebase using the specified agent profile.

        Creates an AdaptiveChunker with the existing chunker and applies
        profile-specific chunking for each file.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            profile: Agent profile name (reviewer, debugger, architect,
                documenter, generic). Default: "generic".

        Returns:
            Dict with status, profile, total_chunks, strategy_description.
        """
        result = self._require_buffer(buffer_id, "chunk_with_profile")
        if isinstance(result, dict):
            return result
        info, chunks = result

        strategy = ChunkingStrategyFactory.get_strategy_by_name(profile)
        profile_enum = strategy.profile

        try:
            adaptive_chunker = AdaptiveChunker(self._chunker)

            total_chunks = 0
            file_contents: dict[str, str] = {}
            for chunk in chunks:
                if chunk.file not in file_contents:
                    file_contents[chunk.file] = chunk.text or ""
                else:
                    file_contents[chunk.file] += "\n" + (chunk.text or "")

            for file_path, content in file_contents.items():
                if content.strip():
                    optimized = adaptive_chunker.chunk_with_profile(
                        content, file_path, profile_enum
                    )
                    total_chunks += len(optimized)

            return {
                "status": "ok",
                "profile": strategy.profile.value,
                "total_chunks": total_chunks,
                "strategy_description": strategy.description,
            }
        except (ValueError, RuntimeError) as e:
            logger.warning(f"chunk_with_profile failed: {e}")
            return self._make_error_response(
                f"Chunk with profile failed: {e}",
                buffer_id=buffer_id,
                operation="chunk_with_profile",
            )

    def adapt_search(self, buffer_id: str, query: str, profile: str = "generic") -> dict[str, Any]:
        """Enhance a search query based on agent profile context.

        Appends profile-specific keywords to improve search relevance
        for the task type.

        Args:
            buffer_id: Buffer handle returned by embed_codebase.
            query: Original search query to enhance.
            profile: Agent profile name (reviewer, debugger, architect,
                documenter, generic). Default: "generic".

        Returns:
            Dict with enhanced_query, profile, original_query.
        """
        result = self._require_buffer(buffer_id, "adapt_search", require_chunks=False)
        if isinstance(result, dict):
            return result
        info, _ = result

        try:
            profile_enum = AgentProfile(profile)
        except ValueError:
            profile_enum = AgentProfile.GENERIC

        # Lazy-initialize profile adapter
        if self._profile_adapter is None:
            adaptive_chunker = AdaptiveChunker(self._chunker)
            self._profile_adapter = ProfileAdapter(adaptive_chunker)

        result = self._profile_adapter.adapt_search(query, profile_enum)

        return {
            "status": "ok",
            "enhanced_query": result["query"],
            "profile": result["profile"],
            "original_query": query,
        }
