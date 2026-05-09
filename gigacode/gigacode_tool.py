"""Agent-facing tool interface for GPU-accelerated code embedding.

Chunks code at AST boundaries, keeps a persistent FAISS index in GPU
memory when available, and exposes a read-write-commit workflow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from gigacode.buffer_state import BufferState, BufferStateTransition
from gigacode.chunker import CodeChunk, chunk_file, chunk_text
from gigacode.context_packer import pack_context
from gigacode.diff_engine import compute_diff, hash_lines
from gigacode.duplicate_detector import find_duplicates
from gigacode.embedder import Embedder
from gigacode.gpu_index import GpuIndex
from gigacode.hybrid_search import reciprocal_rank_fusion
from gigacode.json_logger import StructuredJsonLogger
from gigacode.lexical_index import LexicalIndex
from gigacode.metadata_store import load_metadata, save_metadata
from gigacode.metrics import get_metrics
from gigacode.metrics_exporter import configure_prometheus
from gigacode.exceptions import (
    GigaCodeError,
    BufferNotFound,
    InvalidPathError,
    EmbeddingError,
    SearchError,
    CommitError,
    RateLimitExceeded,
    QueryLimitExceeded,
)
from gigacode.response_types import (
    SearchMatch,
    SearchResponse,
    ResponseStatus,
    EmbedResponse,
)
from gigacode.size_guard import check_size
from gigacode.path_utils import validate_buffer_path
from gigacode import response_adapters
from gigacode import tool_validation
from gigacode.tool_security import ToolSecurityLayer
from gigacode.context_assembler import ContextAssembler
from gigacode.refactor_engine import RefactorEngine
from gigacode.retry_utils import retry_on_io_error
from gigacode.faceted_search import FacetedSearcher, SearchFilter
from gigacode.symbol_index import SymbolIndex
from gigacode.type_search import TypeSearcher
from gigacode.multi_buffer import MultiBufferManager
from gigacode.resource_budget import estimate_budget as _estimate_budget, ConfidenceScorer

logger = logging.getLogger(__name__)
json_logger = StructuredJsonLogger('tool')

_MAX_DIRTY_BEFORE_AUTO_REBUILD = 3


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

    def __init__(
        self,
        work_dir: str | Path,
        model_name: str | None = None,
        device: str | None = None,
        threshold_mb: float = 500.0,
        use_gpu: bool = True,
        gpu_id: int = 0,
        max_buffers: int = 10,
        enable_prometheus: bool = False,
        prometheus_port: int = 9090,
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

        # Phase 4: State manager for crash recovery and transaction safety
        # Enables write-ahead logging (WAL) for commit operations
        self._state_manager = StateManager(self.work_dir)
        
        # Phase 6: Health status tracker for state-based access control
        self._health_tracker = HealthStatusTracker()
        
        # Phase 7: Security layer (RBAC, audit logging, and rate limiting)
        self._security = ToolSecurityLayer(self.work_dir)

        # Phase 4 Integration: Initialize the three manager layers
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
                audit_logger=self._security._audit_logger,
                user_id=self._security._current_user_id,
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

            logger.info("Phase 4 integration: BufferManager, IndexManager, SearchService initialized")
        except (ImportError, ModuleNotFoundError) as e:
            # Optional dependencies (sklearn for SearchService) may be missing
            # Managers are still required for core functionality
            logger.warning(f"Optional dependency unavailable during manager init: {type(e).__name__}: {e}")
            if self._buffer_manager is None:
                logger.error("CRITICAL: BufferManager initialization failed. Core operations will fail.")
            if self._index_manager is None:
                logger.error("CRITICAL: IndexManager initialization failed. Indexing will fail.")
            if self._search_service is None:
                logger.warning("SearchService unavailable: sklearn or other optional dependency missing. Search operations will fail.")
        
        # Multi-buffer orchestration manager (aliases and virtual buffers)
        self._multi_buffer_manager = MultiBufferManager(self.work_dir)

        # Phase 7: Prometheus metrics export via HTTP endpoint
        self._prometheus_exporter = None
        if enable_prometheus:
            try:
                self._prometheus_exporter = configure_prometheus(
                    port=prometheus_port,
                    start_server=True,
                )
                self._prometheus_exporter.set_embedding_dimension(self._embedding_dim)
                
                # Share Prometheus exporter with managers (always available now)
                self._index_manager._prometheus_exporter = self._prometheus_exporter
                self._search_service._prometheus_exporter = self._prometheus_exporter
                
                logger.info(f"Prometheus metrics endpoint enabled on port {prometheus_port}")
            except ImportError:
                logger.warning(
                    "Prometheus metrics disabled: prometheus-client not installed. "
                    "Install with: pip install prometheus-client"
                )

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
            validate_schemas_against_code,
            report_schema_validation,
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
        message: str, buffer_id: str | None = None, operation: str = "", context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Thin wrapper to tool_validation.make_error_response."""
        return tool_validation.make_error_response(message, buffer_id=buffer_id, operation=operation, context=context)

    @staticmethod
    def _validate_search_params(
        query: str,
        top_k: int | None = None,
        max_results: int | None = None,
    ) -> dict[str, Any] | None:
        """Thin wrapper to tool_validation.validate_search_params."""
        return tool_validation.validate_search_params(query, top_k=top_k, max_results=max_results)

    # ------------------------------------------------------------------
    # Phase 5: Response adapters (delegated to response_adapters module)
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
                operation='chunk_file',
                message=f'Could not count lines in {f}: {exc}',
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
            # Validate path to prevent traversal attacks
            try:
                validated_path = validate_buffer_path(path, self._work_dir)
            except ValueError as e:
                return self._make_error_response(
                    f"Invalid path: {e}",
                    operation="embed_codebase",
                    context={"path": str(path)}
                )
            
            # Delegate to BufferManager: handle chunking, validation, registration
            buffer_id, chunks, files = self._buffer_manager.embed_codebase(
                path=validated_path,
                language_hint=language_hint,
                pattern=pattern,
                sliding_window_size=sliding_window_size,
            )
            
            # Embed chunks
            texts = [ch.text for ch in chunks]
            embeddings = self._embedder.encode(texts, batch_size=64)
            
            # Create and cache indices via IndexManager
            self._index_manager.create_indices(buffer_id, embeddings, chunks)
            
            elapsed = time.perf_counter() - t0
            
            # Record metrics
            if self._prometheus_exporter:
                self._prometheus_exporter.record_operation(
                    operation='embed_codebase',
                    duration_s=elapsed,
                    status='ok',
                    chunk_count=len(chunks),
                )
            
            metrics = get_metrics()
            metrics.record_histogram("embed_codebase_latency_s", elapsed)
            metrics.record_histogram("embed_chunks_count", len(chunks))
            metrics.increment_counter("embed_codebase_calls")
            
            json_logger.info(
                operation='embed_codebase',
                buffer_id=buffer_id,
                elapsed_s=elapsed,
                status='ok',
                message=f'Embedded {len(chunks)} chunks from {len(files)} files',
                details={
                    'files_count': len(files),
                    'chunks_count': len(chunks),
                    'size_bytes': embeddings.nbytes,
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
                operation='embed_codebase',
                status='error',
                message=str(e),
            )
            return {"status": "warning", "message": str(e)}
        except (OSError, ValueError, RuntimeError) as e:
            json_logger.error(
                operation='embed_codebase',
                status='error',
                message=f'Unexpected error: {e}',
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
    ) -> dict[str, Any]:
        """Semantic search via embeddings.
        
        Delegates to SearchService when available (Phase 5), otherwise uses monolithic implementation.
        
        Args:
            buffer_id: Buffer ID to search
            query: Search query
            top_k: Number of top results to return
            offset: Pagination offset
            
        Returns:
            Dict with search results or error
        """
        # Validate parameters first (before delegation)
        err = self._validate_search_params(query, top_k=top_k)
        if err is not None:
            return err
        
        # Phase 5: Delegate to SearchService
        if self._search_service:
            try:
                result = self._search_service.semantic_search(
                    buffer_id=buffer_id,
                    query=query,
                    top_k=top_k + offset,
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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="faceted_search"
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="faceted_search"
            )

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

        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="hybrid_search")

        # Phase 5: Delegate to SearchService
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

        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}
        
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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

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

        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="search_symbols")

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="search_by_type"
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="search_by_type"
            )

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="find_implementations"
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="find_implementations"
            )

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="symbol_search"
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="symbol_search"
            )

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="get_symbol_definition"
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="get_symbol_definition"
            )

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="get_symbol_references"
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="get_symbol_references"
            )

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

    def list_file_symbols(
        self,
        buffer_id: str,
        file: str,
    ) -> dict[str, Any]:
        """List all symbols defined in a specific file."""
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="list_file_symbols"
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="list_file_symbols"
            )

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="cluster_code")

        if not self._search_service:
            return self._make_error_response(
                "SearchService unavailable",
                buffer_id=buffer_id,
                operation="cluster_code",
            )

        try:
            result = self._search_service.cluster_code(
                buffer_id=buffer_id,
                n_clusters=max(2, int(1.0 / (1.0 - threshold + 0.1))),  # Convert threshold to n_clusters estimate
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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="find_duplicates")

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return {"status": "error", "message": "No chunks loaded."}

        # Use hybrid search for relevance scoring (always delegates to SearchService)
        search_result = self.hybrid_search(buffer_id, query, top_k=top_k, offset=0)
        if search_result.get("status") != "ok":
            return search_result

        matches = search_result.get("matches", [])
        if not matches:
            return {"status": "ok", "packed_chunks": [], "total_tokens": 0, "remaining_tokens": max_tokens, "count": 0}

        # Build score map by doc_id
        scores = [0.0] * len(chunks)
        for m in matches:
            did = m.get("doc_id")
            if did is not None and 0 <= did < len(chunks):
                scores[did] = m.get("rrf_score", m.get("score", 0.0))

        return pack_context(chunks, scores, max_tokens=max_tokens)

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="related_code"
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="related_code"
            )

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="refactor_rename"
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="refactor_rename"
            )

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="add_import"
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="add_import"
            )

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="remove_import"
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="remove_import"
            )

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
        t0 = time.perf_counter()
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="read_code")

        snapshot_mgr = self._get_snapshot_manager(buffer_id)
        if snapshot_mgr is None:
            return self._make_error_response(
                "Snapshot not available", buffer_id=buffer_id, operation="read_code",
                context={"reason": "snapshot_load_failed"}
            )

        if file is not None:
            if file not in snapshot_mgr.manifest.files:
                return self._make_error_response(
                    f"File not in buffer: {file}", buffer_id=buffer_id, operation="read_code",
                    context={"requested_file": file}
                )
            
            # Validate file path to prevent traversal attacks
            try:
                buffer_root = Path(info.get("buffer_dir", self._work_dir))
                validate_buffer_path(file, buffer_root)
            except ValueError as e:
                return self._make_error_response(
                    f"Invalid file path: {e}", buffer_id=buffer_id, operation="read_code",
                    context={"requested_file": file}
                )
            
            # Read file on-demand from disk
            lines = snapshot_mgr.read_lines(file)
            if lines is None:
                return self._make_error_response(
                    f"Failed to read file: {file}", buffer_id=buffer_id, operation="read_code",
                    context={"requested_file": file}
                )
            
            end = end_line if end_line is not None else len(lines) + 1
            selected = lines[start_line - 1:end - 1]
            result = {
                "status": "ok",
                "file": file,
                "start_line": start_line,
                "end_line": end,
                "lines": selected,
            }
            
            # Audit log successful file read
            self._security.log_success(
                operation="read_code",
                buffer_id=buffer_id,
                details={
                    "file": file,
                    "start_line": start_line,
                    "end_line": end,
                    "lines_count": len(selected),
                }
            )
            
            # Phase 7: Record operation metrics
            elapsed = time.perf_counter() - t0
            if self._prometheus_exporter:
                self._prometheus_exporter.record_operation(
                    operation='read_code',
                    duration_s=elapsed,
                    status='ok',
                    chunk_count=len(selected),
                )
            
            return result

        # Read all files on-demand
        result: dict[str, list[str]] = {}
        for fname in snapshot_mgr.manifest.files.keys():
            lines = snapshot_mgr.read_lines(fname)
            if lines is None:
                json_logger.warning(
                    operation='read_code',
                    message=f'Failed to read file {fname}',
                )
                continue
            end = end_line if end_line is not None else len(lines) + 1
            result[fname] = lines[start_line - 1:end - 1]
        
        final_result = {"status": "ok", "files": result}
        
        # Audit log successful read-all operation
        self._security.log_success(
            operation="read_code",
            buffer_id=buffer_id,
            details={
                "files_count": len(result),
                "start_line": start_line,
                "end_line": end_line,
            }
        )
        
        # Phase 7: Record operation metrics for read-all
        elapsed = time.perf_counter() - t0
        if self._prometheus_exporter:
            self._prometheus_exporter.record_operation(
                operation='read_code',
                duration_s=elapsed,
                status='ok',
                chunk_count=sum(len(lines) for lines in result.values()),
            )
        
        return final_result

    def write_code(
        self,
        buffer_id: str,
        file: str,
        start_line: int,
        new_lines: list[str],
        end_line: int | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        # Check current buffer state - must be READY to write
        try:
            current_state = self._get_buffer_state(buffer_id)
            if current_state != BufferState.READY:
                return {
                    "status": "error",
                    "message": f"Cannot write code: buffer is in {current_state} state. "
                               f"Valid transitions: {BufferStateTransition.VALID_TRANSITIONS.get(current_state, [])}"
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
            buffer_root = Path(info.get("buffer_dir", self._work_dir))
            validate_buffer_path(file, buffer_root)
        except ValueError as e:
            return {"status": "error", "message": f"Invalid file path: {e}"}

        # Phase 2: Detect 3-way merge conflicts before allowing write
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
        new_file_lines = old_lines[:start_line - 1] + sanitized_new_lines + old_lines[end:]
        snapshot[file] = new_file_lines
        self._save_source_snapshot(buffer_id, snapshot)

        dirty = info.setdefault("dirty_files", {})
        dirty[file] = True
        self._save_registry()
        
        # Transition state from READY → DIRTY if not already dirty
        try:
            if current_state == BufferState.READY:
                self._set_buffer_state(buffer_id, BufferState.DIRTY)
        except ValueError:
            # State transition failed, but don't fail the write
            logger.warning(f"Failed to transition buffer {buffer_id} to DIRTY state")


        # Auto-rebuild if too many dirty files
        if len(dirty) >= _MAX_DIRTY_BEFORE_AUTO_REBUILD:
            self._rebuild_dirty(buffer_id)

        result = {
            "status": "ok",
            "file": file,
            "changed_lines": len(sanitized_new_lines),
            "replaced_lines": end - start_line,
            "total_lines": len(new_file_lines),
        }
        
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
            }
        )
        
        # Phase 7: Record operation metrics
        elapsed = time.perf_counter() - t0
        if self._prometheus_exporter:
            self._prometheus_exporter.record_operation(
                operation='write_code',
                duration_s=elapsed,
                status='ok',
                chunk_count=len(sanitized_new_lines),
            )
        
        return result

    def diff(self, buffer_id: str) -> dict[str, Any]:
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        root = Path(info["root"])
        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}

        old_hashes = info.get("file_hashes", {})
        changed: list[dict[str, Any]] = []
        for rel_path, lines in snapshot.items():
            current_text = "\n".join(lines)
            current_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
            if old_hashes.get(rel_path) != current_hash:
                disk_path = root / rel_path
                disk_lines = disk_path.read_text(encoding="utf-8").splitlines() if disk_path.exists() else []
                changed.append({
                    "file": rel_path,
                    "buffer_lines": len(lines),
                    "disk_lines": len(disk_lines),
                    "dirty": info.get("dirty_files", {}).get(rel_path, False),
                })
        return {"status": "ok", "changed_files": changed}

    def discard(
        self,
        buffer_id: str,
        file: str | None = None,
    ) -> dict[str, Any]:
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

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
    ) -> dict[str, Any]:
        """Commit buffer changes to disk with 3-way merge conflict handling and crash recovery.
        
        Phase 3: Uses SnapshotManager for merge conflicts.
        Phase 4: Wraps with StateManager transactions for crash recovery.
        
        - If both buffer and disk modified: returns "conflict" status instead of aborting
        - Rebuilds embeddings before writing
        - Updates snapshot manifest after successful writes
        - Returns detailed conflict information for user resolution
        - Uses write-ahead logging (WAL) for crash recovery
        
        Args:
            buffer_id: Buffer to commit
            dry_run: If True, check what would be written without modifying files
        
        Returns:
            Dict with:
            - "status": "ok", "conflict", or "error"
            - "written_files": List of successfully written files
            - "conflict_files": List of dicts with conflict details (file, message, line counts)
            - "dry_run": Whether this was a dry run
            - "transaction_id": Transaction ID (for debugging)
        """
        t0 = time.perf_counter()
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        root = Path(info["root"])
        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}

        dirty = info.get("dirty_files", {})
        if not dirty:
            return {"status": "ok", "written_files": [], "conflict_files": [], "dry_run": dry_run, "transaction_id": None}

        # Phase 4: Begin transaction for crash recovery
        transaction_id = None
        if not dry_run:
            transaction_id = self._state_manager.start_transaction(
                operation="commit",
                buffer_id=buffer_id,
                file_path=",".join(dirty.keys()),  # Comma-separated list of files
                start_line=0,
                end_line=None,
                new_lines=None
            )

        try:
            # Rebuild embeddings for dirty files before writing to disk
            if not dry_run:
                self._rebuild_files(buffer_id, list(dirty.keys()))

            # Phase 3: Use SnapshotManager for 3-way merge conflict handling
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
                    conflicts.append({
                        "file": rel_path,
                        "message": f"Invalid file path: {e}",
                    })
                    continue

                if dry_run:
                    # Dry run: just check for conflicts without writing
                    diff_result = snapshot_mgr.compute_diff(rel_path, lines)
                    if diff_result.get("has_conflict"):
                        conflicts.append({
                            "file": rel_path,
                            "message": "3-way merge conflict: disk and buffer both modified",
                        })
                    else:
                        written.append(rel_path)
                else:
                    # Real commit: use 3-way merge with conflict detection
                    merge_result = snapshot_mgr.write_file_with_merge(rel_path, lines, allow_conflicts=False)
                    
                    if merge_result["status"] == "conflict":
                        # Conflict detected - record it but continue processing other files
                        diff_result = snapshot_mgr.compute_diff(rel_path, lines)
                        conflicts.append({
                            "file": rel_path,
                            "message": merge_result.get("message", "3-way merge conflict"),
                            "disk_lines": len(diff_result.get("disk_lines") or []),
                            "buffer_lines": len(lines),
                            "snapshot_line_count": diff_result.get("snapshot_line_count"),
                        })
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
                            "transaction_id": transaction_id
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

                # Phase 4: Commit transaction (WAL is updated)
                self._state_manager.commit_transaction(transaction_id)
                self._state_manager.save_registry()

            result = {
                "status": "conflict" if conflicts else "ok",
                "written_files": written,
                "conflict_files": conflicts,
                "dry_run": dry_run,
                "transaction_id": transaction_id,
            }
            
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
                    }
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
                    }
                )
            
            # Phase 7: Record operation metrics for Prometheus
            elapsed = time.perf_counter() - t0
            if self._prometheus_exporter:
                self._prometheus_exporter.record_operation(
                    operation='commit',
                    duration_s=elapsed,
                    status='conflict' if conflicts else 'ok',
                    chunk_count=len(written),
                )
            
            return result

        except (OSError, ValueError, RuntimeError) as e:
            # Phase 4: Rollback on any error
            if transaction_id:
                json_logger.error(
                    operation='commit',
                    buffer_id=buffer_id,
                    message=f'Commit failed with exception: {e}; rolling back transaction {transaction_id}',
                )
                self._state_manager.rollback_transaction(transaction_id)
            raise

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

    def check_buffer_state(self, buffer_id: str) -> dict[str, Any]:
        """Return the current state of a buffer (READY, DIRTY, REBUILDING).

        Args:
            buffer_id: Buffer handle.

        Returns:
            Dict with ``status``, ``buffer_id``, ``state``, and optional
            ``state_changed_at`` timestamp.
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="check_buffer_state"
            )

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
                operation='_rebuild_files',
                buffer_id=buffer_id,
                message='Snapshot manager not found',
            )
            return
        
        language_hint = info.get("language_hint")
        sliding_window_size = info.get("sliding_window_size", 30)
        new_embeddings_list: list[np.ndarray] = []
        for rel_path in files:
            lines = snapshot_mgr.read_lines(rel_path)
            if lines is None:
                json_logger.warning(
                    operation='_rebuild_files',
                    buffer_id=buffer_id,
                    message=f'Could not read {rel_path}',
                )
                continue
            text = "\n".join(lines)
            file_chunks = chunk_text(text, language_hint=language_hint, filename_hint=rel_path, sliding_window_size=sliding_window_size)
            for ch in file_chunks:
                ch.file = rel_path
                ch.id = next_id
                new_chunks.append(ch)
                next_id += 1
            if file_chunks:
                texts = [ch.text for ch in file_chunks]
                emb = self._embedder.encode(texts, batch_size=64)
                new_embeddings_list.append(emb)

        # Rebuild embeddings array
        kept_mask = np.ones(len(chunks), dtype=bool)
        for idx in removed_ids:
            kept_mask[idx] = False
        kept_embeddings = embeddings[kept_mask]
        if new_embeddings_list:
            new_embeddings = np.vstack(new_embeddings_list)
            final_embeddings = np.vstack([kept_embeddings, new_embeddings]) if kept_embeddings.size else new_embeddings
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
        self._save_buffer_state(buffer_dir, new_chunks, final_embeddings, index, lexical_index=lexical)

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
            lexical_path.write_bytes(json.dumps(lexical_data, separators=(",", ":")).encode("utf-8"))
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
                operation='_load_lexical_index',
                message=f'Failed to load lexical index from {path}: {exc}',
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
            raise RuntimeError(f"Cannot build lexical index for buffer_id={buffer_id}: chunks not found")
        
        json_logger.warning(
            operation='_get_lexical_index',
            buffer_id=buffer_id,
            message=f'Lexical index missing; building from {len(chunks)} chunks',
            details={'chunks_count': len(chunks)},
        )
        lexical = LexicalIndex()
        for i, ch in enumerate(chunks):
            lexical.add(i, ch.text)
        self._lexical_cache[buffer_id] = lexical
        return lexical

    def _get_buffer_info(self, buffer_id: str) -> dict[str, Any] | None:
        return self._registry.get(buffer_id)
    
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
        
        # Update health tracker (Phase 6)
        self._health_tracker.update_buffer_state(buffer_id, new_state)
        
        # Log state change
        self._security.log_success(
            operation="state_transition",
            buffer_id=buffer_id,
            details={
                "from_state": str(current_state),
                "to_state": str(new_state),
            }
        )

    def _get_snapshot_manager(self, buffer_id: str) -> SnapshotManager | None:
        """Get or load SnapshotManager for buffer.
        
        Caches instances to avoid repeated loads from disk.
        
        Args:
            buffer_id: Buffer identifier
        
        Returns:
            SnapshotManager instance or None if buffer not found or snapshot invalid
        """
        if buffer_id in self._snapshot_managers:
            return self._snapshot_managers[buffer_id]
        
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return None
        
        buffer_dir = Path(info["buffer_dir"])
        try:
            snapshot_mgr = SnapshotManager(buffer_dir)
            if snapshot_mgr.manifest is None:
                json_logger.warning(
                    operation='_get_snapshot_manager',
                    buffer_id=buffer_id,
                    message='Manifest is None',
                )
                return None
            
            self._snapshot_managers[buffer_id] = snapshot_mgr
            return snapshot_mgr
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            json_logger.error(
                operation='_get_snapshot_manager',
                buffer_id=buffer_id,
                message=f'Failed to load snapshot: {e}',
            )
            return None

    @retry_on_io_error(max_attempts=3, delay_s=0.5)
    def _save_registry(self) -> None:
        """Save registry to disk with retry on transient I/O errors."""
        # Delegate to StateManager which handles atomicity, locking, and recovery
        if self._state_manager is not None:
            self._state_manager.save_registry()
        else:
            # Fallback for cases where StateManager is not available
            with self._registry_path.open("wb") as fh:
                fh.write(
                    json.dumps(self._registry, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                )

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
        index_cache_size = len(getattr(self, '_index_cache', {}))
        lexical_cache_size = len(getattr(self, '_lexical_cache', {}))
        query_cache_size = len(getattr(self, '_query_cache', {}))

        # Approximate memory for loaded buffers
        buffer_memory_mb: dict[str, float] = {}
        for buffer_id in getattr(self, '_index_cache', {}):
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
    # Phase 6: State-Based Access Control
    # ------------------------------------------------------------------
    def _check_state_for_operation(
        self, 
        buffer_id: str, 
        operation_type: OperationType
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
            return False, f"Cannot write to buffer in {current_state.value} state. Buffer must be in {BufferState.READY.value} state."
        
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
    # Phase 7: RBAC, Audit Logging, and Rate Limiting
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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="get_test_context"
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="get_test_context"
            )

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
        for idx, test_idx in enumerate(test_indices):
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
                test_files.append({
                    "file": chunk.file,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "type": chunk.type,
                    "name": chunk.name,
                    "score": round(score, 4),
                })

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response(
                "Unknown buffer_id", buffer_id=buffer_id, operation="suggest_tests"
            )

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return self._make_error_response(
                "No chunks loaded", buffer_id=buffer_id, operation="suggest_tests"
            )

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
        changed_emb_norm = changed_emb / (np.linalg.norm(changed_emb, axis=1, keepdims=True) + 1e-10)
        test_emb_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-10)
        similarities = test_emb_norm @ changed_emb_norm.T

        # Max similarity per test chunk
        max_scores = similarities.max(axis=1)

        # Build and rank results
        scored_tests: list[tuple[CodeChunk, float]] = []
        for idx, test_idx in enumerate(test_indices):
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
                suggested_tests.append({
                    "file": chunk.file,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "type": chunk.type,
                    "name": chunk.name,
                    "score": round(score, 4),
                })

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
            operation='close',
            message='CodeEmbeddingTool closed; all caches released',
        )

    def __enter__(self) -> CodeEmbeddingTool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
