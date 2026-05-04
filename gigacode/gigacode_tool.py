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
from gigacode.query_cache import QueryCache
from gigacode.snapshot_manager import SnapshotManager
from gigacode.state_manager import StateManager
from gigacode.response_types import (
    SearchMatch,
    SearchResponse,
    ResponseStatus,
    EmbedResponse,
    ClusterResponse,
    ClusterItem,
    ErrorResponse,
)
from gigacode.retry_utils import retry_on_io_error
from gigacode.size_guard import check_size
from gigacode.operation_config import OperationType, OperationConfig
from gigacode.health_status import HealthStatus, HealthStatusTracker
from gigacode.access_control import User, AccessControl, Permission, Role
from gigacode.audit_logger import AuditLogger, AuditStatus
from gigacode.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)
json_logger = StructuredJsonLogger('tool')

_MAX_DIRTY_BEFORE_AUTO_REBUILD = 3


class LRUDict(OrderedDict):
    """Bounded LRU dict. Evicts least-recently-used item when max_size exceeded.
    
    Usage:
        cache = LRUDict(max_size=10)
        cache['key'] = value  # Auto-evicts oldest if size exceeds max_size
        val = cache['key']    # Moves 'key' to end (most recently used)
    """

    def __init__(self, max_size: int = 10) -> None:
        super().__init__()
        self.max_size = max_size

    def __getitem__(self, key: str) -> Any:
        """Get item and move to end (mark as recently used)."""
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item and move to end. Evict LRU item if size exceeded."""
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            # Pop the oldest (first) item
            oldest_key = next(iter(self))
            self.pop(oldest_key)
            json_logger.debug(
                operation='lru_eviction',
                details={'removed_key': oldest_key, 'new_size': len(self)},
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get with default; moves to end if found."""
        if key in self:
            return self[key]
        return default

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": len(self),
            "maxsize": self.max_size,
            "utilization": len(self) / self.max_size if self.max_size > 0 else 0
        }



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

        self._embedder = Embedder(model_name=model_name, device=device)
        self._embedding_dim = self._embedder.embedding_dim

        # Query result cache with semantic similarity matching
        # (Managers will use this cache)
        self._query_cache = QueryCache(
            maxsize=256,
            embedder=self._embedder,
            similarity_threshold=0.95,  # Catch paraphrased queries (e.g., "find add" vs "locate addition")
        )

        # Phase 4: State manager for crash recovery and transaction safety
        # Enables write-ahead logging (WAL) for commit operations
        self._state_manager = StateManager(self.work_dir)
        
        # Phase 6: Health status tracker for state-based access control
        self._health_tracker = HealthStatusTracker()
        
        # Phase 7: RBAC, audit logging, and rate limiting
        self._access_control = AccessControl()
        self._audit_logger = AuditLogger(self.work_dir / "audit.jsonl")
        self._rate_limiter = RateLimiter()
        self._current_user_id = "default"  # Default user for local development
        # Register default user with AGENT role (full operational access for AI agents)
        self._access_control.register_user("default", Role.AGENT)

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

    # Delegation properties for backward compatibility with registry access
    @property
    def _registry(self) -> dict[str, Any]:
        """Access registry from StateManager (backward compatibility)."""
        if self._state_manager is None:
            return {}
        return self._state_manager.registry
    
    @property
    def _registry_path(self) -> Path:
        """Access registry path from StateManager (backward compatibility)."""
        if self._state_manager is None:
            return self.work_dir / ".registry.json"
        return self._state_manager.registry_path

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
    # Input validation
    # ------------------------------------------------------------------
    @staticmethod
    def _make_error_response(
        message: str, buffer_id: str | None = None, operation: str = "", context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create a structured error response with context.
        
        Args:
            message: Human-readable error message.
            buffer_id: Buffer ID if applicable.
            operation: Operation name (e.g., 'semantic_search', 'write_code').
            context: Additional context dict.
        
        Returns:
            Dict with ErrorResponse.to_dict() format (includes context field).
        """
        ctx = context or {}
        if buffer_id is not None:
            ctx["buffer_id"] = buffer_id
        if operation:
            ctx["operation"] = operation
        response = ErrorResponse(message=message, context=ctx)
        return response.to_dict()

    @staticmethod
    def _validate_search_params(
        query: str,
        top_k: int | None = None,
        max_results: int | None = None,
    ) -> dict[str, Any] | None:
        """Return an error dict when params are invalid, else ``None``."""
        if not query or not query.strip():
            return {"status": "error", "message": "query must be a non-empty string."}
        if top_k is not None:
            if not isinstance(top_k, int) or top_k < 1 or top_k > 10_000:
                return {"status": "error", "message": "top_k must be an integer between 1 and 10000."}
        if max_results is not None:
            if not isinstance(max_results, int) or max_results < 1 or max_results > 100_000:
                return {"status": "error", "message": "max_results must be an integer between 1 and 100000."}
        return None

    # ------------------------------------------------------------------
    # Phase 5: Response adapters for manager delegation
    # ------------------------------------------------------------------
    @staticmethod
    def _adapt_search_response(
        service_result: Any,
        offset: int = 0,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Adapt SearchService response to CodeEmbeddingTool format.
        
        Args:
            service_result: SearchService.SearchResponse or error dict
            offset: Results offset for slicing
            top_k: Number of results to return (if None, return all)
        
        Returns:
            Dict in CodeEmbeddingTool response format
        """
        # If it's already an error dict, return as-is
        if isinstance(service_result, dict) and service_result.get("status") == "error":
            return service_result
        
        # Convert SearchService.SearchResponse to CodeEmbeddingTool format
        if hasattr(service_result, "to_dict"):
            service_dict = service_result.to_dict()
        else:
            service_dict = service_result
        
        # Extract matches and apply offset/slicing
        matches = service_dict.get("matches", [])
        if offset > 0 or top_k:
            end = offset + top_k if top_k else None
            matches = matches[offset:end]
        
        # Convert SearchService.SearchMatch to CodeEmbeddingTool.SearchMatch
        converted_matches = []
        for idx, match_dict in enumerate(matches):
            if isinstance(match_dict, dict):
                # Build CodeEmbeddingTool.SearchMatch
                converted = SearchMatch(
                    file=match_dict.get("file", ""),
                    start_line=match_dict.get("start_line", 0),
                    end_line=match_dict.get("end_line", 0),
                    score=float(match_dict.get("score", 0.0)),
                    doc_id=match_dict.get("doc_id", idx),  # Use index if no doc_id
                    type=match_dict.get("type"),
                    name=match_dict.get("name"),
                    match_type=match_dict.get("match_type", "semantic"),
                )
                converted_matches.append(converted)
            else:
                # Already a SearchMatch object
                converted_matches.append(match_dict)
        
        # Build response in CodeEmbeddingTool format
        response = SearchResponse(
            status=ResponseStatus.OK,
            matches=converted_matches,
            cached=service_dict.get("cache_hit", False),
        )
        return response.to_dict()

    @staticmethod
    def _adapt_file_response(
        service_result: Any,
    ) -> dict[str, Any]:
        """Adapt SearchService look_for_file response to CodeEmbeddingTool format.
        
        SearchService returns: {"status", "files", "count", "pattern"}
        CodeEmbeddingTool expects: {"status", "file_location", "absolute_path", "match_type"}
        """
        if isinstance(service_result, dict):
            if service_result.get("status") == "error":
                return service_result
            
            files = service_result.get("files", [])
            if not files:
                return {"status": "error", "message": "File not found"}
            elif len(files) == 1:
                # Single match
                return {
                    "status": "ok",
                    "file_location": files[0],
                    "match_type": "found",
                }
            else:
                # Multiple matches
                return {
                    "status": "ok",
                    "candidates": files,
                    "match_type": "multiple",
                    "message": f"Found {len(files)} matching files.",
                }
        return service_result

    @staticmethod
    def _adapt_cluster_response(
        service_result: Any,
    ) -> dict[str, Any]:
        """Adapt SearchService ClusterResult to CodeEmbeddingTool format.
        
        SearchService returns ClusterResult with dict of clusters
        CodeEmbeddingTool expects list of ClusterItem with file/start_line/end_line/size/avg_score
        """
        if isinstance(service_result, dict) and service_result.get("status") == "error":
            return service_result
        
        # Convert ClusterResult to dict if needed
        if hasattr(service_result, "to_dict"):
            service_dict = service_result.to_dict()
        else:
            service_dict = service_result
        
        clusters_dict = service_dict.get("clusters", {})
        cluster_list = []
        
        # Convert dict-based clusters to list format with aggregated metadata
        for cluster_id, chunks in clusters_dict.items():
            if chunks:
                first_chunk = chunks[0]
                last_chunk = chunks[-1]
                cluster_list.append(ClusterItem(
                    file=first_chunk.get("file", ""),
                    start_line=first_chunk.get("start_line", 0),
                    end_line=last_chunk.get("end_line", 0),
                    size=len(chunks),
                    avg_score=0.0,  # Would need more data to calculate
                ))
        
        response = ClusterResponse(
            status=ResponseStatus.OK,
            clusters=cluster_list,
        )
        return response.to_dict()

    @staticmethod
    def _adapt_duplicate_response(
        service_result: Any,
    ) -> dict[str, Any]:
        """Adapt SearchService DuplicateResult to CodeEmbeddingTool format.
        
        Both formats are similar, just ensure proper dict conversion.
        """
        if isinstance(service_result, dict):
            if service_result.get("status") == "error":
                return service_result
            return service_result
        
        # Convert DuplicateResult to dict if needed
        if hasattr(service_result, "to_dict"):
            return service_result.to_dict()
        
        return service_result

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
            except Exception as exc:
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
            # Delegate to BufferManager: handle chunking, validation, registration
            buffer_id, chunks, files = self._buffer_manager.embed_codebase(
                path=path,
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
        except Exception as e:
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
        
        # Phase 5: Delegate to SearchService if available
        if self._search_service:
            try:
                result = self._search_service.semantic_search(
                    buffer_id=buffer_id,
                    query=query,
                    top_k=top_k + offset,
                )
                return self._adapt_search_response(result, offset=offset, top_k=top_k)
            except Exception as e:
                logger.warning(f"SearchService delegation failed: {e}. Falling back to monolithic implementation.")
                # Fall through to monolithic implementation below
        
        # Monolithic implementation (fallback)
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="semantic_search")

        cached = self._query_cache.get(buffer_id, query, top_k + offset, "semantic")
        if cached is not None:
            cached_matches = cached.get("matches", [])
            # Slice and convert to SearchMatch if needed
            sliced = cached_matches[offset:offset + top_k]
            if sliced and isinstance(sliced[0], dict):
                sliced = [SearchMatch(**m) if isinstance(m, dict) else m for m in sliced]
            metrics = get_metrics()
            metrics.record_cache_hit("semantic_search_cache")
            response = SearchResponse(
                status=ResponseStatus.OK,
                matches=sliced,
                cached=True,
            )
            return response.to_dict()

        index = self._get_index(buffer_id)
        chunks = self._load_chunks(buffer_id)
        if chunks is None:
            return {"status": "error", "message": "Chunk metadata missing."}

        t0 = time.perf_counter()
        q_emb = self._embedder.encode([query], batch_size=1)
        distances, indices = index.search(q_emb, top_k + offset)
        elapsed = time.perf_counter() - t0

        matches_list: list[SearchMatch] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            ch = chunks[idx]
            matches_list.append(SearchMatch(
                file=ch.file,
                start_line=ch.start_line,
                end_line=ch.end_line,
                type=ch.type,
                name=ch.name,
                score=float(score),
                doc_id=int(idx),
                match_type="semantic",
            ))

        # Log search with JSON structure
        json_logger.debug(
            operation='semantic_search',
            buffer_id=buffer_id,
            elapsed_s=elapsed,
            details={'top_k': top_k, 'gpu': index.is_gpu, 'matches': len(matches_list)},
        )

        # Phase 7: Record operation metrics for Prometheus
        if self._prometheus_exporter:
            self._prometheus_exporter.record_operation(
                operation='semantic_search',
                duration_s=elapsed,
                status='ok',
                chunk_count=len(matches_list),
            )

        self._query_cache.set(buffer_id, query, top_k + offset, "semantic", {"matches": matches_list})
        metrics = get_metrics()
        metrics.record_cache_miss("semantic_search_cache")
        metrics.record_histogram("semantic_search_latency_s", elapsed)
        metrics.increment_counter("semantic_search_calls")
        
        response = SearchResponse(
            status=ResponseStatus.OK,
            matches=matches_list[offset:offset + top_k],
            cached=False,
        )
        return response.to_dict()

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

        # Phase 5: Delegate to SearchService if available
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
            except Exception as e:
                logger.warning(f"SearchService delegation failed: {e}. Falling back to monolithic implementation.")
                # Fall through to monolithic implementation below
        
        # Monolithic implementation (fallback)
        cached = self._query_cache.get(buffer_id, query, top_k + offset, "hybrid")
        if cached is not None:
            cached_matches = cached.get("matches", [])
            # Slice and convert to SearchMatch if needed
            sliced = cached_matches[offset:offset + top_k]
            if sliced and isinstance(sliced[0], dict):
                sliced = [SearchMatch(**m) if isinstance(m, dict) else m for m in sliced]
            metrics = get_metrics()
            metrics.record_cache_hit("hybrid_search_cache")
            response = SearchResponse(
                status=ResponseStatus.OK,
                matches=sliced,
                cached=True,
            )
            return response.to_dict()

        t0_hybrid = time.perf_counter()
        chunks = self._load_chunks(buffer_id)
        if chunks is None:
            return {"status": "error", "message": "Chunk metadata missing."}

        # Semantic results
        index = self._get_index(buffer_id)
        t0 = time.perf_counter()
        q_emb = self._embedder.encode([query], batch_size=1)
        distances, indices = index.search(q_emb, top_k * 4 + offset)
        elapsed = time.perf_counter() - t0
        json_logger.debug(
            operation='hybrid_search_semantic',
            buffer_id=buffer_id,
            elapsed_s=elapsed,
            details={'top_k': top_k, 'gpu': index.is_gpu},
        )
        semantic_results: list[dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            semantic_results.append({
                "doc_id": int(idx),
                "score": float(score),
            })

        # Lexical results
        lexical = self._get_lexical_index(buffer_id)
        lexical_results = lexical.search(query, top_k=top_k * 4 + offset)

        merged = reciprocal_rank_fusion(
            semantic_results,
            lexical_results,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            top_k=top_k + offset,
        )

        # Enrich merged results with chunk metadata
        matches_list: list[SearchMatch] = []
        for m in merged:
            idx = m["doc_id"]
            ch = chunks[idx]
            matches_list.append(SearchMatch(
                file=ch.file,
                start_line=ch.start_line,
                end_line=ch.end_line,
                type=ch.type,
                name=ch.name,
                rrf_score=m.get("rrf_score", 0.0),
                semantic_rank=m.get("semantic_rank"),
                lexical_rank=m.get("lexical_rank"),
                doc_id=idx,
                match_type="hybrid",
            ))

        self._query_cache.set(buffer_id, query, top_k + offset, "hybrid", {"matches": matches_list})
        elapsed_hybrid = time.perf_counter() - t0_hybrid
        
        # Phase 7: Record operation metrics for Prometheus
        if self._prometheus_exporter:
            self._prometheus_exporter.record_operation(
                operation='hybrid_search',
                duration_s=elapsed_hybrid,
                status='ok',
                chunk_count=len(matches_list),
            )
        
        metrics = get_metrics()
        metrics.record_cache_miss("hybrid_search_cache")
        metrics.record_histogram("hybrid_search_latency_s", elapsed_hybrid)
        metrics.increment_counter("hybrid_search_calls")
        
        response = SearchResponse(
            status=ResponseStatus.OK,
            matches=matches_list[offset:offset + top_k],
            cached=False,
        )
        return response.to_dict()

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
        
        result = self._search_service.cluster_code(
            buffer_id=buffer_id,
            n_clusters=max(2, int(1.0 / (1.0 - threshold + 0.1))),  # Convert threshold to n_clusters estimate
        )
        return self._adapt_cluster_response(result)

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
        
        result = self._search_service.find_duplicates(
            buffer_id=buffer_id,
            threshold=threshold,
        )
        return self._adapt_duplicate_response(result)

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

        # Use hybrid search for relevance scoring
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
            self._audit_logger.log_success(
                operation="read_code",
                user_id=self._current_user_id,
                role=self._access_control.get_user(self._current_user_id).role.name,
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
        self._audit_logger.log_success(
            operation="read_code",
            user_id=self._current_user_id,
            role=self._access_control.get_user(self._current_user_id).role.name,
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
        self._audit_logger.log_success(
            operation="write_code",
            user_id=self._current_user_id,
            role=self._access_control.get_user(self._current_user_id).role.name,
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
                self._audit_logger.log_failure(
                    operation="commit",
                    user_id=self._current_user_id,
                    role=self._access_control.get_user(self._current_user_id).role.name,
                    error_message=f"Commit conflict: {len(conflicts)} files",
                    buffer_id=buffer_id,
                    details={
                        "dry_run": dry_run,
                        "written_files_count": len(written),
                        "conflict_files_count": len(conflicts),
                        "transaction_id": transaction_id,
                    }
                )
            else:
                self._audit_logger.log_success(
                    operation="commit",
                    user_id=self._current_user_id,
                    role=self._access_control.get_user(self._current_user_id).role.name,
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

        except Exception as e:
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
        except Exception as exc:
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
        self._audit_logger.log_success(
            operation="state_transition",
            user_id=self._current_user_id,
            role=self._access_control.get_user(self._current_user_id).role.name,
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
        except Exception as e:
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
    # Lifecycle
    # ------------------------------------------------------------------
    def get_cache_stats(self) -> dict[str, Any]:
        """Return cache utilization statistics (delegated to IndexManager)."""
        # Delegation attempt
        if self._index_manager:
            try:
                result = self._index_manager.get_cache_stats()
                return result
            except Exception as e:
                logger.warning(f"IndexManager.get_cache_stats delegation failed: {e}. Falling back...")
        
        # Monolithic fallback
        return {
            "index_cache_size": len(self._index_cache),
            "index_cache_max": self._index_cache.max_size,
            "lexical_cache_size": len(self._lexical_cache),
            "lexical_cache_max": self._lexical_cache.max_size,
            "query_cache_stats": self._query_cache.stats(),
        }

    def health_check(self) -> dict[str, Any]:
        """Perform a health check and return system status (delegated to IndexManager)."""
        # Delegation attempt
        if self._index_manager:
            try:
                result = self._index_manager.health_check()
                return result
            except Exception as e:
                logger.warning(f"IndexManager.health_check delegation failed: {e}. Falling back...")
        
        # Monolithic fallback
        import datetime
        
        warnings = []
        status = "healthy"
        
        # Check cache utilization
        index_util = (len(self._index_cache) / self._index_cache.max_size) * 100
        lexical_util = (len(self._lexical_cache) / self._lexical_cache.max_size) * 100
        query_stats = self._query_cache.stats()
        query_util = (query_stats["size"] / query_stats["maxsize"]) * 100
        
        max_util = max(index_util, lexical_util, query_util)
        if max_util > 90:
            warnings.append(f"High cache utilization: {max_util:.1f}% (may evict buffers soon)")
            status = "degraded"
        elif max_util > 75:
            warnings.append(f"Elevated cache utilization: {max_util:.1f}%")
        
        # Check GPU availability
        faiss_available = False
        gpu_available = False
        try:
            import faiss
            faiss_available = True
            ngpu = faiss.get_num_gpus()
            gpu_available = ngpu > 0
        except Exception:
            pass
        
        if self.use_gpu and not gpu_available:
            warnings.append("GPU requested but not available; using CPU")
            status = "degraded"
        
        # Check embedder
        embedder_ready = self._embedder is not None and self._embedder._model is not None
        if not embedder_ready:
            warnings.append("Embedder not initialized")
            status = "unhealthy"
        
        return {
            "status": status,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "buffers_registered": len(self._registry),
            "buffers_loaded": len(self._index_cache),
            "cache_utilization_percent": {
                "index_cache": round(index_util, 1),
                "lexical_cache": round(lexical_util, 1),
                "query_cache": round(query_util, 1),
                "max": round(max_util, 1),
            },
            "embedder_ready": embedder_ready,
            "faiss_available": faiss_available,
            "faiss_gpu_available": gpu_available,
            "warnings": warnings,
        }

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
        self._current_user_id = user_id
        return self._access_control.register_user(user_id, role_enum)
    
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
        allowed, reason = self._access_control.check_operation(
            self._current_user_id,
            operation,
            buffer_owner,
        )
        
        # Log denied operation
        if not allowed:
            self._audit_logger.log_denied(
                operation=operation,
                user_id=self._current_user_id,
                role=self._access_control.get_user(self._current_user_id).role.value,
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
        user = self._access_control.get_user(self._current_user_id)
        allowed, reason = self._rate_limiter.check_all_limits(
            self._current_user_id,
            user.role,
            buffer_id,
            operation_type="operations",
        )
        
        return allowed, reason
    
    def get_audit_log(self, buffer_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Get audit log entries.
        
        Args:
            buffer_id: Filter by buffer (optional)
            limit: Max entries to return
            
        Returns:
            List of audit log entries
        """
        # Check permission to view audit logs
        allowed, reason = self._check_permission("view_audit_log")
        if not allowed:
            return []
        
        # Get entries
        if buffer_id:
            entries = self._audit_logger.get_buffer_history(buffer_id, limit)
        else:
            entries = self._audit_logger.query_logs(
                user_id=self._current_user_id,
                limit=limit,
            )
        
        return [e.to_dict() for e in entries]
    
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
