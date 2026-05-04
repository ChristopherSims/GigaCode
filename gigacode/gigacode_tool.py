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

        # Registry of embedded codebases: buffer_id -> metadata dict
        self._registry_path = self.work_dir / "registry.json"
        self._registry: dict[str, dict[str, Any]] = (
            json.loads(self._registry_path.read_text(encoding="utf-8"))
            if self._registry_path.exists()
            else {}
        )
        
        # Audit log: append-only JSON lines file for operation tracking
        self._audit_log_path = self.work_dir / "audit.jsonl"

        # In-memory index cache: buffer_id -> GpuIndex (LRU-bounded)
        self._index_cache: LRUDict = LRUDict(max_size=max_buffers)

        # In-memory lexical index cache: buffer_id -> LexicalIndex (LRU-bounded)
        self._lexical_cache: LRUDict = LRUDict(max_size=max_buffers)

        # Query result cache with semantic similarity matching
        self._query_cache = QueryCache(
            maxsize=256,
            embedder=self._embedder,
            similarity_threshold=0.95,  # Catch paraphrased queries (e.g., "find add" vs "locate addition")
        )

        # Snapshot managers: buffer_id -> SnapshotManager (LRU-bounded, one per buffer)
        # Caches instances to avoid repeated loads from disk
        self._snapshot_managers: dict[str, SnapshotManager] = {}

        # Phase 4: State manager for crash recovery and transaction safety
        # Enables write-ahead logging (WAL) for commit operations
        self._state_manager = StateManager(self.work_dir)

        # Phase 4 Integration: Initialize the three manager layers
        # These provide separation of concerns for buffer mgmt, indexing, and search
        # Note: SearchService imports sklearn, which has Windows compatibility issues
        # If sklearn fails, fall back to monolithic implementation gracefully
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
        except (ImportError, OSError, RuntimeError) as e:
            # OSError/RuntimeError catch sklearn Windows fatal exception (0xc0000139)
            logger.warning(f"Phase 4 integration unavailable: {type(e).__name__}: {e}. Falling back to monolithic implementation.")
            self._buffer_manager = None
            self._index_manager = None
            self._search_service = None
        
        # Phase 7: Prometheus metrics export via HTTP endpoint
        self._prometheus_exporter = None
        if enable_prometheus:
            try:
                self._prometheus_exporter = configure_prometheus(
                    port=prometheus_port,
                    start_server=True,
                )
                self._prometheus_exporter.set_embedding_dimension(self._embedding_dim)
                
                # Share Prometheus exporter with managers
                if self._index_manager:
                    self._index_manager._prometheus_exporter = self._prometheus_exporter
                if self._search_service:
                    self._search_service._prometheus_exporter = self._prometheus_exporter
                
                logger.info(f"Prometheus metrics endpoint enabled on port {prometheus_port}")
            except ImportError:
                logger.warning(
                    "Prometheus metrics disabled: prometheus-client not installed. "
                    "Install with: pip install prometheus-client"
                )

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------
    def _audit_log(self, operation: str, buffer_id: str | None = None, status: str = "ok", details: dict[str, Any] | None = None) -> None:
        """Log an audit entry to the audit trail.
        
        Args:
            operation: Operation name (e.g., 'embed_codebase', 'write_code', 'commit').
            buffer_id: Buffer ID associated with the operation.
            status: Status (e.g., 'ok', 'error', 'conflict').
            details: Additional operation details (files, line counts, conflicts, etc.).
        """
        try:
            entry = {
                "timestamp": time.time(),
                "operation": operation,
                "buffer_id": buffer_id,
                "status": status,
            }
            if details:
                entry["details"] = details
            
            with self._audit_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            json_logger.warning(
                operation='audit_log',
                status='error',
                message=f'Could not write audit log: {exc}',
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
        t0 = time.perf_counter()
        root = Path(path)
        files = [root] if root.is_file() else sorted(root.rglob(pattern))
        if not files:
            return {"status": "warning", "message": f"No files matched '{pattern}' in {root}"}

        preflight = self.check_codebase(path, pattern)
        if preflight["status"] == "exceeds_threshold":
            return {
                "status": "warning",
                "message": (
                    f"Codebase too large ({preflight['estimated_mb']:.1f} MB exceeds "
                    f"threshold {preflight['threshold_mb']:.1f} MB)."
                ),
                "suggested_max": f"{preflight['threshold_mb']:.0f} MB",
                "estimated_mb": preflight["estimated_mb"],
            }

        # Stage 1: chunk all files
        all_chunks: list[CodeChunk] = []
        file_chunks_map: dict[str, list[int]] = {}  # rel_path -> list of chunk indices in all_chunks
        for f in files:
            try:
                chunks = chunk_file(f, language_hint=language_hint, sliding_window_size=sliding_window_size)
            except Exception as exc:
                json_logger.warning(
                    operation='chunk_file',
                    message=f'Failed to chunk {f}: {exc}',
                )
                continue
            rel = str(f.relative_to(root))
            file_chunks_map[rel] = []
            for ch in chunks:
                ch.file = rel
                file_chunks_map[rel].append(len(all_chunks))
                all_chunks.append(ch)

        if not all_chunks:
            return {"status": "warning", "message": "No chunks extracted from input files."}

        token_count = len(all_chunks)
        size_check = check_size(token_count, self._embedding_dim, self.threshold_mb)
        if size_check["status"] == "exceeds_threshold":
            return {
                "status": "warning",
                "message": (
                    f"Codebase too large ({size_check['estimated_mb']:.1f} MB exceeds "
                    f"threshold {size_check['threshold_mb']:.1f} MB)."
                ),
                "suggested_max": f"{size_check['threshold_mb']:.0f} MB",
                "estimated_mb": size_check["estimated_mb"],
            }

        # Stage 2: embed chunks in batches
        texts = [ch.text for ch in all_chunks]
        embeddings = self._embedder.encode(texts, batch_size=64)

        # Stage 3: build FAISS index
        index = GpuIndex(self._embedding_dim, use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        ids = index.new_ids(len(all_chunks))
        index.add(ids, embeddings)
        
        # Pre-sync GPU to avoid lazy sync latency on first search (Phase 5 optimization)
        index.sync_gpu()

        # Stage 4: persist  (buffer_id assigned here so later stages can reference it)
        buffer_id = str(uuid.uuid4())
        buffer_dir = self.work_dir / f"{buffer_id}.gcbuff"
        buffer_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata-only snapshot (no full source code stored)
        snapshot_mgr = SnapshotManager(buffer_dir)
        files_dict = {str(f.relative_to(root)): f for f in files}
        manifest = snapshot_mgr.create_snapshot(buffer_id, root, files_dict)

        # Create initial source_snapshot.json (buffer state at embed time)
        # This tracks buffer edits separately from the on-disk manifest
        source_snapshot: dict[str, list[str]] = {}
        for rel_path, file_path in files_dict.items():
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_snapshot[rel_path] = f.read().splitlines()
        snapshot_path = buffer_dir / "source_snapshot.json"
        snapshot_path.write_bytes(
            json.dumps(source_snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )

        # Store file hashes from metadata for quick change detection
        file_hashes: dict[str, str] = {
            rel_path: meta.hash 
            for rel_path, meta in manifest.files.items()
        }

        # Cache the snapshot manager for later use
        self._snapshot_managers[buffer_id] = snapshot_mgr

        # Stage 5: register
        self._registry[buffer_id] = {
            "root": str(root),
            "buffer_dir": str(buffer_dir),
            "chunk_count": token_count,
            "embedding_dim": self._embedding_dim,
            "size_bytes": embeddings.nbytes,
            "file_hashes": file_hashes,
            "pattern": pattern,
            "language_hint": language_hint,
            "sliding_window_size": sliding_window_size,
            "dirty_files": {},
        }
        # Stage 5b: build lexical BM25 index (now that buffer_id is defined)
        lexical = LexicalIndex()
        for i, ch in enumerate(all_chunks):
            lexical.add(i, ch.text)
        self._lexical_cache[buffer_id] = lexical

        self._save_registry()
        self._index_cache[buffer_id] = index
        
        # Persist all state including lexical index
        self._save_buffer_state(buffer_dir, all_chunks, embeddings, index, lexical_index=lexical, file_chunks_map=file_chunks_map)
        
        # Calculate elapsed time
        elapsed = time.perf_counter() - t0
        
        # Phase 7: Record operation metrics for Prometheus
        if self._prometheus_exporter:
            self._prometheus_exporter.record_operation(
                operation='embed_codebase',
                duration_s=elapsed,
                status='ok',
                chunk_count=token_count,
            )
        
        # Log embedding completion with JSON structure
        json_logger.info(
            operation='embed_codebase',
            buffer_id=buffer_id,
            elapsed_s=elapsed,
            status='ok',
            message=f'Embedded {token_count} chunks from {len(files)} files',
            details={
                'files_count': len(files),
                'chunks_count': token_count,
                'size_bytes': embeddings.nbytes,
            },
        )
        metrics = get_metrics()
        metrics.record_histogram("embed_codebase_latency_s", elapsed)
        metrics.record_histogram("embed_chunks_count", token_count)
        metrics.increment_counter("embed_codebase_calls")
        
        response = EmbedResponse(
            status=ResponseStatus.OK,
            buffer_id=buffer_id,
            chunk_count=token_count,
            size_bytes=embeddings.nbytes,
            message=f"Embedded {token_count} chunks from {len(files)} files.",
        )
        result = response.to_dict()
        
        # Audit log successful embedding
        self._audit_log(
            operation="embed_codebase",
            buffer_id=buffer_id,
            status="ok",
            details={
                "files_count": len(files),
                "chunks_count": token_count,
                "size_bytes": embeddings.nbytes,
                "elapsed_s": elapsed,
            }
        )
        
        return result

    # ------------------------------------------------------------------
    # Reload without re-embedding
    # ------------------------------------------------------------------
    def reload_codebase(self, buffer_id: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        root = Path(info["root"])
        
        # Use SnapshotManager to detect external changes
        snapshot_mgr = self._get_snapshot_manager(buffer_id)
        if snapshot_mgr is None:
            return {"status": "error", "message": "Snapshot not available"}
        
        changes = snapshot_mgr.detect_external_changes()
        mismatched = changes["changed"] + changes["deleted"]
        
        if not mismatched:
            # Warm index cache if cold
            self._get_index(buffer_id)
            if self._prometheus_exporter:
                elapsed = time.perf_counter() - t0
                self._prometheus_exporter.record_operation(
                    operation='reload_codebase',
                    duration_s=elapsed,
                    status='ok',
                    chunk_count=info.get("chunk_count", 0),
                )
            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "chunk_count": info.get("chunk_count", 0),
                "size_bytes": info.get("size_bytes", 0),
                "message": "All file hashes match; reloaded without re-embedding.",
            }

        json_logger.info(
            operation='reload_codebase',
            buffer_id=buffer_id,
            message=f'{len(mismatched)} file(s) changed on disk; re-embedding only changed files',
            details={'changed_files': len(mismatched)},
        )

        # Update snapshot and registry for changed files
        pattern = info.get("pattern", "*.py")
        files = [root] if root.is_file() else sorted(root.rglob(pattern))
        
        # Rebuild mismatched files
        self._rebuild_files(buffer_id, mismatched)
        
        # Update file hashes in registry from updated snapshot manifest
        new_hashes: dict[str, str] = {}
        for rel_path, meta in snapshot_mgr.manifest.files.items():
            if rel_path in mismatched:
                new_hashes[rel_path] = meta.hash
        
        if new_hashes:
            info.setdefault("file_hashes", {}).update(new_hashes)
            self._save_registry()

        elapsed = time.perf_counter() - t0
        if self._prometheus_exporter:
            self._prometheus_exporter.record_operation(
                operation='reload_codebase',
                duration_s=elapsed,
                status='ok',
                chunk_count=len(mismatched),
            )

        return {
            "status": "ok",
            "buffer_id": buffer_id,
            "chunk_count": info.get("chunk_count", 0),
            "size_bytes": info.get("size_bytes", 0),
            "message": f"Re-embedded {len(mismatched)} changed file(s).",
        }

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
        err = self._validate_search_params(query, top_k=top_k)
        if err is not None:
            return err

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
        err = self._validate_search_params(query, top_k=top_k)
        if err is not None:
            return err

        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="hybrid_search")

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
        err = self._validate_search_params(query, max_results=max_results)
        if err is not None:
            return err

        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}
        
        snapshot_mgr = self._get_snapshot_manager(buffer_id)
        if snapshot_mgr is None:
            return {"status": "error", "message": "Snapshot not available."}

        matches: list[dict[str, Any]] = []
        target = query if case_sensitive else query.lower()

        for rel_path in snapshot_mgr.manifest.files.keys():
            lines = snapshot_mgr.read_lines(rel_path)
            if lines is None:
                json_logger.warning(
                    operation='search_for',
                    message=f'Failed to read file {rel_path}',
                )
                continue
            
            for line_no, raw_line in enumerate(lines, start=1):
                haystack = raw_line if case_sensitive else raw_line.lower()
                if target in haystack:
                    matches.append({
                        "file": rel_path,
                        "line": line_no,
                        "content": raw_line,
                    })
                    if len(matches) >= max_results:
                        return {"status": "ok", "matches": matches, "total": len(matches)}

        return {"status": "ok", "matches": matches, "total": len(matches)}

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

        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}

        root = Path(info["root"])
        files = list(snapshot.keys())

        # 1) Exact relative-path match
        if file_name in files:
            return {
                "status": "ok",
                "file_location": file_name,
                "absolute_path": str(root / file_name),
                "match_type": "exact",
            }

        # 2) Basename exact match
        for rel_path in files:
            if Path(rel_path).name == file_name:
                return {
                    "status": "ok",
                    "file_location": rel_path,
                    "absolute_path": str(root / rel_path),
                    "match_type": "basename",
                }

        # 3) Partial (substring) match
        query_lower = file_name.lower()
        candidates = [rel_path for rel_path in files if query_lower in rel_path.lower()]

        if len(candidates) == 1:
            return {
                "status": "ok",
                "file_location": candidates[0],
                "absolute_path": str(root / candidates[0]),
                "match_type": "partial",
            }
        if len(candidates) > 1:
            return {
                "status": "ok",
                "candidates": candidates,
                "match_type": "multiple",
                "message": f"Found {len(candidates)} matching files.",
            }

        return {"status": "error", "message": f"File '{file_name}' not found in buffer."}

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
        err = self._validate_search_params(query, top_k=top_k)
        if err is not None:
            return err

        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="search_symbols")

        chunks = self._load_chunks(buffer_id)
        if chunks is None:
            return self._make_error_response(
                "Chunk metadata missing", buffer_id=buffer_id, operation="search_symbols"
            )

        _SYMBOL_TYPES = {
            "function",
            "class",
            "method",
            "struct_item",
            "impl_item",
            "trait_item",
            "interface_definition",
            "enum_item",
        }

        query_lower = query.lower()
        name_matches: list[tuple[int, SearchMatch]] = []  # (chunk_idx, match)
        for idx, ch in enumerate(chunks):
            if ch.name and query_lower in ch.name.lower():
                name_matches.append((
                    idx,
                    SearchMatch(
                        file=ch.file,
                        start_line=ch.start_line,
                        end_line=ch.end_line,
                        type=ch.type,
                        name=ch.name,
                        score=1.0,
                        doc_id=idx,
                        match_type="name",
                    )
                ))

        # Semantic search filtered to symbols
        index = self._get_index(buffer_id)
        q_emb = self._embedder.encode([query], batch_size=1)
        distances, indices = index.search(q_emb, top_k * 3)

        semantic_matches: list[tuple[int, SearchMatch]] = []  # (chunk_idx, match)
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            ch = chunks[idx]
            if ch.type in _SYMBOL_TYPES:
                semantic_matches.append((
                    int(idx),
                    SearchMatch(
                        file=ch.file,
                        start_line=ch.start_line,
                        end_line=ch.end_line,
                        type=ch.type,
                        name=ch.name,
                        score=float(score),
                        doc_id=int(idx),
                        match_type="semantic",
                    )
                ))
            if len(semantic_matches) >= top_k:
                break

        # Merge with dedup (name matches first)
        seen: set[int] = set()  # Track chunk indices to avoid duplicates
        merged: list[SearchMatch] = []
        for chunk_idx, match in name_matches + semantic_matches:
            if chunk_idx not in seen:
                seen.add(chunk_idx)
                merged.append(match)
            if len(merged) >= top_k:
                break

        response = SearchResponse(
            status=ResponseStatus.OK,
            matches=merged,
        )
        return response.to_dict()

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
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return {"status": "error", "message": "No chunks loaded."}

        # Load embeddings
        emb_path = Path(info["buffer_dir"]) / "embeddings.npy"
        if not emb_path.exists():
            return {"status": "error", "message": "Embeddings file missing."}
        embeddings = np.load(emb_path)

        # Greedy clustering on chunk embeddings (simpler than line-level)
        n = embeddings.shape[0]
        cluster_items: list[ClusterItem] = []
        visited = set()
        for i in range(n):
            if i in visited:
                continue
            cluster_indices = [i]
            visited.add(i)
            for j in range(i + 1, min(n, i + 64)):  # look ahead window
                if j in visited:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim > threshold:
                    cluster_indices.append(j)
                    visited.add(j)
            if len(cluster_indices) >= 2:
                start_ch = chunks[cluster_indices[0]]
                end_ch = chunks[cluster_indices[-1]]
                avg_score = float(np.mean([
                    float(np.dot(embeddings[cluster_indices[a]], embeddings[cluster_indices[a + 1]]))
                    for a in range(len(cluster_indices) - 1)
                ]))
                cluster_items.append(ClusterItem(
                    file=start_ch.file,
                    start_line=start_ch.start_line,
                    end_line=end_ch.end_line,
                    size=len(cluster_indices),
                    avg_score=round(avg_score, 4),
                ))

        response = ClusterResponse(
            status=ResponseStatus.OK,
            clusters=cluster_items,
        )
        return response.to_dict()

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
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        chunks = self._load_chunks(buffer_id)
        if chunks is None or not chunks:
            return {"status": "error", "message": "No chunks loaded."}

        duplicates = find_duplicates(chunks, threshold=threshold)
        return {"status": "ok", "duplicates": duplicates, "total": len(duplicates)}

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
            self._audit_log(
                operation="read_code",
                buffer_id=buffer_id,
                status="ok",
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
        self._audit_log(
            operation="read_code",
            buffer_id=buffer_id,
            status="ok",
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
        self._audit_log(
            operation="write_code",
            buffer_id=buffer_id,
            status="ok",
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
            status = "conflict" if conflicts else "ok"
            self._audit_log(
                operation="commit",
                buffer_id=buffer_id,
                status=status,
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
        return {
            "status": "ok",
            "buffers": [{"buffer_id": bid, **info} for bid, info in self._registry.items()],
        }

    def delete_buffer(self, buffer_id: str) -> dict[str, Any]:
        info = self._registry.pop(buffer_id, None)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}
        self._save_registry()
        self._index_cache.pop(buffer_id, None)
        self._lexical_cache.pop(buffer_id, None)
        self._query_cache.invalidate_buffer(buffer_id)
        shutil.rmtree(info["buffer_dir"], ignore_errors=True)
        return {"status": "ok", "message": f"Deleted buffer {buffer_id}"}

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
        with self._registry_path.open("wb") as fh:
            fh.write(
                json.dumps(self._registry, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def get_cache_stats(self) -> dict[str, Any]:
        """Return cache utilization statistics.

        Returns a dict with:
        - index_cache_size: Current number of cached indices
        - index_cache_max: Maximum cached indices (LRU limit)
        - lexical_cache_size: Current number of cached lexical indices
        - lexical_cache_max: Maximum cached lexical indices (LRU limit)
        - query_cache_stats: Hit/miss counts from query cache
        """
        return {
            "index_cache_size": len(self._index_cache),
            "index_cache_max": self._index_cache.max_size,
            "lexical_cache_size": len(self._lexical_cache),
            "lexical_cache_max": self._lexical_cache.max_size,
            "query_cache_stats": self._query_cache.stats(),
        }

    def health_check(self) -> dict[str, Any]:
        """Perform a health check and return system status.

        Returns a dict with:
        - status: "healthy" if all checks pass, "degraded" if warnings, "unhealthy" if critical issues
        - timestamp: ISO 8601 timestamp of check
        - buffers_registered: Number of embedded codebases in registry
        - buffers_loaded: Number of indices currently in memory
        - cache_utilization: Current cache usage as percentage
        - embedder_ready: Whether embedder is initialized
        - faiss_gpu_available: Whether GPU FAISS is available
        - warnings: List of any warnings (e.g., high cache usage, GPU not available)

        Can be called periodically (e.g., every 60 seconds) to monitor system health.
        """
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
        
        # Phase 7: Record buffer count metrics for Prometheus
        if self._prometheus_exporter:
            self._prometheus_exporter.set_buffer_counts(
                registered=len(self._registry),
                loaded=len(self._index_cache),
            )
        
        return result

    @staticmethod
    def get_metrics() -> dict[str, Any]:
        """Export current metrics snapshot.

        Returns a dict with counters (operation counts), gauges, histograms
        (latency and size distributions), and cache hit rates.  Can be
        serialized to JSON for monitoring integration.
        """
        return get_metrics().dump_metrics()

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
