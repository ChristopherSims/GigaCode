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

from src.chunker import CodeChunk, chunk_file, chunk_text
from src.context_packer import pack_context
from src.diff_engine import compute_diff, hash_lines
from src.duplicate_detector import find_duplicates
from src.embedder import Embedder
from src.gpu_index import GpuIndex
from src.hybrid_search import reciprocal_rank_fusion
from src.lexical_index import LexicalIndex
from src.metadata_store import load_metadata, save_metadata
from src.metrics import get_metrics
from src.query_cache import QueryCache
from src.response_types import (
    SearchMatch,
    SearchResponse,
    ResponseStatus,
    EmbedResponse,
    ClusterResponse,
    ClusterItem,
    ErrorResponse,
)
from src.size_guard import check_size
from src.buffer_state import BufferState, BufferStateTransition
from src.operation_config import OperationType, OperationConfig
from src.health_status import HealthStatus, HealthStatusTracker
from gigacode.dependency_graph import DependencyGraph
from gigacode.dead_code_detector import DeadCodeDetector
from gigacode.todo_tracker import TodoTracker
from gigacode.quality_scorer import QualityScorer
from gigacode.progress_stream import ProgressReporter
from gigacode.conversation_memory import ConversationMemory

logger = logging.getLogger(__name__)

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
            logger.debug("LRU eviction: removed key=%s (size now %d)", oldest_key, len(self))

    def get(self, key: str, default: Any = None) -> Any:
        """Get with default; moves to end if found."""
        if key in self:
            return self[key]
        return default



class CodeEmbeddingTool:
    """Embed a codebase into GPU/CPU buffers and expose search + cluster.

    Args:
        work_dir: Directory where buffer files and metadata are persisted.
        model_name: Sentence-transformers model name.
        device: torch device (``"cuda"``, ``"cpu"``, or ``None`` for auto).
        threshold_mb: Size-guard threshold in megabytes.
        use_gpu: Whether to mirror the FAISS index to GPU when possible.
        max_buffers: Maximum number of in-memory indices to keep (LRU eviction).
    """

    def __init__(
        self,
        work_dir: str | Path,
        model_name: str | None = None,
        device: str | None = None,
        threshold_mb: float = 500.0,
        use_gpu: bool = True,
        max_buffers: int = 10,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_mb = threshold_mb
        self.use_gpu = use_gpu
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

        # In-memory index cache: buffer_id -> GpuIndex (LRU-bounded)
        self._index_cache: LRUDict = LRUDict(max_size=max_buffers)

        # In-memory lexical index cache: buffer_id -> LexicalIndex (LRU-bounded)
        self._lexical_cache: LRUDict = LRUDict(max_size=max_buffers)

        # Query result cache
        self._query_cache = QueryCache(maxsize=256)

        # Health status tracker (Phase 6)
        self._health_tracker = HealthStatusTracker()
        
        # Conversation memory (Phase 10)
        self._conversation_memory = ConversationMemory(self.work_dir / "memories.json")

        # Multi-turn conversation memory (Phase 10)
        self._conversation_memory = ConversationMemory(self.work_dir / "memories.json")

    # ------------------------------------------------------------------
    # Schema exposure
    # ------------------------------------------------------------------
    @staticmethod
    def get_tool_schemas() -> list[dict[str, Any]]:
        from src.tool_schema import get_all_schemas
        return get_all_schemas()

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
                logger.debug("Could not count lines in %s: %s", f, exc)

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
                logger.warning("Failed to chunk %s: %s", f, exc)
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
        index = GpuIndex(self._embedding_dim, use_gpu=self.use_gpu)
        ids = index.new_ids(len(all_chunks))
        index.add(ids, embeddings)

        # Stage 4: persist  (buffer_id assigned here so later stages can reference it)
        buffer_id = str(uuid.uuid4())
        buffer_dir = self.work_dir / f"{buffer_id}.gcbuff"
        buffer_dir.mkdir(parents=True, exist_ok=True)

        # Source snapshot (line-based, for read/write)
        source_snapshot: dict[str, list[str]] = {}
        file_hashes: dict[str, str] = {}
        for f in files:
            rel = str(f.relative_to(root))
            raw = f.read_text(encoding="utf-8", errors="replace")
            normalized = "\n".join(raw.splitlines())
            source_snapshot[rel] = normalized.splitlines()
            file_hashes[rel] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        snapshot_path = buffer_dir / "source_snapshot.json"
        snapshot_path.write_bytes(
            json.dumps(source_snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )

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
        logger.info(
            "embed_codebase completed: buffer_id=%s chunks=%d files=%d elapsed=%.3fs",
            buffer_id, token_count, len(files), elapsed,
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
        return response.to_dict()

    # ------------------------------------------------------------------
    # Reload without re-embedding
    # ------------------------------------------------------------------
    def reload_codebase(self, buffer_id: str) -> dict[str, Any]:
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        root = Path(info["root"])
        pattern = info.get("pattern", "*.py")
        files = [root] if root.is_file() else sorted(root.rglob(pattern))

        old_hashes = info.get("file_hashes", {})
        mismatched: list[str] = []
        for f in files:
            rel = str(f.relative_to(root))
            raw = f.read_text(encoding="utf-8", errors="replace")
            normalized = "\n".join(raw.splitlines())
            current = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if old_hashes.get(rel) != current:
                mismatched.append(rel)

        if not mismatched:
            # Warm index cache if cold
            self._get_index(buffer_id)
            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "chunk_count": info.get("chunk_count", 0),
                "size_bytes": info.get("size_bytes", 0),
                "message": "All file hashes match; reloaded without re-embedding.",
            }

        logger.info(
            "reload_codebase(%s): %d file(s) changed on disk; re-embedding only changed files.",
            buffer_id,
            len(mismatched),
        )

        # Update source snapshot and hashes for changed files before rebuilding
        snapshot = self._load_source_snapshot(buffer_id) or {}
        new_hashes: dict[str, str] = {}
        for rel_path in mismatched:
            f = root / rel_path
            try:
                raw = f.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("reload_codebase: cannot read %s: %s", rel_path, exc)
                continue
            normalized = "\n".join(raw.splitlines())
            snapshot[rel_path] = normalized.splitlines()
            new_hashes[rel_path] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        self._save_source_snapshot(buffer_id, snapshot)
        info.setdefault("file_hashes", {}).update(new_hashes)
        self._save_registry()

        self._rebuild_files(buffer_id, mismatched)

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
        logger.debug(
            "semantic_search: buffer_id=%s top_k=%d elapsed=%.3fs gpu=%s",
            buffer_id, top_k, elapsed, index.is_gpu,
        )

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
        logger.debug(
            "hybrid_search semantic leg: buffer_id=%s top_k=%d elapsed=%.3fs gpu=%s",
            buffer_id, top_k, elapsed, index.is_gpu,
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
        """Find every occurrence of *query* in the buffered source snapshot.

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
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}

        matches: list[dict[str, Any]] = []
        target = query if case_sensitive else query.lower()

        for rel_path, lines in snapshot.items():
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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="read_code")

        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return self._make_error_response(
                "Source snapshot missing", buffer_id=buffer_id, operation="read_code",
                context={"reason": "snapshot_file_not_found"}
            )

        if file is not None:
            if file not in snapshot:
                return self._make_error_response(
                    f"File not in buffer: {file}", buffer_id=buffer_id, operation="read_code",
                    context={"requested_file": file}
                )
            lines = snapshot[file]
            end = end_line if end_line is not None else len(lines) + 1
            selected = lines[start_line - 1:end - 1]
            return {
                "status": "ok",
                "file": file,
                "start_line": start_line,
                "end_line": end,
                "lines": selected,
            }

        result: dict[str, list[str]] = {}
        for fname, lines in snapshot.items():
            end = end_line if end_line is not None else len(lines) + 1
            result[fname] = lines[start_line - 1:end - 1]
        return {"status": "ok", "files": result}

    def write_code(
        self,
        buffer_id: str,
        file: str,
        start_line: int,
        new_lines: list[str],
        end_line: int | None = None,
    ) -> dict[str, Any]:
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}
        if file not in snapshot:
            return {"status": "error", "message": f"File not in buffer: {file}"}

        old_lines = snapshot[file]
        end = end_line if end_line is not None else len(old_lines) + 1
        new_file_lines = old_lines[:start_line - 1] + new_lines + old_lines[end - 1:]
        snapshot[file] = new_file_lines
        self._save_source_snapshot(buffer_id, snapshot)

        dirty = info.setdefault("dirty_files", {})
        dirty[file] = True
        self._save_registry()

        # Auto-rebuild if too many dirty files
        if len(dirty) >= _MAX_DIRTY_BEFORE_AUTO_REBUILD:
            self._rebuild_dirty(buffer_id)

        return {
            "status": "ok",
            "file": file,
            "changed_lines": len(new_lines),
            "replaced_lines": end - start_line,
            "total_lines": len(new_file_lines),
        }

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        root = Path(info["root"])
        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}

        dirty = info.get("dirty_files", {})
        if not dirty:
            return {"status": "ok", "written_files": [], "dry_run": dry_run}

        # Rebuild embeddings for dirty files before writing to disk
        if not dry_run:
            self._rebuild_dirty(buffer_id)

        written: list[str] = []
        new_hashes: dict[str, str] = {}
        for rel_path in dirty:
            lines = snapshot.get(rel_path, [])
            disk_path = root / rel_path
            if not dry_run:
                if disk_path.exists():
                    raw_text = disk_path.read_text(encoding="utf-8", errors="replace")
                    normalized = "\n".join(raw_text.splitlines())
                    disk_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                    old_hash = info.get("file_hashes", {}).get(rel_path)
                    if old_hash is not None and disk_hash != old_hash:
                        return {
                            "status": "error",
                            "message": (
                                f"File {rel_path} changed on disk since embedding "
                                f"(hash mismatch). Aborting commit."
                            ),
                        }
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                disk_path.write_text("\n".join(lines), encoding="utf-8")
                new_hashes[rel_path] = hashlib.sha256(
                    "\n".join(lines).encode("utf-8")
                ).hexdigest()
            written.append(rel_path)

        if not dry_run:
            info["file_hashes"].update(new_hashes)
            info["dirty_files"] = {}
            self._save_registry()

        return {"status": "ok", "written_files": written, "dry_run": dry_run}

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
        snapshot = self._load_source_snapshot(buffer_id)
        language_hint = info.get("language_hint")
        sliding_window_size = info.get("sliding_window_size", 30)
        new_embeddings_list: list[np.ndarray] = []
        for rel_path in files:
            lines = snapshot.get(rel_path, [])
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
        index = GpuIndex(info["embedding_dim"], use_gpu=self.use_gpu)
        index_path = Path(info["buffer_dir"]) / "index.faiss"
        if index_path.exists():
            index.load(index_path)
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
            logger.warning("Failed to load lexical index from %s: %s", path, exc)
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
        
        logger.warning(
            "Lexical index missing for buffer_id=%s; building from chunks "
            "(%d items). Consider warmup via embed_codebase or commit.",
            buffer_id, len(chunks),
        )
        lexical = LexicalIndex()
        for i, ch in enumerate(chunks):
            lexical.add(i, ch.text)
        self._lexical_cache[buffer_id] = lexical
        return lexical

    def _get_buffer_info(self, buffer_id: str) -> dict[str, Any] | None:
        return self._registry.get(buffer_id)

    def _save_registry(self) -> None:
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
        embedder_ready = self._embedder is not None and self._embedder.model is not None
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

    def close(self) -> None:
        """Release all in-memory resources.

        Clears the FAISS index cache (drops GPU VRAM), the lexical index
        cache, and the query result cache.  The on-disk registry and buffer
        files are left intact — they can be reloaded by constructing a new
        ``CodeEmbeddingTool`` with the same *work_dir*.
        """
        self._index_cache.clear()
        self._lexical_cache.clear()
        self._query_cache.clear()
        logger.debug("CodeEmbeddingTool closed; all in-memory caches released.")

    # ------------------------------------------------------------------
    # Phases 5-10: Call/Dependency Graph, Dead Code, TODOs, Quality,
    #              Progress Streaming, Conversation Memory
    # ------------------------------------------------------------------
    def trace_call_chain(
        self,
        buffer_id: str,
        from_symbol: str,
        to_symbol: str,
        max_depth: int = 10,
    ) -> dict[str, Any]:
        """Find call chain between two symbols."""
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="trace_call_chain")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="trace_call_chain")
        graph = DependencyGraph(chunks)
        result = graph.trace_call_chain(from_symbol, to_symbol, max_depth)
        return {"status": "ok", **result.to_dict()}

    def get_dependencies(
        self,
        buffer_id: str,
        file: str,
        direction: str = "both",
    ) -> dict[str, Any]:
        """Get file dependencies."""
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="get_dependencies")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="get_dependencies")
        graph = DependencyGraph(chunks)
        deps = graph.get_dependencies(file, direction)
        return {"status": "ok", "file": file, "direction": direction, "dependencies": deps}

    def find_circular_dependencies(self, buffer_id: str) -> dict[str, Any]:
        """Find circular dependencies in a buffer."""
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="find_circular_dependencies")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="find_circular_dependencies")
        graph = DependencyGraph(chunks)
        cycles = graph.find_cycles()
        return {"status": "ok", "cycles": cycles, "cycle_count": len(cycles)}

    def export_dependency_graph(self, buffer_id: str, format: str = "json") -> dict[str, Any]:
        """Export dependency graph."""
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="export_dependency_graph")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="export_dependency_graph")
        graph = DependencyGraph(chunks)
        return graph.export_graph(format)

    def find_dead_code(self, buffer_id: str, min_confidence: str = "medium") -> dict[str, Any]:
        """Find dead code and unused imports in a buffer."""
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="find_dead_code")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="find_dead_code")
        detector = DeadCodeDetector(chunks)
        dead = detector.find_dead_code(min_confidence)
        unused = detector.find_unused_imports()
        return {"status": "ok", "dead_symbols": [s.to_dict() for s in dead], "unused_imports": unused}

    def extract_todos(self, buffer_id: str, tag: str | None = None) -> dict[str, Any]:
        """Extract TODO/FIXME comments from a buffer."""
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="extract_todos")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="extract_todos")
        tracker = TodoTracker()
        todos = tracker.extract_todos(chunks)
        if tag:
            todos = [t for t in todos if t.tag == tag.upper()]
        return {"status": "ok", "todos": [t.to_dict() for t in todos], "total": len(todos)}

    def score_code_quality(self, buffer_id: str, file: str) -> dict[str, Any]:
        """Score code quality for a specific file in a buffer."""
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="score_code_quality")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="score_code_quality")
        scorer = QualityScorer()
        result = scorer.score_file(chunks, file)
        if not result:
            return self._make_error_response("File not found in buffer", buffer_id=buffer_id, operation="score_code_quality")
        return {"status": "ok", **result.to_dict()}

    def _create_progress_reporter(self, phases: list[str]) -> "ProgressReporter":
        """Create a progress reporter for long-running operations."""
        return ProgressReporter(phases)

    def remember(self, key: str, value: str, tags: list[str] | None = None) -> dict[str, Any]:
        """Store a fact in conversation memory."""
        return self._conversation_memory.remember(key, value, tags)

    def recall(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Recall facts from conversation memory."""
        memories = self._conversation_memory.recall(query, top_k)
        return {"status": "ok", "memories": [m.to_dict() for m in memories]}

    def list_memories(self, tag: str | None = None) -> dict[str, Any]:
        """List all stored memories, optionally filtered by tag."""
        memories = self._conversation_memory.list_memories(tag)
        return {"status": "ok", "memories": [m.to_dict() for m in memories], "total": len(memories)}

    def forget(self, key: str) -> dict[str, Any]:
        """Forget a stored memory by key."""
        return self._conversation_memory.forget(key)
    
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
    
    def _get_buffer_state(self, buffer_id: str) -> BufferState:
        """Get current buffer state from registry.
        
        Args:
            buffer_id: Buffer identifier
        
        Returns:
            BufferState enum value
        
        Raises:
            ValueError: If buffer_id not found
        """
        if buffer_id not in self._registry:
            raise ValueError(f"Unknown buffer_id: {buffer_id}")
        
        info = self._registry[buffer_id]
        state_str = info.get("state", BufferState.READY.value)
        return BufferState(state_str)
    
    def _set_buffer_state(self, buffer_id: str, new_state: BufferState) -> None:
        """Set buffer state with validation and logging.
        
        Args:
            buffer_id: Buffer identifier
            new_state: New buffer state
        
        Raises:
            ValueError: If transition is invalid
        """
        if buffer_id not in self._registry:
            raise ValueError(f"Unknown buffer_id: {buffer_id}")
        
        info = self._registry[buffer_id]
        current_state_str = info.get("state", BufferState.READY.value)
        current_state = BufferState(current_state_str)
        
        # Validate transition
        BufferStateTransition.validate_or_raise(current_state, new_state)
        
        # Update registry
        info["state"] = new_state.value
        info["state_changed_at"] = time.time()
        
        # Persist registry
        self._save_registry()
        
        # Update health tracker
        self._health_tracker.update_buffer_state(buffer_id, new_state)
        
        # Log transition
        logger.info(
            f"Buffer state transition: {buffer_id} {current_state.value} → {new_state.value}"
        )
    
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
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        self._registry_path.write_text(
            json.dumps(self._registry, separators=(",", ":"), indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Phase 5: Dependency Graph
    # ------------------------------------------------------------------
    def trace_call_chain(self, buffer_id: str, from_symbol: str, to_symbol: str, max_depth: int = 10) -> dict[str, Any]:
        """Find call chain between two symbols."""
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="trace_call_chain")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="trace_call_chain")
        graph = DependencyGraph(chunks)
        result = graph.trace_call_chain(from_symbol, to_symbol, max_depth)
        return {"status": "ok", **result.to_dict()}

    def get_dependencies(self, buffer_id: str, file: str, direction: str = "both") -> dict[str, Any]:
        """Get file dependencies."""
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="get_dependencies")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="get_dependencies")
        graph = DependencyGraph(chunks)
        deps = graph.get_dependencies(file, direction)
        return {"status": "ok", "file": file, "direction": direction, "dependencies": deps}

    def find_circular_dependencies(self, buffer_id: str) -> dict[str, Any]:
        """Find circular dependencies in a buffer."""
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="find_circular_dependencies")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="find_circular_dependencies")
        graph = DependencyGraph(chunks)
        cycles = graph.find_cycles()
        return {"status": "ok", "cycles": cycles, "cycle_count": len(cycles)}

    def export_dependency_graph(self, buffer_id: str, format: str = "json") -> dict[str, Any]:
        """Export dependency graph."""
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="export_dependency_graph")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="export_dependency_graph")
        graph = DependencyGraph(chunks)
        return graph.export_graph(format)

    # ------------------------------------------------------------------
    # Phase 6: Dead Code Detection
    # ------------------------------------------------------------------
    def find_dead_code(self, buffer_id: str, min_confidence: str = "medium") -> dict[str, Any]:
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="find_dead_code")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="find_dead_code")
        detector = DeadCodeDetector(chunks)
        dead = detector.find_dead_code(min_confidence)
        unused = detector.find_unused_imports()
        return {"status": "ok", "dead_symbols": [s.to_dict() for s in dead], "unused_imports": unused}

    # ------------------------------------------------------------------
    # Phase 7: TODO/FIXME Tracker
    # ------------------------------------------------------------------
    def extract_todos(self, buffer_id: str, tag: str | None = None) -> dict[str, Any]:
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="extract_todos")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="extract_todos")
        tracker = TodoTracker()
        todos = tracker.extract_todos(chunks)
        if tag:
            todos = [t for t in todos if t.tag == tag.upper()]
        return {"status": "ok", "todos": [t.to_dict() for t in todos], "total": len(todos)}

    # ------------------------------------------------------------------
    # Phase 8: Code Quality Scoring
    # ------------------------------------------------------------------
    def score_code_quality(self, buffer_id: str, file: str) -> dict[str, Any]:
        info = self._get_buffer_info(buffer_id)
        if not info:
            return self._make_error_response("Unknown buffer_id", buffer_id=buffer_id, operation="score_code_quality")
        chunks = self._load_chunks(buffer_id)
        if not chunks:
            return self._make_error_response("No chunks loaded", buffer_id=buffer_id, operation="score_code_quality")
        scorer = QualityScorer()
        result = scorer.score_file(chunks, file)
        if not result:
            return self._make_error_response("File not found in buffer", buffer_id=buffer_id, operation="score_code_quality")
        return {"status": "ok", **result.to_dict()}

    # ------------------------------------------------------------------
    # Phase 9: Streaming Support Infrastructure
    # ------------------------------------------------------------------
    def _create_progress_reporter(self, phases: list[str]) -> "ProgressReporter":
        from gigacode.progress_stream import ProgressReporter
        return ProgressReporter(phases)

    # ------------------------------------------------------------------
    # Phase 10: Multi-turn Conversation Memory
    # ------------------------------------------------------------------
    def remember(self, key: str, value: str, tags: list[str] | None = None) -> dict[str, Any]:
        return self._conversation_memory.remember(key, value, tags)

    def recall(self, query: str, top_k: int = 5) -> dict[str, Any]:
        memories = self._conversation_memory.recall(query, top_k)
        return {"status": "ok", "memories": [m.to_dict() for m in memories]}

    def list_memories(self, tag: str | None = None) -> dict[str, Any]:
        memories = self._conversation_memory.list_memories(tag)
        return {"status": "ok", "memories": [m.to_dict() for m in memories], "total": len(memories)}

    def forget(self, key: str) -> dict[str, Any]:
        return self._conversation_memory.forget(key)

    def __enter__(self) -> CodeEmbeddingTool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
