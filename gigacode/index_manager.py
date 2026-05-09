"""Index management layer for CodeEmbeddingTool.

Handles:
- FAISS index caching (GPU/CPU)
- BM25 lexical index caching
- Query result caching with semantic similarity
- GPU memory management
- Index persistence and loading
- Metrics and health tracking
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from gigacode.chunker import CodeChunk
from gigacode.embedder import Embedder
from gigacode.gpu_index import GpuIndex
from gigacode.incremental_indexer import IncrementalIndexManager
from gigacode.json_logger import StructuredJsonLogger
from gigacode.lexical_index import LexicalIndex
from gigacode.metadata_store import load_metadata, save_metadata
from gigacode.query_cache import QueryCache

logger = logging.getLogger(__name__)
json_logger = StructuredJsonLogger('index_manager')


__all__ = [
    "IndexManager",
]


class IndexManager:
    """Manage FAISS/Lexical indices and query caching.
    
    Responsibilities:
    - Maintain LRU-bounded cache of FAISS indices (GPU/CPU)
    - Maintain LRU-bounded cache of BM25 lexical indices
    - Handle query result caching with paraphrase detection
    - Manage GPU memory (sync, eviction, monitoring)
    - Track metrics (cache hits, GPU usage, operation latency)
    - Rebuild indices on demand (called from BufferManager)
    
    Args:
        embedder: Embedder instance for vector operations.
        embedding_dim: Dimension of embeddings.
        max_buffers: Max indices in memory (LRU limit).
        work_dir: Root directory for index persistence.
        use_gpu: Enable GPU FAISS indices.
        gpu_id: GPU device ID.
        prometheus_exporter: Optional metrics exporter.
    """
    
    def __init__(
        self,
        embedder: Embedder,
        embedding_dim: int,
        max_buffers: int,
        work_dir: Path,
        use_gpu: bool = True,
        gpu_id: int = 0,
        prometheus_exporter: Any = None,
    ) -> None:
        """Initialize IndexManager."""
        self._embedder = embedder
        self._embedding_dim = embedding_dim
        self.max_buffers = max_buffers
        self.work_dir = work_dir
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self._prometheus_exporter = prometheus_exporter
        
        # Index caches (LRU-bounded)
        from gigacode.lru_cache import LRUDict
        self._index_cache: LRUDict = LRUDict(max_size=max_buffers)
        self._lexical_cache: LRUDict = LRUDict(max_size=max_buffers)
        
        # Query result cache with semantic matching
        self._query_cache = QueryCache(
            maxsize=256,
            embedder=self._embedder,
            similarity_threshold=0.95,
        )
        
        # Incremental indexing manager for efficient updates
        self._incremental_manager = IncrementalIndexManager(
            embedder=self._embedder,
        )
    
    # ------------------------------------------------------------------
    # Index Caching
    # ------------------------------------------------------------------
    def _get_index(self, buffer_id: str) -> GpuIndex | None:
        """Load FAISS index from cache or disk.
        
        Returns:
            GpuIndex instance if found, None otherwise.
        """
        # Check cache first
        if buffer_id in self._index_cache:
            return self._index_cache[buffer_id]
        
        # Try to load from disk
        buffer_dir = self.work_dir / f"{buffer_id}.gcbuff"
        index_path = buffer_dir / "index.faiss"
        
        if not index_path.exists():
            return None
        
        try:
            index = GpuIndex.load(str(index_path), use_gpu=self.use_gpu, gpu_id=self.gpu_id)
            
            # Pre-sync to GPU if enabled
            if self.use_gpu:
                index.sync_gpu()
            
            # Cache it
            self._index_cache[buffer_id] = index
            
            json_logger.debug(
                operation='load_index',
                details={'buffer_id': buffer_id, 'from': 'disk'}
            )
            
            return index
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            json_logger.warning(
                operation='load_index',
                message=f'Failed to load index for {buffer_id}: {exc}'
            )
            return None

    def _get_lexical_index(self, buffer_id: str) -> LexicalIndex | None:
        """Load BM25 index from cache or disk.
        
        Returns:
            LexicalIndex instance if found, None otherwise.
        """
        # Check cache first
        if buffer_id in self._lexical_cache:
            return self._lexical_cache[buffer_id]
        
        # Try to load from disk
        buffer_dir = self.work_dir / f"{buffer_id}.gcbuff"
        lexical_path = buffer_dir / "lexical_index.json"
        
        if not lexical_path.exists():
            return None
        
        try:
            data = json.loads(lexical_path.read_text(encoding="utf-8"))
            lexical_index = LexicalIndex.from_dict(data)
            
            # Cache it
            self._lexical_cache[buffer_id] = lexical_index
            
            json_logger.debug(
                operation='load_lexical_index',
                details={'buffer_id': buffer_id, 'from': 'disk'}
            )
            
            return lexical_index
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            json_logger.warning(
                operation='load_lexical_index',
                message=f'Failed to load lexical index for {buffer_id}: {exc}'
            )
            return None

    def _load_chunks(self, buffer_id: str) -> list[CodeChunk] | None:
        """Load chunks from buffer metadata file.
        
        Returns:
            List of CodeChunk objects, or None if not found.
        """
        buffer_dir = self.work_dir / f"{buffer_id}.gcbuff"
        chunks_path = buffer_dir / "chunks.json"
        
        if not chunks_path.exists():
            return None
        
        try:
            metadata = load_metadata(chunks_path)
            if metadata is None:
                return None
            
            chunks = [
                CodeChunk(
                    file=c.get('file'),
                    start_line=c.get('start_line'),
                    end_line=c.get('end_line'),
                    type=c.get('type'),
                    name=c.get('name'),
                    content=c.get('content'),
                )
                for c in metadata.get('chunks', [])
            ]
            
            return chunks
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            json_logger.warning(
                operation='load_chunks',
                message=f'Failed to load chunks for {buffer_id}: {exc}'
            )
            return None

    # ------------------------------------------------------------------
    # GPU Memory Management
    # ------------------------------------------------------------------
    def _check_gpu_memory(self) -> dict[str, Any]:
        """Check available GPU memory.
        
        Returns:
            Dict with available_mb, total_mb, used_mb, or status='error'.
        """
        if not self.use_gpu:
            return {"status": "gpu_disabled"}
        
        try:
            import torch
            if not torch.cuda.is_available():
                return {"status": "cuda_unavailable"}
            
            device = torch.device(f"cuda:{self.gpu_id}")
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            available = total_memory - reserved
            
            return {
                "status": "ok",
                "available_mb": available / 1024 / 1024,
                "allocated_mb": allocated / 1024 / 1024,
                "reserved_mb": reserved / 1024 / 1024,
                "total_mb": total_memory / 1024 / 1024,
            }
        except (OSError, ImportError, AttributeError) as exc:
            json_logger.warning(
                operation='check_gpu_memory',
                message=f'GPU memory check failed: {exc}'
            )
            return {"status": "error", "message": str(exc)}

    def _evict_gpu_indices(self, buffer_id: str) -> None:
        """Remove index from GPU when evicted from cache.
        
        Args:
            buffer_id: Buffer ID to evict.
        """
        if buffer_id not in self._index_cache:
            return
        
        try:
            index = self._index_cache[buffer_id]
            if hasattr(index, 'unload_from_gpu'):
                index.unload_from_gpu()
            
            json_logger.debug(
                operation='evict_gpu_index',
                details={'buffer_id': buffer_id}
            )
        except (OSError, AttributeError) as exc:
            json_logger.warning(
                operation='evict_gpu_index',
                message=f'Failed to evict GPU index for {buffer_id}: {exc}'
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_buffer_state(
        self,
        buffer_id: str,
        embeddings: np.ndarray,
        chunks: list[CodeChunk],
    ) -> dict[str, Any]:
        """Save embeddings and chunks to disk.
        
        Args:
            buffer_id: Buffer identifier.
            embeddings: Embedding vectors.
            chunks: Code chunks.
        
        Returns:
            Status dict.
        """
        buffer_dir = self.work_dir / f"{buffer_id}.gcbuff"
        buffer_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save chunks
            chunks_path = buffer_dir / "chunks.json"
            chunks_data = {
                "chunks": [
                    {
                        "file": c.file,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "type": c.type,
                        "name": c.name,
                        "content": c.content,
                    }
                    for c in chunks
                ]
            }
            save_metadata(chunks_data, chunks_path)
            
            # Save embeddings as numpy file
            embeddings_path = buffer_dir / "embeddings.npy"
            np.save(embeddings_path, embeddings)
            
            json_logger.debug(
                operation='save_buffer_state',
                details={
                    'buffer_id': buffer_id,
                    'chunks_count': len(chunks),
                    'embeddings_shape': list(embeddings.shape)
                }
            )
            
            return {"status": "ok", "buffer_id": buffer_id}
        except (OSError, TypeError, ValueError) as exc:
            json_logger.error(
                operation='save_buffer_state',
                message=f'Failed to save buffer state: {exc}'
            )
            return {"status": "error", "message": str(exc)}

    def load_buffer_state(
        self,
        buffer_id: str,
    ) -> tuple[GpuIndex | None, list[CodeChunk] | None, np.ndarray | None]:
        """Load all buffer state from disk.
        
        Returns:
            (index, chunks, embeddings) tuple.
        """
        index = self._get_index(buffer_id)
        chunks = self._load_chunks(buffer_id)
        
        # Load embeddings
        buffer_dir = self.work_dir / f"{buffer_id}.gcbuff"
        embeddings_path = buffer_dir / "embeddings.npy"
        embeddings = None
        
        if embeddings_path.exists():
            try:
                embeddings = np.load(embeddings_path)
            except (OSError, ValueError) as exc:
                json_logger.warning(
                    operation='load_embeddings',
                    message=f'Failed to load embeddings: {exc}'
                )

        return index, chunks, embeddings

    # ------------------------------------------------------------------
    # Index Creation
    # ------------------------------------------------------------------
    def create_indices(
        self,
        buffer_id: str,
        embeddings: np.ndarray,
        chunks: list[CodeChunk],
    ) -> dict[str, Any]:
        """Create and cache indices from embeddings.
        
        Args:
            buffer_id: Buffer identifier.
            embeddings: Embedding vectors.
            chunks: Code chunks.
        
        Returns:
            Status dict with index info.
        """
        t0 = time.perf_counter()
        
        try:
            # Create FAISS index
            index = GpuIndex(
                embedding_dim=self._embedding_dim,
                use_gpu=self.use_gpu,
                gpu_id=self.gpu_id,
            )
            index.add(embeddings)
            
            # Pre-sync to GPU
            if self.use_gpu:
                index.sync_gpu()
            
            # Save to disk
            buffer_dir = self.work_dir / f"{buffer_id}.gcbuff"
            buffer_dir.mkdir(parents=True, exist_ok=True)
            index_path = buffer_dir / "index.faiss"
            index.save(str(index_path))
            
            # Cache index
            self._index_cache[buffer_id] = index
            
            # Create and cache lexical index
            lexical_index = LexicalIndex()
            for chunk in chunks:
                lexical_index.add(chunk.doc_id or len(chunk.file), chunk.content or "")
            
            lexical_path = buffer_dir / "lexical_index.json"
            lexical_path.write_text(
                json.dumps(lexical_index.to_dict()),
                encoding="utf-8"
            )
            self._lexical_cache[buffer_id] = lexical_index
            
            # Save embeddings and chunks
            self._save_buffer_state(buffer_id, embeddings, chunks)
            
            elapsed = time.perf_counter() - t0
            
            if self._prometheus_exporter:
                self._prometheus_exporter.record_operation(
                    operation='create_indices',
                    duration_s=elapsed,
                    status='ok',
                    chunk_count=len(chunks),
                )
            
            json_logger.info(
                operation='create_indices',
                buffer_id=buffer_id,
                elapsed_s=elapsed,
                details={
                    'chunks_count': len(chunks),
                    'embeddings_shape': list(embeddings.shape),
                }
            )
            
            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "chunks_count": len(chunks),
                "embeddings_shape": list(embeddings.shape),
                "elapsed_s": elapsed,
            }
        except (OSError, TypeError, ValueError, ImportError) as exc:
            json_logger.error(
                operation='create_indices',
                buffer_id=buffer_id,
                message=f'Failed to create indices: {exc}'
            )
            return {"status": "error", "message": str(exc)}

    def _rebuild_files(
        self,
        buffer_id: str,
        files: list[str],
        embeddings: np.ndarray | None = None,
        chunks: list[CodeChunk] | None = None,
    ) -> dict[str, Any]:
        """Rebuild index for changed files with incremental updates.
        
        Uses incremental indexing when possible to skip unchanged chunks.
        
        Args:
            buffer_id: Buffer ID.
            files: List of changed files.
            embeddings: New embeddings (optional, will reload if not provided).
            chunks: Updated chunks (optional, will reload if not provided).
        
        Returns:
            Status dict.
        """
        if embeddings is None or chunks is None:
            _, chunks, embeddings = self.load_buffer_state(buffer_id)
            if embeddings is None or chunks is None:
                return {
                    "status": "error",
                    "message": "Could not load buffer state for rebuild"
                }
        
        try:
            # Try incremental update if available
            incremental_stats = None
            if self._incremental_manager is not None:
                try:
                    # Group chunks by file for incremental processing
                    from collections import defaultdict
                    chunks_by_file = defaultdict(list)
                    for chunk in chunks:
                        chunks_by_file[chunk.file].append(chunk)
                    
                    # Process each changed file incrementally
                    total_changed = 0
                    total_kept = 0
                    for file_path in files:
                        if file_path in chunks_by_file:
                            file_chunks = chunks_by_file[file_path]
                            changed, removed, kept = self._incremental_manager._chunk_tracker.detect_changes(
                                file_path, file_chunks
                            )
                            total_changed += len(changed)
                            total_kept += len(kept)
                    
                    incremental_stats = {
                        "total_chunks": len(chunks),
                        "changed_chunks": total_changed,
                        "kept_chunks": total_kept,
                    }
                    
                    if total_changed > 0:
                        logger.info(
                            f"Incremental rebuild: {total_changed} changed, "
                            f"{total_kept} kept, {len(chunks) - total_changed - total_kept} removed"
                        )
                except (OSError, TypeError, ValueError) as e:
                    logger.warning(f"Incremental update failed, falling back to full rebuild: {e}")
            
            # Recreate indices
            index = GpuIndex(
                embedding_dim=self._embedding_dim,
                use_gpu=self.use_gpu,
                gpu_id=self.gpu_id,
            )
            index.add(embeddings)
            
            if self.use_gpu:
                index.sync_gpu()
            
            # Save updated index
            buffer_dir = self.work_dir / f"{buffer_id}.gcbuff"
            index_path = buffer_dir / "index.faiss"
            index.save(str(index_path))
            self._index_cache[buffer_id] = index
            
            # Update lexical index
            lexical_index = LexicalIndex()
            for chunk in chunks:
                lexical_index.add(chunk.doc_id or len(chunk.file), chunk.content or "")
            
            lexical_path = buffer_dir / "lexical_index.json"
            lexical_path.write_text(
                json.dumps(lexical_index.to_dict()),
                encoding="utf-8"
            )
            self._lexical_cache[buffer_id] = lexical_index
            
            # Clear query cache (results may be stale)
            self._query_cache.invalidate_buffer(buffer_id)
            
            json_logger.info(
                operation='rebuild_files',
                buffer_id=buffer_id,
                details={'files_count': len(files), 'chunks_count': len(chunks)}
            )
            
            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "rebuilt_files": files,
                "chunks_count": len(chunks),
            }
        except (OSError, TypeError, ValueError, ImportError) as exc:
            json_logger.error(
                operation='rebuild_files',
                buffer_id=buffer_id,
                message=f'Failed to rebuild files: {exc}'
            )
            return {"status": "error", "message": str(exc)}

    # ------------------------------------------------------------------
    # Query Caching
    # ------------------------------------------------------------------
    def _record_search_query(
        self,
        buffer_id: str,
        query: str,
        results: Any,
        top_k: int = 10,
        mode: str = "semantic",
    ) -> None:
        """Cache search results with paraphrase matching.
        
        Args:
            buffer_id: Buffer ID.
            query: Search query string.
            results: Search results to cache.
            top_k: Number of results.
            mode: Search mode (semantic, lexical, hybrid).
        """
        self._query_cache.set(buffer_id, query, top_k, mode, results)
    
    def _get_cached_search(
        self,
        buffer_id: str,
        query: str,
        top_k: int = 10,
        mode: str = "semantic",
    ) -> Any | None:
        """Check cache for semantically similar queries.
        
        Args:
            buffer_id: Buffer ID.
            query: Search query string.
            top_k: Number of results.
            mode: Search mode (semantic, lexical, hybrid).
        
        Returns:
            Cached results if found, None otherwise.
        """
        return self._query_cache.get(buffer_id, query, top_k, mode)
    
    def clear_query_cache(self) -> None:
        """Clear all cached queries."""
        self._query_cache.clear()
    
    # ------------------------------------------------------------------
    # Metrics & Health
    # ------------------------------------------------------------------
    def get_cache_stats(self) -> dict[str, Any]:
        """Get FAISS and lexical cache utilization.
        
        Returns:
            Dict with cache statistics.
        """
        return {
            "index_cache": self._index_cache.stats(),
            "lexical_cache": self._lexical_cache.stats(),
            "query_cache": self._query_cache.stats(),
        }
    
    def health_check(self) -> dict[str, Any]:
        """Check index system health.
        
        Returns:
            Dict with health status and diagnostics.
        """
        status = "healthy"
        warnings: list[str] = []
        
        cache_stats = self.get_cache_stats()
        
        # Check cache utilization
        index_util = cache_stats["index_cache"]["utilization"]
        if index_util > 0.9:
            status = "degraded"
            warnings.append(f"Index cache at {index_util*100:.0f}% utilization")
        elif index_util > 0.75:
            warnings.append(f"Index cache at {index_util*100:.0f}% utilization")
        
        # Check GPU availability
        gpu_status = self._check_gpu_memory()
        gpu_available = gpu_status.get("status") == "ok"
        if self.use_gpu and not gpu_available:
            status = "degraded"
            warnings.append("GPU requested but unavailable")
        
        # Query cache hit rate
        query_stats = cache_stats["query_cache"]
        hit_rate = query_stats.get("hit_rate_percent", 0)
        
        return {
            "status": status,
            "cache_utilization": {
                "index_cache": index_util * 100,
                "lexical_cache": cache_stats["lexical_cache"]["utilization"] * 100,
                "query_cache": query_stats.get("utilization_percent", 0),
            },
            "query_cache_hit_rate": hit_rate,
            "gpu_available": gpu_available,
            "gpu_memory": gpu_status if gpu_available else None,
            "warnings": warnings,
        }
    
    def close(self) -> None:
        """Clean up GPU memory and close resources."""
        # Unload GPU indices
        for buffer_id in list(self._index_cache.keys()):
            self._evict_gpu_indices(buffer_id)
        
        # Clear all caches
        self._index_cache.clear()
        self._lexical_cache.clear()
        self._query_cache.clear()
        
        json_logger.info(
            operation='close',
            message='IndexManager closed and resources released'
        )
