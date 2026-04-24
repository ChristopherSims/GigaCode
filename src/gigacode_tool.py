"""Agent-facing tool interface for GPU-accelerated code embedding.

Chunks code at AST boundaries, keeps a persistent FAISS index in GPU
memory when available, and exposes a read-write-commit workflow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
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
from src.query_cache import QueryCache
from src.size_guard import check_size

logger = logging.getLogger(__name__)

_MAX_DIRTY_BEFORE_AUTO_REBUILD = 3


class CodeEmbeddingTool:
    """Embed a codebase into GPU/CPU buffers and expose search + cluster.

    Args:
        work_dir: Directory where buffer files and metadata are persisted.
        model_name: Sentence-transformers model name.
        device: torch device (``"cuda"``, ``"cpu"``, or ``None`` for auto).
        threshold_mb: Size-guard threshold in megabytes.
        use_gpu: Whether to mirror the FAISS index to GPU when possible.
    """

    def __init__(
        self,
        work_dir: str | Path,
        model_name: str | None = None,
        device: str | None = None,
        threshold_mb: float = 500.0,
        use_gpu: bool = True,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_mb = threshold_mb
        self.use_gpu = use_gpu

        self._embedder = Embedder(model_name=model_name, device=device)
        self._embedding_dim = self._embedder.embedding_dim

        # Registry of embedded codebases: buffer_id -> metadata dict
        self._registry_path = self.work_dir / "registry.json"
        self._registry: dict[str, dict[str, Any]] = (
            json.loads(self._registry_path.read_text(encoding="utf-8"))
            if self._registry_path.exists()
            else {}
        )

        # In-memory index cache: buffer_id -> GpuIndex
        self._index_cache: dict[str, GpuIndex] = {}

        # In-memory lexical index cache: buffer_id -> LexicalIndex
        self._lexical_cache: dict[str, LexicalIndex] = {}

        # Query result cache
        self._query_cache = QueryCache(maxsize=256)

    # ------------------------------------------------------------------
    # Schema exposure
    # ------------------------------------------------------------------
    @staticmethod
    def get_tool_schemas() -> list[dict[str, Any]]:
        from src.tool_schema import get_all_schemas
        return get_all_schemas()

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
    ) -> dict[str, Any]:
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
                chunks = chunk_file(f, language_hint=language_hint)
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

        # Stage 3b: build lexical BM25 index
        lexical = LexicalIndex()
        for i, ch in enumerate(all_chunks):
            lexical.add(i, ch.text)
        self._lexical_cache[buffer_id] = lexical

        # Stage 4: persist
        buffer_id = str(uuid.uuid4())
        buffer_dir = self.work_dir / f"{buffer_id}.gcbuff"
        buffer_dir.mkdir(parents=True, exist_ok=True)

        self._save_buffer_state(buffer_dir, all_chunks, embeddings, index, file_chunks_map)

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
            "dirty_files": {},
        }
        self._save_registry()
        self._index_cache[buffer_id] = index

        return {
            "status": "ok",
            "buffer_id": buffer_id,
            "chunk_count": token_count,
            "size_bytes": embeddings.nbytes,
            "message": f"Embedded {token_count} chunks from {len(files)} files.",
        }

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

        logger.info("Hash mismatch for %s in files %s; re-embedding.", buffer_id, mismatched)
        return self.embed_codebase(root, language_hint=info.get("language_hint"), pattern=pattern)

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        cached = self._query_cache.get(buffer_id, query, top_k + offset, "semantic")
        if cached is not None:
            matches = cached["matches"][offset:offset + top_k]
            return {"status": "ok", "matches": matches, "cached": True}

        index = self._get_index(buffer_id)
        chunks = self._load_chunks(buffer_id)
        if chunks is None:
            return {"status": "error", "message": "Chunk metadata missing."}

        q_emb = self._embedder.encode([query], batch_size=1)
        distances, indices = index.search(q_emb, top_k + offset)

        matches = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            ch = chunks[idx]
            matches.append({
                "file": ch.file,
                "start_line": ch.start_line,
                "end_line": ch.end_line,
                "type": ch.type,
                "name": ch.name,
                "score": float(score),
                "doc_id": int(idx),
            })

        self._query_cache.set(buffer_id, query, top_k + offset, "semantic", {"matches": matches})
        return {"status": "ok", "matches": matches[offset:offset + top_k], "cached": False}

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        cached = self._query_cache.get(buffer_id, query, top_k + offset, "hybrid")
        if cached is not None:
            matches = cached["matches"][offset:offset + top_k]
            return {"status": "ok", "matches": matches, "cached": True}

        chunks = self._load_chunks(buffer_id)
        if chunks is None:
            return {"status": "error", "message": "Chunk metadata missing."}

        # Semantic results
        index = self._get_index(buffer_id)
        q_emb = self._embedder.encode([query], batch_size=1)
        distances, indices = index.search(q_emb, top_k * 4 + offset)
        semantic_results: list[dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            semantic_results.append({
                "doc_id": int(idx),
                "score": float(score),
            })

        # Lexical results
        lexical = self._lexical_cache.get(buffer_id)
        if lexical is None:
            # Rebuild lexical index from chunks if missing
            lexical = LexicalIndex()
            for i, ch in enumerate(chunks):
                lexical.add(i, ch.text)
            self._lexical_cache[buffer_id] = lexical
        lexical_results = lexical.search(query, top_k=top_k * 4 + offset)

        merged = reciprocal_rank_fusion(
            semantic_results,
            lexical_results,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            top_k=top_k + offset,
        )

        # Enrich merged results with chunk metadata
        matches = []
        for m in merged:
            idx = m["doc_id"]
            ch = chunks[idx]
            matches.append({
                "file": ch.file,
                "start_line": ch.start_line,
                "end_line": ch.end_line,
                "type": ch.type,
                "name": ch.name,
                "rrf_score": m.get("rrf_score", 0.0),
                "semantic_rank": m.get("semantic_rank"),
                "lexical_rank": m.get("lexical_rank"),
                "doc_id": idx,
            })

        self._query_cache.set(buffer_id, query, top_k + offset, "hybrid", {"matches": matches})
        return {"status": "ok", "matches": matches[offset:offset + top_k], "cached": False}

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
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        snapshot = self._load_source_snapshot(buffer_id)
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
        1. **Name match** — chunks whose ``name`` contains the query string.
        2. **Semantic match** — top-K chunks by embedding similarity, filtered to
           symbol types (function, class, method, struct, trait, enum, interface).

        Results are deduplicated and merged (name matches rank first).

        Args:
            buffer_id: Buffer handle.
            query: Word or phrase to look for.
            top_k: Maximum number of symbol matches to return.

        Returns:
            Dict with ``status`` and ``matches`` (list of symbol metadata).
        """
        info = self._get_buffer_info(buffer_id)
        if info is None:
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        chunks = self._load_chunks(buffer_id)
        if chunks is None:
            return {"status": "error", "message": "Chunk metadata missing."}

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
        name_matches: list[dict[str, Any]] = []
        for ch in chunks:
            if ch.name and query_lower in ch.name.lower():
                name_matches.append({
                    "file": ch.file,
                    "start_line": ch.start_line,
                    "end_line": ch.end_line,
                    "type": ch.type,
                    "name": ch.name,
                    "score": 1.0,
                    "match_type": "name",
                })

        # Semantic search filtered to symbols
        index = self._get_index(buffer_id)
        q_emb = self._embedder.encode([query], batch_size=1)
        distances, indices = index.search(q_emb, top_k * 3)

        semantic_matches: list[dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            ch = chunks[idx]
            if ch.type in _SYMBOL_TYPES:
                semantic_matches.append({
                    "file": ch.file,
                    "start_line": ch.start_line,
                    "end_line": ch.end_line,
                    "type": ch.type,
                    "name": ch.name,
                    "score": float(score),
                    "match_type": "semantic",
                })
            if len(semantic_matches) >= top_k:
                break

        # Merge with dedup (name matches first)
        seen: set[tuple[str, int, int, str | None]] = set()
        merged: list[dict[str, Any]] = []
        for m in name_matches + semantic_matches:
            key = (m["file"], m["start_line"], m["end_line"], m.get("name"))
            if key not in seen:
                seen.add(key)
                merged.append(m)
            if len(merged) >= top_k:
                break

        return {"status": "ok", "matches": merged, "total": len(merged)}

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    def cluster_code(
        self,
        buffer_id: str,
        threshold: float = 0.75,
    ) -> dict[str, Any]:
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
        clusters: list[dict[str, Any]] = []
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
                clusters.append({
                    "file": start_ch.file,
                    "start_line": start_ch.start_line,
                    "end_line": end_ch.end_line,
                    "size": len(cluster_indices),
                    "avg_score": round(avg_score, 4),
                })

        return {"status": "ok", "clusters": clusters}

    def find_duplicates(
        self,
        buffer_id: str,
        threshold: float = 0.85,
    ) -> dict[str, Any]:
        """Find near-duplicate code chunks within a buffer."""
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
            return {"status": "error", "message": f"Unknown buffer_id: {buffer_id}"}

        snapshot = self._load_source_snapshot(buffer_id)
        if snapshot is None:
            return {"status": "error", "message": "Source snapshot missing."}

        if file is not None:
            if file not in snapshot:
                return {"status": "error", "message": f"File not in buffer: {file}"}
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
        """Re-chunk, re-embed, and patch index for the given files."""
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
        new_embeddings_list: list[np.ndarray] = []
        for rel_path in files:
            lines = snapshot.get(rel_path, [])
            text = "\n".join(lines)
            file_chunks = chunk_text(text, language_hint=language_hint, filename_hint=rel_path)
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

        # Persist
        self._save_buffer_state(buffer_dir, new_chunks, final_embeddings, index)

        # Rebuild lexical index
        lexical = LexicalIndex()
        for i, ch in enumerate(new_chunks):
            lexical.add(i, ch.text)
        self._lexical_cache[buffer_id] = lexical
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
        file_chunks_map: dict[str, list[int]] | None = None,
    ) -> None:
        emb_path = buffer_dir / "embeddings.npy"
        chunks_path = buffer_dir / "chunks.json"
        index_path = buffer_dir / "index.faiss"
        fcm_path = buffer_dir / "file_chunks_map.json"

        np.save(emb_path, embeddings)
        save_metadata(chunks_path, [ch.dict() for ch in chunks], compact=True)
        index.save(index_path)
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
    def close(self) -> None:
        self._index_cache.clear()

    def __enter__(self) -> CodeEmbeddingTool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
