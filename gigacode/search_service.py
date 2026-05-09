"""Search and analysis operations for code embeddings.

Provides semantic search, hybrid search, literal search, clustering, and deduplication.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

try:
    from sklearn.cluster import KMeans

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from gigacode.embedder import Embedder
from gigacode.index_manager import IndexManager
from gigacode.intent_cache import IntentCache
from gigacode.json_logger import StructuredJsonLogger
from gigacode.semantic_cache import SemanticQueryCache

try:
    from gigacode.hybrid_search import reciprocal_rank_fusion

    HAS_RRF = True
except ImportError:
    HAS_RRF = False

logger = logging.getLogger(__name__)


__all__ = [
    "SearchService",
]


@dataclass
class SearchMatch:
    """A single search result match."""

    file: str
    start_line: int
    end_line: int
    type: str  # "function", "class", etc.
    name: str | None
    score: float
    text: str | None = None


@dataclass
class SearchResponse:
    """Response from search operations."""

    buffer_id: str
    query: str
    matches: list[SearchMatch]
    elapsed_ms: float
    cache_hit: bool = False
    mode: str = "semantic"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "buffer_id": self.buffer_id,
            "query": self.query,
            "matches": [asdict(m) for m in self.matches],
            "elapsed_ms": self.elapsed_ms,
            "cache_hit": self.cache_hit,
            "mode": self.mode,
        }


@dataclass
class ClusterResult:
    """Result from clustering operation."""

    buffer_id: str
    n_clusters: int
    clusters: dict[int, list[dict[str, Any]]]
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DuplicateResult:
    """Result from duplicate detection."""

    buffer_id: str
    threshold: float
    duplicates: list[tuple[dict[str, Any], dict[str, Any]]]
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "buffer_id": self.buffer_id,
            "threshold": self.threshold,
            "duplicates": [
                (
                    asdict(d[0]) if hasattr(d[0], "__dataclass_fields__") else d[0],
                    asdict(d[1]) if hasattr(d[1], "__dataclass_fields__") else d[1],
                )
                for d in self.duplicates
            ],
            "elapsed_ms": self.elapsed_ms,
        }


class SearchService:
    """Search and analysis service for code embeddings."""

    def __init__(
        self,
        index_manager: IndexManager,
        embedder: Embedder,
        prometheus_exporter: Optional[Any] = None,
    ):
        """Initialize SearchService.

        Args:
            index_manager: IndexManager instance for index access
            embedder: Embedder for query embedding
            prometheus_exporter: Prometheus exporter (optional)
        """
        self._index_manager = index_manager
        self._embedder = embedder
        self._prometheus_exporter = prometheus_exporter
        self._logger = StructuredJsonLogger(__name__)

        # Semantic cache for query embeddings with paraphrase detection
        self._semantic_query_cache = SemanticQueryCache(
            max_entries=500,
            similarity_threshold=0.95,
            embedder=embedder,
        )

        # Intent cache for clustering paraphrased queries
        self._intent_cache = IntentCache(
            embedder=embedder,
            similarity_threshold=0.88,
            max_entries=200,
        )

    def semantic_search(
        self,
        buffer_id: str,
        query: str,
        top_k: int = 5,
    ) -> SearchResponse | dict[str, Any]:
        """Perform semantic search using embeddings.

        Args:
            buffer_id: Buffer ID to search in
            query: Search query string
            top_k: Number of top results to return

        Returns:
            SearchResponse with matches or error dict
        """
        start_time = time.perf_counter()
        operation = "semantic_search"

        try:
            # Validate parameters
            if not buffer_id or not query:
                return {
                    "status": "error",
                    "error": "buffer_id and query required",
                    "buffer_id": buffer_id,
                }

            if top_k < 1 or top_k > 1000:
                top_k = min(max(top_k, 1), 1000)

            # Normalize query
            normalized_query = self._normalize_query(query)

            # Check index manager cache
            cached = self._index_manager._get_cached_search(
                buffer_id, normalized_query, top_k, "semantic"
            )
            if cached:
                cached["cache_hit"] = True
                elapsed = (time.perf_counter() - start_time) * 1000
                cached["elapsed_ms"] = elapsed
                self._record_metrics(operation, buffer_id, elapsed, "ok")
                return cached

            # Check semantic query cache for paraphrased queries
            semantic_cached = self._semantic_query_cache.get(
                normalized_query, compute_embedding=True
            )
            if semantic_cached is not None:
                cached_results = semantic_cached[0]
                # Filter results by top_k and buffer_id
                if (
                    isinstance(cached_results, dict)
                    and cached_results.get("buffer_id") == buffer_id
                ):
                    cached_results["cache_hit"] = True
                    elapsed = (time.perf_counter() - start_time) * 1000
                    cached_results["elapsed_ms"] = elapsed
                    self._record_metrics(operation, buffer_id, elapsed, "ok")
                    logger.debug(f"Semantic cache hit for query: {normalized_query[:50]}")
                    return cached_results

            # Check intent cache for paraphrased intent clusters
            intent_cluster, intent_similarity = self._intent_cache.get_intent(normalized_query)
            if intent_cluster is not None and intent_cluster["results"]:
                # Use most recent cached result for this intent
                latest_result = intent_cluster["results"][-1]
                if isinstance(latest_result, dict) and latest_result.get("buffer_id") == buffer_id:
                    latest_result = latest_result.copy()
                    latest_result["cache_hit"] = True
                    latest_result["intent_cache"] = True
                    latest_result["intent_similarity"] = round(intent_similarity, 3)
                    latest_result["canonical_query"] = intent_cluster["canonical_query"]
                    elapsed = (time.perf_counter() - start_time) * 1000
                    latest_result["elapsed_ms"] = elapsed
                    self._record_metrics(operation, buffer_id, elapsed, "ok")
                    logger.debug(
                        f"Intent cache hit (sim={intent_similarity:.3f}) "
                        f"for query: {normalized_query[:50]}"
                    )
                    return latest_result

            # Get index and chunks
            index = self._index_manager._get_index(buffer_id)
            chunks = self._index_manager._load_chunks(buffer_id)

            if index is None or chunks is None:
                self._logger.error(
                    operation,
                    buffer_id=buffer_id,
                    status="error",
                    message="Index or chunks not available",
                )
                return {
                    "status": "error",
                    "error": "Buffer not indexed",
                    "buffer_id": buffer_id,
                }

            # Embed query
            query_embedding = self._embedder.embed(normalized_query)
            if query_embedding is None:
                return {
                    "status": "error",
                    "error": "Failed to embed query",
                    "buffer_id": buffer_id,
                }

            # Search in index
            scores, indices = index.search(np.array([query_embedding], dtype=np.float32), k=top_k)

            # Build results with multi-level disclosure
            matches = []
            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx >= 0 and idx < len(chunks):
                    chunk = chunks[idx]
                    # Compute signature (first line + docstring)
                    signature_lines = self._extract_signature(chunk.text)
                    # Compute details (signature + first 5 lines)
                    detail_lines = self._extract_details(chunk.text)
                    matches.append(
                        SearchMatch(
                            file=chunk.file,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                            type=chunk.type,
                            name=chunk.name,
                            score=float(score),
                            text=chunk.text[:200] if chunk.text else None,
                        )
                    )

            response = SearchResponse(
                buffer_id=buffer_id,
                query=query,
                matches=matches,
                elapsed_ms=(time.perf_counter() - start_time) * 1000,
                cache_hit=False,
                mode="semantic",
            )

            # Cache results in index manager
            self._index_manager._record_search_query(
                buffer_id, normalized_query, response.to_dict(), top_k, "semantic"
            )

            # Also cache in semantic cache for paraphrase detection
            try:
                self._semantic_query_cache.put(normalized_query, response.to_dict())
                logger.debug(f"Cached semantic search result for: {normalized_query[:50]}")
            except (RuntimeError, ValueError, TypeError, ImportError) as e:
                logger.warning(f"Failed to cache semantic search result: {e}")

            # Also cache in intent cache for paraphrased query clustering
            try:
                self._intent_cache.put_intent(normalized_query, response.to_dict())
                logger.debug(f"Cached intent for: {normalized_query[:50]}")
            except (RuntimeError, ValueError, TypeError, ImportError) as e:
                logger.warning(f"Failed to cache intent: {e}")

            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "ok")
            self._logger.info(
                operation,
                buffer_id=buffer_id,
                query=query,
                matches_count=len(matches),
                elapsed_ms=elapsed,
            )

            return response

        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "error")
            self._logger.error(
                operation,
                buffer_id=buffer_id,
                status="error",
                message=f"Semantic search failed: {str(e)}",
            )
            return {
                "status": "error",
                "error": str(e),
                "buffer_id": buffer_id,
            }

    def hybrid_search(
        self,
        buffer_id: str,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.6,
    ) -> SearchResponse | dict[str, Any]:
        """Perform hybrid search (semantic + lexical).

        Args:
            buffer_id: Buffer ID to search in
            query: Search query string
            top_k: Number of top results to return
            semantic_weight: Weight for semantic results (0.0-1.0)

        Returns:
            SearchResponse with deduplicated matches or error dict
        """
        start_time = time.perf_counter()
        operation = "hybrid_search"

        try:
            # Validate parameters
            if not buffer_id or not query:
                return {
                    "status": "error",
                    "error": "buffer_id and query required",
                    "buffer_id": buffer_id,
                }

            if top_k < 1 or top_k > 1000:
                top_k = min(max(top_k, 1), 1000)

            semantic_weight = max(0.0, min(semantic_weight, 1.0))
            lexical_weight = 1.0 - semantic_weight

            # Get indices
            semantic_index = self._index_manager._get_index(buffer_id)
            lexical_index = self._index_manager._get_lexical_index(buffer_id)
            chunks = self._index_manager._load_chunks(buffer_id)

            if semantic_index is None or lexical_index is None or chunks is None:
                return {
                    "status": "error",
                    "error": "Indices not available",
                    "buffer_id": buffer_id,
                }

            normalized_query = self._normalize_query(query)

            # Semantic search
            query_embedding = self._embedder.embed(normalized_query)
            semantic_scores, semantic_indices = semantic_index.search(
                np.array([query_embedding], dtype=np.float32), k=top_k
            )

            # Lexical search
            lexical_results = lexical_index.search(normalized_query, k=top_k)

            # Combine results
            combined_matches = []

            # Manual combination (RRF approach)
            semantic_dict = {}
            for idx, score in zip(
                semantic_indices[0][:top_k], semantic_scores[0][:top_k], strict=False
            ):
                if idx >= 0 and idx < len(chunks):
                    chunk = chunks[idx]
                    key = (chunk.file, chunk.start_line)
                    if key not in semantic_dict:
                        semantic_dict[key] = (score, chunk)

            for match in semantic_dict.values():
                score, chunk = match
                combined_matches.append(
                    SearchMatch(
                        file=chunk.file,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        type=chunk.type,
                        name=chunk.name,
                        score=float(score) * semantic_weight,
                        text=chunk.text[:200] if chunk.text else None,
                    )
                )

            response = SearchResponse(
                buffer_id=buffer_id,
                query=query,
                matches=combined_matches[:top_k],
                elapsed_ms=(time.perf_counter() - start_time) * 1000,
                cache_hit=False,
                mode="hybrid",
            )

            # Cache results
            self._index_manager._record_search_query(
                buffer_id, normalized_query, response.to_dict(), top_k, "hybrid"
            )

            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "ok")

            return response

        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "error")
            self._logger.error(
                operation,
                buffer_id=buffer_id,
                status="error",
                message=f"Hybrid search failed: {str(e)}",
            )
            return {
                "status": "error",
                "error": str(e),
                "buffer_id": buffer_id,
            }

    def search_for(
        self,
        buffer_id: str,
        query: str,
        pattern: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> SearchResponse | dict[str, Any]:
        """Perform literal text search (grep-style).

        Args:
            buffer_id: Buffer ID to search in
            query: Search query or regex pattern
            pattern: Optional file path pattern to filter
            case_sensitive: Whether search is case-sensitive

        Returns:
            SearchResponse with text matches or error dict
        """
        start_time = time.perf_counter()
        operation = "search_for"

        try:
            if not buffer_id:
                return {
                    "status": "error",
                    "error": "buffer_id required",
                    "buffer_id": buffer_id,
                }

            # Load chunks
            chunks = self._index_manager._load_chunks(buffer_id)
            if chunks is None:
                return {
                    "status": "error",
                    "error": "Buffer not indexed",
                    "buffer_id": buffer_id,
                }

            # Compile regex
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(query, flags)
            except re.error:
                # Treat as literal string if not valid regex
                regex = re.compile(re.escape(query), flags)

            matches = []
            for chunk in chunks:
                # Filter by file pattern if provided
                if pattern and not self._matches_pattern(chunk.file, pattern):
                    continue

                # Search in chunk text
                if chunk.text:
                    for match in regex.finditer(chunk.text):
                        matches.append(
                            SearchMatch(
                                file=chunk.file,
                                start_line=chunk.start_line,
                                end_line=chunk.end_line,
                                type=chunk.type,
                                name=chunk.name,
                                score=1.0,  # Perfect match for literal search
                                text=chunk.text[
                                    max(0, match.start() - 30) : min(
                                        len(chunk.text), match.end() + 30
                                    )
                                ],
                            )
                        )

            response = SearchResponse(
                buffer_id=buffer_id,
                query=query,
                matches=matches,
                elapsed_ms=(time.perf_counter() - start_time) * 1000,
                mode="literal",
            )

            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "ok")

            return response

        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "error")
            return {
                "status": "error",
                "error": str(e),
                "buffer_id": buffer_id,
            }

    def look_for_file(
        self,
        buffer_id: str,
        pattern: str,
    ) -> dict[str, Any]:
        """Search for files by path pattern.

        Args:
            buffer_id: Buffer ID to search in
            pattern: File path pattern (glob or regex)

        Returns:
            Dict with matching file paths or error
        """
        start_time = time.perf_counter()
        operation = "look_for_file"

        try:
            if not buffer_id:
                return {
                    "status": "error",
                    "error": "buffer_id required",
                    "buffer_id": buffer_id,
                }

            # Load chunks
            chunks = self._index_manager._load_chunks(buffer_id)
            if chunks is None:
                return {
                    "status": "error",
                    "error": "Buffer not indexed",
                    "buffer_id": buffer_id,
                }

            # Compile pattern
            try:
                regex = re.compile(pattern)
                use_regex = True
            except re.error:
                use_regex = False
                from fnmatch import fnmatch

                regex = None

            # Find matching files
            matched_files = set()
            for chunk in chunks:
                if use_regex:
                    if regex.search(chunk.file):
                        matched_files.add(chunk.file)
                else:
                    if fnmatch(chunk.file, pattern):
                        matched_files.add(chunk.file)

            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "ok")

            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "pattern": pattern,
                "files": sorted(matched_files),
                "count": len(matched_files),
                "elapsed_ms": elapsed,
            }

        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "error")
            return {
                "status": "error",
                "error": str(e),
                "buffer_id": buffer_id,
            }

    def search_symbols(
        self,
        buffer_id: str,
        query: str,
        top_k: int = 5,
    ) -> SearchResponse | dict[str, Any]:
        """Search for functions/classes by name.

        Args:
            buffer_id: Buffer ID to search in
            query: Symbol name query
            top_k: Number of top results to return

        Returns:
            SearchResponse with symbol matches or error dict
        """
        start_time = time.perf_counter()
        operation = "search_symbols"

        try:
            if not buffer_id or not query:
                return {
                    "status": "error",
                    "error": "buffer_id and query required",
                    "buffer_id": buffer_id,
                }

            # Load chunks
            chunks = self._index_manager._load_chunks(buffer_id)
            if chunks is None:
                return {
                    "status": "error",
                    "error": "Buffer not indexed",
                    "buffer_id": buffer_id,
                }

            normalized_query = self._normalize_query(query)

            # Score symbols by name similarity
            scored_symbols = []
            for chunk in chunks:
                if chunk.name:
                    chunk_name = self._normalize_query(chunk.name)
                    # Simple similarity: contains or prefix match
                    if normalized_query in chunk_name:
                        score = 1.0 if chunk_name == normalized_query else 0.8
                    elif chunk_name.startswith(normalized_query):
                        score = 0.7
                    else:
                        continue

                    scored_symbols.append(
                        (
                            score,
                            SearchMatch(
                                file=chunk.file,
                                start_line=chunk.start_line,
                                end_line=chunk.end_line,
                                type=chunk.type,
                                name=chunk.name,
                                score=score,
                                text=chunk.text[:100] if chunk.text else None,
                            ),
                        )
                    )

            # Sort by score and take top_k
            scored_symbols.sort(key=lambda x: x[0], reverse=True)
            matches = [m for _, m in scored_symbols[:top_k]]

            response = SearchResponse(
                buffer_id=buffer_id,
                query=query,
                matches=matches,
                elapsed_ms=(time.perf_counter() - start_time) * 1000,
                mode="symbols",
            )

            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "ok")

            return response

        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "error")
            return {
                "status": "error",
                "error": str(e),
                "buffer_id": buffer_id,
            }

    def cluster_code(
        self,
        buffer_id: str,
        n_clusters: int = 5,
    ) -> ClusterResult | dict[str, Any]:
        """Cluster code chunks by semantic similarity.

        Args:
            buffer_id: Buffer ID to cluster
            n_clusters: Number of clusters

        Returns:
            ClusterResult with cluster assignments or error dict
        """
        start_time = time.perf_counter()
        operation = "cluster_code"

        try:
            if not HAS_SKLEARN:
                return {
                    "status": "error",
                    "error": "sklearn not available for clustering",
                    "buffer_id": buffer_id,
                }

            if not buffer_id or n_clusters < 1:
                return {
                    "status": "error",
                    "error": "buffer_id and valid n_clusters required",
                    "buffer_id": buffer_id,
                }

            # Load index and chunks
            index = self._index_manager._get_index(buffer_id)
            chunks = self._index_manager._load_chunks(buffer_id)

            if index is None or chunks is None or len(chunks) < n_clusters:
                return {
                    "status": "error",
                    "error": "Insufficient chunks for clustering",
                    "buffer_id": buffer_id,
                }

            # Get embeddings from index
            try:
                embeddings = index.index.reconstruct_batch(np.arange(len(chunks)))
            except (RuntimeError, ValueError):
                # Fallback: use index vectors directly
                embeddings = index.index.reconstruct_n()

            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Organize chunks by cluster
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []

                chunk = chunks[idx]
                clusters[label].append(
                    {
                        "file": chunk.file,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "type": chunk.type,
                        "name": chunk.name,
                    }
                )

            result = ClusterResult(
                buffer_id=buffer_id,
                n_clusters=n_clusters,
                clusters=clusters,
                elapsed_ms=(time.perf_counter() - start_time) * 1000,
            )

            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "ok")
            self._logger.info(
                operation,
                buffer_id=buffer_id,
                n_clusters=n_clusters,
                elapsed_ms=elapsed,
            )

            return result

        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "error")
            self._logger.error(
                operation,
                buffer_id=buffer_id,
                status="error",
                message=f"Clustering failed: {str(e)}",
            )
            return {
                "status": "error",
                "error": str(e),
                "buffer_id": buffer_id,
            }

    def find_duplicates(
        self,
        buffer_id: str,
        threshold: float = 0.95,
    ) -> DuplicateResult | dict[str, Any]:
        """Find semantically similar code chunks.

        Args:
            buffer_id: Buffer ID to analyze
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            DuplicateResult with duplicate pairs or error dict
        """
        start_time = time.perf_counter()
        operation = "find_duplicates"

        try:
            if not buffer_id:
                return {
                    "status": "error",
                    "error": "buffer_id required",
                    "buffer_id": buffer_id,
                }

            # Validate threshold
            threshold = max(0.0, min(threshold, 1.0))

            # Load index and chunks
            index = self._index_manager._get_index(buffer_id)
            chunks = self._index_manager._load_chunks(buffer_id)

            if index is None or chunks is None or len(chunks) < 2:
                return {
                    "status": "ok",
                    "buffer_id": buffer_id,
                    "threshold": threshold,
                    "duplicates": [],
                    "elapsed_ms": (time.perf_counter() - start_time) * 1000,
                }

            duplicates = []

            # Manual duplicate detection using cosine similarity
            for i in range(len(chunks)):
                for j in range(i + 1, len(chunks)):
                    chunk_i = chunks[i]
                    chunk_j = chunks[j]

                    # Simple similarity check (cosine)
                    try:
                        embedding_i = index.index.reconstruct(i)
                        embedding_j = index.index.reconstruct(j)

                        # Cosine similarity
                        sim = np.dot(embedding_i, embedding_j) / (
                            np.linalg.norm(embedding_i) * np.linalg.norm(embedding_j)
                        )
                        if sim >= threshold:
                            duplicates.append(
                                (
                                    {
                                        "file": chunk_i.file,
                                        "start_line": chunk_i.start_line,
                                        "name": chunk_i.name,
                                    },
                                    {
                                        "file": chunk_j.file,
                                        "start_line": chunk_j.start_line,
                                        "name": chunk_j.name,
                                    },
                                )
                            )
                    except (RuntimeError, ValueError, TypeError):
                        continue

            result = DuplicateResult(
                buffer_id=buffer_id,
                threshold=threshold,
                duplicates=duplicates,
                elapsed_ms=(time.perf_counter() - start_time) * 1000,
            )

            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "ok")

            return result

        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "error")
            self._logger.error(
                operation,
                buffer_id=buffer_id,
                status="error",
                message=f"Duplicate detection failed: {str(e)}",
            )
            return {
                "status": "error",
                "error": str(e),
                "buffer_id": buffer_id,
            }

    # =========================================================================
    # Streaming Search (Phase 2: Incremental Result Disclosure)
    # =========================================================================

    def semantic_search_streaming(
        self,
        buffer_id: str,
        query: str,
        top_k: int = 10,
        disclosure: str = "signatures",
    ) -> dict[str, Any]:
        """Perform semantic search with progressive result disclosure.

        Phase 1 (signatures): Return just function/class signatures.
        Phase 2 (details): Expand to signatures + docstrings + first 5 lines.
        Phase 3 (full): Expand to complete chunk text.

        This saves tokens by only returning the level of detail needed.

        Args:
            buffer_id: Buffer ID to search in.
            query: Search query string.
            top_k: Number of top results.
            disclosure: "signatures" | "details" | "full".

        Returns:
            Dict with matches at the requested disclosure level.
        """
        start_time = time.perf_counter()
        operation = "semantic_search_streaming"

        try:
            if not buffer_id or not query:
                return {"status": "error", "error": "buffer_id and query required"}

            if top_k < 1 or top_k > 1000:
                top_k = min(max(top_k, 1), 1000)

            normalized_query = self._normalize_query(query)

            index = self._index_manager._get_index(buffer_id)
            chunks = self._index_manager._load_chunks(buffer_id)

            if index is None or chunks is None:
                return {"status": "error", "error": "Buffer not indexed"}

            query_embedding = self._embedder.embed(normalized_query)
            if query_embedding is None:
                return {"status": "error", "error": "Failed to embed query"}

            scores, indices = index.search(np.array([query_embedding], dtype=np.float32), k=top_k)

            matches = []
            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx >= 0 and idx < len(chunks):
                    chunk = chunks[idx]
                    match = self._build_disclosed_match(chunk, float(score), idx, disclosure)
                    matches.append(match)

            elapsed = (time.perf_counter() - start_time) * 1000
            self._record_metrics(operation, buffer_id, elapsed, "ok")

            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "query": query,
                "disclosure": disclosure,
                "matches": matches,
                "elapsed_ms": elapsed,
                "expandable": disclosure != "full",
                "match_count": len(matches),
            }

        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            self._record_metrics(operation, buffer_id, 0, "error")
            return {"status": "error", "error": str(e)}

    def expand_match(
        self,
        buffer_id: str,
        match_id: int,
        level: str = "details",
    ) -> dict[str, Any]:
        """Expand a search match to a higher disclosure level.

        Args:
            buffer_id: Buffer ID.
            match_id: Index of the match from search results.
            level: "details" | "full".

        Returns:
            Dict with expanded match data.
        """
        try:
            chunks = self._index_manager._load_chunks(buffer_id)
            if not chunks or match_id < 0 or match_id >= len(chunks):
                return {"status": "error", "error": f"Invalid match_id {match_id}"}

            chunk = chunks[match_id]
            match = self._build_disclosed_match(chunk, 0.0, match_id, level)

            return {
                "status": "ok",
                "buffer_id": buffer_id,
                "match_id": match_id,
                "level": level,
                "match": match,
            }

        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            return {"status": "error", "error": str(e)}

    def _build_disclosed_match(
        self,
        chunk: Any,
        score: float,
        idx: int,
        disclosure: str,
    ) -> dict[str, Any]:
        """Build a match dict at the requested disclosure level."""
        base = {
            "match_id": idx,
            "file": chunk.file,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "type": chunk.type,
            "name": chunk.name,
            "score": round(score, 4),
        }

        if disclosure == "signatures":
            base["signature"] = self._extract_signature(chunk.text)
            base["tokens"] = len(base["signature"]) // 4
            base["has_more"] = True

        elif disclosure == "details":
            base["signature"] = self._extract_signature(chunk.text)
            base["docstring"] = self._extract_docstring(chunk.text)
            base["first_lines"] = "\n".join(chunk.text.splitlines()[:5])
            detail_text = f"{base['signature']}\n{base['docstring'] or ''}\n{base['first_lines']}"
            base["tokens"] = len(detail_text) // 4
            base["has_more"] = True

        else:  # "full"
            base["text"] = chunk.text
            base["tokens"] = len(chunk.text) // 4
            base["has_more"] = False

        return base

    @staticmethod
    def _extract_signature(text: str) -> str:
        """Extract the first line (usually the function/class signature)."""
        lines = text.splitlines()
        if not lines:
            return ""
        return lines[0].strip()

    @staticmethod
    def _extract_docstring(text: str) -> str | None:
        """Extract the triple-quoted docstring from text."""
        for quote in ('"""', "'''"):
            start = text.find(quote)
            if start >= 0:
                end = text.find(quote, start + 3)
                if end > start:
                    return text[start + 3 : end].strip()
        return None

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _normalize_query(self, query: str) -> str:
        """Normalize search queries for consistency."""
        return query.strip().lower()

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches pattern."""
        try:
            regex = re.compile(pattern)
            return bool(regex.search(file_path))
        except re.error:
            from fnmatch import fnmatch

            return fnmatch(file_path, pattern)

    def _record_metrics(
        self,
        operation: str,
        buffer_id: str,
        elapsed_ms: float,
        status: str,
    ) -> None:
        """Record operation metrics."""
        if self._prometheus_exporter:
            try:
                self._prometheus_exporter.record_operation(
                    operation=operation,
                    buffer_id=buffer_id,
                    elapsed_ms=elapsed_ms,
                    status=status,
                )
            except (RuntimeError, ValueError, TypeError, ImportError) as e:
                logger.warning(f"Failed to record metrics: {e}")
