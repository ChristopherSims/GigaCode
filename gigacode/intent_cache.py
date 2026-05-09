"""Intent cache for query embeddings with semantic clustering.

Reduces token costs when AI agents make paraphrased queries by grouping
semantically similar queries into intent clusters. Each cluster stores
cached results and tracks hit statistics.
"""

import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


__all__ = ["IntentCache"]


class IntentCache:
    """Cache for query intents using semantic clustering.

    Groups paraphrased queries into clusters based on cosine similarity
    of their embeddings. Each cluster stores the most recent results and
    tracks hit statistics to optimize repeated intent lookups.

    Features:
    - Semantic clustering via configurable similarity threshold
    - Per-cluster stats: queries, hits, average similarity
    - Most-recent-5 result retention per cluster
    - Hit/miss statistics with average similarity tracking
    - LRU-style cluster management up to max_entries limit
    """

    def __init__(
        self,
        embedder: Any,
        similarity_threshold: float = 0.88,
        max_entries: int = 200,
    ):
        """Initialize intent cache.

        Args:
            embedder: Embedder for computing query embeddings.
                Must provide an ``encode`` method that accepts a list of
                strings and returns an iterable of embedding vectors.
            similarity_threshold: Cosine similarity threshold (0-1) for
                assigning a query to an existing intent cluster.
            max_entries: Maximum number of intent clusters to retain.
                Oldest clusters are evicted when the limit is exceeded.
        """
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold
        self._max_entries = max_entries

        # Mapping from cluster_id -> cluster dict
        self._clusters: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self._cluster_counter = 0

        # Global stats
        self._hits = 0
        self._misses = 0
        self._total_similarity = 0.0
        self._similarity_count = 0

    def _compute_embedding(self, query: str) -> np.ndarray:
        """Compute normalized embedding for a query.

        Args:
            query: Query string.

        Returns:
            Normalized embedding vector as a 1-D numpy array.

        Raises:
            ValueError: If no embedder was provided.
            RuntimeError: If the embedder fails to encode the query.
        """
        if self._embedder is None:
            raise ValueError("Embedder required for intent caching")

        embedding = self._embedder.encode([query])
        emb = np.asarray(embedding[0], dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def _compute_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two *already normalised* embeddings.

        Args:
            emb1: First normalised embedding vector.
            emb2: Second normalised embedding vector.

        Returns:
            Cosine similarity in the range [-1, 1]. For unit vectors this
            is equivalent to the dot product.
        """
        return float(np.dot(emb1, emb2))

    def get_intent(self, query: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """Retrieve cached intent for a query if a semantically similar cluster exists.

        Args:
            query: Query string to look up.

        Returns:
            A tuple ``(cluster, similarity_score)`` when a matching cluster
            is found, otherwise ``(None, 0.0)``.
        """
        try:
            query_emb = self._compute_embedding(query)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to compute embedding for intent lookup: {e}")
            self._misses += 1
            return None, 0.0

        best_cluster: Optional[Dict[str, Any]] = None
        best_similarity = 0.0

        for cluster in self._clusters.values():
            similarity = self._compute_similarity(query_emb, cluster["embedding"])
            if similarity >= self._similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if best_cluster is not None:
            best_cluster["hits"] += 1
            best_cluster["queries"].append(query)
            best_cluster["last_accessed"] = time.time()
            best_cluster["similarity_sum"] = (
                best_cluster.get("similarity_sum", 0.0) + best_similarity
            )
            best_cluster["similarity_count"] = best_cluster.get("similarity_count", 0) + 1

            self._hits += 1
            self._total_similarity += best_similarity
            self._similarity_count += 1
            logger.debug(f"IntentCache HIT (similarity={best_similarity:.3f}): {query[:50]}")
            # Move cluster to end (LRU)
            cid = best_cluster["cluster_id"]
            self._clusters.move_to_end(cid)
            return best_cluster, best_similarity

        self._misses += 1
        logger.debug(f"IntentCache MISS: {query[:50]}")
        return None, 0.0

    def put_intent(
        self,
        query: str,
        results: Any,
        intent_label: Optional[str] = None,
    ) -> None:
        """Store results for a query, creating or reusing an intent cluster.

        If a semantically similar cluster already exists, the query is added
        to that cluster and the new results are appended (most recent 5 kept).
        Otherwise a new cluster is created.

        Args:
            query: Query string to store.
            results: Results to cache for this intent.
            intent_label: Optional human-readable label for the intent.
        """
        try:
            query_emb = self._compute_embedding(query)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to compute embedding for intent caching: {e}")
            return

        # Search for an existing matching cluster
        best_cluster: Optional[Dict[str, Any]] = None
        best_similarity = 0.0

        for cluster in self._clusters.values():
            similarity = self._compute_similarity(query_emb, cluster["embedding"])
            if similarity >= self._similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if best_cluster is not None:
            # Add to existing cluster
            best_cluster["queries"].append(query)
            best_cluster["results"].append(results)
            # Keep most recent 5 results
            if len(best_cluster["results"]) > 5:
                best_cluster["results"] = best_cluster["results"][-5:]
            best_cluster["last_accessed"] = time.time()
            if intent_label is not None:
                best_cluster["intent_label"] = intent_label

            cid = best_cluster["cluster_id"]
            self._clusters.move_to_end(cid)
            logger.debug(f"IntentCache updated cluster {cid} (similarity={best_similarity:.3f})")
            return

        # Create new cluster
        self._cluster_counter += 1
        cluster_id = self._cluster_counter
        cluster = {
            "cluster_id": cluster_id,
            "canonical_query": query,
            "embedding": query_emb,
            "queries": [query],
            "results": [results],
            "hits": 0,
            "created": time.time(),
            "last_accessed": time.time(),
            "intent_label": intent_label,
            "similarity_sum": 0.0,
            "similarity_count": 0,
        }

        self._clusters[cluster_id] = cluster
        self._clusters.move_to_end(cluster_id)

        # Evict oldest if over limit
        if len(self._clusters) > self._max_entries:
            oldest_id = next(iter(self._clusters))
            self._clusters.pop(oldest_id)
            logger.debug(f"IntentCache evicted cluster {oldest_id} (LRU)")

        logger.debug(f"IntentCache created cluster {cluster_id} for: {query[:50]}")

    def find_similar_intents(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find the top-k most similar cached intent clusters for a query.

        Args:
            query: Query string to compare.
            top_k: Maximum number of clusters to return.

        Returns:
            List of dictionaries containing ``cluster``, ``similarity``, and
            ``rank`` keys, sorted by similarity descending.
        """
        try:
            query_emb = self._compute_embedding(query)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to compute embedding for similar intent search: {e}")
            return []

        scored = []
        for cluster in self._clusters.values():
            similarity = self._compute_similarity(query_emb, cluster["embedding"])
            scored.append((similarity, cluster))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for rank, (similarity, cluster) in enumerate(scored[:top_k], start=1):
            results.append(
                {
                    "cluster": cluster,
                    "similarity": similarity,
                    "rank": rank,
                }
            )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dictionary with hit rate, miss rate, cluster count, and average
            similarity across all hits.
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        miss_rate = (self._misses / total * 100) if total > 0 else 0.0
        avg_similarity = (
            (self._total_similarity / self._similarity_count * 100)
            if self._similarity_count > 0
            else 0.0
        )

        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "cluster_count": len(self._clusters),
            "max_entries": self._max_entries,
            "avg_similarity": avg_similarity,
            "similarity_threshold": self._similarity_threshold,
        }

    def clear(self) -> None:
        """Clear all intent clusters and reset statistics."""
        self._clusters.clear()
        self._cluster_counter = 0
        self._hits = 0
        self._misses = 0
        self._total_similarity = 0.0
        self._similarity_count = 0
        logger.info("IntentCache cleared")

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"IntentCache("
            f"clusters={stats['cluster_count']}/{self._max_entries}, "
            f"hit_rate={stats['hit_rate']:.1f}%, "
            f"threshold={self._similarity_threshold})"
        )
