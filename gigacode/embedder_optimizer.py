"""Embedder optimization wrapper for batch processing.

This module provides an optimized wrapper around the Embedder that uses
BatchEmbedder for large batches while maintaining backward compatibility.
"""

import logging
from typing import Optional, Any

import numpy as np

from gigacode.batch_embedder import BatchEmbedder
from gigacode.embedder import Embedder

logger = logging.getLogger(__name__)


__all__ = [
    "OptimizedEmbedder",
    "wrap_embedder_with_optimization",
]


class OptimizedEmbedder:
    """Wrapper around Embedder that automatically uses batch optimization.
    
    This wrapper:
    - Uses standard Embedder for small batches (<100 texts)
    - Uses BatchEmbedder for large batches (>=100 texts)
    - Caches embedding results to avoid recomputation
    - Maintains full backward compatibility with Embedder API
    """
    
    def __init__(
        self,
        embedder: Embedder,
        use_batch_optimization: bool = True,
        batch_threshold: int = 100,
    ):
        """Initialize OptimizedEmbedder.
        
        Args:
            embedder: Base Embedder instance
            use_batch_optimization: Whether to use batch optimization
            batch_threshold: Number of texts above which to use batch optimization
        """
        self._embedder = embedder
        self._use_batch_optimization = use_batch_optimization
        self._batch_threshold = batch_threshold
        self._batch_processor: Optional[BatchEmbedder] = None
        
        if use_batch_optimization:
            try:
                self._batch_processor = BatchEmbedder(
                    model=embedder._model,
                    device=embedder.device or "cpu",
                    cache_enabled=True,
                )
                logger.info(
                    "OptimizedEmbedder initialized with batch optimization "
                    "(threshold: %d texts)", batch_threshold
                )
            except (RuntimeError, OSError, ImportError, ValueError) as e:
                logger.warning(
                    "Failed to initialize BatchEmbedder: %s. "
                    "Will use standard embedder.", e
                )
                self._batch_processor = None
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._embedder.embedding_dim
    
    @property
    def device(self) -> str:
        """Get device (cpu or cuda)."""
        return self._embedder.device or "cpu"
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._embedder.model_name
    
    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings with automatic optimization.
        
        Automatically selects between standard and batch optimization based on
        number of texts.
        
        Args:
            texts: Input texts to encode
            batch_size: Batch size for encoding (ignored for batch optimization)
            show_progress: Whether to show progress bar
            use_cache: Whether to use embedding cache (only for batch processor)
        
        Returns:
            Embeddings as (N, embedding_dim) float32 array
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        # Use batch optimization for large batches if available
        if (
            self._use_batch_optimization
            and self._batch_processor is not None
            and len(texts) >= self._batch_threshold
        ):
            logger.debug(
                "Using batch optimization for %d texts (threshold: %d)",
                len(texts),
                self._batch_threshold,
            )
            try:
                embeddings = self._batch_processor.embed_batch(
                    texts,
                    show_progress=show_progress,
                )
                return np.asarray(embeddings, dtype=np.float32)
            except (RuntimeError, OSError, ImportError, ValueError) as e:
                logger.warning(
                    "Batch optimization failed: %s. Falling back to standard encoding.",
                    e,
                )
        
        # Fall back to standard encoder
        logger.debug(
            "Using standard encoding for %d texts", len(texts)
        )
        return self._embedder.encode(
            texts,
            batch_size=batch_size,
        )
    
    def get_batch_processor(self) -> Optional[BatchEmbedder]:
        """Get the underlying batch processor (if available).
        
        Returns:
            BatchEmbedder instance or None if not initialized
        """
        return self._batch_processor
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics from batch processor.
        
        Returns:
            Dictionary with cache hit rate, hits, misses, etc.
            Returns empty dict if batch processor not available.
        """
        if self._batch_processor is None:
            return {}
        return self._batch_processor.get_cache_stats() or {}
    
    def clear_cache(self) -> None:
        """Clear embedding cache in batch processor."""
        if self._batch_processor is not None:
            self._batch_processor.clear_cache()
            logger.debug("Embedding cache cleared")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OptimizedEmbedder("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"batch_optimization={self._use_batch_optimization}, "
            f"dim={self.embedding_dim}"
            ")"
        )


def wrap_embedder_with_optimization(
    embedder: Embedder,
    use_batch_optimization: bool = True,
    batch_threshold: int = 100,
) -> OptimizedEmbedder:
    """Wrap an Embedder with optimization capabilities.
    
    Args:
        embedder: Base Embedder instance to wrap
        use_batch_optimization: Whether to enable batch optimization
        batch_threshold: Texts threshold for batch optimization
    
    Returns:
        OptimizedEmbedder wrapper
    """
    return OptimizedEmbedder(
        embedder=embedder,
        use_batch_optimization=use_batch_optimization,
        batch_threshold=batch_threshold,
    )
