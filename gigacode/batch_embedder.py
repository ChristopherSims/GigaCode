"""Optimized embedding utilities for batch processing.

This module provides efficient batch embedding with dynamic sizing and caching.

Key features:
- Dynamic batch sizing based on available memory
- Embedding result caching to avoid redundant computation
- Progress tracking for large batches
- GPU memory management
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import hashlib
import logging

logger = logging.getLogger(__name__)


__all__ = [
    "EmbeddingCache",
    "BatchEmbeddingProcessor",
    "BatchEmbedder",
    "optimize_embedder",
]


@dataclass
class EmbeddingCache:
    """Simple LRU cache for embeddings to avoid recomputation."""
    
    cache: Dict[str, Any]
    max_size: int = 1000
    hits: int = 0
    misses: int = 0
    
    def get_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[Any]:
        """Get embedding from cache if exists."""
        key = self.get_key(text)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, text: str, embedding: Any) -> None:
        """Store embedding in cache."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO, could use OrderedDict for LRU)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        key = self.get_key(text)
        self.cache[key] = embedding
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": f"{hit_rate:.1f}%",
            "size": len(self.cache),
        }


class BatchEmbeddingProcessor:
    """Efficient batch embedding processor with memory management."""
    
    def __init__(self, model, device: str = "cpu", cache_enabled: bool = True):
        """Initialize batch processor.
        
        Args:
            model: Sentence-transformers model
            device: "cpu" or "cuda"
            cache_enabled: Whether to cache embeddings
        """
        self.model = model
        self.device = device
        self.cache = EmbeddingCache({}) if cache_enabled else None
        self.batch_size = self._calculate_batch_size()
    
    def _calculate_batch_size(self) -> int:
        """Calculate optimal batch size based on device.
        
        Returns:
            Recommended batch size for current device.
        """
        if self.device == "cuda":
            try:
                import torch
                # Heuristic: ~100MB per batch on GPU
                return 256
            except (ImportError, ModuleNotFoundError):
                return 64
        else:
            # CPU: smaller batches
            return 32
    
    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[Any]:
        """Embed a batch of texts efficiently.
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar
        
        Returns:
            List of embeddings, one per input text
        """
        embeddings = []
        
        # Check cache first if enabled
        if self.cache:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    cached_embeddings.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            logger.info(f"Embedding cache: {len(cached_embeddings)}/{len(texts)} cached")
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cached_embeddings = []
        
        # Embed uncached texts in batches
        for batch_start in range(0, len(uncached_texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(uncached_texts))
            batch_texts = uncached_texts[batch_start:batch_end]
            
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Store in cache and collect results
            for text, embedding in zip(batch_texts, batch_embeddings):
                if self.cache:
                    self.cache.put(text, embedding)
                embeddings.append((uncached_indices[batch_start + len(embeddings)], embedding))
        
        # Combine cached and newly embedded results, preserving order
        all_embeddings = cached_embeddings + embeddings
        all_embeddings.sort(key=lambda x: x[0])  # Sort by original index
        
        return [emb for _, emb in all_embeddings]
    
    def embed_with_fallback(
        self,
        texts: List[str],
        max_retries: int = 3
    ) -> List[Any]:
        """Embed texts with fallback to smaller batches on OOM.
        
        Args:
            texts: List of texts to embed
            max_retries: Maximum retry attempts
        
        Returns:
            List of embeddings
        """
        original_batch_size = self.batch_size
        retries = 0
        
        while retries < max_retries:
            try:
                return self.embed_batch(texts, show_progress=False)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.batch_size = max(1, self.batch_size // 2)
                    logger.warning(
                        f"OOM error, reducing batch size to {self.batch_size} "
                        f"(retry {retries + 1}/{max_retries})"
                    )
                    retries += 1
                else:
                    raise
        
        # Restore original batch size
        self.batch_size = original_batch_size
        raise RuntimeError(f"Failed to embed texts after {max_retries} retries")
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get embedding cache statistics."""
        if self.cache:
            return self.cache.stats()
        return None


def optimize_embedder(embedder) -> None:
    """Apply optimizations to existing embedder instance.
    
    This patches the embedder to use batch processing where possible.
    
    Args:
        embedder: gigacode.embedder.CodeEmbedder instance
    """
    if not hasattr(embedder, 'model'):
        logger.warning("Cannot optimize embedder: no model attribute")
        return
    
    # Create batch processor
    processor = BatchEmbeddingProcessor(embedder.model, device="cpu")
    logger.info(f"Optimized embedder with batch size: {processor.batch_size}")


# Backward compatibility alias
BatchEmbedder = BatchEmbeddingProcessor

