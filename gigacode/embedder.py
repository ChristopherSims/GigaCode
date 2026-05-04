"""Code embedding model with batching and GPU acceleration.

Defaults to a code-specific model (jina-embeddings-v2-base-code) and falls
back to all-MiniLM-L6-v2 if unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Prefer code-specific models; fallback to general MiniLM
_CODE_MODELS = [
    "jinaai/jina-embeddings-v2-base-code",
    "Salesforce/codet5p-110m-embedding",
]
_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Lightweight wrapper around sentence-transformers with code defaults."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = device
        self.model_name = model_name
        self._model: Any = None
        self._embedding_dim: int = 0

        if model_name:
            self._load(model_name)
        else:
            # Try code models in order, then fallback
            loaded = False
            for name in _CODE_MODELS:
                try:
                    self._load(name)
                    loaded = True
                    break
                except Exception as exc:
                    logger.debug("Code model %s unavailable (%s)", name, exc)
            if not loaded:
                self._load(_FALLBACK_MODEL)

        logger.info("Embedder ready: %s (%s dim) on %s", self.model_name, self._embedding_dim, self.device)

    def _load(self, name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(name, device=self.device)
        self.model_name = name
        self._embedding_dim = int(self._model.get_embedding_dimension())

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed texts and L2-normalize so dot-product == cosine similarity.

        Args:
            texts: Input strings (code chunks).
            batch_size: Forward-pass batch size.

        Returns:
            float32 ndarray of shape ``(len(texts), embedding_dim)``.
        """
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms
