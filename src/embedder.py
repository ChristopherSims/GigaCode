"""Sentence-transformers embedder with pre-normalized outputs."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Lightweight wrapper around sentence-transformers.

    Loads ``all-MiniLM-L6-v2`` by default and produces L2-normalized
    embeddings so that dot-product equals cosine similarity.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        """Initialise the embedder.

        Args:
            model_name: Hugging Face model name or local path.
                Defaults to ``all-MiniLM-L6-v2``.
            device: torch device (``"cpu"``, ``"cuda"``, etc.).
                ``None`` lets sentence-transformers auto-select.
        """
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        self.model_name = model_name or _DEFAULT_MODEL
        logger.info("Loading embedding model %s ...", self.model_name)
        self._model: Any = SentenceTransformer(self.model_name, device=device)
        logger.info("Model loaded (%s)", self._model.get_embedding_dimension())

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of produced vectors."""
        return int(self._model.get_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of strings and L2-normalize the result.

        Args:
            texts: Input strings.
            batch_size: Inference batch size.

        Returns:
            Array of shape ``(len(texts), embedding_dim)`` with dtype float32
            and L2 norm 1.0 along the last axis.
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms
        return normalized


def embed_tokens(token_texts: list[str]) -> np.ndarray:
    """Convenience function: embed token strings with the default model.

    Args:
        token_texts: List of token strings.

    Returns:
        float32 ndarray of shape ``(len(token_texts), embedding_dim)``.
        Vectors are L2-normalized so that dot-product == cosine similarity.
    """
    embedder = Embedder()
    return embedder.encode(token_texts)
