"""Code embedding model with batching and GPU acceleration.

Defaults to a code-specific model (jina-embeddings-v2-base-code) and falls
back to all-MiniLM-L6-v2 if unavailable.

Model loading can be deferred with ``lazy=True`` so that constructing a tool
does not require the embedding dependency or a network download until a
semantic operation actually needs vectors.  ``local_files_only=True`` and
``cache_folder`` support fully offline / pre-cached deployments.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from gigacode.constants import DEFAULT_BATCH_SIZE

# NOTE: torch._dynamo env vars are set in gigacode/__init__.py on package import.

logger = logging.getLogger(__name__)

# Prefer code-specific models; fallback to general MiniLM
_CODE_MODELS = [
    "jinaai/jina-embeddings-v2-base-code",
    "Salesforce/codet5p-110m-embedding",
]
_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


__all__ = [
    "Embedder",
]


class Embedder:
    """Lightweight wrapper around sentence-transformers with code defaults."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        lazy: bool = False,
        local_files_only: bool = False,
        cache_folder: str | None = None,
    ) -> None:

        self.device = device
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.cache_folder = cache_folder
        self._model: Any = None
        self._embedding_dim: int = 0
        self._loaded = False

        if not lazy:
            self.ensure_loaded()

        logger.info(
            "Embedder ready: %s (%s dim) on %s", self.model_name, self._embedding_dim, self.device
        )

    @property
    def is_loaded(self) -> bool:
        """Return True once the underlying model has been loaded."""
        return self._loaded

    def ensure_loaded(self) -> None:
        """Load the embedding model on first use.

        Raises:
            ImportError: If sentence-transformers (or its native backend) is
                not installed, with an actionable install hint.
        """
        if self._loaded:
            return

        if self.model_name:
            self._load(self.model_name)
        else:
            # Try code models in order, then fallback
            loaded = False
            for name in _CODE_MODELS:
                try:
                    self._load(name)
                    loaded = True
                    break
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    logger.debug("Code model %s unavailable (%s)", name, exc)
            if not loaded:
                self._load(_FALLBACK_MODEL)

        self._loaded = True
        logger.info(
            "Embedder ready: %s (%s dim) on %s", self.model_name, self._embedding_dim, self.device
        )

    def _load(self, name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install the embedding extra with: pip install 'gigacode[embed]', "
                "or supply your own embedder via CodeEmbeddingTool(embedder=...)."
            ) from exc

        kwargs: dict[str, Any] = {"device": self.device}
        if self.local_files_only:
            kwargs["local_files_only"] = True
        if self.cache_folder:
            kwargs["cache_folder"] = self.cache_folder

        logger.info(
            "Loading embedding model '%s' (first use may download; pass "
            "local_files_only=True and cache_folder for offline use)...",
            name,
        )
        self._model = SentenceTransformer(name, **kwargs)
        self.model_name = name
        # SentenceTransformers 6.x renamed the dimension accessor to
        # get_embedding_dimension(); 5.x only exposes
        # get_sentence_embedding_dimension(). Support both.
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if not callable(get_dim):
            get_dim = self._model.get_sentence_embedding_dimension
        self._embedding_dim = int(get_dim())

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode(self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """Embed texts and L2-normalize so dot-product == cosine similarity.

        Args:
            texts: Input strings (code chunks).
            batch_size: Forward-pass batch size.

        Returns:
            float32 ndarray of shape ``(len(texts), embedding_dim)``.
        """
        self.ensure_loaded()
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

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string, returning a 1-D vector."""
        vectors = self.encode([text])
        if vectors.shape[0] == 0:
            return np.zeros((self._embedding_dim,), dtype=np.float32)
        return vectors[0]
