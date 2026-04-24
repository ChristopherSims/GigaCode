"""Lightweight in-memory BM25 lexical index for code search.

No external dependencies beyond Python stdlib.  Tokenisation is whitespace +
symbol splitting so that ``fetch_data`` becomes ``["fetch", "data"]``.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Lower-case token list with CamelCase / snake_case splitting."""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        word = match.group(0)
        # Split snake_case
        for part in word.split("_"):
            if not part:
                continue
            # Split CamelCase
            split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
            for sub in split.split():
                if sub:
                    tokens.append(sub.lower())
    return tokens


class LexicalIndex:
    """BM25 index over a collection of documents (code chunks).

    Each document is identified by an integer ``doc_id`` and stored as a list
    of tokens.  The index is rebuilt from scratch when documents are added or
    removed, which is fine for codebases that change relatively slowly.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[int, list[str]] = {}
        self._doc_len: dict[int, int] = {}
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}
        self._tf: dict[int, dict[str, int]] = {}
        self._dirty = True

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add(self, doc_id: int, text: str) -> None:
        """Add or overwrite a document."""
        self._docs[doc_id] = _tokenize(text)
        self._dirty = True

    def remove(self, doc_id: int) -> None:
        """Remove a document."""
        self._docs.pop(doc_id, None)
        self._dirty = True

    def clear(self) -> None:
        """Drop all documents."""
        self._docs.clear()
        self._dirty = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """Return top-K BM25 scores as ``[{doc_id, score}]`` sorted descending."""
        if self._dirty:
            self._rebuild()
        if not self._docs:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scores: dict[int, float] = {}
        for token in q_tokens:
            idf = self._idf.get(token, 0.0)
            if idf == 0.0:
                continue
            for doc_id, tf_map in self._tf.items():
                f = tf_map.get(token, 0)
                if f == 0:
                    continue
                dl = self._doc_len[doc_id]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * f * (self.k1 + 1) / denom

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"doc_id": did, "score": float(sc)} for did, sc in sorted_scores[:top_k]]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _rebuild(self) -> None:
        self._tf = {}
        self._doc_len = {}
        total_len = 0
        for doc_id, tokens in self._docs.items():
            self._doc_len[doc_id] = len(tokens)
            total_len += len(tokens)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._tf[doc_id] = tf

        self._avgdl = total_len / len(self._docs) if self._docs else 1.0

        # Compute IDF for every token that appears in the corpus
        df: dict[str, int] = {}
        for tf_map in self._tf.values():
            for t in tf_map:
                df[t] = df.get(t, 0) + 1

        N = len(self._docs)
        self._idf = {}
        for t, d in df.items():
            # Standard BM25 IDF with smoothing
            self._idf[t] = math.log((N - d + 0.5) / (d + 0.5) + 1)

        self._dirty = False
