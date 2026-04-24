"""LLM context packing — greedily pack high-relevance chunks into a token budget.

Token counting is approximate (1 token ≈ 4 chars for English/code) since we
do not require tiktoken as a dependency.  If ``tiktoken`` is installed it is
used for more accurate counts.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Approximate tokens per character for code
_CHARS_PER_TOKEN = 4.0


def _approx_token_count(text: str) -> int:
    """Fast approximate token count."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _exact_token_count(text: str) -> int:
    """Try tiktoken; fall back to approximation."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return _approx_token_count(text)


def pack_context(
    chunks: list[Any],
    scores: list[float],
    max_tokens: int = 8192,
    use_tiktoken: bool = False,
) -> dict[str, Any]:
    """Greedy-pack chunks by relevance until *max_tokens* is reached.

    Args:
        chunks: List of CodeChunk-like objects (must have ``.text``, ``.file``, ``.start_line``, ``.end_line``, ``.name``).
        scores: Parallel list of relevance scores (higher = better).
        max_tokens: Target token budget.
        use_tiktoken: If True, use tiktoken for exact counts (slower).

    Returns:
        Dict with ``status``, ``packed_chunks``, ``total_tokens``, and ``remaining_tokens``.
    """
    if len(chunks) != len(scores):
        return {
            "status": "error",
            "message": f"chunks ({len(chunks)}) and scores ({len(scores)}) length mismatch",
        }

    token_fn = _exact_token_count if use_tiktoken else _approx_token_count

    # Sort by descending score
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    packed: list[dict[str, Any]] = []
    total_tokens = 0

    for idx, score in indexed:
        ch = chunks[idx]
        cost = token_fn(ch.text)
        if total_tokens + cost > max_tokens and packed:
            # Skip if over budget and we already have something
            continue
        packed.append({
            "file": ch.file,
            "start_line": ch.start_line,
            "end_line": ch.end_line,
            "name": ch.name,
            "type": ch.type,
            "score": round(score, 4),
            "tokens": cost,
        })
        total_tokens += cost
        if total_tokens >= max_tokens:
            break

    # Re-sort packed chunks by original file order for coherent reading
    packed.sort(key=lambda x: (x["file"], x["start_line"]))

    return {
        "status": "ok",
        "packed_chunks": packed,
        "total_tokens": total_tokens,
        "remaining_tokens": max(0, max_tokens - total_tokens),
        "count": len(packed),
    }
