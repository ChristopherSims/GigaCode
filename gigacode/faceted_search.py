"""Faceted search with result ranking explanation.

Extends semantic search with:
- Pre/post filters (language, path, type, line count)
- Confidence scoring based on score distribution
- Score breakdown (semantic, lexical, symbol match)
- Uncertain matches below threshold
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "SearchFilter",
    "ScoreBreakdown",
    "FacetedSearchMatch",
    "FacetedSearchResult",
    "FacetedSearcher",
    "faceted_search",
]


@dataclass
class SearchFilter:
    """Filter criteria for faceted search."""

    language: str | None = None
    path_regex: str | None = None
    type_in: list[str] | None = None  # chunk types: ["function", "class"]
    min_lines: int | None = None
    max_lines: int | None = None
    file_pattern: str | None = None  # glob-like: "*.py", "src/**/*.js"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreBreakdown:
    """Decomposed score components for transparency."""

    semantic_similarity: float
    lexical_match: float
    symbol_name_match: float
    type_bonus: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FacetedSearchMatch:
    """Enhanced search match with confidence and score breakdown."""

    file: str
    start_line: int
    end_line: int
    type: str
    name: str | None
    score: float
    confidence: str  # "high" | "medium" | "low"
    score_breakdown: ScoreBreakdown
    why: str  # Human-readable explanation
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "type": self.type,
            "name": self.name,
            "score": self.score,
            "confidence": self.confidence,
            "score_breakdown": self.score_breakdown.to_dict() if self.score_breakdown else None,
            "why": self.why,
            "text": self.text,
        }


@dataclass
class FacetedSearchResult:
    """Complete faceted search response."""

    buffer_id: str
    query: str
    matches: list[FacetedSearchMatch]
    uncertain_matches: list[FacetedSearchMatch]
    total_matches: int
    filtered_out: int
    elapsed_ms: float
    mode: str = "faceted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "buffer_id": self.buffer_id,
            "query": self.query,
            "matches": [m.to_dict() for m in self.matches],
            "uncertain_matches": [m.to_dict() for m in self.uncertain_matches],
            "total_matches": self.total_matches,
            "filtered_out": self.filtered_out,
            "elapsed_ms": self.elapsed_ms,
            "mode": self.mode,
        }


def _compile_file_pattern(pattern: str | None) -> re.Pattern | None:
    """Convert glob-like pattern to regex."""
    if not pattern:
        return None
    # Convert simple glob to regex: *.py -> .*\.py$
    regex_pattern = pattern.replace(".", r"\.")
    regex_pattern = regex_pattern.replace("*", ".*")
    regex_pattern = regex_pattern.replace("?", ".")
    return re.compile(regex_pattern)


def _matches_filter(chunk, filter: SearchFilter) -> bool:
    """Check if a chunk matches the filter criteria."""
    # Language filter (from file extension)
    if filter.language:
        ext_map = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "rust": ".rs",
            "go": ".go",
            "java": ".java",
            "cpp": ".cpp",
            "c": ".c",
        }
        expected_ext = ext_map.get(filter.language.lower())
        if expected_ext and not chunk.file.endswith(expected_ext):
            return False

    # Path regex filter
    if filter.path_regex:
        if not re.search(filter.path_regex, chunk.file):
            return False

    # Type filter
    if filter.type_in and chunk.type not in filter.type_in:
        return False

    # Line count filters
    line_count = chunk.end_line - chunk.start_line + 1
    if filter.min_lines is not None and line_count < filter.min_lines:
        return False
    if filter.max_lines is not None and line_count > filter.max_lines:
        return False

    # File pattern filter
    if filter.file_pattern:
        compiled = _compile_file_pattern(filter.file_pattern)
        if compiled and not compiled.search(chunk.file):
            return False

    return True


def _compute_score_breakdown(
    semantic_score: float,
    chunk,
    query: str,
) -> ScoreBreakdown:
    """Compute decomposed score components."""
    # Semantic similarity (already computed)
    semantic = float(semantic_score)

    # Lexical match: check if query terms appear in text
    query_terms = set(query.lower().split())
    text_lower = (chunk.text or "").lower()
    lexical_terms_found = sum(1 for term in query_terms if term in text_lower)
    lexical = lexical_terms_found / max(len(query_terms), 1)

    # Symbol name match: check if query matches chunk name
    symbol_match = 0.0
    if chunk.name:
        name_lower = chunk.name.lower()
        query_lower = query.lower()
        if query_lower in name_lower or name_lower in query_lower:
            # Partial match
            symbol_match = 0.5
        # Check individual terms
        for term in query_terms:
            if term in name_lower:
                symbol_match = max(symbol_match, 0.7)
        # Exact or near-exact match
        if any(term == name_lower for term in query_terms):
            symbol_match = 1.0

    # Type bonus: definitions get a small boost
    type_bonus = 0.05 if chunk.type in ("function", "class", "method") else 0.0

    # Total is not a simple sum — semantic dominates
    total = semantic * 0.7 + lexical * 0.15 + symbol_match * 0.15 + type_bonus

    return ScoreBreakdown(
        semantic_similarity=round(semantic, 3),
        lexical_match=round(lexical, 3),
        symbol_name_match=round(symbol_match, 3),
        type_bonus=round(type_bonus, 3),
        total=round(total, 3),
    )


def _compute_confidence(
    score: float,
    all_scores: list[float],
    top_k: int = 5,
) -> str:
    """Compute confidence level based on score distribution.

    High: top score is significantly better than rest (gap > 0.2)
    Medium: moderate gap or above-average score
    Low: weak signal, near noise floor
    """
    if not all_scores:
        return "low"

    # Score absolute threshold
    if score < 0.3:
        return "low"
    if score > 0.85:
        return "high"

    # Gap analysis
    sorted_scores = sorted(all_scores, reverse=True)
    if len(sorted_scores) >= 2:
        gap = sorted_scores[0] - sorted_scores[1]
        if gap > 0.2 and score == sorted_scores[0]:
            return "high"
        if gap < 0.05:
            return "low"

    # Compare to mean
    mean_score = np.mean(all_scores) if all_scores else 0
    if score > mean_score + 0.3:
        return "high"
    if score > mean_score + 0.1:
        return "medium"

    return "medium" if score > 0.5 else "low"


def _build_why(
    breakdown: ScoreBreakdown,
    confidence: str,
    chunk_name: str | None,
    chunk_type: str,
) -> str:
    """Build human-readable explanation of why this match ranked."""
    parts = []

    if breakdown.semantic_similarity > 0.8:
        parts.append("Strong semantic similarity with query concept.")
    elif breakdown.semantic_similarity > 0.5:
        parts.append("Moderate semantic alignment with query.")

    if breakdown.symbol_name_match > 0.7 and chunk_name:
        parts.append(f"Symbol name '{chunk_name}' closely matches query terms.")
    elif breakdown.symbol_name_match > 0.4 and chunk_name:
        parts.append(f"Symbol name '{chunk_name}' partially matches query.")

    if breakdown.lexical_match > 0.5:
        parts.append("Query terms appear directly in code text.")

    if chunk_type in ("function", "method"):
        parts.append("This is a callable definition.")
    elif chunk_type == "class":
        parts.append("This is a class definition.")

    if confidence == "high":
        parts.append("High confidence: this result stands out from others.")
    elif confidence == "low":
        parts.append("Low confidence: result may be incidental.")

    if not parts:
        return "Ranked by general semantic similarity."

    return " ".join(parts)


class FacetedSearcher:
    """Faceted search with filtering, confidence scoring, and explanations."""

    def __init__(self, chunks: list[Any], embeddings: np.ndarray | None = None):
        self.chunks = chunks
        self.embeddings = embeddings

    def search(
        self,
        query_embedding: np.ndarray,
        query: str,
        filter: SearchFilter | None = None,
        top_k: int = 10,
        include_uncertain: bool = True,
        confidence_threshold: float = 0.5,
    ) -> FacetedSearchResult:
        """Perform faceted search with ranking explanation.

        Args:
            query_embedding: Pre-computed query embedding vector.
            query: Original query string (for lexical/symbol matching).
            filter: Optional filter criteria.
            top_k: Number of top results to return.
            include_uncertain: Whether to include uncertain matches.
            confidence_threshold: Minimum total score for confident matches.

        Returns:
            FacetedSearchResult with matches, uncertain matches, and explanations.
        """
        import time

        start_time = time.perf_counter()

        if self.embeddings is None or len(self.chunks) == 0:
            return FacetedSearchResult(
                buffer_id="",
                query=query,
                matches=[],
                uncertain_matches=[],
                total_matches=0,
                filtered_out=0,
                elapsed_ms=0.0,
            )

        # Compute semantic scores
        scores = np.dot(self.embeddings, query_embedding)
        sorted_indices = np.argsort(scores)[::-1]

        # Gather all scores for confidence computation
        all_scores = [float(scores[i]) for i in sorted_indices]

        matches: list[FacetedSearchMatch] = []
        uncertain: list[FacetedSearchMatch] = []
        filtered_out = 0

        for idx in sorted_indices:
            chunk = self.chunks[idx]
            semantic_score = float(scores[idx])

            # Apply filters
            if filter and not _matches_filter(chunk, filter):
                filtered_out += 1
                continue

            # Compute score breakdown
            breakdown = _compute_score_breakdown(semantic_score, chunk, query)

            # Determine confidence
            confidence = _compute_confidence(breakdown.total, all_scores[: max(top_k * 3, 20)])

            # Build explanation
            why = _build_why(breakdown, confidence, chunk.name, chunk.type)

            match = FacetedSearchMatch(
                file=chunk.file,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                type=chunk.type,
                name=chunk.name,
                score=round(breakdown.total, 3),
                confidence=confidence,
                score_breakdown=breakdown,
                why=why,
                text=chunk.text,
            )

            # Classify as confident or uncertain
            if confidence == "high" or breakdown.total >= confidence_threshold:
                matches.append(match)
            elif include_uncertain and breakdown.total >= confidence_threshold * 0.5:
                uncertain.append(match)

            # Stop once we have enough matches
            if len(matches) >= top_k and (not include_uncertain or len(uncertain) >= top_k):
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return FacetedSearchResult(
            buffer_id="",
            query=query,
            matches=matches[:top_k],
            uncertain_matches=uncertain[:top_k] if include_uncertain else [],
            total_matches=len(matches) + len(uncertain),
            filtered_out=filtered_out,
            elapsed_ms=elapsed_ms,
        )


def faceted_search(
    chunks: list[Any],
    query_embedding: np.ndarray,
    query: str,
    filter: dict[str, Any] | None = None,
    top_k: int = 10,
    include_uncertain: bool = True,
) -> dict[str, Any]:
    """Convenience function: faceted search from dict filter."""
    search_filter = SearchFilter(**filter) if filter else None
    searcher = FacetedSearcher(chunks)
    result = searcher.search(
        query_embedding=query_embedding,
        query=query,
        filter=search_filter,
        top_k=top_k,
        include_uncertain=include_uncertain,
    )
    return result.to_dict()
