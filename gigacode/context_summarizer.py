"""Hierarchical context packing with file-level summarization.

Generates three-level context for LLM consumption:
1. File-level summaries (overview, top-level definitions)
2. Chunk-level summaries (function/class signatures + docstrings)
3. Line-level details (specific code lines with context)

This mirrors how humans read code: overview → sections → details.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "FileSummary",
    "ContextLevel",
    "HierarchicalContext",
    "ContextSummarizer",
    "summarize_file",
]


@dataclass
class FileSummary:
    """Summary of a source file."""

    file: str
    description: str
    definitions: list[dict[str, Any]]  # [{name, type, line, docstring}]
    imports: list[str]
    total_lines: int
    chunk_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextLevel:
    """A single level in the hierarchical context."""

    level: str  # "file_summary", "chunk", "lines"
    file: str
    name: str | None
    start_line: int
    end_line: int
    text: str
    relevance_score: float
    tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HierarchicalContext:
    """Complete hierarchical context assembly."""

    query: str
    hierarchy: list[ContextLevel]
    total_tokens: int
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextSummarizer:
    """Generate hierarchical context for LLM consumption."""

    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self._file_summaries: dict[str, FileSummary] = {}
        self._build_file_summaries()

    def _build_file_summaries(self) -> None:
        """Build summaries from chunks."""
        file_chunks: dict[str, list[Any]] = {}
        for ch in self.chunks:
            file_chunks.setdefault(ch.file, []).append(ch)

        for file_path, chunks in file_chunks.items():
            chunks.sort(key=lambda c: c.start_line)

            # Extract top-level definitions
            definitions: list[dict[str, Any]] = []
            for ch in chunks:
                if ch.type in ("function", "class", "method", "trait", "interface"):
                    # Extract docstring (first triple-quoted string after definition)
                    docstring = self._extract_docstring(ch.text)
                    definitions.append(
                        {
                            "name": ch.name,
                            "type": ch.type,
                            "start_line": ch.start_line,
                            "end_line": ch.end_line,
                            "docstring": docstring[:200] if docstring else None,
                        }
                    )

            # Gather imports
            all_imports: set[str] = set()
            for ch in chunks:
                for imp in ch.imports or []:
                    all_imports.add(imp)

            # Estimate total lines
            total_lines = max(
                (ch.end_line for ch in chunks),
                default=0,
            )

            # Build description from module docstring or first chunk
            description = self._extract_module_docstring(chunks)
            if not description and definitions:
                types_count: dict[str, int] = {}
                for d in definitions:
                    types_count[d["type"]] = types_count.get(d["type"], 0) + 1
                parts = [f"{count} {t}s" for t, count in types_count.items()]
                description = f"File with {', '.join(parts)}."

            self._file_summaries[file_path] = FileSummary(
                file=file_path,
                description=description or f"Source file with {len(chunks)} chunks.",
                definitions=definitions,
                imports=sorted(all_imports),
                total_lines=total_lines,
                chunk_count=len(chunks),
            )

    @staticmethod
    def _extract_docstring(text: str) -> str | None:
        """Extract the first triple-quoted docstring from text."""
        # Match """...""" or '''...'''
        for quote in ('"""', "'''"):
            idx = text.find(quote)
            if idx >= 0:
                end = text.find(quote, idx + 3)
                if end > idx:
                    return text[idx + 3 : end].strip()
        return None

    @staticmethod
    def _extract_module_docstring(chunks: list[Any]) -> str | None:
        """Extract module-level docstring from first chunk."""
        for ch in chunks:
            if ch.type == "orphan" or ch.start_line == 1:
                docstring = ContextSummarizer._extract_docstring(ch.text)
                if docstring:
                    return docstring
        return None

    def summarize_file(self, file_path: str) -> FileSummary | None:
        """Get summary for a specific file."""
        return self._file_summaries.get(file_path)

    def pack_hierarchical(
        self,
        query_embedding: Any,
        query: str,
        max_tokens: int = 8192,
        levels: list[str] | None = None,
        top_k_files: int = 5,
        top_k_chunks: int = 10,
    ) -> HierarchicalContext:
        """Pack context hierarchically: file summaries → chunks → lines.

        Args:
            query_embedding: Query embedding vector for relevance scoring.
            query: Original query string.
            max_tokens: Maximum tokens for context.
            levels: Which levels to include ["file_summary", "chunk", "lines"].
            top_k_files: Number of top files to include summaries for.
            top_k_chunks: Number of top chunks to include per file.

        Returns:
            HierarchicalContext with assembled levels.
        """
        levels = levels or ["file_summary", "chunk", "lines"]
        hierarchy: list[ContextLevel] = []
        total_tokens = 0
        chars_per_token = 4  # Approximate

        # Score files by relevance (simplified: use first chunk of each file)
        file_scores: dict[str, float] = {}
        for file_path, summary in self._file_summaries.items():
            # Score by query terms in file path, imports, and definitions
            score = 0.0
            query_lower = query.lower()

            # File name match
            if any(term in file_path.lower() for term in query_lower.split()):
                score += 0.3

            # Definition name match
            for d in summary.definitions:
                if d["name"] and query_lower in d["name"].lower():
                    score += 0.5

            # Import match
            for imp in summary.imports:
                if query_lower in imp.lower():
                    score += 0.2

            file_scores[file_path] = score

        # Sort files by score
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)

        for file_path, file_score in sorted_files[:top_k_files]:
            summary = self._file_summaries[file_path]

            # Level 1: File summary
            if "file_summary" in levels:
                summary_text = (
                    f"# File: {file_path}\n"
                    f"Description: {summary.description}\n"
                    f"Lines: {summary.total_lines}, Chunks: {summary.chunk_count}\n"
                    f"Top definitions: {', '.join(d['name'] for d in summary.definitions[:5] if d['name'])}\n"
                )
                tokens = len(summary_text) // chars_per_token
                if total_tokens + tokens > max_tokens:
                    break

                hierarchy.append(
                    ContextLevel(
                        level="file_summary",
                        file=file_path,
                        name=None,
                        start_line=1,
                        end_line=summary.total_lines,
                        text=summary_text,
                        relevance_score=round(file_score, 3),
                        tokens=tokens,
                    )
                )
                total_tokens += tokens

            # Level 2: Chunks
            if "chunk" in levels:
                file_chunks = [ch for ch in self.chunks if ch.file == file_path]
                # Score chunks
                chunk_scores: list[tuple[Any, float]] = []
                for ch in file_chunks:
                    score = 0.0
                    if ch.name and query.lower() in ch.name.lower():
                        score += 0.8
                    if query.lower() in ch.text.lower():
                        score += 0.3
                    chunk_scores.append((ch, score))

                chunk_scores.sort(key=lambda x: x[1], reverse=True)

                for ch, ch_score in chunk_scores[:top_k_chunks]:
                    # Extract signature + docstring as chunk summary
                    first_lines = ch.text.splitlines()[:5]
                    signature = "\n".join(first_lines)
                    docstring = self._extract_docstring(ch.text)

                    chunk_text = f"## {ch.type} {ch.name or 'anonymous'} (lines {ch.start_line}-{ch.end_line})\n"
                    if docstring:
                        chunk_text += f"Docstring: {docstring[:150]}\n"
                    chunk_text += f"```\n{signature}\n```\n"

                    tokens = len(chunk_text) // chars_per_token
                    if total_tokens + tokens > max_tokens:
                        return HierarchicalContext(
                            query=query,
                            hierarchy=hierarchy,
                            total_tokens=total_tokens,
                            truncated=True,
                        )

                    hierarchy.append(
                        ContextLevel(
                            level="chunk",
                            file=file_path,
                            name=ch.name,
                            start_line=ch.start_line,
                            end_line=ch.end_line,
                            text=chunk_text,
                            relevance_score=round(ch_score, 3),
                            tokens=tokens,
                        )
                    )
                    total_tokens += tokens

            # Level 3: Specific lines (for highest-scoring chunks)
            if "lines" in levels:
                # Add most relevant lines from top chunk
                if chunk_scores:
                    top_chunk = chunk_scores[0][0]
                    # Find query-relevant lines
                    query_terms = query.lower().split()
                    relevant_lines: list[tuple[int, str, float]] = []
                    for i, line in enumerate(
                        top_chunk.text.splitlines(), start=top_chunk.start_line
                    ):
                        line_lower = line.lower()
                        score = sum(1 for term in query_terms if term in line_lower) / max(
                            len(query_terms), 1
                        )
                        if score > 0:
                            relevant_lines.append((i, line, score))

                    relevant_lines.sort(key=lambda x: x[2], reverse=True)
                    for line_num, line_text, line_score in relevant_lines[:3]:
                        line_text_full = f"  Line {line_num}: {line_text}\n"
                        tokens = len(line_text_full) // chars_per_token
                        if total_tokens + tokens > max_tokens:
                            break

                        hierarchy.append(
                            ContextLevel(
                                level="lines",
                                file=file_path,
                                name=top_chunk.name,
                                start_line=line_num,
                                end_line=line_num,
                                text=line_text_full,
                                relevance_score=round(line_score, 3),
                                tokens=tokens,
                            )
                        )
                        total_tokens += tokens

        return HierarchicalContext(
            query=query,
            hierarchy=hierarchy,
            total_tokens=total_tokens,
            truncated=False,
        )

    def save_summaries(self, buffer_dir: Path) -> None:
        """Save file summaries to disk for caching."""
        summaries_path = buffer_dir / "summaries.json"
        data = {k: v.to_dict() for k, v in self._file_summaries.items()}
        try:
            summaries_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except (OSError, TypeError) as e:
            logger.warning(f"Failed to save summaries: {e}")

    @classmethod
    def load_summaries(cls, buffer_dir: Path) -> dict[str, FileSummary] | None:
        """Load file summaries from disk."""
        summaries_path = buffer_dir / "summaries.json"
        if not summaries_path.exists():
            return None
        try:
            data = json.loads(summaries_path.read_text(encoding="utf-8"))
            return {k: FileSummary(**v) for k, v in data.items()}
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to load summaries: {e}")
            return None


def summarize_file(chunks: list[Any], file_path: str) -> dict[str, Any] | None:
    """Convenience function: summarize a single file."""
    summarizer = ContextSummarizer(chunks)
    summary = summarizer.summarize_file(file_path)
    return summary.to_dict() if summary else None
