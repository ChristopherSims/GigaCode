"""Code quality and complexity scoring."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["QualityScorer", "QualityResult"]

@dataclass
class QualityResult:
    file: str
    complexity: dict[str, Any]
    maintainability: dict[str, Any]
    documentation: dict[str, Any]
    overall: str
    suggestions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class QualityScorer:
    def score_file(self, chunks: list[Any], file_path: str) -> QualityResult | None:
        file_chunks = [ch for ch in chunks if ch.file == file_path]
        if not file_chunks:
            return None

        # Complexity
        total_branches = 0
        total_lines = 0
        max_nesting = 0
        function_count = 0
        docstring_count = 0
        type_hint_count = 0

        suggestions: list[str] = []

        for ch in file_chunks:
            lines = ch.text.splitlines()
            total_lines += len(lines)

            if ch.type in ("function", "method"):
                function_count += 1
                # Count branches
                branches = len(re.findall(r"\b(if|for|while|and|or|except|with)\b", ch.text))
                total_branches += branches

                # Check docstring
                if '"""' in ch.text or "'''" in ch.text:
                    docstring_count += 1
                else:
                    if ch.name:
                        suggestions.append(f"Function '{ch.name}' (line {ch.start_line}) missing docstring.")

                # Check type hints
                if "->" in ch.text or ": " in "\n".join(lines[:2]):
                    type_hint_count += 1

                # Nesting depth
                nesting = 0
                max_local_nesting = 0
                for line in lines:
                    indent = len(line) - len(line.lstrip())
                    if indent > 0:
                        depth = indent // 4
                        max_local_nesting = max(max_local_nesting, depth)
                max_nesting = max(max_nesting, max_local_nesting)

                if branches > 8:
                    suggestions.append(
                        f"Function '{ch.name}' (line {ch.start_line}) has {branches} branches. "
                        "Consider splitting into smaller functions."
                    )
                if len(lines) > 50:
                    suggestions.append(
                        f"Function '{ch.name}' (line {ch.start_line}) is {len(lines)} lines long. "
                        "Consider extracting helper functions."
                    )

        cyclomatic = total_branches if function_count > 0 else 0
        avg_cyclomatic = cyclomatic / max(function_count, 1)

        complexity_rating = "low" if avg_cyclomatic < 4 else "medium" if avg_cyclomatic < 8 else "high"
        doc_coverage = docstring_count / max(function_count, 1)
        type_coverage = type_hint_count / max(function_count, 1)

        doc_rating = "high" if doc_coverage > 0.8 else "medium" if doc_coverage > 0.4 else "low"

        # Overall rating
        scores = []
        if complexity_rating == "high":
            scores.append(1)
        elif complexity_rating == "medium":
            scores.append(2)
        else:
            scores.append(3)

        if doc_rating == "low":
            scores.append(1)
        elif doc_rating == "medium":
            scores.append(2)
        else:
            scores.append(3)

        avg_score = sum(scores) / len(scores)
        overall = "low" if avg_score < 1.5 else "medium" if avg_score < 2.5 else "high"

        return QualityResult(
            file=file_path,
            complexity={
                "cyclomatic_total": cyclomatic,
                "average_per_function": round(avg_cyclomatic, 1),
                "max_nesting_depth": max_nesting,
                "rating": complexity_rating,
            },
            maintainability={
                "total_lines": total_lines,
                "function_count": function_count,
                "average_function_length": round(total_lines / max(function_count, 1), 1),
                "rating": "medium" if avg_cyclomatic < 6 else "low",
            },
            documentation={
                "docstring_coverage": round(doc_coverage, 2),
                "type_hint_coverage": round(type_coverage, 2),
                "rating": doc_rating,
            },
            overall=overall,
            suggestions=suggestions[:10],  # Cap suggestions
        )

def score_file_quality(chunks: list[Any], file_path: str) -> dict[str, Any] | None:
    scorer = QualityScorer()
    result = scorer.score_file(chunks, file_path)
    return result.to_dict() if result else None
