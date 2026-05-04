"""Typed response dataclasses for GigaCode API.

Replaces ad-hoc `dict[str, Any]` responses with validated dataclasses.
Each dataclass provides a `to_dict()` method for backward compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class ResponseStatus(str, Enum):
    """Standard response status values."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class SearchMatch:
    """A single match in a search result."""
    file: str
    start_line: int
    end_line: int
    score: float
    doc_id: int
    type: Optional[str] = None
    name: Optional[str] = None
    match_type: Optional[str] = None  # "semantic", "lexical", "name", "rrf"
    semantic_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    rrf_score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SearchResponse:
    """Generic search result response."""
    status: ResponseStatus
    matches: list[SearchMatch] = field(default_factory=list)
    cached: bool = False
    total: Optional[int] = None
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        result = {
            "status": self.status.value,
            "matches": [m.to_dict() for m in self.matches],
            "cached": self.cached,
        }
        if self.total is not None:
            result["total"] = self.total
        if self.message is not None:
            result["message"] = self.message
        return result


@dataclass
class ClusterItem:
    """A single cluster of similar code chunks."""
    file: str
    start_line: int
    end_line: int
    size: int
    avg_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterResponse:
    """Code clustering result."""
    status: ResponseStatus
    clusters: list[ClusterItem] = field(default_factory=list)
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status.value,
            "clusters": [c.to_dict() for c in self.clusters],
        }
        if self.message is not None:
            result["message"] = self.message
        return result


@dataclass
class DuplicateItem:
    """A single duplicate pair."""
    file1: str
    start_line1: int
    end_line1: int
    file2: str
    start_line2: int
    end_line2: int
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DuplicateResponse:
    """Duplicate detection result."""
    status: ResponseStatus
    duplicates: list[DuplicateItem] = field(default_factory=list)
    total: Optional[int] = None
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status.value,
            "duplicates": [d.to_dict() for d in self.duplicates],
        }
        if self.total is not None:
            result["total"] = self.total
        if self.message is not None:
            result["message"] = self.message
        return result


@dataclass
class CodeLine:
    """A single line of code in a read/write response."""
    line_no: int
    content: str


@dataclass
class ReadResponse:
    """Code read result."""
    status: ResponseStatus
    file: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    lines: list[str] = field(default_factory=list)
    files: dict[str, list[str]] = field(default_factory=dict)
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {"status": self.status.value}
        if self.file is not None:
            result["file"] = self.file
            result["start_line"] = self.start_line
            result["end_line"] = self.end_line
            result["lines"] = self.lines
        else:
            result["files"] = self.files
        if self.message is not None:
            result["message"] = self.message
        return result


@dataclass
class WriteResponse:
    """Code write result."""
    status: ResponseStatus
    file: Optional[str] = None
    changed_lines: Optional[int] = None
    replaced_lines: Optional[int] = None
    total_lines: Optional[int] = None
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {"status": self.status.value}
        if self.file is not None:
            result["file"] = self.file
            result["changed_lines"] = self.changed_lines
            result["replaced_lines"] = self.replaced_lines
            result["total_lines"] = self.total_lines
        if self.message is not None:
            result["message"] = self.message
        return result


@dataclass
class CommitResponse:
    """Commit result."""
    status: ResponseStatus
    written_files: list[str] = field(default_factory=list)
    dry_run: bool = False
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status.value,
            "written_files": self.written_files,
            "dry_run": self.dry_run,
        }
        if self.message is not None:
            result["message"] = self.message
        return result


@dataclass
class EmbedResponse:
    """Embedding result."""
    status: ResponseStatus
    buffer_id: Optional[str] = None
    chunk_count: Optional[int] = None
    size_bytes: Optional[int] = None
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {"status": self.status.value}
        if self.buffer_id is not None:
            result["buffer_id"] = self.buffer_id
            result["chunk_count"] = self.chunk_count
            result["size_bytes"] = self.size_bytes
        if self.message is not None:
            result["message"] = self.message
        return result


@dataclass
class ErrorResponse:
    """Generic error response."""
    status: ResponseStatus = ResponseStatus.ERROR
    message: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status.value,
            "message": self.message,
        }
        if self.context:
            result["context"] = self.context
        return result


@dataclass
class ListResponse:
    """List buffers response."""
    status: ResponseStatus
    buffers: list[dict[str, Any]] = field(default_factory=list)
    message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status.value,
            "buffers": self.buffers,
        }
        if self.message is not None:
            result["message"] = self.message
        return result
