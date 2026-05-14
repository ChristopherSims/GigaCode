"""Pydantic request/response models for GigaCode API.

Full typed models for Phase 2 request/response validation.
Complements the existing tool_schema.py JSON schemas with
Pydantic models that can be used for FastAPI dependency injection,
OpenAPI schema generation, and runtime validation.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field

__all__ = [
    # Request models
    "GetReferencesRequest",
    "GetFullContextRequest",
    "AnalyzeChangeRequest",
    "GetTestCoverageRequest",
    "InferTypesRequest",
    "GetSymbolMetadataRequest",
    "SearchBatchRequest",
    "AutoFormatRequest",
    "AutoLintRequest",
    "AutoPolishRequest",
    "PolishBeforeCommitRequest",
    # Response models
    "CallerInfo",
    "CalleeInfo",
    "GetReferencesResponse",
    "DefinitionInfo",
    "TypeInfo",
    "TestInfo",
    "ErrorInfo",
    "GetFullContextResponse",
    "AnalyzeChangeResponse",
    "CoverageEntry",
    "GetTestCoverageResponse",
    "InferTypesResponse",
    "SymbolMetadataResponse",
    "SearchBatchResponse",
    "FormatResult",
    "LintIssue",
    "LintResult",
    "AutoPolishResponse",
    "PolishBeforeCommitResponse",
]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class GetReferencesRequest(BaseModel):
    buffer_id: str
    symbol: str
    direction: str = Field("both", pattern="^(both|calls|called_by)$")
    top_k: int = Field(50, ge=1, le=200)
    expand_depth: Optional[int] = Field(None, ge=1, le=10)


class GetFullContextRequest(BaseModel):
    buffer_id: str
    symbol: str
    include: Optional[List[str]] = Field(None, description="Sections: definition, callers, callees, tests, related_code, type_hints, errors")
    type_inference_method: str = Field("llm", pattern="^(llm|ast)$")


class AnalyzeChangeRequest(BaseModel):
    buffer_id: str
    file: str
    start_line: Optional[int] = Field(None, ge=1)
    end_line: Optional[int] = Field(None, ge=1)
    max_depth: int = Field(6, ge=1, le=20)


class GetTestCoverageRequest(BaseModel):
    buffer_id: str


class InferTypesRequest(BaseModel):
    buffer_id: str
    symbol: str
    method: str = Field("llm", pattern="^(llm|ast)$")


class GetSymbolMetadataRequest(BaseModel):
    buffer_id: str
    symbol: str
    include_types: bool = True
    type_inference_method: str = Field("ast", pattern="^(llm|ast)$")


class SearchBatchRequest(BaseModel):
    buffer_id: str
    queries: List[str] = Field(..., min_length=1, max_length=20)
    top_k: int = Field(5, ge=1, le=50)
    include_types: bool = False
    type_inference_method: str = Field("llm", pattern="^(llm|ast)$")


class AutoFormatRequest(BaseModel):
    buffer_id: str
    files: Optional[List[str]] = None
    formatter: str = Field("black", pattern="^(black|ruff\\.format)$")
    line_length: int = Field(88, ge=1, le=200)
    skip_magic_trailing_comma: bool = False
    dry_run: bool = True
    exclude_patterns: Optional[List[str]] = None


class AutoLintRequest(BaseModel):
    buffer_id: str
    files: Optional[List[str]] = None
    select: Optional[List[str]] = None
    ignore: Optional[List[str]] = None
    auto_fix: bool = False
    dry_run: bool = True
    exclude_patterns: Optional[List[str]] = None


class AutoPolishRequest(BaseModel):
    buffer_id: str
    files: Optional[List[str]] = None
    format_with: str = Field("black", pattern="^(black|ruff\\.format)$")
    auto_fix_lints: bool = True
    line_length: int = Field(88, ge=1, le=200)
    ruff_select: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    dry_run: bool = True


class PolishBeforeCommitRequest(BaseModel):
    buffer_id: str
    files_to_commit: Optional[List[str]] = None
    format_with: str = Field("black", pattern="^(black|ruff\\.format)$")
    lint_with: str = "ruff"
    check_only: bool = False


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class CallerInfo(BaseModel):
    file: str
    line: int
    symbol: str
    context: str
    confidence: str  # "high" | "medium"
    via: Optional[str] = None
    depth: Optional[int] = None


class CalleeInfo(BaseModel):
    file: str
    line: int
    symbol: str
    context: str
    confidence: str
    via: Optional[str] = None
    depth: Optional[int] = None


class GetReferencesResponse(BaseModel):
    status: str
    symbol: str
    file: str
    line: int
    callers: List[CallerInfo] = []
    callees: List[CalleeInfo] = []
    direction: str = "both"
    depth: int = 1
    cached: bool = False


class DefinitionInfo(BaseModel):
    name: str
    file: str
    start_line: int
    end_line: int
    type: str
    source: Optional[str] = None


class TypeInfo(BaseModel):
    parameters: List[dict[str, str]] = []
    return_type: Optional[str] = None
    type_confidence: Optional[float] = None
    inference_method: Optional[str] = None


class TestInfo(BaseModel):
    file: str
    start_line: int
    end_line: int
    name: str
    type: str
    target_symbol: Optional[str] = None


class ErrorInfo(BaseModel):
    file: str
    line: int
    type: str
    context: str


class GetFullContextResponse(BaseModel):
    status: str
    symbol: str
    definition: Optional[DefinitionInfo] = None
    callers: List[CallerInfo] = []
    callees: List[CalleeInfo] = []
    types: Optional[TypeInfo] = None
    tests: List[TestInfo] = []
    related_code: List[dict[str, Any]] = []
    errors: List[ErrorInfo] = []


class AnalyzeChangeResponse(BaseModel):
    status: str
    file: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    affected_symbols: List[str] = []
    direct_callers: List[dict[str, Any]] = []
    dependent_symbols: int = 0
    files_affected: int = 0
    impacted_files: List[str] = []
    test_coverage: List[dict[str, Any]] = []
    has_tests: bool = False
    risk_level: str = "low"
    risk_score: float = 0.0


class CoverageEntry(BaseModel):
    line_range: str
    test_names: List[str]


class GetTestCoverageResponse(BaseModel):
    status: str
    coverage: dict[str, List[CoverageEntry]] = {}


class InferTypesResponse(BaseModel):
    status: str
    symbol: str
    parameters: List[dict[str, str]] = []
    return_type: Optional[str] = None
    is_async: bool = False
    signature: Optional[str] = None
    type_confidence: Optional[float] = None
    method: str = "llm"
    cached: bool = False


class SymbolMetadataResponse(BaseModel):
    status: str
    name: str
    file: str
    line: int
    end_line: int
    type: str
    lines_of_code: int = 0
    cyclomatic_complexity: int = 1
    called_by_count: int = 0
    calls_count: int = 0
    docstring: Optional[str] = None
    parameters: List[dict[str, str]] = []
    return_type: Optional[str] = None
    type_confidence: Optional[float] = None
    inference_method: Optional[str] = None
    parent: Optional[str] = None


class SearchBatchResponse(BaseModel):
    status: str
    results: dict[str, List[dict[str, Any]]] = {}
    query_count: int = 0


class FormatResult(BaseModel):
    status: str
    formatter: str = "black"
    formatted_files: int = 0
    already_formatted: int = 0
    changes: List[dict[str, Any]] = []
    summary: str = ""


class LintIssue(BaseModel):
    file: str
    line: int
    code: str
    message: str
    fixed: bool = False


class LintResult(BaseModel):
    status: str
    linter: str = "ruff"
    files_with_issues: int = 0
    total_issues: int = 0
    issues: List[LintIssue] = []
    fixed_count: int = 0
    unfixed_count: int = 0
    by_rule: dict[str, dict[str, Any]] = {}
    auto_fixed_code_available: bool = False


class AutoPolishResponse(BaseModel):
    status: str
    formatting: FormatResult = Field(default_factory=lambda: FormatResult(status="ok"))
    linting: LintResult = Field(default_factory=lambda: LintResult(status="ok"))
    ready_to_commit: bool = False
    summary: str = ""


class PolishBeforeCommitResponse(BaseModel):
    status: str
    formatting: FormatResult = Field(default_factory=lambda: FormatResult(status="ok"))
    linting: LintResult = Field(default_factory=lambda: LintResult(status="ok"))
    ready_to_commit: bool = False
    pre_commit_warnings: List[str] = []
    summary: str = ""
