"""Async FastAPI server for GigaCode agent tools.

Replaces the synchronous stdlib HTTPServer with an ASGI app for concurrency,
middleware, and WebSocket support.

Usage:
    uvicorn gigacode.gigacode_api:app --host 0.0.0.0 --port 8765

Or import and embed your own codebase first:

    from gigacode.gigacode_api import create_app
    from gigacode.gigacode_tool import CodeEmbeddingTool

    tool = CodeEmbeddingTool(work_dir='./buffers', device='cpu', use_gpu=False)
    res = tool.embed_codebase('./my_project', pattern='*.py')
    app = create_app(tool)

FastAPI dependency injection:
    from gigacode.gigacode_api import get_tool
    from fastapi import Depends

    @app.post("/my-endpoint")
    async def my_endpoint(tool: CodeEmbeddingTool = Depends(get_tool)):
        ...
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List

from gigacode.gigacode_tool import CodeEmbeddingTool
from gigacode.pydantic_models import (
    GetReferencesRequest,
    AnalyzeChangeRequest,
    InferTypesRequest,
    AutoFormatRequest,
    AutoLintRequest,
    AutoPolishRequest,
    PolishBeforeCommitRequest,
    GetFullContextRequest as FullContextRequest,
    GetSymbolMetadataRequest as SymbolMetadataRequest,
    SearchBatchRequest as BatchSearchRequest,
    GetTestCoverageRequest,
    # Response models
    GetReferencesResponse,
    GetFullContextResponse,
    AnalyzeChangeResponse,
    GetTestCoverageResponse,
    InferTypesResponse,
    SymbolMetadataResponse,
    SearchBatchResponse,
    AutoPolishResponse,
    PolishBeforeCommitResponse,
)
from gigacode.server_dispatch import resolve_tool_method

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


def get_tool(request: Request) -> CodeEmbeddingTool:
    """FastAPI dependency that provides the CodeEmbeddingTool from app state.

    Usage:
        @app.post("/my-endpoint")
        async def my_endpoint(tool: CodeEmbeddingTool = Depends(get_tool)):
            return tool.semantic_search(...)
    """
    return request.app.state.tool


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    path: str = Field(description="Path to codebase directory.")
    pattern: str = Field(default="*.py", description="Glob pattern for file inclusion.")
    language_hint: Optional[str] = Field(
        default=None, description="Optional language hint for better parsing."
    )


class SearchRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    query: str = Field(description="Natural language search query.")
    top_k: int = Field(default=5, description="Maximum number of results to return.")
    offset: int = Field(default=0, ge=0, description="Offset for paginating results.")
    include_types: bool = Field(default=False, description="Include type inference in results.")
    type_inference_method: str = Field(
        default="llm", description="Type inference method: llm (accurate) or ast (fast)."
    )


class HybridSearchRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    query: str = Field(description="Natural language search query.")
    top_k: int = Field(default=5, description="Maximum number of results to return.")
    offset: int = Field(default=0, ge=0, description="Offset for paginating results.")
    semantic_weight: float = Field(default=1.0, description="Weight for semantic similarity.")
    lexical_weight: float = Field(default=1.0, description="Weight for lexical (keyword) matching.")


class LiteralSearchRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    query: str = Field(description="Natural language search query.")
    case_sensitive: bool = Field(default=False, description="Whether search is case sensitive.")
    max_results: int = Field(default=50, description="Maximum number of results.")


class SymbolSearchRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    query: str = Field(description="Natural language search query.")
    top_k: int = Field(default=10, description="Maximum number of results to return.")


class ReadRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    file: Optional[str] = Field(default=None, description="File path within the buffer.")
    start_line: int = Field(default=1, ge=1, description="Start line number (1-indexed).")
    end_line: Optional[int] = Field(
        default=None, ge=1, description="End line number. If None, reads to end of file."
    )


class LookForFileRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    file_name: str = Field(description="File name to search for.")


class WriteRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    file: str = Field(description="File path within the buffer.")
    start_line: int = Field(description="Start line number (1-indexed).")
    new_lines: List[str] = Field(description="New lines to write.")
    end_line: Optional[int] = Field(
        default=None, description="End line number. If None, replaces to end of file."
    )


class CommitRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    dry_run: bool = Field(default=False, description="Preview only without making changes.")


class DiscardRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    file: Optional[str] = Field(default=None, description="File path within the buffer.")


class DeleteBufferRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")


class CallRequest(BaseModel):
    tool: str = Field(description="Tool method name to invoke.")
    args: dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for the tool method."
    )


class TraceExecutionPathsRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    symbol: str = Field(description="Symbol name to analyze.")
    max_depth: int = Field(default=3, description="Maximum call depth to trace.")


class DependencyGraphRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    symbol: Optional[str] = Field(default=None, description="Symbol name to analyze.")
    depth: int = Field(default=2, description="Depth of dependencies to include.")


class CodeSmellsRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    types: Optional[List[str]] = Field(default=None, description="Code smell types to check.")
    severity_min: str = Field(
        default="low", description="Minimum severity to report: low, medium, or high."
    )


class SecurityScanRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    severity_min: str = Field(
        default="medium", description="Minimum severity to report: low, medium, or high."
    )


class SuggestRefactoringsRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    symbol: str = Field(description="Symbol name to analyze.")


class LintBufferRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    files: Optional[List[str]] = Field(
        default=None, description="Specific files to analyze. If None, processes entire buffer."
    )
    select: Optional[List[str]] = Field(
        default=None, description="Lint rule categories to check (e.g. E, F, W)."
    )
    exclude_patterns: Optional[List[str]] = Field(
        default=None, description="Glob patterns to exclude."
    )
    group_by: str = Field(
        default="file", description="How to organize results: file, severity, or rule."
    )


class FormatBufferRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    files: Optional[List[str]] = Field(
        default=None, description="Specific files to analyze. If None, processes entire buffer."
    )
    formatter: str = Field(default="black", description="Formatter to use: black or ruff.format.")
    line_length: int = Field(default=88, description="Maximum line length for formatting.")
    exclude_patterns: Optional[List[str]] = Field(
        default=None, description="Glob patterns to exclude."
    )
    dry_run: bool = Field(default=True, description="Preview only without making changes.")
    summary_only: bool = Field(
        default=False, description="Return only summary statistics, not full diffs."
    )


class FindPerformanceHotspotsRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")


class GenerateDocumentationRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    symbol: str = Field(description="Symbol name to analyze.")
    style: str = Field(default="google", description="Docstring style: google, numpy, or sphinx.")


class FindSimilarPatternsRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    code_snippet: str = Field(description="Code snippet to find similar patterns for.")
    min_similarity: float = Field(
        default=0.7, description="Minimum similarity threshold (0.0-1.0)."
    )
    top_k: int = Field(default=10, description="Maximum number of results to return.")
    cluster: bool = Field(default=False, description="Group similar matches into clusters.")
    clustering_method: str = Field(
        default="agglomerative", description="Clustering method to apply."
    )
    cluster_threshold: float = Field(
        default=0.8, description="Cluster grouping threshold (0.0-1.0)."
    )
    expected_clusters: Optional[int] = Field(
        default=None, description="Preferred number of clusters when the method uses it."
    )


class FindDeprecatedRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")


class ValidateChangesRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    dry_run: bool = Field(default=True, description="Preview only without making changes.")


class ExtractConfigurationRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")


class AnalyzeLoggingPatternsRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")


class AnalyzeErrorHandlingRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")


class GenerateChangelogRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    since_commit: Optional[str] = Field(
        default=None, description="Git commit hash to compare against."
    )


class DetectApiChangesRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    since_commit: Optional[str] = Field(
        default=None, description="Git commit hash to compare against."
    )


class GetRollbackInfoRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    file: str = Field(description="File path within the buffer.")


class GenerateChangeTemplateRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    request: str = Field(description="Natural language description of the desired change.")


class MapApiEndpointsRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")


class AnalyzeCachePatternsRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")


class AnalyzeThreadSafetyRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")


class DetectMemoryIssuesRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")


class LintWithConfigRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    config_file: Optional[str] = Field(
        default=None, description="Path to config file. Auto-detected if None."
    )
    files: Optional[List[str]] = Field(
        default=None, description="Specific files to analyze. If None, processes entire buffer."
    )
    auto_fix: bool = Field(
        default=False, description="Automatically fix issues instead of just reporting."
    )


class FormatWithConfigRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    config_file: Optional[str] = Field(
        default=None, description="Path to config file. Auto-detected if None."
    )
    files: Optional[List[str]] = Field(
        default=None, description="Specific files to analyze. If None, processes entire buffer."
    )
    dry_run: bool = Field(default=True, description="Preview only without making changes.")


class FindDuplicatesRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    threshold: float = Field(
        default=0.85, description="Minimum Jaccard similarity threshold (0.0-1.0)."
    )


class PackContextRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    query: str = Field(description="Natural language search query.")
    max_tokens: int = Field(default=8192, description="Maximum tokens for packed context.")
    top_k: int = Field(default=10, description="Maximum number of chunks to include.")


class BufferIdRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")


class SolveRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    task: str = Field(description="Natural language description of the task to solve.")
    max_iterations: int = Field(default=10, description="Maximum number of solve loop iterations.")


class UndoRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    steps: int = Field(default=1, description="Number of operations to undo.")


class RedoRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    steps: int = Field(default=1, description="Number of operations to redo.")


class CreateBranchRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    name: str = Field(description="Name for the new branch.")


class ListBranchesRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")


class CheckoutBranchRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    name: str = Field(description="Name of the branch to switch to.")


class AnnotateSearchRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    query: str = Field(description="Natural language search query.")
    top_k: int = Field(default=5, description="Maximum number of annotated results to return.")


class PredictConflictsRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")


class SearchModifiedOnlyRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    query: str = Field(description="Natural language search query.")
    scope: str = Field(
        default="changes+deps", description="Search scope: changes, changes+deps, or all."
    )
    top_k: int = Field(default=10, description="Maximum number of results to return.")


class GetChunkingStrategyRequest(BaseModel):
    profile: str = Field(
        default="generic",
        description="Agent profile name: reviewer, debugger, architect, documenter, or generic.",
    )


class SetAgentProfileRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    profile: str = Field(
        default="generic",
        description="Agent profile name: reviewer, debugger, architect, documenter, or generic.",
    )


class ChunkWithProfileRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    profile: str = Field(
        default="generic",
        description="Agent profile name: reviewer, debugger, architect, documenter, or generic.",
    )


class AdaptSearchRequest(BaseModel):
    buffer_id: str = Field(description="Buffer handle returned by embed_codebase.")
    query: str = Field(description="Original search query to enhance.")
    profile: str = Field(
        default="generic",
        description="Agent profile name: reviewer, debugger, architect, documenter, or generic.",
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(tool: Any) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.tool = tool
        yield

    app = FastAPI(title="GigaCode API", version="2.0.0", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error in %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Internal server error"},
        )

    # ------------------------------------------------------------------
    # Health & schemas
    # ------------------------------------------------------------------
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/schemas")
    def schemas() -> dict[str, Any]:
        return {"schemas": tool.get_tool_schemas()}

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------
    @app.post("/buffers")
    def embed_codebase(req: EmbedRequest) -> dict[str, Any]:
        """Embed a codebase directory into a searchable buffer."""
        result = tool.embed_codebase(req.path, language_hint=req.language_hint, pattern=req.pattern)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.get("/buffers")
    def list_buffers() -> dict[str, Any]:
        return tool.list_buffers()

    @app.delete("/buffers/{buffer_id}")
    def delete_buffer(buffer_id: str) -> dict[str, Any]:
        result = tool.delete_buffer(buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.post("/buffers/{buffer_id}/reload")
    def reload_codebase(buffer_id: str) -> dict[str, Any]:
        """Reload a codebase buffer from its source directory."""
        result = tool.reload_codebase(buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    @app.post("/search/semantic")
    def semantic_search(req: SearchRequest) -> dict[str, Any]:
        """Search codebase using natural language query with semantic embeddings."""
        result = tool.semantic_search(
            req.buffer_id,
            req.query,
            top_k=req.top_k,
            offset=req.offset,
            include_types=req.include_types,
            type_inference_method=req.type_inference_method,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/search/hybrid")
    def hybrid_search(req: HybridSearchRequest) -> dict[str, Any]:
        """Search codebase using combined semantic and lexical matching."""
        result = tool.hybrid_search(
            req.buffer_id,
            req.query,
            top_k=req.top_k,
            offset=req.offset,
            semantic_weight=req.semantic_weight,
            lexical_weight=req.lexical_weight,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/search/literal")
    def literal_search(req: LiteralSearchRequest) -> dict[str, Any]:
        """Search codebase for literal string matches."""
        result = tool.search_for(
            req.buffer_id, req.query, case_sensitive=req.case_sensitive, max_results=req.max_results
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/search/symbols")
    def symbol_search(req: SymbolSearchRequest) -> dict[str, Any]:
        """Search for symbol definitions by name."""
        result = tool.search_symbols(req.buffer_id, req.query, top_k=req.top_k)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/duplicates")
    def find_duplicates(req: FindDuplicatesRequest) -> dict[str, Any]:
        """Find duplicate code blocks across the buffer."""
        result = tool.find_duplicates(req.buffer_id, threshold=req.threshold)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/pack")
    def pack_context(req: PackContextRequest) -> dict[str, Any]:
        """Pack relevant context for a query into a compact summary."""
        result = tool.pack_context(
            req.buffer_id, req.query, max_tokens=req.max_tokens, top_k=req.top_k
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    # Read / Write / Commit
    # ------------------------------------------------------------------
    @app.post("/read")
    def read_code(req: ReadRequest) -> dict[str, Any]:
        """Read code from a file in the buffer."""
        result = tool.read_code(
            req.buffer_id, file=req.file, start_line=req.start_line, end_line=req.end_line
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/look-for-file")
    def look_for_file(req: LookForFileRequest) -> dict[str, Any]:
        """Look up a file by name within the buffer."""
        result = tool.look_for_file(req.buffer_id, req.file_name)
        if result.get("status") != "ok":
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.post("/write")
    def write_code(req: WriteRequest) -> dict[str, Any]:
        """Write or replace lines in a file within the buffer."""
        result = tool.write_code(
            req.buffer_id, req.file, req.start_line, req.new_lines, end_line=req.end_line
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/diff")
    def diff(req: BufferIdRequest) -> dict[str, Any]:
        """Show uncommitted changes in the buffer."""
        result = tool.diff(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/commit")
    def commit(req: CommitRequest) -> dict[str, Any]:
        """Commit current changes in the buffer."""
        result = tool.commit(req.buffer_id, dry_run=req.dry_run)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/discard")
    def discard(req: DiscardRequest) -> dict[str, Any]:
        """Discard uncommitted changes in the buffer."""
        result = tool.discard(req.buffer_id, file=req.file)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    # Additional Endpoints
    # ------------------------------------------------------------------
    @app.post("/search/batch", response_model=SearchBatchResponse)
    def batch_search(req: BatchSearchRequest) -> dict[str, Any]:
        """Run multiple semantic search queries in a single call."""
        result = tool.search_batch(
            req.buffer_id,
            req.queries,
            top_k=req.top_k,
            include_types=req.include_types,
            type_inference_method=req.type_inference_method,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/types/infer", response_model=InferTypesResponse)
    def infer_types(req: InferTypesRequest) -> dict[str, Any]:
        """Infer types for code at a specific location using AST or LLM."""
        result = tool.infer_types(req.buffer_id, req.symbol, method=req.method)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/symbols/metadata", response_model=SymbolMetadataResponse)
    def symbol_metadata(req: SymbolMetadataRequest) -> dict[str, Any]:
        """Get detailed metadata for a symbol: complexity, callers, types, docstring."""
        result = tool.get_symbol_metadata(
            req.buffer_id,
            req.symbol,
            include_types=req.include_types,
            type_inference_method=req.type_inference_method,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/references", response_model=GetReferencesResponse)
    def get_references(req: GetReferencesRequest) -> dict[str, Any]:
        """Get callers, callees, and related references for a symbol."""
        result = tool.get_references(
            req.buffer_id,
            req.symbol,
            direction=req.direction,
            top_k=req.top_k,
            expand_depth=req.expand_depth,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/context/full", response_model=GetFullContextResponse)
    def get_full_context(req: FullContextRequest) -> dict[str, Any]:
        """Get full context for a symbol: definition, callers, tests, types, errors."""
        result = tool.get_full_context(
            req.buffer_id,
            req.symbol,
            include=req.include,
            type_inference_method=req.type_inference_method,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/impact/analyze-change", response_model=AnalyzeChangeResponse)
    def analyze_change(req: AnalyzeChangeRequest) -> dict[str, Any]:
        """Analyze the impact of a code change before editing."""
        result = tool.analyze_change(
            req.buffer_id,
            req.file,
            start_line=req.start_line,
            end_line=req.end_line,
            max_depth=req.max_depth,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/tests/coverage", response_model=GetTestCoverageResponse)
    def get_test_coverage(req: GetTestCoverageRequest) -> dict[str, Any]:
        """Map source files to test coverage with line ranges and test names."""
        result = tool.get_test_coverage(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/polish-before-commit", response_model=PolishBeforeCommitResponse)
    def polish_before_commit(req: PolishBeforeCommitRequest) -> dict[str, Any]:
        """Run format + lint + impact checks before committing."""
        result = tool.polish_before_commit(
            req.buffer_id,
            files_to_commit=req.files_to_commit,
            format_with=req.format_with,
            lint_with=req.lint_with,
            check_only=req.check_only,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    # Advanced Analysis
    # ------------------------------------------------------------------
    @app.post("/execution-paths")
    def trace_execution_paths(req: TraceExecutionPathsRequest) -> dict[str, Any]:
        """Trace all execution paths through a symbol with AST branch detection."""
        result = tool.trace_execution_paths(
            req.buffer_id,
            req.symbol,
            max_depth=req.max_depth,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/dependency-graph")
    def get_dependency_graph(req: DependencyGraphRequest) -> dict[str, Any]:
        """Get dependency graph visualization data (nodes + edges)."""
        result = tool.get_dependency_graph(
            req.buffer_id,
            symbol=req.symbol,
            depth=req.depth,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/analysis/code-smells")
    def detect_code_smells(req: CodeSmellsRequest) -> dict[str, Any]:
        """Detect code smells: long functions, deep nesting, missing docstrings, complex logic."""
        result = tool.detect_code_smells(
            req.buffer_id,
            types=req.types,
            severity_min=req.severity_min,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/security/scan")
    def scan_security(req: SecurityScanRequest) -> dict[str, Any]:
        """Scan for security vulnerabilities: eval, exec, shell injection, SQL injection, secrets."""
        result = tool.scan_security(
            req.buffer_id,
            severity_min=req.severity_min,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/refactoring/suggest")
    def suggest_refactorings(req: SuggestRefactoringsRequest) -> dict[str, Any]:
        """Suggest safe refactorings for a symbol with risk assessment."""
        result = tool.suggest_refactorings(req.buffer_id, req.symbol)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/lint-buffer")
    def lint_buffer(req: LintBufferRequest) -> dict[str, Any]:
        """Deep lint analysis of entire buffer with detailed aggregation."""
        result = tool.lint_buffer(
            req.buffer_id,
            files=req.files,
            select=req.select,
            exclude_patterns=req.exclude_patterns,
            group_by=req.group_by,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/format-buffer")
    def format_buffer(req: FormatBufferRequest) -> dict[str, Any]:
        """Deep format analysis with detailed change tracking across codebase."""
        result = tool.format_buffer(
            req.buffer_id,
            files=req.files,
            formatter=req.formatter,
            line_length=req.line_length,
            exclude_patterns=req.exclude_patterns,
            dry_run=req.dry_run,
            summary_only=req.summary_only,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/format")
    def auto_format_endpoint(req: AutoFormatRequest) -> dict[str, Any]:
        """Auto-format code using Black or ruff.format."""
        result = tool.auto_format(
            req.buffer_id,
            files=req.files,
            formatter=req.formatter,
            line_length=req.line_length,
            skip_magic_trailing_comma=req.skip_magic_trailing_comma,
            dry_run=req.dry_run,
            exclude_patterns=req.exclude_patterns,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/lint")
    def auto_lint_endpoint(req: AutoLintRequest) -> dict[str, Any]:
        """Auto-lint code using Ruff with optional auto-fix."""
        result = tool.auto_lint(
            req.buffer_id,
            files=req.files,
            select=req.select,
            ignore=req.ignore,
            auto_fix=req.auto_fix,
            dry_run=req.dry_run,
            exclude_patterns=req.exclude_patterns,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/polish", response_model=AutoPolishResponse)
    def auto_polish_endpoint(req: AutoPolishRequest) -> dict[str, Any]:
        """Combined format + lint in a single call."""
        result = tool.auto_polish(
            req.buffer_id,
            files=req.files,
            format_with=req.format_with,
            auto_fix_lints=req.auto_fix_lints,
            line_length=req.line_length,
            ruff_select=req.ruff_select,
            exclude_patterns=req.exclude_patterns,
            dry_run=req.dry_run,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    # Advanced Analysis & Configuration
    # ------------------------------------------------------------------
    @app.post("/performance/hotspots")
    def find_performance_hotspots(req: FindPerformanceHotspotsRequest) -> dict[str, Any]:
        """Detect performance hotspots: N+1 queries, inefficient loops, resource leaks."""
        result = tool.find_performance_hotspots(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/docs/generate")
    def generate_documentation(req: GenerateDocumentationRequest) -> dict[str, Any]:
        """Auto-generate docstring from code analysis (Google/NumPy/Sphinx style)."""
        result = tool.generate_documentation(req.buffer_id, req.symbol, style=req.style)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/search/similar-patterns")
    def find_similar_patterns(req: FindSimilarPatternsRequest) -> dict[str, Any]:
        """Find similar code patterns using semantic + syntactic matching."""
        result = tool.find_similar_patterns(
            req.buffer_id,
            req.code_snippet,
            min_similarity=req.min_similarity,
            top_k=req.top_k,
            cluster=req.cluster,
            clustering_method=req.clustering_method,
            cluster_threshold=req.cluster_threshold,
            expected_clusters=req.expected_clusters,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/analysis/deprecated")
    def find_deprecated(req: FindDeprecatedRequest) -> dict[str, Any]:
        """Detect usage of deprecated functions and APIs."""
        result = tool.find_deprecated(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/validation/changes")
    def validate_changes(req: ValidateChangesRequest) -> dict[str, Any]:
        """Validate changes before committing (syntax + import resolution)."""
        result = tool.validate_changes(req.buffer_id, dry_run=req.dry_run)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/config/extract")
    def extract_configuration(req: ExtractConfigurationRequest) -> dict[str, Any]:
        """Extract configuration: env vars, config files, hardcoded secrets."""
        result = tool.extract_configuration(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/analysis/logging")
    def analyze_logging_patterns(req: AnalyzeLoggingPatternsRequest) -> dict[str, Any]:
        """Analyze logging patterns: levels, consistency, gaps."""
        result = tool.analyze_logging_patterns(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/analysis/error-handling")
    def analyze_error_handling_patterns(req: AnalyzeErrorHandlingRequest) -> dict[str, Any]:
        """Analyze error handling: broad catches, missing finally, silent failures."""
        result = tool.analyze_error_handling_patterns(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/changelog/generate")
    def generate_changelog(req: GenerateChangelogRequest) -> dict[str, Any]:
        """Generate changelog from git history with semantic categorization."""
        result = tool.generate_changelog(req.buffer_id, since_commit=req.since_commit)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/api/detect-changes")
    def detect_api_changes(req: DetectApiChangesRequest) -> dict[str, Any]:
        """Detect API-breaking changes between commits."""
        result = tool.detect_api_changes(req.buffer_id, since_commit=req.since_commit)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/rollback/info")
    def get_rollback_info(req: GetRollbackInfoRequest) -> dict[str, Any]:
        """Get rollback information for a file from git history."""
        result = tool.get_rollback_info(req.buffer_id, req.file)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/planning/change-template")
    def generate_change_template(req: GenerateChangeTemplateRequest) -> dict[str, Any]:
        """Generate a change plan template for a natural language request."""
        result = tool.generate_change_template(req.buffer_id, req.request)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/api/endpoints")
    def map_api_endpoints(req: MapApiEndpointsRequest) -> dict[str, Any]:
        """Map all API endpoints from FastAPI and Flask decorators."""
        result = tool.map_api_endpoints(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/analysis/cache-patterns")
    def analyze_cache_patterns(req: AnalyzeCachePatternsRequest) -> dict[str, Any]:
        """Analyze cache usage: invalidation logic, stale data risks."""
        result = tool.analyze_cache_patterns(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/analysis/thread-safety")
    def analyze_thread_safety(req: AnalyzeThreadSafetyRequest) -> dict[str, Any]:
        """Analyze thread safety: shared state, race conditions, deadlocks."""
        result = tool.analyze_thread_safety(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/analysis/memory-issues")
    def detect_memory_issues(req: DetectMemoryIssuesRequest) -> dict[str, Any]:
        """Detect memory issues: circular refs, unbounded collections, resource leaks."""
        result = tool.detect_memory_issues(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/lint-with-config")
    def lint_with_config(req: LintWithConfigRequest) -> dict[str, Any]:
        """Lint using project configuration (ruff.toml, pyproject.toml)."""
        result = tool.lint_with_config(
            req.buffer_id, config_file=req.config_file, files=req.files, auto_fix=req.auto_fix
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/format-with-config")
    def format_with_config(req: FormatWithConfigRequest) -> dict[str, Any]:
        """Format using project configuration (pyproject.toml, .black, ruff.toml)."""
        result = tool.format_with_config(
            req.buffer_id, config_file=req.config_file, files=req.files, dry_run=req.dry_run
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    # Solve, Undo/Redo, Branching, Annotate, Conflict Prediction, Diff-Aware Search
    # ------------------------------------------------------------------
    @app.post("/solve")
    def solve(req: SolveRequest) -> dict[str, Any]:
        """Automatically solve a coding task with a unified loop."""
        result = tool.solve(req.buffer_id, req.task, max_iterations=req.max_iterations)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/undo")
    def undo(req: UndoRequest) -> dict[str, Any]:
        """Undo the last N operations on a buffer."""
        result = tool.undo(req.buffer_id, steps=req.steps)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/redo")
    def redo(req: RedoRequest) -> dict[str, Any]:
        """Redo the last N undone operations on a buffer."""
        result = tool.redo(req.buffer_id, steps=req.steps)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/branch/create")
    def create_branch(req: CreateBranchRequest) -> dict[str, Any]:
        """Create a named branch for experimental edits."""
        result = tool.create_branch(req.buffer_id, req.name)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/branch/list")
    def list_branches(req: ListBranchesRequest) -> dict[str, Any]:
        """List all branches for a buffer."""
        result = tool.list_branches(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/branch/checkout")
    def checkout_branch(req: CheckoutBranchRequest) -> dict[str, Any]:
        """Switch to a different branch on a buffer."""
        result = tool.checkout_branch(req.buffer_id, req.name)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/search/annotate")
    def annotate_search_results(req: AnnotateSearchRequest) -> dict[str, Any]:
        """Search and annotate results with 'why this matters' explanations."""
        result = tool.annotate_search_results(req.buffer_id, req.query, top_k=req.top_k)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/conflict/predict")
    def predict_conflicts(req: PredictConflictsRequest) -> dict[str, Any]:
        """Predict potential merge conflicts by analyzing commits since embed time."""
        result = tool.predict_conflicts(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/search/modified")
    def search_modified_only(req: SearchModifiedOnlyRequest) -> dict[str, Any]:
        """Search only modified files and their dependencies."""
        result = tool.search_modified_only(
            req.buffer_id, req.query, scope=req.scope, top_k=req.top_k
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    # Agent Profile
    # ------------------------------------------------------------------
    @app.post("/profile/strategy")
    def get_chunking_strategy(req: GetChunkingStrategyRequest) -> dict[str, Any]:
        """Get chunking strategy for a given agent profile."""
        result = tool.get_chunking_strategy(profile=req.profile)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/profile/set")
    def set_agent_profile(req: SetAgentProfileRequest) -> dict[str, Any]:
        """Set agent profile for a buffer, affecting future chunking and search."""
        result = tool.set_agent_profile(req.buffer_id, profile=req.profile)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/profile/chunk")
    def chunk_with_profile(req: ChunkWithProfileRequest) -> dict[str, Any]:
        """Re-chunk the buffer's codebase using the specified agent profile."""
        result = tool.chunk_with_profile(req.buffer_id, profile=req.profile)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/profile/adapt-search")
    def adapt_search(req: AdaptSearchRequest) -> dict[str, Any]:
        """Enhance a search query based on agent profile context."""
        result = tool.adapt_search(req.buffer_id, req.query, profile=req.profile)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    @app.post("/call")
    def call(req: CallRequest) -> dict[str, Any]:
        """Generic tool call endpoint (backward compatible)."""
        method = resolve_tool_method(tool, req.tool)
        if method is None:
            raise HTTPException(
                status_code=404, detail={"status": "error", "message": f"Unknown tool: {req.tool}"}
            )
        try:
            result = method(**req.args)
        except TypeError as exc:
            logger.warning("Invalid arguments for %s: %s", req.tool, exc)
            raise HTTPException(
                status_code=400,
                detail={"status": "error", "message": f"Invalid arguments for {req.tool}"},
            ) from exc
        if isinstance(result, dict) and result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    # Schema Export (format-configurable for AI agent integration)
    # ------------------------------------------------------------------
    @app.get("/schemas")
    def export_schemas(
        format: str = "openai",
        include_metadata: bool = True,
        category: Optional[str] = None,
        read_only_only: bool = False,
    ) -> Any:
        """Export all tool schemas in the requested format.

        Supported formats: openai, anthropic, mcp, ollama.
        Query params: format, include_metadata, category, read_only_only.
        """
        from gigacode.tool_schema import export_schemas as _export, SchemaFormat

        try:
            fmt = SchemaFormat(format)
        except ValueError:
            valid = ", ".join(f.value for f in SchemaFormat)
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": f"Invalid format '{format}'. Supported: {valid}",
                },
            )
        return _export(
            format=fmt,
            include_metadata=include_metadata,
            category=category,
            read_only_only=read_only_only,
        )

    @app.get("/schemas/config")
    def get_schema_config() -> dict[str, Any]:
        """Read the current schema export config (from gigacode.toml or pyproject.toml)."""
        from gigacode.tool_schema import SchemaConfig

        config = SchemaConfig()
        return {"status": "ok", "config": config.to_dict()}

    @app.get("/schemas/categories")
    def get_schema_categories() -> dict[str, Any]:
        """List all available tool categories."""
        from gigacode.tool_schema import TOOL_CATEGORIES

        return {"status": "ok", "categories": TOOL_CATEGORIES}

    # ------------------------------------------------------------------
    # Middleware: monitoring + error handling
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def monitoring_middleware(request: Request, call_next):
        import time

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({elapsed_ms:.1f}ms)"
        )
        return response

    return app


# ---------------------------------------------------------------------------
# Starter app template with auth + rate limiting
# ---------------------------------------------------------------------------


def create_production_app(
    tool: CodeEmbeddingTool,
    api_key: str | None = None,
    rate_limit_calls: int = 100,
    rate_limit_period: int = 60,
) -> FastAPI:
    """Create a production-ready FastAPI app with API key auth and rate limiting.

    Usage:
        from gigacode.gigacode_tool import CodeEmbeddingTool
        from gigacode.gigacode_api import create_production_app

        tool = CodeEmbeddingTool(work_dir='./buffers', device='cpu')
        app = create_production_app(tool, api_key="my-secret-key")

    Args:
        tool: CodeEmbeddingTool instance.
        api_key: Optional API key for authentication. If None, auth is disabled.
        rate_limit_calls: Max calls per period (default: 100).
        rate_limit_period: Rate limit period in seconds (default: 60).
    """
    from fastapi.security import APIKeyHeader
    from fastapi import Security

    app = create_app(tool)

    # API Key authentication
    if api_key:
        api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # Skip auth for health endpoint
            if request.url.path == "/health":
                return await call_next(request)
            # Skip for OpenAPI docs
            if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi"):
                return await call_next(request)

            key = request.headers.get("X-API-Key")
            if key != api_key:
                return JSONResponse(
                    status_code=401,
                    content={"status": "error", "message": "Invalid or missing API key"},
                )
            return await call_next(request)

    # Simple in-memory rate limiter
    _rate_tracker: dict[str, list[float]] = {}
    import time as _time

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        # Skip rate limiting for health/docs
        if request.url.path in ("/health", "/docs", "/openapi.json"):
            return await call_next(request)

        client_id = request.client.host if request.client else "unknown"
        now = _time.perf_counter()

        if client_id not in _rate_tracker:
            _rate_tracker[client_id] = []

        # Remove expired entries
        _rate_tracker[client_id] = [
            t for t in _rate_tracker[client_id] if now - t < rate_limit_period
        ]

        if len(_rate_tracker[client_id]) >= rate_limit_calls:
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "message": f"Rate limit exceeded: {rate_limit_calls} calls per {rate_limit_period}s",
                },
            )

        _rate_tracker[client_id].append(now)
        return await call_next(request)

    return app
