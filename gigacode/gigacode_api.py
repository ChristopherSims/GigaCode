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
    path: str
    pattern: str = "*.py"
    language_hint: Optional[str] = None


class SearchRequest(BaseModel):
    buffer_id: str
    query: str
    top_k: int = 5
    offset: int = 0
    include_types: bool = False
    type_inference_method: str = "llm"


class HybridSearchRequest(BaseModel):
    buffer_id: str
    query: str
    top_k: int = 5
    offset: int = 0
    semantic_weight: float = 1.0
    lexical_weight: float = 1.0


class LiteralSearchRequest(BaseModel):
    buffer_id: str
    query: str
    case_sensitive: bool = False
    max_results: int = 50


class SymbolSearchRequest(BaseModel):
    buffer_id: str
    query: str
    top_k: int = 10


class ReadRequest(BaseModel):
    buffer_id: str
    file: Optional[str] = None
    start_line: int = 1
    end_line: Optional[int] = None


class LookForFileRequest(BaseModel):
    buffer_id: str
    file_name: str


class WriteRequest(BaseModel):
    buffer_id: str
    file: str
    start_line: int
    new_lines: List[str]
    end_line: Optional[int] = None


class CommitRequest(BaseModel):
    buffer_id: str
    dry_run: bool = False


class DiscardRequest(BaseModel):
    buffer_id: str
    file: Optional[str] = None


class DeleteBufferRequest(BaseModel):
    buffer_id: str


class CallRequest(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class BatchSearchRequest(BaseModel):
    buffer_id: str
    queries: List[str]
    top_k: int = 5
    include_types: bool = False
    type_inference_method: str = "llm"


class InferTypesRequest(BaseModel):
    buffer_id: str
    symbol: str
    method: str = "llm"


class SymbolMetadataRequest(BaseModel):
    buffer_id: str
    symbol: str
    include_types: bool = True
    type_inference_method: str = "ast"


class GetReferencesRequest(BaseModel):
    buffer_id: str
    symbol: str
    direction: str = "both"
    top_k: int = 50
    expand_depth: Optional[int] = None


class FullContextRequest(BaseModel):
    buffer_id: str
    symbol: str
    include: Optional[List[str]] = None
    type_inference_method: str = "llm"


class AnalyzeChangeRequest(BaseModel):
    buffer_id: str
    file: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    max_depth: int = 6


class PolishBeforeCommitRequest(BaseModel):
    buffer_id: str
    files_to_commit: Optional[List[str]] = None
    format_with: str = "black"
    lint_with: str = "ruff"
    check_only: bool = False


class TraceExecutionPathsRequest(BaseModel):
    buffer_id: str
    symbol: str
    max_depth: int = 3


class DependencyGraphRequest(BaseModel):
    buffer_id: str
    symbol: Optional[str] = None
    depth: int = 2


class CodeSmellsRequest(BaseModel):
    buffer_id: str
    types: Optional[List[str]] = None
    severity_min: str = "low"


class SecurityScanRequest(BaseModel):
    buffer_id: str
    severity_min: str = "medium"


class SuggestRefactoringsRequest(BaseModel):
    buffer_id: str
    symbol: str


class LintBufferRequest(BaseModel):
    buffer_id: str
    files: Optional[List[str]] = None
    select: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    group_by: str = "file"


class FormatBufferRequest(BaseModel):
    buffer_id: str
    files: Optional[List[str]] = None
    formatter: str = "black"
    line_length: int = 88
    exclude_patterns: Optional[List[str]] = None
    dry_run: bool = True
    summary_only: bool = False


class AutoFormatRequest(BaseModel):
    buffer_id: str
    files: Optional[List[str]] = None
    formatter: str = "black"
    line_length: int = 88
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
    format_with: str = "black"
    auto_fix_lints: bool = True
    line_length: int = 88
    ruff_select: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    dry_run: bool = True


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
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})

    # ------------------------------------------------------------------
    # Health & schemas
    # ------------------------------------------------------------------
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/schemas")
    async def schemas() -> dict[str, Any]:
        return {"schemas": tool.get_tool_schemas()}

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------
    @app.post("/buffers")
    async def embed_codebase(req: EmbedRequest) -> dict[str, Any]:
        result = tool.embed_codebase(req.path, language_hint=req.language_hint, pattern=req.pattern)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.get("/buffers")
    async def list_buffers() -> dict[str, Any]:
        return tool.list_buffers()

    @app.delete("/buffers/{buffer_id}")
    async def delete_buffer(buffer_id: str) -> dict[str, Any]:
        result = tool.delete_buffer(buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.post("/buffers/{buffer_id}/reload")
    async def reload_codebase(buffer_id: str) -> dict[str, Any]:
        result = tool.reload_codebase(buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    @app.post("/search/semantic")
    async def semantic_search(req: SearchRequest) -> dict[str, Any]:
        result = tool.semantic_search(
            req.buffer_id, req.query, top_k=req.top_k, offset=req.offset,
            include_types=req.include_types, type_inference_method=req.type_inference_method,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/search/hybrid")
    async def hybrid_search(req: HybridSearchRequest) -> dict[str, Any]:
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
    async def literal_search(req: LiteralSearchRequest) -> dict[str, Any]:
        result = tool.search_for(
            req.buffer_id, req.query, case_sensitive=req.case_sensitive, max_results=req.max_results
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/search/symbols")
    async def symbol_search(req: SymbolSearchRequest) -> dict[str, Any]:
        result = tool.search_symbols(req.buffer_id, req.query, top_k=req.top_k)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/duplicates")
    async def find_duplicates(req: DeleteBufferRequest) -> dict[str, Any]:
        result = tool.find_duplicates(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/pack")
    async def pack_context(req: SearchRequest) -> dict[str, Any]:
        result = tool.pack_context(req.buffer_id, req.query, max_tokens=8192, top_k=req.top_k)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    # Read / Write / Commit
    # ------------------------------------------------------------------
    @app.post("/read")
    async def read_code(req: ReadRequest) -> dict[str, Any]:
        result = tool.read_code(
            req.buffer_id, file=req.file, start_line=req.start_line, end_line=req.end_line
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/look-for-file")
    async def look_for_file(req: LookForFileRequest) -> dict[str, Any]:
        result = tool.look_for_file(req.buffer_id, req.file_name)
        if result.get("status") != "ok":
            raise HTTPException(status_code=404, detail=result)
        return result

    @app.post("/write")
    async def write_code(req: WriteRequest) -> dict[str, Any]:
        result = tool.write_code(
            req.buffer_id, req.file, req.start_line, req.new_lines, end_line=req.end_line
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/diff")
    async def diff(req: DeleteBufferRequest) -> dict[str, Any]:
        result = tool.diff(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/commit")
    async def commit(req: CommitRequest) -> dict[str, Any]:
        result = tool.commit(req.buffer_id, dry_run=req.dry_run)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/discard")
    async def discard(req: DiscardRequest) -> dict[str, Any]:
        result = tool.discard(req.buffer_id, file=req.file)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    # New Feature Endpoints (Phase 1)
    # ------------------------------------------------------------------
    @app.post("/search/batch")
    async def batch_search(req: BatchSearchRequest) -> dict[str, Any]:
        result = tool.search_batch(
            req.buffer_id, req.queries, top_k=req.top_k,
            include_types=req.include_types, type_inference_method=req.type_inference_method,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/types/infer")
    async def infer_types(req: InferTypesRequest) -> dict[str, Any]:
        result = tool.infer_types(req.buffer_id, req.symbol, method=req.method)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/symbols/metadata")
    async def symbol_metadata(req: SymbolMetadataRequest) -> dict[str, Any]:
        result = tool.get_symbol_metadata(
            req.buffer_id, req.symbol, include_types=req.include_types,
            type_inference_method=req.type_inference_method,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/references")
    async def get_references(req: GetReferencesRequest) -> dict[str, Any]:
        result = tool.get_references(
            req.buffer_id, req.symbol, direction=req.direction,
            top_k=req.top_k, expand_depth=req.expand_depth,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/context/full")
    async def get_full_context(req: FullContextRequest) -> dict[str, Any]:
        result = tool.get_full_context(
            req.buffer_id, req.symbol, include=req.include,
            type_inference_method=req.type_inference_method,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/impact/analyze-change")
    async def analyze_change(req: AnalyzeChangeRequest) -> dict[str, Any]:
        result = tool.analyze_change(
            req.buffer_id, req.file, start_line=req.start_line,
            end_line=req.end_line, max_depth=req.max_depth,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/tests/coverage")
    async def get_test_coverage(req: DeleteBufferRequest) -> dict[str, Any]:
        result = tool.get_test_coverage(req.buffer_id)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/polish-before-commit")
    async def polish_before_commit(req: PolishBeforeCommitRequest) -> dict[str, Any]:
        result = tool.polish_before_commit(
            req.buffer_id, files_to_commit=req.files_to_commit,
            format_with=req.format_with, lint_with=req.lint_with,
            check_only=req.check_only,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    # Phase 3: Advanced Analysis
    # ------------------------------------------------------------------
    @app.post("/execution-paths")
    async def trace_execution_paths(req: TraceExecutionPathsRequest) -> dict[str, Any]:
        result = tool.trace_execution_paths(
            req.buffer_id, req.symbol, max_depth=req.max_depth,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/dependency-graph")
    async def get_dependency_graph(req: DependencyGraphRequest) -> dict[str, Any]:
        result = tool.get_dependency_graph(
            req.buffer_id, symbol=req.symbol, depth=req.depth,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/analysis/code-smells")
    async def detect_code_smells(req: CodeSmellsRequest) -> dict[str, Any]:
        result = tool.detect_code_smells(
            req.buffer_id, types=req.types, severity_min=req.severity_min,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/security/scan")
    async def scan_security(req: SecurityScanRequest) -> dict[str, Any]:
        result = tool.scan_security(
            req.buffer_id, severity_min=req.severity_min,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/refactoring/suggest")
    async def suggest_refactorings(req: SuggestRefactoringsRequest) -> dict[str, Any]:
        result = tool.suggest_refactorings(req.buffer_id, req.symbol)
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/lint-buffer")
    async def lint_buffer(req: LintBufferRequest) -> dict[str, Any]:
        result = tool.lint_buffer(
            req.buffer_id, files=req.files, select=req.select,
            exclude_patterns=req.exclude_patterns, group_by=req.group_by,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/format-buffer")
    async def format_buffer(req: FormatBufferRequest) -> dict[str, Any]:
        result = tool.format_buffer(
            req.buffer_id, files=req.files, formatter=req.formatter,
            line_length=req.line_length, exclude_patterns=req.exclude_patterns,
            dry_run=req.dry_run, summary_only=req.summary_only,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/format")
    async def auto_format_endpoint(req: AutoFormatRequest) -> dict[str, Any]:
        result = tool.auto_format(
            req.buffer_id, files=req.files, formatter=req.formatter,
            line_length=req.line_length, skip_magic_trailing_comma=req.skip_magic_trailing_comma,
            dry_run=req.dry_run, exclude_patterns=req.exclude_patterns,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/lint")
    async def auto_lint_endpoint(req: AutoLintRequest) -> dict[str, Any]:
        result = tool.auto_lint(
            req.buffer_id, files=req.files, select=req.select, ignore=req.ignore,
            auto_fix=req.auto_fix, dry_run=req.dry_run, exclude_patterns=req.exclude_patterns,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    @app.post("/quality/polish")
    async def auto_polish_endpoint(req: AutoPolishRequest) -> dict[str, Any]:
        result = tool.auto_polish(
            req.buffer_id, files=req.files, format_with=req.format_with,
            auto_fix_lints=req.auto_fix_lints, line_length=req.line_length,
            ruff_select=req.ruff_select, exclude_patterns=req.exclude_patterns,
            dry_run=req.dry_run,
        )
        if result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    # ------------------------------------------------------------------
    # Generic tool call (backward compatible with old server)
    # ------------------------------------------------------------------
    @app.post("/call")
    async def call(req: CallRequest) -> dict[str, Any]:
        method = getattr(tool, req.tool, None)
        if method is None:
            raise HTTPException(
                status_code=404, detail={"status": "error", "message": f"Unknown tool: {req.tool}"}
            )
        try:
            result = method(**req.args)
        except TypeError as exc:
            raise HTTPException(
                status_code=400, detail={"status": "error", "message": str(exc)}
            ) from exc
        if isinstance(result, dict) and result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

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
        logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed_ms:.1f}ms)")
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
            t for t in _rate_tracker[client_id]
            if now - t < rate_limit_period
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
