"""Async FastAPI server for GigaCode agent tools.

Replaces the synchronous stdlib HTTPServer with an ASGI app for concurrency,
middleware, and WebSocket support.

Usage:
    uvicorn src.gigacode_api:app --host 0.0.0.0 --port 8765

Or import and embed your own codebase first:

    from gigacode.gigacode_api import create_app
    from gigacode.gigacode_tool import CodeEmbeddingTool

    tool = CodeEmbeddingTool(work_dir='./buffers', device='cpu', use_gpu=False)
    res = tool.embed_codebase('./my_project', pattern='*.py')
    app = create_app(tool)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List

logger = logging.getLogger(__name__)

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
        yield {"tool": tool}

    app = FastAPI(title="GigaCode API", version="1.1.0", lifespan=lifespan)

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

    return app
