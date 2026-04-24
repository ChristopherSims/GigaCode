"""Async FastAPI server for GigaCode agent tools.

Replaces the synchronous stdlib HTTPServer with an ASGI app for concurrency,
middleware, and WebSocket support.

Usage:
    uvicorn src.gigacode_api:app --host 0.0.0.0 --port 8765

Or import and embed your own codebase first:

    from src.gigacode_api import create_app
    from src.gigacode_tool import CodeEmbeddingTool

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    path: str
    pattern: str = "*.py"
    language_hint: str | None = None

class SearchRequest(BaseModel):
    buffer_id: str
    query: str
    top_k: int = 5
    offset: int = 0

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
    file: str | None = None
    start_line: int = 1
    end_line: int | None = None

class LookForFileRequest(BaseModel):
    buffer_id: str
    file_name: str

class WriteRequest(BaseModel):
    buffer_id: str
    file: str
    start_line: int
    new_lines: list[str]
    end_line: int | None = None

class CommitRequest(BaseModel):
    buffer_id: str
    dry_run: bool = False

class DiscardRequest(BaseModel):
    buffer_id: str
    file: str | None = None

class DeleteBufferRequest(BaseModel):
    buffer_id: str

class CallRequest(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)

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
        result = tool.semantic_search(req.buffer_id, req.query, top_k=req.top_k, offset=req.offset)
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
        result = tool.search_for(req.buffer_id, req.query, case_sensitive=req.case_sensitive, max_results=req.max_results)
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
        result = tool.read_code(req.buffer_id, file=req.file, start_line=req.start_line, end_line=req.end_line)
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
        result = tool.write_code(req.buffer_id, req.file, req.start_line, req.new_lines, end_line=req.end_line)
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
    # Generic tool call (backward compatible with old server)
    # ------------------------------------------------------------------
    @app.post("/call")
    async def call(req: CallRequest) -> dict[str, Any]:
        method = getattr(tool, req.tool, None)
        if method is None:
            raise HTTPException(status_code=404, detail={"status": "error", "message": f"Unknown tool: {req.tool}"})
        try:
            result = method(**req.args)
        except TypeError as exc:
            raise HTTPException(status_code=400, detail={"status": "error", "message": str(exc)})
        if isinstance(result, dict) and result.get("status") != "ok":
            raise HTTPException(status_code=400, detail=result)
        return result

    return app
