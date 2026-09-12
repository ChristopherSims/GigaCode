"""Model Context Protocol (MCP) server for GigaCode.

Exposes all GigaCode tools via MCP so Claude Desktop, Cursor, and other
MCP-compatible agents can use them natively.

Usage (stdio transport — for Claude Desktop):
    python -m gigacode.mcp_server --work-dir ./buffers

Usage (HTTP-SSE transport):
    python -m gigacode.mcp_server --work-dir ./buffers --transport sse --port 8766
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from gigacode.server_dispatch import get_published_schemas, resolve_tool_method

logger = logging.getLogger(__name__)


__all__ = [
    "main",
]

try:
    from mcp.server import Server
    from mcp.server.lowlevel.server import NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        CallToolResult,
        TextContent,
        Tool,
        ToolAnnotations,
    )

    _HAS_MCP = True
except ImportError as _mcp_err:
    _HAS_MCP = False
    Server = None  # type: ignore
    Tool = None  # type: ignore
    NotificationOptions = None  # type: ignore
    CallToolResult = None  # type: ignore
    ToolAnnotations = None  # type: ignore


# Domain statuses that represent a failed operation rather than a successful
# call with a non-ok payload.
_ERROR_STATUSES = {"error", "conflict", "blocked"}


def _server_version() -> str:
    """Return the package version so the MCP handshake never reports stale data."""
    try:
        from gigacode import __version__

        return __version__
    except (ImportError, AttributeError):
        return "0.0.0"


def _result_to_text(result: dict[str, Any]) -> str:
    """Serialize a tool result dict to a JSON string for MCP TextContent."""
    return json.dumps(result, indent=2, default=str)


def _make_call_result(result: Any, *, is_error: bool = False) -> Any:
    """Build a ``CallToolResult`` with an explicit error status.

    Falls back to the legacy ``list[TextContent]`` shape when the installed
    MCP SDK does not expose ``CallToolResult``.
    """
    if not isinstance(result, dict):
        result = {"status": "ok", "result": result}
    text = _result_to_text(result)
    if CallToolResult is None:
        return [TextContent(type="text", text=text)]
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=result,
        isError=is_error,
    )


def _success_result(result: Any) -> Any:
    """Wrap a tool return value, flagging non-ok domain statuses as errors."""
    status = result.get("status") if isinstance(result, dict) else None
    return _make_call_result(result, is_error=status in _ERROR_STATUSES)


def _failure_result(message: str) -> Any:
    """Wrap an execution/validation failure as an MCP error result."""
    return _make_call_result({"status": "error", "message": message}, is_error=True)


async def _invoke_tool(tool: Any, name: str, arguments: dict[str, Any]) -> Any:
    """Resolve and invoke a tool, returning a protocol-appropriate result.

    Shared by every transport so discovery, invocation, and error handling
    stay consistent.
    """
    method = resolve_tool_method(tool, name)
    if method is None:
        return _failure_result(f"Unknown tool: {name}")
    try:
        result = await asyncio.to_thread(method, **arguments)
    except TypeError as exc:
        logger.warning("Invalid arguments for MCP tool %s: %s", name, exc)
        return _failure_result(f"Invalid arguments for {name}: {exc}")
    except (ValueError, OSError, ImportError, ModuleNotFoundError) as exc:
        logger.exception("MCP tool %s failed", name)
        return _failure_result(f"Tool execution failed: {name}")
    return _success_result(result)


def _build_mcp_tools(tool: Any) -> list[Any]:
    """Build MCP Tool definitions with accurate annotations and output schemas."""
    tools: list[Any] = []
    for schema in get_published_schemas(tool):
        tags = schema.get("tags", []) or []
        annotations = None
        if ToolAnnotations is not None:
            annotations = ToolAnnotations(
                title=schema.get("name"),
                readOnlyHint=bool(schema.get("read_only", True)),
                destructiveHint="destructive" in tags,
                idempotentHint=bool(schema.get("read_only", True)),
                openWorldHint=False,
            )
        output_schema = schema.get("output_schema")
        tools.append(
            Tool(
                name=schema["name"],
                description=schema.get("description", ""),
                inputSchema=schema.get("input_schema", {}),
                outputSchema=output_schema if isinstance(output_schema, dict) else None,
                annotations=annotations,
            )
        )
    return tools


def _build_server(tool: Any) -> Any:
    """Build an MCP server with discovery and invocation handlers registered."""
    server = Server("gigacode", version=_server_version())

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return _build_mcp_tools(tool)

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
        return await _invoke_tool(tool, name, arguments)

    return server


async def _run_stdio(tool: Any) -> None:
    server = _build_server(tool)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="gigacode",
                server_version=_server_version(),
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def _extract_api_key(scope: dict[str, Any]) -> str:
    """Pull an API key from X-API-Key or an Authorization: Bearer header."""
    for raw_name, raw_value in scope.get("headers", []):
        name = raw_name.decode("latin-1").lower()
        value = raw_value.decode("latin-1")
        if name == "x-api-key":
            return value
        if name == "authorization" and value.lower().startswith("bearer "):
            return value[7:].strip()
    return ""


def _auth_asgi_wrapper(app: Any, api_key: str | None) -> Any:
    """Wrap an ASGI app with API-key authentication when a key is configured."""
    if not api_key:
        return app

    async def wrapped(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            supplied = _extract_api_key(scope)
            if not secrets.compare_digest(supplied, api_key):
                body = json.dumps(
                    {"status": "error", "message": "Invalid or missing API key"}
                ).encode("utf-8")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json; charset=utf-8"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await app(scope, receive, send)

    return wrapped


def _build_streamable_http_app(
    tool: Any, api_key: str | None = None, stateless: bool = False
) -> Any:
    """Build an ASGI app serving MCP over Streamable HTTP (mounted at ``/mcp``)."""
    try:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.routing import Mount
    except ImportError as exc:
        raise SystemExit(f"MCP Streamable HTTP transport not available: {exc}") from exc

    server = _build_server(tool)
    session_manager = StreamableHTTPSessionManager(app=server, stateless=stateless)

    @asynccontextmanager
    async def lifespan(_app: Any):
        # Owns the session-manager lifecycle; exiting cancels sessions and
        # releases resources.
        async with session_manager.run():
            yield

    starlette_app = Starlette(
        routes=[Mount("/mcp", app=session_manager.handle_request)],
        lifespan=lifespan,
    )
    return _auth_asgi_wrapper(starlette_app, api_key)


def _run_streamable_http(
    tool: Any,
    host: str,
    port: int,
    api_key: str | None = None,
    stateless: bool = False,
) -> None:
    """Serve MCP over Streamable HTTP (mounted at ``/mcp``)."""
    app = _build_streamable_http_app(tool, api_key=api_key, stateless=stateless)

    import uvicorn

    if api_key:
        logger.info("GigaCode MCP Streamable HTTP server (API key auth) on /mcp")
    else:
        logger.warning(
            "MCP Streamable HTTP server starting without authentication; bind to "
            "localhost or set --api-key / GIGACODE_API_KEY for untrusted networks."
        )
    logger.info("GigaCode MCP Streamable HTTP server on http://%s:%d/mcp", host, port)
    uvicorn.run(app, host=host, port=port)


def _run_sse(
    tool: Any,
    host: str,
    port: int,
    api_key: str | None = None,
) -> None:
    # SSE transport requires mcp >= 1.1 with sse_server support
    try:
        from mcp.server.sse import SseServerTransport
    except ImportError as exc:
        raise SystemExit(f"MCP SSE transport not available: {exc}") from exc

    from starlette.applications import Starlette
    from starlette.routing import Mount, Route

    server = _build_server(tool)
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_session(request.scope, request.receive, request._send) as (
            read_stream,
            write_stream,
        ):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="gigacode",
                    server_version=_server_version(),
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    starlette_app = Starlette(
        debug=False,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )
    app = _auth_asgi_wrapper(starlette_app, api_key)

    import uvicorn

    if not api_key:
        logger.warning(
            "MCP SSE server starting without authentication; bind to localhost or "
            "set --api-key / GIGACODE_API_KEY for untrusted networks."
        )
    logger.info("GigaCode MCP SSE server on http://%s:%d/sse", host, port)
    uvicorn.run(app, host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    if not _HAS_MCP:
        print(
            "ERROR: MCP SDK not installed. Install with:\n  pip install mcp>=1.1.0\n",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description="GigaCode MCP server")
    parser.add_argument("--work-dir", "-w", default="./buffers", help="Buffer working directory")
    parser.add_argument("--device", "-d", default=None, help="torch device (cpu / cuda / auto)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU FAISS mirror")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport type (default: stdio for local integrations)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address for network transports")
    parser.add_argument("--port", type=int, default=8766, help="Port for network transports")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GIGACODE_API_KEY"),
        help=(
            "Require this key (X-API-Key or Authorization: Bearer) for SSE and "
            "Streamable HTTP transports. Defaults to GIGACODE_API_KEY. Network "
            "transports run unauthenticated when unset."
        ),
    )
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Run Streamable HTTP without server-side sessions (for horizontal scale).",
    )
    parser.add_argument(
        "--tool-profile",
        default="read_only",
        choices=["read_only", "editing", "full"],
        help=(
            "Curated tool surface exposed to the client (default: read_only). "
            "Use 'editing' to enable write/commit tools."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from gigacode.gigacode_tool import CodeEmbeddingTool

    tool = CodeEmbeddingTool(
        work_dir=args.work_dir,
        device=args.device,
        use_gpu=not args.no_gpu,
        tool_profile=args.tool_profile,
    )

    try:
        if args.transport == "stdio":
            asyncio.run(_run_stdio(tool))
        elif args.transport == "sse":
            _run_sse(tool, args.host, args.port, api_key=args.api_key)
        else:
            _run_streamable_http(
                tool,
                args.host,
                args.port,
                api_key=args.api_key,
                stateless=args.stateless,
            )
    finally:
        tool.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
