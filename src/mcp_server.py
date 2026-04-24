"""Model Context Protocol (MCP) server for GigaCode.

Exposes all GigaCode tools via MCP so Claude Desktop, Cursor, and other
MCP-compatible agents can use them natively.

Usage (stdio transport — for Claude Desktop):
    python -m src.mcp_server --work-dir ./buffers

Usage (HTTP-SSE transport):
    python -m src.mcp_server --work-dir ./buffers --transport sse --port 8766
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.server.lowlevel.server import NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        TextContent,
        Tool,
    )
    _HAS_MCP = True
except ImportError as _mcp_err:
    _HAS_MCP = False
    Server = None  # type: ignore
    Tool = None  # type: ignore
    NotificationOptions = None  # type: ignore


def _result_to_text(result: dict[str, Any]) -> str:
    """Serialize a tool result dict to a JSON string for MCP TextContent."""
    return json.dumps(result, indent=2, default=str)


async def _run_stdio(tool: Any) -> None:
    server = Server("gigacode")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        schemas = tool.get_tool_schemas()
        tools: list[Tool] = []
        for schema in schemas:
            tools.append(
                Tool(
                    name=schema["name"],
                    description=schema.get("description", ""),
                    inputSchema=schema.get("input_schema", {}),
                )
            )
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        method = getattr(tool, name, None)
        if method is None:
            return [TextContent(type="text", text=json.dumps({"status": "error", "message": f"Unknown tool: {name}"}))]
        try:
            result = method(**arguments)
        except Exception as exc:
            logger.exception("MCP tool %s failed", name)
            return [TextContent(type="text", text=json.dumps({"status": "error", "message": str(exc)}))]
        return [TextContent(type="text", text=_result_to_text(result))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="gigacode",
                server_version="1.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def _run_sse(tool: Any, host: str, port: int) -> None:
    # SSE transport requires mcp >= 1.1 with sse_server support
    try:
        from mcp.server.sse import SseServerTransport
    except ImportError as exc:
        raise SystemExit(f"MCP SSE transport not available: {exc}")

    from starlette.applications import Starlette
    from starlette.routing import Mount, Route

    server = Server("gigacode")
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_session(request.scope, request.receive, request._send) as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="gigacode",
                    server_version="1.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    @server.list_tools()
    async def list_tools() -> list[Any]:
        schemas = tool.get_tool_schemas()
        tools = []
        for schema in schemas:
            tools.append(
                Tool(
                    name=schema["name"],
                    description=schema.get("description", ""),
                    inputSchema=schema.get("input_schema", {}),
                )
            )
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        method = getattr(tool, name, None)
        if method is None:
            return [TextContent(type="text", text=json.dumps({"status": "error", "message": f"Unknown tool: {name}"}))]
        try:
            result = method(**arguments)
        except Exception as exc:
            logger.exception("MCP tool %s failed", name)
            return [TextContent(type="text", text=json.dumps({"status": "error", "message": str(exc)}))]
        return [TextContent(type="text", text=_result_to_text(result))]

    app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    import uvicorn
    logger.info("GigaCode MCP SSE server on http://%s:%d/sse", host, port)
    uvicorn.run(app, host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    if not _HAS_MCP:
        print(
            "ERROR: MCP SDK not installed. Install with:\n"
            "  pip install mcp>=1.1.0\n",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description="GigaCode MCP server")
    parser.add_argument("--work-dir", "-w", default="./buffers", help="Buffer working directory")
    parser.add_argument("--device", "-d", default=None, help="torch device (cpu / cuda / auto)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU FAISS mirror")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio", help="Transport type")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address for SSE transport")
    parser.add_argument("--port", type=int, default=8766, help="Port for SSE transport")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.gigacode_tool import CodeEmbeddingTool

    tool = CodeEmbeddingTool(
        work_dir=args.work_dir,
        device=args.device,
        use_gpu=not args.no_gpu,
    )

    if args.transport == "stdio":
        asyncio.run(_run_stdio(tool))
    else:
        _run_sse(tool, args.host, args.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
