"""HTTP server for GigaCode agent tools.

Tries FastAPI + Uvicorn first for async concurrency.  Falls back to the
stdlib HTTPServer if dependencies are missing.

Usage:
    python -m src.gigacode_server --work-dir ./buffers --port 8765

Or import and embed your own codebase first:

    from src.gigacode_server import run_server
    from src.gigacode_tool import CodeEmbeddingTool

    tool = CodeEmbeddingTool(work_dir='./buffers', device='cpu', use_gpu=False)
    res = tool.embed_codebase('./my_project', pattern='*.py')
    print('Buffer:', res['buffer_id'])
    run_server(tool, port=8765)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI path (preferred)
# ---------------------------------------------------------------------------

def _run_fastapi(tool: Any, host: str, port: int) -> None:
    import uvicorn
    from src.gigacode_api import create_app

    app = create_app(tool)
    logger.info("GigaCode FastAPI server listening on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Fallback stdlib server
# ---------------------------------------------------------------------------

def _make_handler(tool: Any) -> type:
    from http.server import BaseHTTPRequestHandler

    class _GigacodeHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info(fmt, *args)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/schemas":
                schemas = tool.get_tool_schemas()
                self._send_json(200, {"schemas": schemas})
                return
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "Not found. Try POST /call or GET /schemas"})

        def do_POST(self) -> None:
            if self.path == "/call":
                self._handle_call()
                return
            self._send_json(404, {"error": "Not found. Try POST /call or GET /schemas"})

        def _handle_call(self) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._send_json(400, {"error": "Empty body"})
                return

            try:
                raw = self.rfile.read(content_length).decode("utf-8")
                req = json.loads(raw)
            except Exception as exc:
                self._send_json(400, {"error": f"Invalid JSON: {exc}"})
                return

            tool_name = req.get("tool")
            args = req.get("args", {})
            if not tool_name or not isinstance(tool_name, str):
                self._send_json(400, {"error": "Missing or invalid 'tool' field"})
                return

            method = getattr(tool, tool_name, None)
            if method is None or not callable(method):
                self._send_json(400, {"error": f"Unknown tool: {tool_name}"})
                return

            try:
                result = method(**args)
            except TypeError as exc:
                self._send_json(400, {"error": f"Invalid arguments for {tool_name}: {exc}"})
                return
            except Exception as exc:
                logger.exception("Tool %s failed", tool_name)
                self._send_json(500, {"error": f"Tool execution failed: {exc}"})
                return

            self._send_json(200, {"status": "ok", "result": result})

    return _GigacodeHandler


def _run_stdlib(tool: Any, host: str, port: int) -> None:
    from http.server import HTTPServer

    handler = _make_handler(tool)
    server = HTTPServer((host, port), handler)
    logger.info("GigaCode stdlib server listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.server_close()
        tool.close()


# ---------------------------------------------------------------------------
# Unified runner
# ---------------------------------------------------------------------------

def run_server(tool: Any, host: str = "127.0.0.1", port: int = 8765, use_fastapi: bool = True) -> None:
    """Start the HTTP server.

    Args:
        tool: CodeEmbeddingTool instance.
        host: Bind address.
        port: Port number.
        use_fastapi: If True (default), try FastAPI + Uvicorn first.
    """
    if use_fastapi:
        try:
            _run_fastapi(tool, host, port)
            return
        except ImportError:
            logger.warning("FastAPI/uvicorn not installed; falling back to stdlib HTTPServer.")
    _run_stdlib(tool, host, port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GigaCode HTTP agent server")
    parser.add_argument("--work-dir", "-w", default="./buffers", help="Buffer working directory")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    parser.add_argument("--port", "-p", type=int, default=8765, help="Port (default 8765)")
    parser.add_argument("--device", "-d", default=None, help="torch device (cpu / cuda / auto)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU FAISS mirror")
    parser.add_argument("--no-fastapi", action="store_true", help="Force stdlib HTTPServer")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Allow running from repo root
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.gigacode_tool import CodeEmbeddingTool

    tool = CodeEmbeddingTool(
        work_dir=args.work_dir,
        device=args.device,
        use_gpu=not args.no_gpu,
    )
    run_server(tool, host=args.host, port=args.port, use_fastapi=not args.no_fastapi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
