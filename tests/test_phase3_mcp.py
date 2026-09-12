"""Phase 3 MCP tests: tool profiles, annotations, and a real MCP client."""

import types

try:
    import sklearn

    if getattr(sklearn, "__spec__", None) is None:
        sklearn.__spec__ = types.ModuleSpec("sklearn", getattr(sklearn, "__file__", None))
except Exception:
    pass

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gigacode.gigacode_tool import CodeEmbeddingTool
from gigacode.server_dispatch import resolve_tool_method
from gigacode.tool_schema import (
    DEFAULT_TOOL_PROFILE,
    get_profile_schemas,
    get_profile_tool_names,
    list_tool_profiles,
)


def _make_tool(tmpdir: str, profile: str) -> CodeEmbeddingTool:
    return CodeEmbeddingTool(Path(tmpdir) / "tool", use_gpu=False, tool_profile=profile)


class TestToolProfiles:
    def test_profiles_registered(self):
        assert set(list_tool_profiles()) == {"read_only", "editing", "full"}
        assert DEFAULT_TOOL_PROFILE == "read_only"

    def test_default_profile_is_curated_read_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = _make_tool(tmpdir, "read_only")
            names = {s["name"] for s in tool.get_exposed_tool_schemas()}
            assert "embed_codebase" in names
            assert "semantic_search" in names
            assert "write_code" not in names
            assert "commit" not in names
            assert "discard" not in names
            assert "solve" not in names
            assert "cluster_code" not in names
            # Much smaller than the full catalogue.
            assert len(names) < len(tool.get_tool_schemas())
            tool.close()

    def test_editing_profile_enables_editing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = _make_tool(tmpdir, "editing")
            names = {s["name"] for s in tool.get_exposed_tool_schemas()}
            assert {"write_code", "commit", "discard", "reload_codebase"} <= names
            assert "solve" not in names
            tool.close()

    def test_unknown_profile_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                _make_tool(tmpdir, "does-not-exist")

    def test_explicit_allowed_tools_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = CodeEmbeddingTool(
                Path(tmpdir) / "tool",
                use_gpu=False,
                allowed_tools=["semantic_search"],
            )
            assert tool.is_tool_allowed("semantic_search")
            assert not tool.is_tool_allowed("write_code")
            tool.close()

    def test_profile_is_enforced_at_execution_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            read_tool = _make_tool(tmpdir, "read_only")
            assert resolve_tool_method(read_tool, "semantic_search") is not None
            # Hidden tools cannot be resolved/dispatched even if requested.
            assert resolve_tool_method(read_tool, "commit") is None
            assert resolve_tool_method(read_tool, "write_code") is None
            read_tool.close()

            edit_tool = _make_tool(tmpdir, "editing")
            assert resolve_tool_method(edit_tool, "commit") is not None
            edit_tool.close()

    def test_curated_set_is_subset_of_all(self):
        full = {s["name"] for s in get_profile_schemas("full")}
        read_only = get_profile_tool_names("read_only")
        editing = get_profile_tool_names("editing")
        assert read_only <= editing <= full


class _DummyTool:
    """Minimal MCP tool double used to exercise the protocol surface."""

    def get_exposed_tool_schemas(self):
        return [
            {
                "name": "ping",
                "description": "Return a pong",
                "input_schema": {"type": "object", "properties": {}},
                "output_schema": {"type": "object"},
                "read_only": True,
                "tags": ["read-only", "fast"],
            },
            {
                "name": "boom",
                "description": "Always fails",
                "input_schema": {"type": "object", "properties": {}},
                "output_schema": {"type": "object"},
                "read_only": False,
                "tags": ["write", "destructive"],
            },
            {
                "name": "slow",
                "description": "Blocks for a while",
                "input_schema": {"type": "object", "properties": {}},
                "output_schema": {"type": "object"},
                "read_only": True,
                "tags": ["read-only"],
            },
        ]

    def ping(self):
        return {"status": "ok", "pong": True}

    def boom(self):
        raise ValueError("kaboom")

    def slow(self):
        import time

        time.sleep(1.0)
        return {"status": "ok"}


def _require_mcp():
    mcp_server = pytest.importorskip("gigacode.mcp_server")
    if not mcp_server._HAS_MCP:
        pytest.skip("MCP SDK not installed")
    return mcp_server


def _init_request() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"},
        },
    }


class TestStreamableHttpTransport:
    def test_requires_api_key_and_initializes(self):
        mcp_server = _require_mcp()
        from starlette.testclient import TestClient

        app = mcp_server._build_streamable_http_app(_DummyTool(), api_key="secret")
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }

        with TestClient(app) as client:
            unauth = client.post("/mcp", json=_init_request(), headers=headers)
            assert unauth.status_code == 401

            ok = client.post(
                "/mcp", json=_init_request(), headers={**headers, "X-API-Key": "secret"}
            )
            assert ok.status_code == 200
            assert "gigacode" in ok.text

    def test_lifespan_starts_and_stops_cleanly(self):
        mcp_server = _require_mcp()
        from starlette.testclient import TestClient

        # Each app owns one session-manager lifecycle; entering and leaving the
        # lifespan must start and cancel it without leaking tasks.
        for _ in range(2):
            app = mcp_server._build_streamable_http_app(_DummyTool())
            with TestClient(app):
                pass


class TestStdioProtocolSafety:
    def test_build_server_writes_nothing_to_stdout(self):
        mcp_server = _require_mcp()
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            server = mcp_server._build_server(_DummyTool())
            tools = mcp_server._build_mcp_tools(_DummyTool())
        assert buffer.getvalue() == ""
        assert server is not None
        assert tools


class TestMcpClient:
    def test_discovery_and_call_statuses(self):
        mcp_server = pytest.importorskip("gigacode.mcp_server")
        if not mcp_server._HAS_MCP:
            pytest.skip("MCP SDK not installed")

        from mcp.shared.memory import create_connected_server_and_client_session

        server = mcp_server._build_server(_DummyTool())

        async def run() -> None:
            async with create_connected_server_and_client_session(server) as session:
                await session.initialize()

                tools = await session.list_tools()
                by_name = {t.name: t for t in tools.tools}
                assert {"ping", "boom"} <= set(by_name)
                assert by_name["boom"].annotations.destructiveHint is True

                ok = await session.call_tool("ping", {})
                assert ok.isError is False

                failed = await session.call_tool("boom", {})
                assert failed.isError is True

                unknown = await session.call_tool("nope", {})
                assert unknown.isError is True

                invalid = await session.call_tool("ping", {"unexpected": 1})
                assert invalid.isError is True

        asyncio.run(run())

    def test_cancellation_leaves_session_usable(self):
        mcp_server = _require_mcp()
        from mcp.shared.memory import create_connected_server_and_client_session

        server = mcp_server._build_server(_DummyTool())

        async def run() -> None:
            async with create_connected_server_and_client_session(server) as session:
                await session.initialize()
                task = asyncio.create_task(session.call_tool("slow", {}))
                await asyncio.sleep(0.05)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # The session must remain usable after a cancelled call.
                ok = await session.call_tool("ping", {})
                assert ok.isError is False

        asyncio.run(run())
