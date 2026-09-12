"""Shared helpers for safe server-side tool dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def get_published_schemas(tool: Any) -> list[Any]:
    """Return the schemas a tool actually exposes.

    Prefers ``get_exposed_tool_schemas`` (profile-filtered) so the tool
    profile is enforced at execution time, not just during discovery.
    """
    exposed = getattr(tool, "get_exposed_tool_schemas", None)
    if callable(exposed):
        return exposed()
    return tool.get_tool_schemas()


def get_allowed_tool_names(tool: Any) -> set[str]:
    """Return the tool names actually exposed by the tool's profile."""
    schemas = get_published_schemas(tool)
    return {
        schema["name"]
        for schema in schemas
        if isinstance(schema, dict) and isinstance(schema.get("name"), str)
    }


def resolve_tool_method(tool: Any, tool_name: str) -> Callable[..., Any] | None:
    """Resolve a callable only if it is part of the published tool surface."""
    if tool_name not in get_allowed_tool_names(tool):
        return None

    method = getattr(tool, tool_name, None)
    if method is None or not callable(method):
        return None

    return method
