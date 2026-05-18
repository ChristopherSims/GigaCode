"""Shared helpers for safe server-side tool dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def get_allowed_tool_names(tool: Any) -> set[str]:
    """Return the published tool names exposed by the schema layer."""
    schemas = tool.get_tool_schemas()
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
