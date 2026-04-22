"""Tests for formal tool schemas (Phase 6.1)."""

from __future__ import annotations

import pytest

from src.tool_schema import (
    ALL_SCHEMAS,
    get_schema,
    to_mcp_tools,
    to_openai_functions,
)


def test_all_schemas_have_required_fields() -> None:
    for schema in ALL_SCHEMAS:
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema
        assert "output_schema" in schema
        assert schema["input_schema"].get("type") == "object"


def test_get_schema_by_name() -> None:
    assert get_schema("embed_codebase") is not None
    assert get_schema("semantic_search") is not None
    assert get_schema("cluster_code") is not None
    assert get_schema("nonexistent") is None


def test_openai_function_format() -> None:
    functions = to_openai_functions()
    assert len(functions) == len(ALL_SCHEMAS)
    for f in functions:
        assert f["type"] == "function"
        assert "function" in f
        assert "name" in f["function"]
        assert "parameters" in f["function"]


def test_mcp_tool_format() -> None:
    tools = to_mcp_tools()
    assert len(tools) == len(ALL_SCHEMAS)
    for t in tools:
        assert "name" in t
        assert "inputSchema" in t


def test_output_schemas_never_contain_source_text() -> None:
    """Verify that no output schema includes a raw source_text field."""
    for schema in ALL_SCHEMAS:
        out = schema.get("output_schema", {})
        props = out.get("properties", {})
        assert "source_text" not in props, f"{schema['name']} leaks source_text"
        # Check nested items too
        for key, val in props.items():
            if isinstance(val, dict) and val.get("type") == "array":
                items = val.get("items", {})
                item_props = items.get("properties", {})
                assert "source_text" not in item_props, f"{schema['name']} leaks source_text in array items"
