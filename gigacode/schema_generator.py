"""Runtime schema generation from CodeEmbeddingTool introspection (Phase 6.1).

Automatically generates JSON schemas from function signatures and docstrings.
Used to detect schema drift and validate that tool_schema.py matches implementation.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


__all__ = [
    "AGENT_FACING_METHODS",
    "extract_docstring_info",
    "generate_schema_from_method",
    "generate_all_schemas",
    "validate_schemas_against_code",
    "report_schema_validation",
]

# Public agent-facing methods that should have schemas
AGENT_FACING_METHODS = {
    "embed_codebase",
    "semantic_search",
    "hybrid_search",
    "search_symbols",
    "search_for",
    "cluster_code",
    "find_duplicates",
    "pack_context",
    "reload_codebase",
    "check_codebase",
    "list_buffers",
    "delete_buffer",
    "read_code",
    "look_for_file",
    "write_code",
    "diff",
    "discard",
    "commit",
}


def extract_docstring_info(method: Callable) -> dict[str, str]:
    """Extract docstring information (summary, args, returns)."""
    doc = inspect.getdoc(method) or ""
    lines = doc.split("\n")

    summary = ""
    args_section = {}
    returns_info = ""

    in_args = False
    in_returns = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Detect sections
        if stripped == "Args:":
            in_args = True
            in_returns = False
            continue
        elif stripped == "Returns:":
            in_args = False
            in_returns = True
            continue

        # Parse summary (before Args section)
        if not in_args and not in_returns and not summary and stripped:
            summary = stripped
            continue

        # Parse Args section
        if in_args and stripped.startswith(("-", "*")):
            # Example: "- buffer_id: Buffer identifier"
            # Or: "buffer_id: Buffer identifier"
            if ":" in stripped:
                parts = stripped.lstrip("-* ").split(":", 1)
                param_name = parts[0].strip()
                param_desc = parts[1].strip() if len(parts) > 1 else ""
                args_section[param_name] = param_desc

        # Parse Returns section
        if in_returns and not in_args:
            if not returns_info:
                returns_info = stripped

    return {
        "summary": summary,
        "args": args_section,
        "returns": returns_info,
    }


def generate_schema_from_method(method_name: str, method: Callable) -> dict[str, Any]:
    """Generate a JSON schema from a method's signature and docstring."""
    sig = inspect.signature(method)
    doc_info = extract_docstring_info(method)

    # Build input schema from parameters
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Skip 'self' parameter
        if param_name == "self":
            continue

        # Determine type from annotation
        param_type = "string"  # Default
        if param.annotation != inspect.Parameter.empty:
            annotation_str = str(param.annotation).lower()

            # Check annotation
            if param.annotation is int or "int" in annotation_str:
                param_type = "integer"
            elif param.annotation is float or "float" in annotation_str:
                param_type = "number"
            elif param.annotation is bool or "bool" in annotation_str:
                param_type = "boolean"
            elif param.annotation is list or "list" in annotation_str:
                param_type = "array"
            elif param.annotation is dict or "dict" in annotation_str:
                param_type = "object"
            # Try to extract from union types (e.g., "str | None")
            elif "|" in annotation_str and "none" in annotation_str:
                if "int" in annotation_str:
                    param_type = "integer"
                elif "float" in annotation_str:
                    param_type = "number"
                elif "bool" in annotation_str:
                    param_type = "boolean"
                elif "list" in annotation_str:
                    param_type = "array"
                elif "dict" in annotation_str:
                    param_type = "object"
                else:
                    param_type = "string"

        # Get description from docstring
        description = doc_info["args"].get(param_name, f"Parameter {param_name}")

        # Build property
        properties[param_name] = {
            "type": param_type,
            "description": description,
        }

        # Track required parameters (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": method_name,
        "description": doc_info["summary"],
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required if required else None,
        },
        "output_schema": {
            "type": "object",
            "description": doc_info["returns"],
        },
    }


def generate_all_schemas(tool_class: Any) -> dict[str, dict[str, Any]]:
    """Generate schemas for all public agent-facing methods.

    Args:
        tool_class: The CodeEmbeddingTool class (or instance)

    Returns:
        Dict mapping method name to generated schema
    """
    generated = {}

    for method_name in AGENT_FACING_METHODS:
        if not hasattr(tool_class, method_name):
            logger.warning("Method not found in tool class: %s", method_name)
            continue

        method = getattr(tool_class, method_name)
        if not callable(method):
            logger.warning("Not callable: %s", method_name)
            continue

        try:
            schema = generate_schema_from_method(method_name, method)
            generated[method_name] = schema
        except (ImportError, AttributeError, TypeError) as exc:
            logger.error("Failed to generate schema for %s: %s", method_name, exc)

    return generated


def validate_schemas_against_code(
    tool_class: Any, hardcoded_schemas: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    """Validate that hardcoded schemas match the actual code.

    Args:
        tool_class: The CodeEmbeddingTool class
        hardcoded_schemas: Dict of hardcoded schemas from tool_schema.py

    Returns:
        Dict mapping method name to list of validation issues (empty if valid)
    """
    generated = generate_all_schemas(tool_class)
    issues: dict[str, list[str]] = {}

    for method_name, hardcoded in hardcoded_schemas.items():
        if method_name not in generated:
            continue

        generated_schema = generated[method_name]
        method_issues = []

        # Check if method exists
        if not hasattr(tool_class, method_name):
            method_issues.append(f"Method {method_name} not found in CodeEmbeddingTool")

        # Check description consistency
        gen_desc = generated_schema.get("description", "")
        hard_desc = hardcoded.get("description", "")
        if gen_desc and hard_desc and gen_desc != hard_desc:
            method_issues.append(
                f"Description mismatch:\n"
                f"  Generated: {gen_desc[:50]}...\n"
                f"  Hardcoded: {hard_desc[:50]}..."
            )

        # Check required parameters match
        gen_required = set(generated_schema.get("input_schema", {}).get("required") or [])
        hard_required = set(hardcoded.get("input_schema", {}).get("required") or [])

        if gen_required != hard_required:
            missing = hard_required - gen_required
            extra = gen_required - hard_required
            if missing:
                method_issues.append(f"Hardcoded requires params missing in code: {missing}")
            if extra:
                method_issues.append(f"Code has extra required params: {extra}")

        # Check parameter types match
        gen_props = generated_schema.get("input_schema", {}).get("properties", {})
        hard_props = hardcoded.get("input_schema", {}).get("properties", {})

        for param_name in hard_props:
            if param_name not in gen_props:
                method_issues.append(f"Parameter in hardcoded but not in code: {param_name}")
            elif gen_props[param_name].get("type") != hard_props[param_name].get("type"):
                method_issues.append(
                    f"Type mismatch for param {param_name}: "
                    f"generated={gen_props[param_name].get('type')}, "
                    f"hardcoded={hard_props[param_name].get('type')}"
                )

        if method_issues:
            issues[method_name] = method_issues

    return issues


def report_schema_validation(validation_issues: dict[str, list[str]]) -> str:
    """Generate a human-readable validation report."""
    if not validation_issues:
        return "✅ All schemas valid — no drift detected!"

    lines = ["❌ Schema drift detected:\n"]
    for method_name, issues in validation_issues.items():
        lines.append(f"  {method_name}:")
        for issue in issues:
            lines.append(f"    • {issue}")

    return "\n".join(lines)
