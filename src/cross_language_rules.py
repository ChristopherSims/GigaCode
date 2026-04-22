"""Language-agnostic code editing rules using tree-sitter + regex fallbacks.

Applies common improvements across many languages:
- Add documentation comments
- Add type annotations / signatures where missing
- Fix bare exception/catch blocks
- Use context managers for resource handling
- Standardize naming conventions
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.language_detect import detect_language, get_tree_sitter_package

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language-specific comment syntax
# ---------------------------------------------------------------------------
COMMENT_SYNTAX: dict[str, dict[str, str]] = {
    "python": {"block_start": '"""', "block_end": '"""', "line": "#"},
    "javascript": {"block_start": "/**", "block_end": " */", "line": "//"},
    "typescript": {"block_start": "/**", "block_end": " */", "line": "//"},
    "java": {"block_start": "/**", "block_end": " */", "line": "//"},
    "c": {"block_start": "/**", "block_end": " */", "line": "//"},
    "cpp": {"block_start": "/**", "block_end": " */", "line": "//"},
    "rust": {"block_start": "/**", "block_end": " */", "line": "//"},
    "go": {"block_start": "/**", "block_end": " */", "line": "//"},
    "ruby": {"block_start": "=begin", "block_end": "=end", "line": "#"},
    "php": {"block_start": "/**", "block_end": " */", "line": "//"},
    "csharp": {"block_start": "/**", "block_end": " */", "line": "//"},
    "swift": {"block_start": "/**", "block_end": " */", "line": "//"},
    "kotlin": {"block_start": "/**", "block_end": " */", "line": "//"},
    "scala": {"block_start": "/**", "block_end": " */", "line": "//"},
    "bash": {"block_start": "", "block_end": "", "line": "#"},
    "lua": {"block_start": "--[[", "block_end": "--]]", "line": "--"},
    "elixir": {"block_start": "\"\"\"", "block_end": "\"\"\"", "line": "#"},
    "default": {"block_start": "/*", "block_end": " */", "line": "//"},
}

# Language-specific function node types in tree-sitter
FUNCTION_NODE_TYPES: dict[str, list[str]] = {
    "python": ["function_definition", "async_function_definition"],
    "javascript": ["function_declaration", "method_definition", "arrow_function"],
    "typescript": ["function_declaration", "method_definition", "arrow_function"],
    "java": ["method_declaration", "constructor_declaration"],
    "c": ["function_definition"],
    "cpp": ["function_definition"],
    "rust": ["function_item"],
    "go": ["function_declaration", "method_declaration"],
    "ruby": ["method", "singleton_method"],
    "php": ["function_definition", "method_declaration"],
    "csharp": ["method_declaration", "constructor_declaration"],
    "swift": ["function_declaration"],
    "kotlin": ["function_declaration"],
    "scala": ["function_definition"],
    "lua": ["function_declaration"],
    "elixir": ["function"],
}

# Bare exception patterns by language
BARE_EXCEPTION_RE: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^(\s*)except\s*:\s*$"),
    "javascript": re.compile(r"^(\s*\}?\s*)catch\s*\(\s*\)\s*\{\s*$"),
    "typescript": re.compile(r"^(\s*\}?\s*)catch\s*\(\s*\)\s*\{\s*$"),
    "java": re.compile(r"^(\s*\}?\s*)catch\s*\(\s*\)\s*\{\s*$"),
    "cpp": re.compile(r"^(\s*\}?\s*)catch\s*\(\s*\.\.\.\s*\)\s*\{\s*$"),
    "csharp": re.compile(r"^(\s*\}?\s*)catch\s*\(\s*\)\s*\{\s*$"),
    "go": re.compile(r"^(\s*\}?\s*)catch\s*\(\s*\)\s*\{\s*$"),
    "ruby": re.compile(r"^(\s*)rescue\s*=>\s*\w+\s*$"),
    "php": re.compile(r"^(\s*\}?\s*)catch\s*\(\s*\)\s*\{\s*$"),
    "swift": re.compile(r"^(\s*\}?\s*)catch\s*\{\s*$"),
    "kotlin": re.compile(r"^(\s*\}?\s*)catch\s*\(\s*\)\s*\{\s*$"),
    "scala": re.compile(r"^(\s*\}?\s*)catch\s*\{\s*case\s+_\s*:\s*\w+\s*=>\s*$"),
}

# Resource open patterns (file handles, network, etc.)
RESOURCE_OPEN_RE = re.compile(
    r"^(\s*)(\w+)\s*=\s*(open|fopen|File\.open|new\s+FileInputStream|socket\.\w+|urllib\.request\.urlopen)\s*\(",
    re.IGNORECASE,
)

# Type annotation hints by language
TYPE_HINT_PATTERNS: dict[str, dict[str, Any]] = {
    "python": {
        "func_re": re.compile(r"^(\s*)def\s+(\w+)\s*\(([^)]*)\)\s*:\s*$"),
        "param_split": ",",
        "add_return": lambda line: line.rstrip()[:-1] + " -> Any:" if line.rstrip().endswith(":") else line,
    },
    "javascript": {
        "func_re": re.compile(r"^(\s*)(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*\{\s*$"),
        "param_split": ",",
    },
    "typescript": {
        "func_re": re.compile(r"^(\s*)(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*[:\{]\s*$"),
        "param_split": ",",
    },
    "java": {
        "func_re": re.compile(r"^(\s*)(?:public|private|protected|static|\s)+\s+(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)\s*\{\s*$"),
        "param_split": ",",
    },
    "cpp": {
        "func_re": re.compile(r"^(\s*)(?:\w+(?:<[^>]+>)?\s+)+(\w+)\s*\(([^)]*)\)\s*\{\s*$"),
        "param_split": ",",
    },
    "rust": {
        "func_re": re.compile(r"^(\s*)fn\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*\w+)?\s*\{\s*$"),
        "param_split": ",",
    },
    "go": {
        "func_re": re.compile(r"^(\s*)func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(([^)]*)\)\s*(?:\w+)?\s*\{\s*$"),
        "param_split": ",",
    },
    "ruby": {
        "func_re": re.compile(r"^(\s*)def\s+(\w+)(?:\s*\(([^)]*)\))?\s*$"),
        "param_split": ",",
    },
    "php": {
        "func_re": re.compile(r"^(\s*)(?:public|private|protected|static|\s)+\s+function\s+(\w+)\s*\(([^)]*)\)\s*\{\s*$"),
        "param_split": ",",
    },
    "csharp": {
        "func_re": re.compile(r"^(\s*)(?:public|private|protected|static|internal|\s)+\s+(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)\s*\{\s*$"),
        "param_split": ",",
    },
    "swift": {
        "func_re": re.compile(r"^(\s*)(?:public|private|internal|fileprivate|open|\s)+\s+func\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*\w+)?\s*\{\s*$"),
        "param_split": ",",
    },
    "kotlin": {
        "func_re": re.compile(r"^(\s*)(?:public|private|protected|internal|\s)+\s+fun\s+(\w+)\s*\(([^)]*)\)\s*:\s*\w+\s*\{\s*$"),
        "param_split": ",",
    },
}


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------
@dataclass
class EditResult:
    """Result of applying editing rules to a file."""

    original: str
    modified: str
    language: str | None = None
    changes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.original != self.modified


def edit_file(path: str | Path, language_hint: str | None = None) -> EditResult:
    """Apply language-agnostic editing rules to a source file.

    Args:
        path: Path to the source file.
        language_hint: Optional language override.

    Returns:
        :class:`EditResult` with original and modified source.
    """
    p = Path(path)
    original = p.read_text(encoding="utf-8", errors="replace")
    language = detect_language(p, hint=language_hint)

    modified = original
    changes: list[dict[str, Any]] = []

    # Try tree-sitter based rules first
    ts_modified, ts_changes = _apply_tree_sitter_rules(p, modified, language)
    if ts_changes:
        modified = ts_modified
        changes.extend(ts_changes)

    # Regex-based fallbacks (always run as safety net)
    regex_modified, regex_changes = _apply_regex_rules(modified, language)
    if regex_changes:
        modified = regex_modified
        changes.extend(regex_changes)

    return EditResult(
        original=original,
        modified=modified,
        language=language,
        changes=changes,
    )


# ---------------------------------------------------------------------------
# Tree-sitter rules
# ---------------------------------------------------------------------------
def _apply_tree_sitter_rules(
    path: Path,
    source: str,
    language: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply tree-sitter based editing rules when a parser is available."""
    changes: list[dict[str, Any]] = []
    if language is None:
        return source, changes

    pkg = get_tree_sitter_package(language)
    if pkg is None:
        return source, changes

    try:
        from tree_sitter import Language, Parser  # type: ignore[import-untyped]

        mod = __import__(pkg)
        lang = getattr(mod, "language", None)
        if lang is None:
            return source, changes
        if callable(lang):
            lang = Language(lang())
        else:
            lang = Language(lang)

        parser = Parser()
        parser.set_language(lang)
        tree = parser.parse(bytes(source, "utf8"))
        root = tree.root_node
    except Exception as exc:
        logger.debug("tree-sitter rule application failed: %s", exc)
        return source, changes

    lines = source.splitlines(keepends=True)
    func_types = FUNCTION_NODE_TYPES.get(language, [])

    # Rule: add docstrings / documentation comments
    insertions: list[tuple[int, str]] = []
    for node in _walk(root):
        if node.type in func_types:
            if not _has_documentation(node, source, language):
                doc = _infer_documentation(node, language)
                if doc:
                    indent = " " * (node.start_point[1] + 4)
                    if node.children:
                        body_start_line = node.children[0].start_point[0]
                        # Find first statement inside function
                        for child in node.children:
                            if child.type not in (
                                "parameters",
                                "type_parameters",
                                "return_type",
                                "annotation",
                            ):
                                body_start_line = child.start_point[0]
                                break
                    else:
                        body_start_line = node.end_point[0]

                    syntax = COMMENT_SYNTAX.get(language, COMMENT_SYNTAX["default"])
                    if syntax["block_start"]:
                        doc_text = (
                            f"{indent}{syntax['block_start']}\n"
                            f"{indent} * {doc}\n"
                            f"{indent} {syntax['block_end']}\n"
                        )
                    else:
                        doc_text = f"{indent}{syntax['line']} {doc}\n"
                    insertions.append((body_start_line, doc_text))

    # Apply insertions bottom-up
    for idx, text in sorted(insertions, key=lambda x: x[0], reverse=True):
        if 0 <= idx <= len(lines):
            lines.insert(idx, text)
            changes.append(
                {
                    "rule": "add_documentation",
                    "line": idx + 1,
                    "description": "Added missing documentation comment.",
                }
            )

    modified = "".join(lines)
    return modified, changes


def _walk(node):
    """Yield every node in the tree (pre-order)."""
    yield node
    for child in node.children:
        yield from _walk(child)


def _has_documentation(node, source: str, language: str) -> bool:
    """Heuristic: does the function already have a doc comment?"""
    # Simple check: look at the first line inside the function body
    # for a comment marker or docstring
    body_children = [
        c
        for c in node.children
        if c.type
        not in (
            "parameters",
            "type_parameters",
            "return_type",
            "annotation",
            "identifier",
            "async",
            "static",
            "public",
            "private",
        )
    ]
    if not body_children:
        return False

    first = body_children[0]
    text = source[first.start_byte : first.end_byte]
    syntax = COMMENT_SYNTAX.get(language, COMMENT_SYNTAX["default"])
    if syntax["block_start"] and syntax["block_start"] in text:
        return True
    if syntax["line"] and text.strip().startswith(syntax["line"]):
        return True
    if language == "python" and '"""' in text:
        return True
    return False


def _infer_documentation(node, language: str) -> str:
    """Generate a simple documentation string from a function node."""
    # Try to extract identifier
    name = "function"
    for child in node.children:
        if child.type == "identifier":
            name = child.text.decode("utf8") if isinstance(child.text, bytes) else str(child.text)
            break

    # Extract parameter names if possible
    params: list[str] = []
    for child in node.children:
        if child.type in ("parameters", "formal_parameters", "parameter_list"):
            for p in child.children:
                if p.type in ("identifier", "parameter", "required_parameter"):
                    ptext = p.text.decode("utf8") if isinstance(p.text, bytes) else str(p.text)
                    params.append(ptext)
                elif p.type == "typed_parameter" or p.type == "parameter":
                    for sub in p.children:
                        if sub.type == "identifier":
                            ptext = sub.text.decode("utf8") if isinstance(sub.text, bytes) else str(sub.text)
                            params.append(ptext)

    if params:
        return f"Handle {name} with parameters: {', '.join(params[:4])}."
    return f"Handle {name}."


# ---------------------------------------------------------------------------
# Regex-based rules (fallback for all languages)
# ---------------------------------------------------------------------------
def _apply_regex_rules(
    source: str,
    language: str | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply regex-based editing rules that work without tree-sitter."""
    changes: list[dict[str, Any]] = []
    lines = source.splitlines(keepends=True)

    # Rule 1: Fix bare exceptions
    lines, exc_changes = _fix_bare_exceptions(lines, language)
    changes.extend(exc_changes)

    # Rule 2: Fix resource open patterns (simple heuristics)
    lines, res_changes = _fix_resource_opens(lines, language)
    changes.extend(res_changes)

    # Rule 3: Add explicit visibility modifiers for some languages
    lines, vis_changes = _add_visibility_modifiers(lines, language)
    changes.extend(vis_changes)

    modified = "".join(lines)
    return modified, changes


def _fix_bare_exceptions(
    lines: list[str],
    language: str | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Replace bare catch/except with explicit exception types."""
    changes: list[dict[str, Any]] = []
    if language is None:
        return lines, changes

    pattern = BARE_EXCEPTION_RE.get(language)
    if pattern is None:
        return lines, changes

    out: list[str] = []
    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            indent = match.group(1)
            if language == "python":
                out.append(f"{indent}except Exception:\n")
                changes.append(
                    {"rule": "fix_bare_except", "line": i + 1, "description": "Replaced bare except with except Exception."}
                )
            elif language in ("javascript", "typescript", "java", "csharp", "kotlin", "php"):
                out.append(f"{indent}catch (error) {{\n")
                changes.append(
                    {"rule": "fix_bare_catch", "line": i + 1, "description": "Replaced bare catch with catch (error)."}
                )
            elif language == "cpp":
                out.append(f"{indent}catch (const std::exception& e) {{\n")
                changes.append(
                    {"rule": "fix_bare_catch", "line": i + 1, "description": "Replaced catch (...) with catch (const std::exception& e)."}
                )
            elif language == "ruby":
                out.append(f"{indent}rescue StandardError => e\n")
                changes.append(
                    {"rule": "fix_bare_rescue", "line": i + 1, "description": "Replaced bare rescue with rescue StandardError."}
                )
            elif language == "swift":
                out.append(f"{indent}catch let error {{\n")
                changes.append(
                    {"rule": "fix_bare_catch", "line": i + 1, "description": "Replaced bare catch with catch let error."}
                )
            else:
                out.append(line)
        else:
            out.append(line)
    return out, changes


def _fix_resource_opens(
    lines: list[str],
    language: str | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Suggest context-manager patterns for resource opens (Python only for now)."""
    changes: list[dict[str, Any]] = []
    if language != "python":
        return lines, changes

    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("f = open(") or stripped.startswith("fh = open("):
            var = stripped.split("=")[0].strip()
            # collect read line
            read_line = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(var + ".close()"):
                if ".read(" in lines[j] or ".readline(" in lines[j]:
                    read_line = lines[j].strip()
                j += 1
            indent = len(lines[i]) - len(lines[i].lstrip())
            path_expr = stripped[stripped.find("(") + 1 : stripped.find(")")]
            mode = '"r"'
            if "," in path_expr:
                parts = path_expr.split(",")
                path_expr = parts[0].strip()
                mode = parts[1].strip()
            if read_line:
                lhs = read_line.split("=")[0].strip()
                rhs = read_line.split("=")[1].strip()
                out.append(" " * indent + f"with open({path_expr}, {mode}) as {var}:\n")
                out.append(" " * (indent + 4) + f"{lhs} = {var}.{rhs.split('.')[1].strip()}\n")
                changes.append(
                    {"rule": "fix_resource_open", "line": i + 1, "description": "Replaced open/close with context manager."}
                )
            i = j + 1
            continue
        out.append(lines[i])
        i += 1
    return out, changes


def _add_visibility_modifiers(
    lines: list[str],
    language: str | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Add explicit access modifiers where missing (Java, C#, Kotlin, etc.)."""
    changes: list[dict[str, Any]] = []
    if language not in ("java", "csharp", "kotlin", "scala"):
        return lines, changes

    # Simple heuristic: if a class member declaration lacks public/private/protected
    member_re = re.compile(r"^(\s+)(?!\s*(?:public|private|protected|internal|static|final|abstract|override)\s)(\w+(?:<[^>]+>)?)\s+(\w+)\s*[;=]")
    out: list[str] = []
    for i, line in enumerate(lines):
        match = member_re.match(line)
        if match:
            indent = match.group(1)
            out.append(f"{indent}private {match.group(2)} {match.group(3)}{line[match.end(3):]}")
            changes.append(
                {"rule": "add_visibility", "line": i + 1, "description": "Added explicit private visibility."}
            )
        else:
            out.append(line)
    return out, changes
