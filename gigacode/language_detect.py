"""Language detection from file extension and content heuristics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Mapping from file extension to language name
_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".rb": "ruby",
    ".erb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sc": "scala",
    ".r": "r",
    ".R": "r",
    ".m": "objective_c",
    ".mm": "objective_cpp",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".pl": "perl",
    ".pm": "perl",
    ".lua": "lua",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".xml": "xml",
    ".css": "css",
    ".scss": "scss",
    ".sass": "scss",
    ".less": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".md": "markdown",
    ".rst": "markdown",
    ".dockerfile": "dockerfile",
    ".makefile": "makefile",
    ".mk": "makefile",
    ".cmake": "cmake",
    ".dart": "dart",
    ".elm": "elm",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".fs": "fsharp",
    ".fsx": "fsharp",
    ".fsi": "fsharp",
    ".groovy": "groovy",
    ".gradle": "groovy",
    ".jl": "julia",
    ".nim": "nim",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".pas": "pascal",
    ".pp": "pascal",
    ".tcl": "tcl",
    ".tf": "hcl",
    ".hcl": "hcl",
    ".v": "verilog",
    ".sv": "systemverilog",
    ".vhd": "vhdl",
    ".vhdl": "vhdl",
}

# Content-based shebang detection
_SHEBANG_MAP: dict[str, str] = {
    "python": "python",
    "python3": "python",
    "node": "javascript",
    "bash": "bash",
    "sh": "bash",
    "zsh": "bash",
    "ruby": "ruby",
    "perl": "perl",
    "php": "php",
    "lua": "lua",
    "rscript": "r",
}


def detect_language(path: str | Path, hint: str | None = None) -> str | None:
    """Detect programming language from extension and optional hint.

    Args:
        path: File path to inspect.
        hint: Optional language override. If provided and non-empty,
            it is normalized and returned directly.

    Returns:
        Normalized language string or None if not recognised.
    """
    if hint:
        return _normalize(hint)

    p = Path(path)

    # Extension match
    ext = p.suffix.lower()
    if ext in _EXTENSION_MAP:
        return _EXTENSION_MAP[ext]

    # Special file names
    name = p.name.lower()
    if name == "dockerfile" or name.startswith("dockerfile"):
        return "dockerfile"
    if name in ("makefile", "gnumakefile"):
        return "makefile"
    if name.endswith(".cmake") or name == "cmakelists.txt":
        return "cmake"

    # Shebang detection
    try:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            first = fh.readline(256)
        if first.startswith("#!/"):
            parts = first.split("/")[-1].split()
            interpreter = parts[0].lower()
            # Handle "#!/usr/bin/env python3" -> interpreter is "env", actual lang is next arg
            if interpreter == "env" and len(parts) > 1:
                interpreter = parts[1].lower()
            for key, lang in _SHEBANG_MAP.items():
                if key in interpreter:
                    return lang
    except Exception:
        pass

    return None


def _normalize(hint: str) -> str:
    """Normalize a user-provided language hint."""
    h = hint.lower().strip()
    aliases: dict[str, str] = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "cpp": "cpp",
        "c++": "cpp",
        "cxx": "cpp",
        "rs": "rust",
        "rb": "ruby",
        "cs": "csharp",
        "kt": "kotlin",
        "objc": "objective_c",
        "objcpp": "objective_cpp",
        "shell": "bash",
        "ps": "powershell",
        "docker": "dockerfile",
    }
    return aliases.get(h, h)


def get_tree_sitter_package(language: str) -> str | None:
    """Return the likely tree-sitter language package name.

    Args:
        language: Normalized language string.

    Returns:
        Package name like ``"tree_sitter_python"`` or None.
    """
    mapping: dict[str, str] = {
        "python": "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "tsx": "tree_sitter_typescript",
        "java": "tree_sitter_java",
        "c": "tree_sitter_c",
        "cpp": "tree_sitter_cpp",
        "rust": "tree_sitter_rust",
        "go": "tree_sitter_go",
        "ruby": "tree_sitter_ruby",
        "php": "tree_sitter_php",
        "csharp": "tree_sitter_c_sharp",
        "swift": "tree_sitter_swift",
        "kotlin": "tree_sitter_kotlin",
        "scala": "tree_sitter_scala",
        "bash": "tree_sitter_bash",
        "lua": "tree_sitter_lua",
        "elixir": "tree_sitter_elixir",
        "erlang": "tree_sitter_erlang",
        "haskell": "tree_sitter_haskell",
        "ocaml": "tree_sitter_ocaml",
    }
    return mapping.get(language)
