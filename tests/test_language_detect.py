"""Tests for language detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.language_detect import detect_language, get_tree_sitter_package


def test_detect_from_extension(tmp_path: Path) -> None:
    assert detect_language(tmp_path / "foo.py") == "python"
    assert detect_language(tmp_path / "foo.js") == "javascript"
    assert detect_language(tmp_path / "foo.ts") == "typescript"
    assert detect_language(tmp_path / "foo.rs") == "rust"
    assert detect_language(tmp_path / "foo.go") == "go"
    assert detect_language(tmp_path / "foo.java") == "java"
    assert detect_language(tmp_path / "foo.c") == "c"
    assert detect_language(tmp_path / "foo.cpp") == "cpp"
    assert detect_language(tmp_path / "foo.rb") == "ruby"
    assert detect_language(tmp_path / "foo.php") == "php"


def test_detect_from_hint(tmp_path: Path) -> None:
    assert detect_language(tmp_path / "foo.txt", hint="python") == "python"
    assert detect_language(tmp_path / "foo.txt", hint="js") == "javascript"
    assert detect_language(tmp_path / "foo.txt", hint="cpp") == "cpp"
    assert detect_language(tmp_path / "foo.txt", hint="c++") == "cpp"


def test_detect_special_filenames(tmp_path: Path) -> None:
    assert detect_language(tmp_path / "Dockerfile") == "dockerfile"
    assert detect_language(tmp_path / "Makefile") == "makefile"
    assert detect_language(tmp_path / "CMakeLists.txt") == "cmake"


def test_detect_from_shebang(tmp_path: Path) -> None:
    f = tmp_path / "script"
    f.write_text("#!/usr/bin/env python3\nprint(1)\n", encoding="utf-8")
    assert detect_language(f) == "python"

    f2 = tmp_path / "script2"
    f2.write_text("#!/bin/bash\necho hi\n", encoding="utf-8")
    assert detect_language(f2) == "bash"


def test_unknown_extension_returns_none(tmp_path: Path) -> None:
    assert detect_language(tmp_path / "foo.unknown") is None


def test_get_tree_sitter_package() -> None:
    assert get_tree_sitter_package("python") == "tree_sitter_python"
    assert get_tree_sitter_package("rust") == "tree_sitter_rust"
    assert get_tree_sitter_package("unknown") is None
