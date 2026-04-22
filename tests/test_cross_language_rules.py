"""Tests for language-agnostic editing rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cross_language_rules import edit_file


def test_python_bare_except_fix(tmp_path: Path) -> None:
    src = tmp_path / "test.py"
    src.write_text(
        "def foo():\n    try:\n        pass\n    except:\n        pass\n",
        encoding="utf-8",
    )
    result = edit_file(src, language_hint="python")
    assert result.changed
    assert any(c["rule"] == "fix_bare_except" for c in result.changes)
    assert "except Exception:" in result.modified


def test_javascript_bare_catch_fix(tmp_path: Path) -> None:
    src = tmp_path / "test.js"
    src.write_text(
        "function foo() {\n    try {\n        bar();\n    } catch () {\n        console.log('err');\n    }\n}\n",
        encoding="utf-8",
    )
    result = edit_file(src, language_hint="javascript")
    assert result.changed
    assert any(c["rule"] == "fix_bare_catch" for c in result.changes)
    assert "catch (error)" in result.modified


def test_cpp_catch_all_fix(tmp_path: Path) -> None:
    src = tmp_path / "test.cpp"
    src.write_text(
        "void foo() {\n    try {\n        bar();\n    } catch (...) {\n        std::cerr << \"err\";\n    }\n}\n",
        encoding="utf-8",
    )
    result = edit_file(src, language_hint="cpp")
    assert result.changed
    assert any(c["rule"] == "fix_bare_catch" for c in result.changes)
    assert "catch (const std::exception& e)" in result.modified


def test_python_resource_open_fix(tmp_path: Path) -> None:
    src = tmp_path / "test.py"
    src.write_text(
        "f = open('data.txt', 'r')\ndata = f.read()\nf.close()\n",
        encoding="utf-8",
    )
    result = edit_file(src, language_hint="python")
    assert result.changed
    assert any(c["rule"] == "fix_resource_open" for c in result.changes)
    assert "with open(" in result.modified


def test_java_visibility_modifier(tmp_path: Path) -> None:
    src = tmp_path / "Test.java"
    src.write_text(
        "public class Test {\n    String name;\n    int count;\n}\n",
        encoding="utf-8",
    )
    result = edit_file(src, language_hint="java")
    assert result.changed
    assert any(c["rule"] == "add_visibility" for c in result.changes)
    assert "private String name;" in result.modified


def test_no_change_for_clean_file(tmp_path: Path) -> None:
    src = tmp_path / "clean.py"
    src.write_text(
        "def foo():\n    '''Doc.'''\n    try:\n        pass\n    except Exception:\n        pass\n",
        encoding="utf-8",
    )
    result = edit_file(src, language_hint="python")
    # May still add docs via tree-sitter if docstring is deemed insufficient,
    # but bare except and resource rules should not fire.
    assert not any(c["rule"] == "fix_bare_except" for c in result.changes)
    assert not any(c["rule"] == "fix_resource_open" for c in result.changes)
