"""Behavior and scope checks for the fixed coding benchmark workloads."""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

from scripts.context_benchmark_tasks import CONTEXT_BEHAVIOR


def source_snapshot(root: Path) -> dict[str, str]:
    """Capture Python text and other file digests, excluding generated caches."""
    ignored = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis"}
    snapshot = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in ignored for part in relative.parts):
            continue
        if path.name.startswith(".coverage") or path.suffix == ".pyc":
            continue
        snapshot[relative.as_posix()] = (
            path.read_text(encoding="utf-8")
            if path.suffix == ".py"
            else "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
    return snapshot


def check_scope(task: dict, before: dict[str, str], after: dict[str, str]) -> str | None:
    file = task["check"]["file"]
    changed = {f for f in before.keys() | after.keys() if before.get(f) != after.get(f)}
    if changed != {file}:
        return (
            "Expected exactly the target file to change; unrelated edit, deletion, or no-op found."
        )
    try:
        old = ast.parse(before[file])
        new = ast.parse(after[file])
    except SyntaxError:
        return "Edited file has invalid Python syntax."
    tid = task["id"]
    if tid in {"rename_in_file", "rename_filesize_helper"}:
        old_name, new_name = (
            ("parse_key_value_pairs", "parse_kv_config")
            if tid == "rename_in_file"
            else ("_to_str", "format_size")
        )

        class RestoreRename(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == new_name:
                    node.id = old_name
                return node

            def visit_FunctionDef(self, node):
                if node.name == new_name:
                    node.name = old_name
                return self.generic_visit(node)

        new = RestoreRename().visit(new)
    elif tid == "add_function":
        if (
            sum(isinstance(n, ast.FunctionDef) and n.name == "camel_to_kebab" for n in new.body)
            != 1
        ):
            return "Expected exactly one new camel_to_kebab function."
        new.body = [
            n for n in new.body if not isinstance(n, ast.FunctionDef) or n.name != "camel_to_kebab"
        ]
    else:
        symbol = task["check"]["symbol"]

        class MaskTarget(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == symbol:
                    return ast.Pass()
                return self.generic_visit(node)

        old = MaskTarget().visit(old)
        new = MaskTarget().visit(new)
    if ast.dump(old, include_attributes=False) != ast.dump(new, include_attributes=False):
        return "Code outside the permitted function/rename changed."
    # Docstring-only tasks must preserve the function's executable statements.
    if tid in {"add_docstring", "add_doctest_example", "context_summary_doctest"}:
        symbol = task["check"]["symbol"]
        old_node = next(
            n
            for n in ast.walk(ast.parse(before[file]))
            if isinstance(n, ast.FunctionDef) and n.name == symbol
        )
        new_node = next(
            (
                n
                for n in ast.walk(ast.parse(after[file]))
                if isinstance(n, ast.FunctionDef) and n.name == symbol
            ),
            None,
        )
        if new_node is None:
            return "Target function was removed."
        for node in (old_node, new_node):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
        if ast.dump(old_node) != ast.dump(new_node):
            return "A docstring-only task changed executable code."
    return None


BEHAVIOR = {
    "add_input_validation": """
f = ns['quick_sort']
assert f([3, 1, 2, 1]) == [1, 1, 2, 3]
assert f([]) == []
for value in (None, 'abc', (3, 1), 42):
    try: f(value)
    except TypeError: pass
    else: raise AssertionError('non-list accepted')
""",
    "write_error_handling": """
f = ns['moving_average']
assert f([1, 2], 3) == []
assert f([], 1) == []
assert f([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
""",
    "add_function": """
f = ns['camel_to_kebab']
original = ns['camel_to_snake']
for value in ('camelCase', 'PascalCase', 'HTTPServer', '', 'simple'):
    assert f(value) == original(value).replace('_', '-')
tree = ast.parse(target.read_text(encoding='utf-8'))
names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
assert names.index('camel_to_kebab') == names.index('camel_to_snake') + 1
""",
    "rename_in_file": """
assert 'parse_key_value_pairs' not in ns
assert ns['parse_kv_config']('a=1\\nb=2') == {'a': '1', 'b': '2'}
""",
    "add_method_validation": """
from rich.text import Text
t = Text('hello!')
try: t.remove_suffix('')
except ValueError as exc: assert str(exc)
else: raise AssertionError('empty suffix accepted')
t.remove_suffix('!')
assert t.plain == 'hello'
t.remove_suffix('?')
assert t.plain == 'hello'
""",
    "assert_to_return": """
f = ns['pick_bool']
assert f() is False
assert f(None, True, False) is True
assert f(None, False, True) is False
""",
    "rename_filesize_helper": """
from rich import filesize
assert '_to_str' not in ns
assert callable(ns['format_size'])
assert filesize.decimal(1000) == '1.0 kB'
assert filesize.decimal(0) == '0 bytes'
""",
    "needle_validation": """
from rich.color import ColorTriplet
f = ns['blend_rgb']
a, b = ColorTriplet(10, 20, 30), ColorTriplet(110, 120, 130)
assert f(a, b, 0.0) == a
assert f(a, b, 1.0) == b
assert f(a, b, 0.5) == ColorTriplet(60, 70, 80)
for weight in (-0.1, 1.1):
    try: f(a, b, weight)
    except ValueError as exc: assert str(exc)
    else: raise AssertionError('out-of-range weight accepted')
""",
}
BEHAVIOR.update(CONTEXT_BEHAVIOR)


def check_behavior(task: dict, root: Path) -> str | None:
    """Run focused behavior checks in an isolated, time-bounded subprocess."""
    file = task["check"]["file"]
    symbol = task["check"].get("symbol")
    if task["id"] in {"add_docstring", "add_doctest_example", "context_summary_doctest"}:
        body = f"""
f = ns[{symbol!r}]
examples = doctest.DocTestParser().get_examples(f.__doc__ or '')
assert examples and any(e.want.strip() for e in examples), 'runnable example with expected output required'
test = doctest.DocTestParser().get_doctest(f.__doc__, ns, f.__name__, str(target), 0)
assert doctest.DocTestRunner().run(test).failed == 0
"""
    else:
        body = BEHAVIOR[task["id"]]
    script = (
        "import sys, runpy, ast, doctest, importlib\nfrom pathlib import Path\n"
        "root = Path(sys.argv[1]); target = root / sys.argv[2]\n"
        "sys.path.insert(0, str(root))\n"
        "ns = vars(importlib.import_module(sys.argv[2][:-3].replace('/', '.'))) "
        "if sys.argv[2].startswith('rich/') else runpy.run_path(str(target))\n" + body
    )
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", script, str(root), file],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return "Behavior check timed out."
    if result.returncode:
        return "Focused behavior/doctest check failed."
    return None
