"""Deterministic, source-backed navigation. No model or embedding dependency."""

from __future__ import annotations

import ast
import hashlib
import json
import posixpath
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gigacode.chunker import _get_grammar
from gigacode.language_detect import detect_language
from gigacode.path_utils import resolve_source_path, validate_buffer_path
from gigacode.source_search import SourceSearchIndex

INDEX_VERSION = 1
LANGUAGES = {"python", "javascript", "typescript", "tsx", "cpp", "c", "yaml", "dockerfile", "helm"}
SKIP_DIRS = {
    ".git", ".ai", ".gigacode", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".next", "dist", "build", "_build",
}
MAX_FILE_BYTES = 2_000_000
MAX_FILES = 20_000
MAX_TOTAL_BYTES = 64_000_000


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bounded(rows: list, limit: int = 20, offset: int = 0) -> dict:
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("limit must be 1..100 and offset must be non-negative.")
    return {"items": rows[offset:offset + limit], "total": len(rows),
            "next_offset": offset + limit if offset + limit < len(rows) else None}


def own_nodes(node: ast.AST):
    """Exclude nested definitions from their parent's call/reference ownership."""
    yield node
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield from own_nodes(child)


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


@dataclass
class Symbol:
    id: str
    file: str
    symbol: str
    symbol_type: str
    line_start: int
    line_end: int
    parent: str | None
    signature: str
    exported: bool = False
    doc: str = ""
    calls: list[dict] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    local_bindings: list[str] = field(default_factory=list)
    imports: dict[str, str] = field(default_factory=dict)

    def location(self) -> dict:
        return {key: getattr(self, key) for key in (
            "id", "file", "symbol", "symbol_type", "line_start", "line_end", "parent", "signature", "exported",
        )}


@dataclass
class ParsedFile:
    file: str
    language: str
    hash: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[dict] = field(default_factory=list)
    bindings: dict[str, str] = field(default_factory=dict)
    purpose: str = ""
    diagnostics: list[str] = field(default_factory=list)
    database: list[str] = field(default_factory=list)
    api_routes: list[dict] = field(default_factory=list)

    def add(self, name, kind, start, end, parent, signature, exported=False, doc="") -> Symbol:
        base = f"{self.file}::{name}"
        number = sum(s.symbol == name for s in self.symbols)
        symbol = Symbol(base if not number else f"{base}#{number + 1}", self.file,
                        name, kind, start, end, parent, signature[:300], exported, doc[:500])
        self.symbols.append(symbol)
        return symbol


def parse_python(record: ParsedFile, text: str) -> None:
    tree = ast.parse(text)
    lines = text.splitlines()
    record.purpose = (ast.get_docstring(tree) or "")[:400]
    exports = None
    route_prefixes = {}
    module_nodes = {id(node) for node in own_nodes(tree)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    record.imports.append({"kind": "import", "target": alias.name, "line": node.lineno})
                    if id(node) in module_nodes:
                        record.bindings[alias.asname or alias.name.split(".")[0]] = alias.name
            else:
                module = "." * node.level + (node.module or "")
                for alias in node.names:
                    target = module + ("." if node.module else "") + alias.name
                    record.imports.append({"kind": "import", "target": target, "module": module,
                                           "line": node.lineno})
                    if id(node) in module_nodes:
                        record.bindings[alias.asname or alias.name] = target
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                for keyword in node.value.keywords:
                    if keyword.arg == "prefix" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                route_prefixes[target.id] = keyword.value.value
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        value = ast.literal_eval(node.value)
                        if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
                            exports = set(value)
                    except (ValueError, TypeError):
                        record.diagnostics.append("Dynamic __all__; exports are syntactic candidates.")
                if isinstance(target, ast.Name) and target.id == "__tablename__":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        record.database.append(node.value.value)

    def visit(container, parent=None, prefix="", parent_kind=None):
        for node in ast.iter_child_nodes(container):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                kind = "class" if isinstance(node, ast.ClassDef) else "method" if parent_kind == "class" else "function"
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                # Multiline signatures stop at the first statement, not the whole body.
                signature_end = node.body[0].lineno - 1 if node.body else node.lineno
                signature = " ".join(line.strip() for line in lines[node.lineno - 1:max(node.lineno, signature_end)])
                exported = parent is None and (node.name in exports if exports is not None else not node.name.startswith("_"))
                sym = record.add(name, kind, start, node.end_lineno or node.lineno,
                                 parent, signature, exported, ast.get_docstring(node) or "")
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                        if decorator.func.attr in {"get", "post", "put", "patch", "delete", "route"} and decorator.args:
                            argument = decorator.args[0]
                            if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and argument.value.startswith("/"):
                                path = route_prefixes.get(dotted(decorator.func.value), "") + argument.value
                                record.api_routes.append({"path": path, "method": decorator.func.attr.upper(),
                                                          "symbol_id": sym.id, "line": decorator.lineno})
                if kind != "class":
                    decorator_nodes = {id(child) for decorator in node.decorator_list for child in ast.walk(decorator)}
                    arguments = node.args
                    sym.local_bindings = [a.arg for a in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]]
                    sym.local_bindings.extend(a.arg for a in (arguments.vararg, arguments.kwarg) if a)
                    for child in own_nodes(node):
                        if isinstance(child, ast.Import):
                            for alias in child.names:
                                sym.imports[alias.asname or alias.name.split(".")[0]] = alias.name
                        elif isinstance(child, ast.ImportFrom):
                            module = "." * child.level + (child.module or "")
                            for alias in child.names:
                                sym.imports[alias.asname or alias.name] = module + ("." if child.module else "") + alias.name
                        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                            sym.local_bindings.append(child.id)
                        if isinstance(child, ast.Call):
                            target = dotted(child.func)
                            if target:
                                sym.calls.append({"target": target, "line": child.lineno, "kind": "call"})
                                if id(child) not in decorator_nodes and target.rsplit(".", 1)[-1] in {"get", "post", "put", "patch", "delete", "request"} and child.args:
                                    argument = child.args[0]
                                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                                        _add_api(record, argument.value, child.lineno)
                            if isinstance(child.func, ast.Attribute):
                                if child.func.attr in {"endswith", "removesuffix"}:
                                    sym.signals.extend(["suffix", "end", "substring", "match"])
                                elif child.func.attr in {"startswith", "removeprefix"}:
                                    sym.signals.extend(["prefix", "start", "substring", "match"])
                        if isinstance(child, (ast.Name, ast.Attribute)) and isinstance(child.ctx, ast.Load):
                            target = dotted(child)
                            if target:
                                sym.references.append({"target": target, "line": child.lineno, "kind": "reference"})
                visit(node, sym.id, name + ".", kind)
            elif isinstance(node, ast.Assign) and parent is None:
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        record.add(target.id, "variable", node.lineno, node.end_lineno or node.lineno,
                                   None, target.id, target.id in exports if exports is not None else True)
            else:
                visit(node, parent, prefix, parent_kind)
    visit(tree)


def parse_tree(record: ParsedFile, text: str) -> None:
    from tree_sitter import Parser

    grammar = _get_grammar(record.language)
    if grammar is None:
        record.diagnostics.append("Parser unavailable; structural navigation is incomplete.")
        return
    data = text.encode("utf-8")
    root = Parser(grammar).parse(data).root_node
    if root.has_error:
        record.diagnostics.append("Syntax errors; recovered nodes are syntactic candidates.")

    def value(node):
        return data[node.start_byte:node.end_byte].decode("utf-8", errors="replace") if node else ""

    def function_name(node):
        name = node.child_by_field_name("name")
        declarator = node.child_by_field_name("declarator")
        while name is None and declarator is not None:
            if declarator.type in {"identifier", "field_identifier", "qualified_identifier", "destructor_name"}:
                name = declarator
                break
            name = declarator.child_by_field_name("name")
            declarator = declarator.child_by_field_name("declarator")
        if name is None and node.parent and node.parent.type in {"variable_declarator", "pair", "field_definition", "public_field_definition"}:
            name = node.parent.child_by_field_name("name") or node.parent.child_by_field_name("key") or node.parent.child_by_field_name("property")
        return value(name)

    class_types = {"class_declaration", "class", "class_specifier", "struct_specifier",
                   "interface_declaration", "enum_declaration", "namespace_definition"}
    function_types = {"function_declaration", "function_definition", "method_definition",
                      "arrow_function", "function_expression", "generator_function_declaration", "method_signature"}
    variable_types = {"variable_declarator"}

    def visit(node, owner=None, exported=False):
        current = owner
        declaration = node.child_by_field_name("declarator")
        is_prototype = node.type in {"declaration", "field_declaration"} and declaration is not None and declaration.type == "function_declarator"
        if node.type == "export_statement":
            exported = True
            source = node.child_by_field_name("source")
            if source:
                record.imports.append({"kind": "reexport", "target": value(source).strip("'\""),
                                       "line": node.start_point[0] + 1})
        if node.type == "import_statement":
            source = node.child_by_field_name("source")
            if source:
                module = value(source).strip("'\"")
                record.imports.append({"kind": "import", "target": module, "line": node.start_point[0] + 1})
                # Syntax nodes distinguish imported names, aliases and namespace imports.
                def bindings(child):
                    if child.type == "import_specifier":
                        imported = child.child_by_field_name("name")
                        alias = child.child_by_field_name("alias") or imported
                        record.bindings[value(alias)] = module + "::" + value(imported)
                    elif child.type == "namespace_import":
                        identifiers = [c for c in child.named_children if c.type == "identifier"]
                        if identifiers:
                            record.bindings[value(identifiers[-1])] = module
                    elif child.type == "import_clause":
                        for c in child.named_children:
                            if c.type == "identifier":
                                record.bindings[value(c)] = module + "::default"
                    for c in child.named_children:
                        bindings(c)
                bindings(node)
        if node.type == "preproc_include":
            target = node.child_by_field_name("path")
            record.imports.append({"kind": "include", "target": value(target).strip('<>"'),
                                   "line": node.start_point[0] + 1})
        if node.type in class_types | function_types or is_prototype:
            name = function_name(node)
            if name:
                kind = "class" if node.type in class_types else "method" if owner and owner.symbol_type in {"class", "interface"} else "function"
                if node.type == "namespace_definition":
                    kind = "namespace"
                if node.type == "interface_declaration":
                    kind = "interface"
                name = name.replace("::", ".")
                qualified = owner.symbol + "." + name if owner and "." not in name else name
                body = node.child_by_field_name("body")
                signature = value(node)[:body.start_byte - node.start_byte] if body else value(node).splitlines()[0]
                current = record.add(qualified, kind, node.start_point[0] + 1,
                                     node.end_point[0] + (1 if node.end_point[1] else 0),
                                     owner.id if owner else None, " ".join(signature.split()), exported and owner is None)
                if node.type == "function_definition" and node.child_by_field_name("declarator"):
                    current.exported = True  # Definition visibility, not C++ linker certainty.
                parameters = node.child_by_field_name("parameters")
                if parameters:
                    current.local_bindings = re.findall(r"\b[A-Za-z_]\w*\b", value(parameters))
        elif node.type in variable_types and owner is None:
            name = value(node.child_by_field_name("name"))
            initializer = node.child_by_field_name("value")
            if name and (not initializer or initializer.type not in function_types):
                record.add(name, "variable", node.start_point[0] + 1,
                           node.end_point[0] + 1, None, name, exported)
        if node.type in {"call_expression", "new_expression"}:
            callee = node.child_by_field_name("function") or node.child_by_field_name("constructor")
            target = value(callee).replace("::", ".")
            if current and target:
                current.calls.append({"target": target, "line": node.start_point[0] + 1, "kind": "call"})
            if target in {"require", "import"}:
                arguments = node.child_by_field_name("arguments")
                if arguments and arguments.named_children and arguments.named_children[0].type == "string":
                    record.imports.append({"kind": "import", "target": value(arguments.named_children[0]).strip("'\""),
                                           "line": node.start_point[0] + 1})
            if target == "fetch" or target.rsplit(".", 1)[-1] in {"get", "post", "put", "patch", "delete"}:
                arguments = node.child_by_field_name("arguments")
                if arguments and arguments.named_children and arguments.named_children[0].type == "string":
                    _add_api(record, value(arguments.named_children[0]).strip("'\""), node.start_point[0] + 1)
            if current and target.rsplit(".", 1)[-1] in {"endsWith", "endswith"}:
                current.signals.extend(["suffix", "end", "substring", "match"])
        if node.type in {"identifier", "field_identifier", "property_identifier"} and current:
            # Syntactic references, not a claim of type-resolved dispatch.
            current.references.append({"target": value(node), "line": node.start_point[0] + 1, "kind": "reference"})
        for child in node.named_children:
            visit(child, current, exported)
    visit(root)


def _add_api(record: ParsedFile, target: str, line: int) -> None:
    if target.startswith(("http://", "https://")):
        from urllib.parse import urlsplit
        url = urlsplit(target)
        # Never index query strings, credentials or fragments.
        target = f"{url.scheme}://{url.hostname or ''}"
    elif not target.startswith("/"):
        return
    record.imports.append({"kind": "api", "target": target, "line": line})


def parse_yaml(record: ParsedFile, text: str) -> None:
    import yaml
    from yaml.nodes import MappingNode, ScalarNode, SequenceNode

    # compose does not construct Python objects, evaluate tags or templates.
    documents = list(yaml.compose_all(text, Loader=yaml.SafeLoader))
    visited = set()

    def scalar(node):
        return node.value if isinstance(node, ScalarNode) else None

    def mapping(node):
        return {scalar(k): v for k, v in node.value if scalar(k) is not None} if isinstance(node, MappingNode) else {}

    def visit(node, prefix="", parent=None):
        if id(node) in visited:
            return
        visited.add(id(node))
        if isinstance(node, MappingNode):
            for key, child in node.value:
                name = scalar(key)
                if name is None:
                    continue
                path = prefix + "." + name if prefix else name
                sym = record.add(path, "key", key.start_mark.line + 1, max(key.start_mark.line + 1, child.end_mark.line + 1),
                                 parent, path, parent is None)
                visit(child, path, sym.id)
        elif isinstance(node, SequenceNode):
            for i, child in enumerate(node.value):
                visit(child, f"{prefix}[{i}]", parent)
    for document in documents:
        if document is not None:
            visit(document)
        data = mapping(document)
        if "kind" in data:
            kind = scalar(data["kind"])
            if kind:
                record.purpose = f"Kubernetes {kind} manifest."
        # Chart dependency names/repositories are facts, never rendered values.
        deps = data.get("dependencies")
        if Path(record.file).name.lower() == "chart.yaml" and isinstance(deps, SequenceNode):
            for child in deps.value:
                dep = mapping(child)
                name = scalar(dep.get("name"))
                repository = scalar(dep.get("repository")) or ""
                if name:
                    target = repository[7:] if repository.startswith("file://") else name
                    record.imports.append({"kind": "helm_chart", "target": target,
                                           "line": child.start_mark.line + 1, "local": repository.startswith("file://")})
        services = mapping(data.get("services"))
        for service in services.values():
            fields = mapping(service)
            depends = fields.get("depends_on")
            targets = [scalar(n) for n in depends.value] if isinstance(depends, SequenceNode) else list(mapping(depends))
            for target in targets:
                if target:
                    record.imports.append({"kind": "docker_service", "target": f"{record.file}::services.{target}",
                                           "line": service.start_mark.line + 1})
            build = fields.get("build")
            context = scalar(build) or scalar(mapping(build).get("context"))
            if context:
                record.imports.append({"kind": "docker_build", "target": context,
                                       "line": build.start_mark.line + 1, "local": True})


def parse_helm(record: ParsedFile, text: str) -> None:
    record.purpose = "Helm template (unrendered)."
    stack = []
    for match in re.finditer(r"{{-?(.*?)-?}}", text, re.DOTALL):
        expression = match.group(1).strip()
        line = text.count("\n", 0, match.start()) + 1
        definition = re.match(r'(define|block)\s+"([^"]+)"', expression)
        if definition:
            symbol = record.add(definition.group(2), "template", line, line, None,
                                expression, True)
            stack.append((definition.group(1), symbol))
        elif re.match(r"(if|range|with)\b", expression):
            stack.append(("control", None))
        elif expression == "end" and stack:
            _, symbol = stack.pop()
            if symbol:
                symbol.line_end = line
        owner = next((s for _, s in reversed(stack) if s), None)
        for call in re.finditer(r'\b(include|template)\s+"([^"]+)"', expression):
            edge = {"kind": "helm_template", "target": call.group(2), "line": line}
            record.imports.append(edge)
            if owner:
                owner.calls.append(edge)
        for reference in re.finditer(r"\.Values(?:\.[A-Za-z_]\w*)+", expression):
            target = reference.group(0)
            record.imports.append({"kind": "helm_value", "target": target, "line": line})
            if owner:
                owner.references.append({"kind": "helm_value", "target": target, "line": line})
    if stack:
        record.diagnostics.append("Unclosed template blocks; navigation is incomplete.")


def parse_docker(record: ParsedFile, text: str) -> None:
    record.purpose = "Docker build instructions."
    lines = text.splitlines()
    stage = None
    i = 0
    while i < len(lines):
        start = i + 1
        instruction = lines[i].strip()
        while instruction.endswith("\\") and i + 1 < len(lines):
            i += 1
            instruction = instruction[:-1] + " " + lines[i].strip()
        i += 1
        if not instruction or instruction.startswith("#"):
            continue
        keyword, _, rest = instruction.partition(" ")
        keyword = keyword.upper()
        if keyword == "FROM":
            if stage:
                stage.line_end = start - 1
            match = re.match(r"(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?", rest, re.IGNORECASE)
            if match:
                image, alias = match.groups()
                stage = record.add(alias or f"stage{sum(s.symbol_type == 'stage' for s in record.symbols)}",
                                   "stage", start, len(lines), None, f"FROM {image}", True)
                record.imports.append({"kind": "docker_image", "target": image, "line": start})
        else:
            sym = record.add(f"{stage.symbol + '.' if stage else ''}{keyword}@{start}",
                             "instruction", start, i, stage.id if stage else None, keyword)
            if keyword == "COPY":
                previous = re.search(r"--from=(\S+)", rest)
                if previous:
                    edge = {"kind": "docker_stage", "target": previous.group(1), "line": start}
                    record.imports.append(edge)
                    sym.calls.append(edge)
                else:
                    source = rest.split()
                    if len(source) > 1 and not source[0].startswith("--"):
                        record.imports.append({"kind": "docker_copy", "target": source[0],
                                               "line": start, "local": True})


def parse_file(file: str, text: str) -> ParsedFile:
    language = detect_language(file) or ""
    if file.lower().endswith(".tsx"):
        language = "tsx"
    if Path(file).suffix.lower() == ".tpl" or ("templates" in Path(file).parts and "{{" in text):
        language = "helm"
    record = ParsedFile(file, language, source_hash(text))
    try:
        if language == "python":
            parse_python(record, text)
        elif language in {"javascript", "typescript", "tsx", "cpp", "c"}:
            parse_tree(record, text)
        elif language == "yaml":
            parse_yaml(record, text)
        elif language == "helm":
            parse_helm(record, text)
        elif language == "dockerfile":
            parse_docker(record, text)
    except (SyntaxError, ValueError, TypeError, ImportError, RecursionError) as exc:
        record.diagnostics.append(f"Structural parse failed: {type(exc).__name__}.")
    except Exception as exc:
        # YAML/parser errors may embed scalar values. Never publish their text.
        record.diagnostics.append(f"Structural parse failed: {type(exc).__name__}.")
    return record


class NavigationIndex:
    """One per-project index, reparsing only content whose hash changed."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.files: dict[str, ParsedFile] = {}
        self.sources: dict[str, str] = {}
        self.symbols: dict[str, Symbol] = {}
        self.edges: list[dict] = []
        self.parse_count = 0
        self.revision = ""
        self.search_view = None
        self.search_views = {}
        self.diagnostics: list[str] = []
        self.by_file = {}
        self.by_name = {}
        self.by_leaf = {}
        self.outgoing = {}
        self.incoming = {}
        self.callers = {}
        self._summaries_revision = None

    def refresh(self, overlay: dict[str, list[str]] | None = None) -> None:
        sources = {}
        diagnostics = []
        total_bytes = 0
        # Walk deterministically; prune caches and symlink directories.
        import os
        for directory, dirs, names in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.endswith(".gcbuff")
                             and not (Path(directory) / d).is_symlink())
            for name in sorted(names):
                path = Path(directory) / name
                relative = path.relative_to(self.root).as_posix()
                language = detect_language(relative)
                if language not in LANGUAGES and path.suffix.lower() != ".tpl":
                    continue
                try:
                    if path.is_symlink():
                        validate_buffer_path(relative, self.root)
                    size = path.stat().st_size
                    if size > MAX_FILE_BYTES or total_bytes + size > MAX_TOTAL_BYTES:
                        diagnostics.append(f"Skipped oversized source: {relative}")
                        continue
                    sources[relative] = path.read_text(encoding="utf-8")
                    total_bytes += size
                except (OSError, ValueError, UnicodeError):
                    diagnostics.append(f"Skipped unreadable or outside-root source: {relative}")
                if len(sources) >= MAX_FILES:
                    diagnostics.append("File limit reached; project index is incomplete.")
                    break
            if len(sources) >= MAX_FILES:
                break
        # Buffer changes take precedence, but must not introduce outside-root paths.
        for file, lines in (overlay or {}).items():
            validate_buffer_path(file.replace("\\", "/"), self.root)
            key = file.replace("\\", "/")
            text = "\n".join(lines) + "\n"
            size = len(text.encode("utf-8"))
            old_size = len(sources.get(key, "").encode("utf-8"))
            if size > MAX_FILE_BYTES or total_bytes - old_size + size > MAX_TOTAL_BYTES or (key not in sources and len(sources) >= MAX_FILES):
                sources.pop(key, None)
                total_bytes -= old_size
                diagnostics.append(f"Skipped oversized pending source: {key}")
                continue
            sources[key] = text
            total_bytes += size - old_size
        self.update(sources)
        self.diagnostics = diagnostics

    def update(self, sources: dict[str, str]) -> None:
        normalized = {file.replace("\\", "/"): text for file, text in sources.items()}
        changed = set(normalized) != set(self.files)
        for file, text in normalized.items():
            if file not in self.files or self.files[file].hash != source_hash(text):
                self.files[file] = parse_file(file, text)
                self.parse_count += 1
                changed = True
        for file in set(self.files) - set(normalized):
            del self.files[file]
        self.sources = normalized
        if changed:
            self.symbols = {s.id: s for record in self.files.values() for s in record.symbols}
            self.by_file = defaultdict(list)
            self.by_name = defaultdict(list)
            self.by_leaf = defaultdict(list)
            for symbol in self.symbols.values():
                self.by_file[symbol.file].append(symbol)
                self.by_name[symbol.symbol].append(symbol)
                self.by_leaf[symbol.symbol.rsplit(".", 1)[-1]].append(symbol)
            self._build_edges()
            self.revision = source_hash(json.dumps({f: r.hash for f, r in sorted(self.files.items())}))
            self.search_view = None
            self.search_views.clear()

    def source_search(self, files=None) -> SourceSearchIndex:
        key = tuple(sorted(f.replace("\\", "/") for f in files)) if files is not None else None
        if key not in self.search_views:
            # Reuse the existing behavioral ranker, but do not parse source again.
            view = SourceSearchIndex({})
            for record in self.files.values():
                if key is not None and record.file not in key:
                    continue
                lines = self.sources[record.file].splitlines()
                for symbol in record.symbols:
                    body = "\n".join(lines[symbol.line_start - 1:symbol.line_end])
                    kind = "class" if symbol.symbol_type in {"class", "interface"} else "definition"
                    if kind == "class":
                        body = symbol.signature + "\n" + symbol.doc
                    view._add(record.file, symbol.line_start, symbol.line_end, symbol.symbol,
                              symbol.doc, body, kind, record.purpose,
                              symbol.symbol_type == "method", " ".join(symbol.signals))
            if files is not None and isinstance(files, dict):
                for file, lines in files.items():
                    canonical = file.replace("\\", "/")
                    if canonical in self.files and self.files[canonical].symbols:
                        continue
                    for start in range(1, len(lines) + 1, 60):
                        end = min(start + 59, len(lines))
                        view._add(canonical, start, end, "", "", "\n".join(lines[start - 1:end]), "window")
            if len(self.search_views) >= 8:
                self.search_views.clear()
            self.search_views[key] = view
        return self.search_views[key]

    def file_key(self, file: str) -> str:
        return resolve_source_path(file, self.root, self.files)

    def select(self, symbol: str, file: str | None = None) -> list[Symbol]:
        file = self.file_key(file) if file else None
        entries = self.by_file.get(file, []) if file else None
        exact = [self.symbols[symbol]] if symbol in self.symbols else self.by_name.get(symbol, [])
        if file:
            exact = [s for s in exact if s.file == file]
        if exact and ("." in symbol or symbol in self.symbols):
            return exact
        module_qualified = [
            s for s in (entries if entries is not None else self.symbols.values())
            if symbol == s.file.removesuffix(".py").replace("/", ".").removesuffix(".__init__") + "." + s.symbol
        ]
        if module_qualified:
            return module_qualified
        # Unqualified lookup may return multiple candidates; qualified lookup
        # never silently degrades to a leaf name.
        return [s for s in self.by_leaf.get(symbol, []) if file is None or s.file == file] if "." not in symbol else []

    def _module_files(self, file: str, target: str, language: str) -> list[str]:
        if language == "python":
            dots = len(target) - len(target.lstrip("."))
            module = target[dots:].replace(".", "/")
            base = posixpath.dirname(file)
            if dots:
                for _ in range(dots - 1):
                    base = posixpath.dirname(base)
                path = posixpath.normpath(posixpath.join(base, module))
            else:
                path = module
            paths = [path]
            # A from-import may end in a symbol rather than a module.
            if "/" in path:
                paths.append(path.rsplit("/", 1)[0])
            candidates = [p + suffix for p in paths for suffix in (".py", "/__init__.py")]
        elif language in {"javascript", "typescript", "tsx"}:
            module = target.split("::", 1)[0]
            if not module.startswith("."):
                return []  # Packages/path aliases need resolver configuration.
            path = posixpath.normpath(posixpath.join(posixpath.dirname(file), module))
            candidates = [path]
            if path.endswith(".js"):
                candidates.extend([path[:-3] + ".ts", path[:-3] + ".tsx"])
            candidates.extend(path + suffix for suffix in (".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx", "/index.js"))
        else:
            path = posixpath.normpath(posixpath.join(posixpath.dirname(file), target))
            candidates = [path, target]
        return list(dict.fromkeys(p for p in candidates if p in self.files))

    def resolve_call(self, symbol: Symbol, target: str) -> list[Symbol]:
        record = self.files[symbol.file]
        if target.startswith(("self.", "cls.", "this.")):
            if symbol.parent and symbol.parent in self.symbols:
                target = self.symbols[symbol.parent].symbol + "." + target.split(".", 1)[1]
                return self.select(target, symbol.file)
        parts = target.split(".")
        if parts[0] in symbol.local_bindings:
            return []
        bindings = {**record.bindings, **symbol.imports}
        if parts[0] in bindings:
            imported = bindings[parts[0]]
            qualified = imported + ("." + ".".join(parts[1:]) if len(parts) > 1 else "")
            module_files = self._module_files(symbol.file, imported, record.language)
            name = qualified.split("::")[-1] if "::" in qualified else qualified.rsplit(".", 1)[-1]
            matches = [s for f in module_files for s in self.select(name, f)]
            return matches
        local = self.select(target, symbol.file)
        if local:
            return local
        if symbol.parent and symbol.parent in self.symbols:
            parent = self.symbols[symbol.parent]
            scoped = self.select(parent.symbol + "." + target, symbol.file)
            if scoped:
                return scoped
        if record.language in {"cpp", "c"}:
            included = [f for imp in record.imports if imp["kind"] == "include"
                        for f in self._module_files(record.file, imp["target"], record.language)]
            return [s for f in included for s in self.select(target, f)]
        return []  # A same-named symbol elsewhere is not evidence of dispatch.

    def _build_edges(self) -> None:
        edges = []
        for record in self.files.values():
            for imp in record.imports:
                target = imp["target"]
                resolved = []
                if imp["kind"] == "helm_template":
                    resolved = [s.file for s in self.symbols.values() if s.symbol_type == "template" and s.symbol == target]
                elif imp["kind"] == "helm_value":
                    key = target.removeprefix(".Values.")
                    chart_root = record.file.split("/templates/", 1)[0] if "/templates/" in record.file else ""
                    value_file = (chart_root + "/values.yaml").lstrip("/")
                    resolved = [s.file for s in self.symbols.values() if s.file == value_file and s.symbol == key]
                elif imp["kind"] == "docker_service":
                    file, name = target.split("::", 1)
                    resolved = [file] if self.select(name, file) else []
                elif imp["kind"] in {"docker_stage", "docker_image"}:
                    resolved = [record.file] if any(s.symbol == target and s.symbol_type == "stage" and s.line_start < imp["line"] for s in record.symbols) else []
                elif imp["kind"] == "api":
                    resolved = [r.file for r in self.files.values() if any(route["path"] == target for route in r.api_routes)]
                elif imp.get("local"):
                    path = posixpath.normpath(posixpath.join(posixpath.dirname(record.file), target))
                    resolved = [f for f in self.files if f == path or f.startswith(path.rstrip("/") + "/")]
                else:
                    resolved = self._module_files(record.file, target, record.language)
                resolved = sorted(set(resolved))
                status = "resolved" if len(resolved) == 1 else "ambiguous" if resolved else "external"
                edges.append({"source": record.file, "target": target, "kind": imp["kind"],
                              "line": imp["line"], "resolution": status, "files": resolved})
        self.edges = edges
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        for edge in edges:
            self.outgoing[edge["source"]].append(edge)
            for file in edge["files"]:
                self.incoming[file].append(edge)
        self.callers = defaultdict(list)
        for caller in self.symbols.values():
            for call in caller.calls:
                targets = self.resolve_call(caller, call["target"])
                for target in targets:
                    self.callers[target.id].append({**caller.location(), "call_line": call["line"],
                                                   "resolution": "resolved" if len(targets) == 1 else "ambiguous"})

    def relations(self, symbol: Symbol, direction: str, limit: int = 20, offset: int = 0) -> dict:
        rows = []
        if direction in {"callees", "references"}:
            for edge in symbol.calls if direction == "callees" else symbol.references:
                targets = self.resolve_call(symbol, edge["target"])
                rows.append({**edge, "resolution": "resolved" if len(targets) == 1 else "ambiguous" if targets else "unresolved",
                             "symbols": [s.location() for s in targets[:10]]})
        elif direction == "callers":
            rows = self.callers.get(symbol.id, [])
        return bounded(rows, limit, offset)

    def dependency_rows(self, file: str, incoming=False) -> list[dict]:
        file = self.file_key(file)
        return (self.incoming if incoming else self.outgoing).get(file, [])

    def summary(self, file: str, limit: int = 20) -> dict:
        file = self.file_key(file)
        record = self.files[file]
        return {
            "file": file, "language": record.language, "source_hash": record.hash,
            "purpose": record.purpose or "No documented purpose; inspect exported symbols.",
            "exports": bounded([s.location() for s in record.symbols if s.exported], limit),
            "dependencies": bounded(self.dependency_rows(file), limit),
            "used_by": bounded(sorted({e["source"] for e in self.dependency_rows(file, True)}), limit),
            "database": sorted(set(record.database))[:limit],
            "api_routes": record.api_routes[:limit],
            "diagnostics": record.diagnostics,
        }

    def save_summaries(self, cache_dir: Path) -> None:
        """Persist derived facts only, outside the indexed source tree."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        destination = cache_dir / "navigation_summaries.json"
        if self._summaries_revision == self.revision and destination.exists():
            return
        payload = {"version": INDEX_VERSION, "revision": self.revision,
                   "files": {f: self.summary(f) for f in sorted(self.files)}}
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if not destination.exists() or destination.read_text(encoding="utf-8") != encoded:
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(destination)
        self._summaries_revision = self.revision

    def task_context(self, task: str, limit: int = 10) -> dict:
        if not isinstance(task, str) or not task.strip() or len(task) > 10_000:
            raise ValueError("task must contain 1..10000 characters.")
        hits = self.source_search().search(task, min(limit * 3, 100))["matches"]
        files = []
        evidence = []
        for hit in hits:
            if hit["file"] not in files and len(files) < limit:
                files.append(hit["file"])
                evidence.append({"file": hit["file"], "symbol": hit["name"],
                                 "confidence": hit["confidence"], "reason": "Definition/name/documentation overlap."})
        related_tests = []
        for edge in self.edges:
            if any(f in files for f in edge["files"]) and re.search(r"(^|/)(tests?|test_[^/]+)(/|\.|$)", edge["source"]):
                related_tests.append(edge["source"])
        frameworks = []
        for edge in self.edges:
            package = edge["target"].split(".", 1)[0].split("/", 1)[0]
            if package in {"fastapi", "flask", "django", "react", "express", "vue", "stripe"}:
                frameworks.append({"technology": package, "file": edge["source"], "line": edge["line"],
                                   "evidence": "Import declaration, not inferred architecture."})
        auth = []
        for s in self.symbols.values():
            tokens = set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", s.symbol).lower()))
            if tokens & {"auth", "oauth", "login", "jwt", "token"}:
                auth.append(s.location())
        return {
            "task": task, "revision": self.revision,
            "likely_files": evidence,
            "relevant_symbols": [{k: h[k] for k in ("file", "name", "start_line", "end_line", "confidence")} for h in hits[:limit]],
            "architecture_evidence": bounded(frameworks, limit),
            "existing_authentication": bounded(auth, limit) if re.search(r"auth|oauth|login|jwt", task, re.IGNORECASE) else None,
            "relevant_tests": sorted(set(related_tests))[:limit],
            "summaries": [self.summary(f, min(limit, 5)) for f in files[:3]],
            "selection_note": "Likely context, not a completeness or semantic-resolution guarantee.",
            "diagnostics": self.diagnostics[:10],
        }
