"""Reusable definition-first lexical retrieval for the coding loop."""

from __future__ import annotations

import ast
import re
from typing import Any

from gigacode.coding_safety import DEFINITIONS
from gigacode.lexical_index import LexicalIndex, _tokenize

_STOP_WORDS = frozenset(
    "a an the that this it its of to for from by in into on at with and or "
    "is are was be as which where there somewhere currently given offers "
    "helper helpers function functions method methods class internal small tiny "
    "find locate add argument checking clear message valueerror typeerror".split()
)
_ALIASES = {
    "boolean": "bool", "booleans": "bool", "missing": "none",
    "hexadecimal": "hex", "hexadecimalcolor": "hex color",
    "string": "str string", "strips": "strip remove", "strip": "strip remove",
    "fractional": "fraction", "weight": "weight fade", "mixes": "mix blend",
    "mix": "mix blend", "mixing": "mix blend",
    "formats": "format", "formatting": "format",
    "colors": "color", "colours": "color", "colour": "color",
    "readers": "reader", "assembles": "assemble",
    "matches": "match", "matching": "match",
    "removes": "remove", "removing": "remove",
    "parsing": "parse", "parser": "parse",
    "sizes": "size",
    "authentication": "auth", "authorization": "auth", "oauth2": "oauth",
}
_AUXILIARY = frozenset({"examples", "example", "benchmarks", "benchmark", "tests", "test", "docs"})


def query_terms(query: str, deduplicate: bool = True) -> list[str]:
    terms = []
    for token in _tokenize(query):
        if token not in _STOP_WORDS:
            terms.extend(_ALIASES.get(token, token).split())
    if "file" in terms and "size" in terms:
        terms.append("filesize")
    if "substring" in terms and re.search(r"\b(?:end|ending|tail)\b", query.lower()):
        terms.append("suffix")
    if "substring" in terms and re.search(r"\b(?:start|beginning|head)\b", query.lower()):
        terms.append("prefix")
    if re.search(r"\bhuman[\s_-]+readable\b", query.lower()):
        terms.append("humanreadable")
    return list(dict.fromkeys(terms)) if deduplicate else terms


class SourceSearchIndex:
    """Index actual definitions, keeping declarations and file provenance."""

    def __init__(self, snapshot: dict[str, list[str]]):
        self.entries: list[dict[str, Any]] = []
        self.lexical = LexicalIndex()
        for file, lines in snapshot.items():
            text = "\n".join(lines)
            if file.endswith(".py"):
                try:
                    tree = ast.parse(text)
                except SyntaxError:
                    tree = None
                if tree is not None:
                    self._add_python(tree, file, lines, module_doc=ast.get_docstring(tree) or "")
                    continue
            # Preserve retrieval for other languages and temporarily invalid
            # Python without claiming that a whole file is a definition.
            for start in range(1, len(lines) + 1, 60):
                end = min(start + 59, len(lines))
                self._add(file, start, end, "", "", "\n".join(lines[start - 1:end]), "window")

    def _add_python(self, tree: ast.AST, file: str, lines: list[str], parent: str = "", module_doc: str = "", in_class: bool = False):
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, DEFINITIONS):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                end = node.end_lineno or node.lineno
                name = f"{parent}.{node.name}" if parent else node.name
                doc = ast.get_docstring(node, clean=False) or ""
                body = "\n".join(lines[start - 1:end])
                kind = "class" if isinstance(node, ast.ClassDef) else "definition"
                # Class bodies contain many unrelated child methods. Index
                # their own header/docs, then index children separately.
                if kind == "class":
                    body = lines[node.lineno - 1] + "\n" + doc
                behaviors = set()
                for child in (ast.walk(node) if kind != "class" else ()):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                        if child.func.attr in {"endswith", "removesuffix"}:
                            behaviors.update(("suffix", "end", "substring", "match"))
                        elif child.func.attr in {"startswith", "removeprefix"}:
                            behaviors.update(("prefix", "start", "substring", "match"))
                self._add(file, start, end, name, doc, body, kind, module_doc,
                          in_class and kind == "definition", " ".join(sorted(behaviors)) if kind != "class" else "")
                self._add_python(node, file, lines, name, module_doc, kind == "class")
            else:
                self._add_python(node, file, lines, parent, module_doc, in_class)

    def _add(self, file, start, end, name, doc, body, kind, module_doc="", is_method=False, behaviors=""):
        canonical = file.replace("\\", "/")
        parts = canonical.lower().split("/")
        stem = parts[-1].rsplit(".", 1)[0]
        declaration = body.splitlines()[0] if body else ""
        name_terms = set(query_terms(name + " " + canonical))
        body_terms = set(query_terms(doc + " " + body[:2400] + " " + module_doc[:1200] + " " + behaviors))
        self.entries.append({
            "file": file, "start_line": start, "end_line": end, "name": name,
            "signature": declaration, "kind": kind,
            "definition_match": kind in {"definition", "class"},
            "_name_terms": name_terms, "_body_terms": body_terms,
            "_module_terms": set(query_terms(stem)), "_is_method": is_method,
            "_auxiliary": bool(set(parts[:-1]) & _AUXILIARY) or stem in _AUXILIARY
            or stem.startswith(("test_", "bench_", "benchmark_", "example_")),
        })
        self.lexical.add(
            len(self.entries) - 1,
            " ".join(query_terms(
                (name + " ") * 4 + (canonical + " ") * 2 + (doc + " ") * 2 + body[:2400] + " " + module_doc[:1200] + " " + behaviors,
                deduplicate=False,
            )),
        )

    def search(self, query: str, top_k: int = 5, file: str | None = None) -> dict[str, Any]:
        terms = query_terms(query)
        if not terms:
            return {"status": "ok", "mode": "definitions", "matches": [], "hint": "Use a symbol, filename or specific behavior."}
        query_set = set(terms)
        # Ranking is internal; only the compact selected candidates are exposed.
        hits = self.lexical.search(" ".join(terms), top_k=len(self.entries))
        matches = []
        auxiliary_requested = bool(re.search(r"\b(?:benchmark|test_[a-z]|example script|example file|test function)\b", query.lower()))
        for hit in hits:
            entry = self.entries[hit["doc_id"]]
            if file is not None and entry["file"] != file:
                continue
            name_overlap = len(query_set & entry["_name_terms"])
            overlap = len(query_set & (entry["_name_terms"] | entry["_body_terms"]))
            if not overlap:
                continue
            score = hit["score"] * (1 + 0.75 * name_overlap)
            if entry["_module_terms"] and entry["_module_terms"] <= query_set:
                score *= 1.75
            # A compound domain match is stronger than one generic component,
            # e.g. filesize rather than File.readable's isolated "file".
            if "filesize" in query_set & entry["_module_terms"]:
                score *= 2
            if re.search(r"\bmethod\b", query.lower()) and entry["kind"] == "definition":
                score *= 1.5 if entry["_is_method"] else 0.5
            if entry["kind"] == "class" and re.search(r"\b(?:helper|function|method)\b", query.lower()):
                score *= 0.35
            if "internal" in query.lower().split() and entry["name"].rsplit(".", 1)[-1].startswith("_"):
                score *= 2
            elif "helper" in query.lower().split() and entry["name"].rsplit(".", 1)[-1].startswith("_"):
                score *= 1.25
            if entry["_auxiliary"] and file is None and not auxiliary_requested:
                score *= 0.2
            match = {k: v for k, v in entry.items() if not k.startswith("_")}
            match.update(
                score=round(score, 4), matched_terms=overlap,
                confidence="candidate",
            )
            matches.append(match)
        matches.sort(key=lambda m: (-m["score"], m["end_line"] - m["start_line"], m["file"]))
        if matches:
            first = matches[0]
            ambiguous = (
                len(matches) > 1 and matches[1]["score"] >= first["score"] * 0.9
                and (matches[1]["file"], matches[1]["start_line"]) != (first["file"], first["start_line"])
            )
            first["confidence"] = "high" if first["matched_terms"] >= 2 and not ambiguous else "uncertain"
        return {
            "status": "ok", "mode": "definitions", "matches": matches[:top_k],
            "selection_hint": "Read the returned declaration; uncertain candidates require target confirmation.",
        }

    def qualified_matches(self, symbol: str, top_k: int, file: str | None = None) -> dict[str, Any] | None:
        """Resolve a class.method or module.class.method without semantic guessing."""
        matches = []
        for entry in self.entries:
            if file is not None and entry["file"] != file:
                continue
            module = entry["file"].replace("\\", "/").removesuffix(".py").replace("/", ".").removesuffix(".__init__")
            if symbol not in {entry["name"], module + "." + entry["name"]}:
                continue
            match = {k: v for k, v in entry.items() if not k.startswith("_")}
            match.update(score=0.2 if entry["_auxiliary"] and file is None else 1.0, confidence="exact")
            matches.append(match)
        if not matches:
            return None
        matches.sort(key=lambda match: (-match["score"], match["file"]))
        return {"status": "ok", "mode": "exact", "matches": matches[:top_k]}
