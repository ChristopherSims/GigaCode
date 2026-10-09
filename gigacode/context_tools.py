"""Thin, read-only context APIs sharing a deterministic navigation index."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import subprocess
from collections import OrderedDict
from pathlib import Path

from gigacode.navigation_index import NavigationIndex, bounded
from gigacode.path_utils import SourcePathError, validate_buffer_path


class GitContext:
    """Read-only Git operations; revisions are resolved before use as arguments."""

    def __init__(self, root: Path):
        self.root = root
        repo = self.run(["rev-parse", "--show-toplevel"]).strip()
        self.repo = Path(repo).resolve()
        self.prefix = root.relative_to(self.repo).as_posix()
        if self.prefix == ".":
            self.prefix = ""

    def run(self, args: list[str]) -> str:
        result = subprocess.run(
            ["git", "-C", str(getattr(self, "repo", self.root)), "--no-pager", *args], capture_output=True,
            encoding="utf-8", errors="replace", timeout=15,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
        if result.returncode:
            raise ValueError("Git read failed; check repository, revision and file scope.")
        return result.stdout

    def ref(self, ref: str) -> str:
        if not isinstance(ref, str) or not ref or ref.startswith("-") or len(ref) > 200:
            raise ValueError("Use a valid Git revision, not an option.")
        return self.run(["rev-parse", "--verify", "--end-of-options", ref + "^{commit}"]).strip()

    def relative(self, path: str) -> str | None:
        candidate = (self.repo / path).resolve()
        try:
            return candidate.relative_to(self.root).as_posix()
        except ValueError:
            return None

    def path(self, file: str) -> str:
        resolved = validate_buffer_path(file, self.root)
        return resolved.relative_to(self.repo).as_posix()

    def status(self, limit=20, offset=0) -> dict:
        # NUL-delimited porcelain preserves spaces, quotes and rename pairs.
        pieces = self.run(["status", "--porcelain=v1", "-z", "--untracked-files=all"]).split("\0")
        rows = []
        i = 0
        while i < len(pieces):
            entry = pieces[i]
            i += 1
            if not entry:
                continue
            code, path = entry[:2], entry[3:]
            previous = pieces[i] if ("R" in code or "C" in code) and i < len(pieces) else None
            if previous is not None:
                i += 1
            file = self.relative(path)
            if file is not None:
                rows.append({"file": file, "index_status": code[0], "worktree_status": code[1],
                             "previous_file": self.relative(previous) if previous else None})
        try:
            branch = self.run(["symbolic-ref", "--quiet", "--short", "HEAD"]).strip()
        except ValueError:
            branch = self.run(["rev-parse", "--short", "HEAD"]).strip()
        result = {"branch": branch, **bounded(sorted(rows, key=lambda r: r["file"]), limit, offset)}
        selected = result["items"]
        result.update(
            modified=[r["file"] for r in selected if r["worktree_status"] not in {" ", "?"}],
            staged=[r["file"] for r in selected if r["index_status"] not in {" ", "?"}],
            untracked=[r["file"] for r in selected if r["index_status"] == "?"],
            clean=not rows,
        )
        return result

    def blob(self, ref: str, file: str) -> str:
        try:
            return self.run(["show", f"{ref}:{self.path(file)}"])
        except ValueError:
            return ""

    def staged_blob(self, file: str) -> str:
        try:
            return self.run(["show", ":" + self.path(file)])
        except ValueError:
            return ""

    def diff(self, file=None, against="HEAD", staged=False, max_chars=8000) -> dict:
        if against == "STAGED":
            staged = True
            against = "HEAD"
        ref = self.ref(against)
        args = ["diff", "--no-ext-diff", "--no-textconv", "--unified=3"]
        if staged:
            args.append("--cached")
        args.extend([ref, "--", self.path(file) if file else self.prefix or "."])
        text = self.run(args)
        return {"against": ref, "staged": staged, "diff": text[:max_chars], "truncated": len(text) > max_chars}

    def commits(self, file=None, limit=10, offset=0) -> dict:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("limit must be 1..100 and offset must be non-negative.")
        args = ["log", f"--max-count={limit + 1}", f"--skip={offset}",
                "--format=%H%x00%cs%x00%s"]
        args.extend(["--", self.path(file) if file else self.prefix or "."])
        rows = []
        for line in self.run(args).splitlines():
            parts = line.split("\0", 2)
            if len(parts) == 3:
                rows.append({"commit": parts[0], "date": parts[1], "subject": parts[2]})
        return {"items": rows[:limit], "next_offset": offset + limit if len(rows) > limit else None}

    def blame(self, file, line=1, limit=20, commit="HEAD") -> dict:
        if line < 1 or not 1 <= limit <= 100:
            raise ValueError("line must be positive and limit must be 1..100.")
        ref = self.ref(commit)
        text = self.run(["blame", "--no-textconv", "--line-porcelain", "-L", f"{line},+{limit}", ref, "--", self.path(file)])
        entries = []
        current = None
        for value in text.splitlines():
            header = re.match(r"^([0-9a-f]{40,64}) \d+ (\d+)", value)
            if header:
                current = {"commit": header.group(1), "line": int(header.group(2))}
            elif current is not None and value.startswith("summary "):
                current["subject"] = value[8:]
            elif current is not None and value.startswith("\t"):
                # No author/email/source dump by default.
                entries.append(current)
                current = None
        return {"file": file, "items": entries}


class ContextTools:
    """Mixin: public methods operate without bootstrapping an embedding model."""

    def _context_index(self, buffer_id=None, root=None):
        info = None
        if buffer_id is not None:
            info = self._get_buffer_info(buffer_id)
            if info is None:
                raise ValueError("Unknown buffer_id.")
        elif root is None and getattr(self, "_last_buffer_id", None):
            info = self._get_buffer_info(self._last_buffer_id)
        if info:
            configured = Path(info["root"]).resolve()
            if root is not None and Path(root).resolve() != configured:
                raise ValueError("root and buffer_id must identify the same project.")
            root = configured
        if root is None:
            root = getattr(self, "_navigation_root", None)
        if root is None:
            cwd = Path.cwd()
            root = cwd / "project" if (cwd / "project").is_dir() else cwd
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError("root must be an existing project directory.")
        with self._hashline_lock:
            cache = getattr(self, "_navigation_indexes", None)
            if cache is None:
                cache = self._navigation_indexes = OrderedDict()
            key = str(root)
            if key not in cache:
                cache[key] = NavigationIndex(root)
            index = cache[key]
            cache.move_to_end(key)
            while len(cache) > 4:
                cache.popitem(last=False)
            snapshot = self._load_source_snapshot(buffer_id or self._last_buffer_id) if info else {}
            dirty = info.get("dirty_files", {}) if info else {}
            overlay = {f: lines for f, lines in (snapshot or {}).items() if f in dirty}
            index.refresh(overlay)
            self._navigation_root = root
            return index

    def _context_call(self, operation, buffer_id=None, root=None):
        try:
            with self._hashline_lock:
                return {"status": "ok", **operation(self._context_index(buffer_id, root))}
        except SourcePathError as exc:
            return {"status": "error", "code": exc.code, "message": str(exc), "candidates": exc.candidates}
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            return {"status": "error", "message": str(exc)}

    @staticmethod
    def _one_symbol(index, symbol, file=None):
        matches = index.select(symbol, file)
        if not matches:
            raise ValueError("Symbol not found. Use an exact name, qualified name or symbol ID.")
        if len(matches) > 1:
            return None, {"status": "ambiguous", "candidates": bounded([s.location() for s in matches]),
                          "message": "Select a returned symbol ID or provide file; no target was guessed."}
        return matches[0], None

    def code_navigate(self, action, file=None, symbol=None, include_source=False,
                      limit=20, offset=0, buffer_id=None, root=None):
        """Compact structural navigation; bodies are opt-in."""
        def operation(index):
            bounded([], limit, offset)
            if action == "summary":
                return index.summary(file, limit)
            if action in {"dependencies", "dependents"}:
                rows = index.dependency_rows(file, action == "dependents")
                return {"file": index.file_key(file), **bounded(rows, limit, offset)}
            if action == "graph":
                key = index.file_key(file)
                seen, queue, edges = set(), [key], []
                while queue and len(seen) < limit:
                    current = queue.pop(0)
                    if current in seen:
                        continue
                    seen.add(current)
                    for edge in index.outgoing.get(current, []):
                        if len(edges) >= limit * 3:
                            break
                        edges.append(edge)
                        queue.extend(f for f in edge["files"] if f not in seen)
                return {"file": key, "nodes": sorted(seen), "edges": edges,
                        "truncated": bool(queue) or len(edges) >= limit * 3}
            if action == "structure":
                key = index.file_key(file)
                record = index.files[key]
                return {"file": key, "language": record.language, "source_hash": record.hash,
                        "imports": bounded(record.imports, limit),
                        "diagnostics": record.diagnostics,
                        **bounded([s.location() for s in record.symbols], limit, offset)}
            if not symbol:
                raise ValueError("This action requires symbol.")
            selected, error = self._one_symbol(index, symbol, file)
            if error:
                return error
            if action in {"callers", "callees", "references"}:
                return {"symbol": selected.location(), **index.relations(selected, action, limit, offset)}
            if action == "children":
                return {"symbol": selected.location(),
                        **bounded([s.location() for s in index.by_file.get(selected.file, []) if s.parent == selected.id], limit, offset)}
            if action != "symbol":
                raise ValueError("Unknown navigation action.")
            result = selected.location()
            result["source_hash"] = index.files[selected.file].hash
            if include_source:
                lines = index.sources[selected.file].splitlines()
                text = "\n".join(lines[selected.line_start - 1:selected.line_end])
                result.update(source=text[:8000], source_truncated=len(text) > 8000,
                              file_hash=hashlib.sha256(json.dumps(lines, ensure_ascii=False).encode("utf-8")).hexdigest())
            return result
        return self._context_call(operation, buffer_id, root)

    def get_symbol(self, symbol, file=None, buffer_id=None, root=None):
        return self.code_navigate("symbol", file, symbol, buffer_id=buffer_id, root=root)

    def read_symbol(self, file, symbol, include=None, limit=20, buffer_id=None, root=None):
        def operation(index):
            selected, error = self._one_symbol(index, symbol, file)
            if error:
                return error
            result = selected.location()
            lines = index.sources[selected.file].splitlines()
            source = "\n".join(lines[selected.line_start - 1:selected.line_end])
            result.update(source=source[:8000], source_truncated=len(source) > 8000,
                          source_hash=index.files[selected.file].hash,
                          file_hash=hashlib.sha256(json.dumps(lines, ensure_ascii=False).encode("utf-8")).hexdigest())
            for section in include or []:
                if section in {"callers", "callees", "references"}:
                    result[section] = index.relations(selected, section, limit)
                elif section == "dependencies":
                    result[section] = bounded(index.dependency_rows(selected.file), limit)
                else:
                    raise ValueError("include supports callers, callees, references and dependencies.")
            return result
        return self._context_call(operation, buffer_id, root)

    def get_callers(self, symbol, file=None, limit=20, offset=0, buffer_id=None, root=None):
        return self.code_navigate("callers", file, symbol, limit=limit, offset=offset, buffer_id=buffer_id, root=root)

    def get_callees(self, symbol, file=None, limit=20, offset=0, buffer_id=None, root=None):
        return self.code_navigate("callees", file, symbol, limit=limit, offset=offset, buffer_id=buffer_id, root=root)

    def get_children(self, symbol, file=None, limit=20, offset=0, buffer_id=None, root=None):
        return self.code_navigate("children", file, symbol, limit=limit, offset=offset, buffer_id=buffer_id, root=root)

    def get_file_structure(self, file, limit=20, offset=0, buffer_id=None, root=None):
        return self.code_navigate("structure", file, limit=limit, offset=offset, buffer_id=buffer_id, root=root)

    def file_summary(self, file, limit=20, buffer_id=None, root=None):
        return self.code_navigate("summary", file, limit=limit, buffer_id=buffer_id, root=root)

    def dependencies(self, file, limit=20, offset=0, buffer_id=None, root=None):
        return self.code_navigate("dependencies", file, limit=limit, offset=offset, buffer_id=buffer_id, root=root)

    def dependents(self, file, limit=20, offset=0, buffer_id=None, root=None):
        return self.code_navigate("dependents", file, limit=limit, offset=offset, buffer_id=buffer_id, root=root)

    def dependency_graph(self, file, limit=20, buffer_id=None, root=None):
        return self.code_navigate("graph", file, limit=limit, buffer_id=buffer_id, root=root)

    def get_task_context(self, task, limit=10, include_git=True, buffer_id=None, root=None):
        def operation(index):
            if not 1 <= limit <= 20:
                raise ValueError("limit must be 1..20.")
            result = index.task_context(task, limit)
            if include_git:
                try:
                    result["git"] = GitContext(index.root).status(limit)
                except (ValueError, OSError, subprocess.SubprocessError):
                    result["git"] = {"available": False}
            cache_dir = self.work_dir / "navigation" / hashlib.sha256(str(index.root).encode()).hexdigest()[:16]
            index.save_summaries(cache_dir)
            return result
        return self._context_call(operation, buffer_id, root)

    def git_status(self, buffer_id=None, root=None, limit=20, offset=0):
        return self._context_call(lambda index: GitContext(index.root).status(limit, offset), buffer_id, root)

    def changed_files(self, buffer_id=None, root=None, limit=20, offset=0):
        return self.git_status(buffer_id, root, limit, offset)

    def git_diff(self, buffer_id=None, file=None, against="HEAD", staged=False, root=None, max_chars=8000):
        def operation(index):
            if not 100 <= max_chars <= 20_000:
                raise ValueError("max_chars must be 100..20000.")
            if file:
                # Deleted files have no current symbol, but are still root-scoped.
                file_path = validate_buffer_path(file.replace("\\", "/"), index.root).relative_to(index.root).as_posix()
            else:
                file_path = None
            return GitContext(index.root).diff(file_path, against, staged, max_chars)
        return self._context_call(operation, buffer_id, root)

    def recent_commits(self, limit=10, offset=0, buffer_id=None, root=None):
        return self._context_call(lambda index: GitContext(index.root).commits(limit=limit, offset=offset), buffer_id, root)

    def file_history(self, file, limit=10, offset=0, buffer_id=None, root=None):
        return self._context_call(lambda index: GitContext(index.root).commits(
            file=validate_buffer_path(file.replace("\\", "/"), index.root).relative_to(index.root).as_posix(),
            limit=limit, offset=offset), buffer_id, root)

    def blame(self, file, line=1, limit=20, commit="HEAD", buffer_id=None, root=None):
        return self._context_call(lambda index: GitContext(index.root).blame(
            index.file_key(file), line, limit, commit), buffer_id, root)

    def changed_symbols(self, against="HEAD", staged=False, limit=20, offset=0, buffer_id=None, root=None):
        def operation(index):
            from gigacode.navigation_index import parse_file
            git = GitContext(index.root)
            ref = git.ref(against)
            status_result = git.status(100)
            status = status_result["items"]
            # Include committed changes relative to a supplied base, even when clean.
            paths = git.run(["diff", "--no-renames", "--name-only", "-z", ref, "--", git.prefix or "."]).split("\0")
            files = {r["file"] for r in status}
            files.update(r["previous_file"] for r in status if r["previous_file"])
            files.update(f for p in paths if p and (f := git.relative(p)) is not None)
            rows = []
            skipped = []
            for file in sorted(files):
                from gigacode.language_detect import detect_language
                from gigacode.navigation_index import LANGUAGES
                if detect_language(file) not in LANGUAGES and not file.endswith(".tpl"):
                    continue
                if not staged and file not in index.sources and (index.root / file).exists():
                    skipped.append({"file": file, "reason": "Current source is outside the supported index/resource bounds."})
                    continue
                before = git.blob(ref, file)
                after = git.staged_blob(file) if staged else index.sources.get(file, "")
                if before == after:
                    continue
                old = parse_file(file, before)
                new = parse_file(file, after)
                if old.diagnostics or new.diagnostics:
                    skipped.append({"file": file, "reason": "Structural parse is incomplete; no deletion/addition was inferred."})
                    continue
                a, b = before.splitlines(), after.splitlines()
                opcodes = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
                old_symbols = {s.id: s for s in old.symbols}
                new_symbols = {s.id: s for s in new.symbols}
                for id in sorted(set(old_symbols) | set(new_symbols)):
                    left, right = old_symbols.get(id), new_symbols.get(id)
                    touches = any(
                        (left and before_start < left.line_end and before_end >= left.line_start - 1)
                        or (right and after_start < right.line_end and after_end >= right.line_start - 1)
                        for _, before_start, before_end, after_start, after_end in opcodes
                    )
                    if not touches:
                        continue
                    source_left = "\n".join(a[left.line_start - 1:left.line_end]) if left else None
                    source_right = "\n".join(b[right.line_start - 1:right.line_end]) if right else None
                    if source_left == source_right:
                        continue
                    rows.append({"change": "added" if left is None else "deleted" if right is None else "modified",
                                 "before": left.location() if left else None,
                                 "after": right.location() if right else None})
            return {"against": ref, "staged": staged, **bounded(rows, limit, offset),
                    "untracked_input_truncated": status_result["total"] > 100,
                    "skipped_files": bounded(skipped, limit),
                    "note": "Renames appear as deletion/addition unless symbol identity is unchanged."}
        return self._context_call(operation, buffer_id, root)
