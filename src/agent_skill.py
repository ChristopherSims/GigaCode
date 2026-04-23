"""GigaCode Agent Skill — language-agnostic code editing assistant.

Uses the GigaCode embedding tool for semantic context, then applies
language-aware refactoring across any supported programming language:

- Documentation comments (tree-sitter or regex)
- Type annotations / signatures
- Error handling (bare catch/except fixes)
- Resource management (context managers)
- Visibility modifiers

Usage:
    python src/agent_skill.py example.py
    python src/agent_skill.py example.js --language javascript

Output:
    example_gigacode_edited.py
    example_gigacode_edited.js
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Allow running directly from repo root
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.cross_language_rules import edit_file
from src.language_detect import detect_language

logger = logging.getLogger(__name__)

# Lazy import of GigaCode tool so the script can still run when heavy
# ML deps are not installed.
_CodeEmbeddingTool = None


def _get_tool() -> Any:
    global _CodeEmbeddingTool
    if _CodeEmbeddingTool is None:
        from src.agent_tool import CodeEmbeddingTool
        _CodeEmbeddingTool = CodeEmbeddingTool
    return _CodeEmbeddingTool


class GigacodeEditAgent:
    """Agent that edits source files in any language using GigaCode context + rules."""

    def __init__(self, work_dir: str = "./buffers") -> None:
        self.work_dir = work_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def edit_file(
        self,
        src_path: str | Path,
        language_hint: str | None = None,
    ) -> dict[str, Any]:
        """Edit a single source file and return the result metadata.

        Args:
            src_path: Path to the source file.
            language_hint: Optional language override.

        Returns:
            Dict with ``language``, ``changed``, ``changes``, ``output_path``.
        """
        src_path = Path(src_path)
        language = detect_language(src_path, hint=language_hint)

        # Step 1: embed via GigaCode tool for semantic context
        buffer_id: str | None = None
        try:
            Tool = _get_tool()
            with Tool(work_dir=self.work_dir, device="cpu") as tool:
                result = tool.embed_codebase(src_path, language_hint=language)
                if result["status"] == "ok":
                    buffer_id = result["buffer_id"]
                    logger.info("[GigaCode] embedded %d lines.", result["token_count"])

                    # Optional: use semantic search to find similar patterns
                    search = tool.semantic_search(
                        buffer_id, "function definition", top_k=3
                    )
                    if search["status"] == "ok":
                        logger.debug(
                            "[GigaCode] semantic context: %s",
                            json.dumps(search["matches"]),
                        )
        except Exception as exc:
            logger.warning("[GigaCode] embedding skipped (%s). Proceeding with rules.", exc)

        # Step 2: apply language-agnostic editing rules
        edit_result = edit_file(src_path, language_hint=language)

        # Step 3: write output
        stem = src_path.stem
        suffix = src_path.suffix
        out_name = f"{stem}_gigacode_edited{suffix}"
        out_path = src_path.with_name(out_name)
        out_path.write_text(edit_result.modified, encoding="utf-8")
        logger.info("[GigaCode] written: %s", out_path)

        return {
            "status": "ok",
            "language": edit_result.language,
            "changed": edit_result.changed,
            "changes": edit_result.changes,
            "output_path": str(out_path),
            "buffer_id": buffer_id,
        }

    def edit_file_commit(
        self,
        src_path: str | Path,
        language_hint: str | None = None,
    ) -> dict[str, Any]:
        """Edit a single source file and overwrite the original in-place.

        Uses the full agent workflow: embed -> read -> apply rules -> write -> commit.

        Args:
            src_path: Path to the source file.
            language_hint: Optional language override.

        Returns:
            Dict with ``language``, ``changed``, ``changes``, ``committed_path``.
        """
        src_path = Path(src_path)
        language = detect_language(src_path, hint=language_hint)
        rel_file = src_path.name

        # Step 1: embed
        Tool = _get_tool()
        with Tool(work_dir=self.work_dir, device="cpu") as tool:
            embed_result = tool.embed_codebase(src_path, language_hint=language)
            if embed_result["status"] != "ok":
                return {
                    "status": "error",
                    "message": f"Embedding failed: {embed_result.get('message')}",
                }
            buffer_id = embed_result["buffer_id"]
            logger.info("[GigaCode] embedded %s -> %s", src_path, buffer_id)

            # Step 2: read current source from buffer
            read_result = tool.read_code(buffer_id, file=rel_file)
            if read_result["status"] != "ok":
                return {
                    "status": "error",
                    "message": f"Read failed: {read_result.get('message')}",
                }
            current_lines = read_result["lines"]
            current_source = "\n".join(current_lines)

            # Step 3: apply language-agnostic editing rules
            from src.cross_language_rules import EditResult

            edit_result = edit_file(src_path, language_hint=language)
            if not edit_result.changed:
                return {
                    "status": "ok",
                    "changed": False,
                    "committed_path": str(src_path),
                    "message": "No changes needed.",
                }

            new_lines = edit_result.modified.splitlines()

            # Step 4: write back to buffer
            write_result = tool.write_code(
                buffer_id,
                file=rel_file,
                start_line=1,
                new_lines=new_lines,
            )
            if write_result["status"] != "ok":
                return {
                    "status": "error",
                    "message": f"Write failed: {write_result.get('message')}",
                }

            # Step 5: commit to disk
            commit_result = tool.commit(buffer_id, dry_run=False)
            if commit_result["status"] != "ok":
                return {
                    "status": "error",
                    "message": f"Commit failed: {commit_result.get('message')}",
                }

            return {
                "status": "ok",
                "language": edit_result.language,
                "changed": True,
                "changes": edit_result.changes,
                "committed_path": str(src_path),
                "buffer_id": buffer_id,
            }

    def edit_directory(
        self,
        dir_path: str | Path,
        pattern: str = "*",
        language_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        """Edit all matching files in a directory.

        Args:
            dir_path: Directory to scan.
            pattern: Glob pattern for files.
            language_hint: Optional language override for all files.

        Returns:
            List of result dicts, one per file.
        """
        dir_path = Path(dir_path)
        results: list[dict[str, Any]] = []
        for f in sorted(dir_path.rglob(pattern)):
            if f.is_file() and not f.name.endswith("_gigacode_edited" + f.suffix):
                try:
                    result = self.edit_file(f, language_hint=language_hint)
                    results.append(result)
                except Exception as exc:
                    logger.error("Failed to edit %s: %s", f, exc)
                    results.append(
                        {
                            "status": "error",
                            "path": str(f),
                            "message": str(exc),
                        }
                    )
        return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GigaCode Agent Skill — edit source files in any language"
    )
    parser.add_argument("input", help="Source file or directory to edit")
    parser.add_argument(
        "--language",
        "-l",
        dest="language_hint",
        default=None,
        help="Override language detection (e.g. python, javascript, rust)",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default="*",
        help="Glob pattern when input is a directory (default '*')",
    )
    parser.add_argument(
        "--work-dir",
        "-w",
        default="./buffers",
        help="Working directory for embedding buffers",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--commit",
        "-c",
        action="store_true",
        help="Overwrite original files in-place instead of writing *_gigacode_edited copies",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    agent = GigacodeEditAgent(work_dir=args.work_dir)
    input_path = Path(args.input)

    edit_method = agent.edit_file_commit if args.commit else agent.edit_file

    if input_path.is_dir():
        results = agent.edit_directory(
            input_path, pattern=args.pattern, language_hint=args.language_hint
        )
        changed = sum(1 for r in results if r.get("changed"))
        print(f"Edited {len(results)} files, {changed} changed.")
        for r in results:
            if r.get("status") == "error":
                print(f"  ERROR {r['path']}: {r['message']}")
            elif r.get("changed"):
                out_key = "committed_path" if args.commit else "output_path"
                print(f"  EDITED {r[out_key]} ({len(r.get('changes', []))} changes)")
    else:
        result = edit_method(input_path, language_hint=args.language_hint)
        if result.get("status") == "error":
            print(f"ERROR: {result['message']}")
        elif result["changed"]:
            out_key = "committed_path" if args.commit else "output_path"
            print(f"Edited -> {result[out_key]}")
            for ch in result.get("changes", []):
                print(f"  Line {ch['line']}: {ch['description']}")
        else:
            print("No changes needed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
