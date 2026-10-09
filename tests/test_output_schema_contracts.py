"""Payload contract tests: real tool responses must match declared output schemas.

Guards against the MCP schema drift that previously caused strict clients to
reject whole tool calls (see suggestions.md R1)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from gigacode.gigacode_tool import CodeEmbeddingTool
from gigacode.token_tools import line_anchor
from gigacode.tool_schema import get_schema

REPO_ROOT = Path(__file__).resolve().parent.parent


class DeterministicEmbedder:
    """Feature-hashing embedder (no torch/sentence-transformers needed)."""

    def __init__(self, dim: int = 96) -> None:
        import hashlib

        self._dim = dim
        self._hashlib = hashlib

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def _featurize(self, text: str) -> dict[int, float]:
        feats: dict[int, float] = {}
        words = re.findall(r"\w+", str(text).lower())
        for w in words:
            h = int.from_bytes(
                self._hashlib.md5(("u:" + w).encode("utf-8")).digest()[:8], "little"
            )
            feats[h % self._dim] = feats.get(h % self._dim, 0.0) + 1.0
        return feats

    def encode(self, texts, batch_size: int = 32, **_kwargs) -> np.ndarray:
        rows = []
        for text in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            for idx, val in self._featurize(text).items():
                vec[idx] = val
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            rows.append(vec)
        if not rows:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.vstack(rows).astype(np.float32)


@pytest.fixture(scope="module")
def tool(tmp_path_factory) -> CodeEmbeddingTool:
    work_dir = tmp_path_factory.mktemp("contract_buffers")
    source = tmp_path_factory.mktemp("contract_source")
    shutil.copytree(
        REPO_ROOT / "examplecode", source / "examplecode", dirs_exist_ok=True
    )
    t = CodeEmbeddingTool(
        work_dir=work_dir,
        device="cpu",
        use_gpu=False,
        embedder=DeterministicEmbedder(),
        tool_profile="full",
    )
    result = t.embed_codebase(source / "examplecode", pattern="*.py")
    assert result["status"] == "ok"
    t._contract_buffer_id = result["buffer_id"]
    yield t
    t.close()


def _invoke(tool: CodeEmbeddingTool, name: str) -> dict[str, Any]:
    buf = tool._contract_buffer_id
    calls: dict[str, dict[str, Any]] = {
        "embed_codebase": {"path": tool._buffer_manager._registry[buf]["root"]},
        "semantic_search": {"buffer_id": buf, "query": "quick sort", "top_k": 3},
        "hybrid_search": {"buffer_id": buf, "query": "quick sort", "top_k": 3},
        "search_for": {"buffer_id": buf, "query": "def ", "max_results": 5},
        "search_symbols": {"buffer_id": buf, "query": "sort", "top_k": 3},
        "code_search": {"buffer_id": buf, "query": "moving average", "mode": "hybrid"},
        "look_for_file": {"buffer_id": buf, "file_name": "math_utils"},
        "read_code": {"buffer_id": buf, "file": "math_utils.py", "start_line": 1},
        "write_code": {
            "buffer_id": buf,
            "file": "string_utils.py",
            "start_line": 10,
            "end_line": 11,
            "new_lines": ["def contract_probe():", '    return "probe"'],
        },
        "commit": {"buffer_id": buf, "dry_run": True},
        "diff": {"buffer_id": buf},
        "semantic_search_streaming": {
            "buffer_id": buf,
            "query": "sort",
            "top_k": 3,
            "disclosure": "signatures",
        },
        "expand_match": {"buffer_id": buf, "match_id": 0, "level": "details"},
        "read_hashlines": {"buffer_id": buf, "file": "math_utils.py", "start_line": 1},
        "edit_hashlines": {
            "buffer_id": buf,
            "file": "math_utils.py",
            "start_anchor": line_anchor(1, screader(tool, buf, "math_utils.py", 1)),
            "end_anchor": line_anchor(1, screader(tool, buf, "math_utils.py", 1)),
            "new_lines": ['"""Anchored contract probe."""'],
            "expected_hash": file_digest(tool, buf, "math_utils.py"),
        },
        "compress_context": {
            "messages": [
                {"role": "user", "content": "probe " + "p" * 1200},
                {"role": "assistant", "tool_calls": [{"function": {"name": "read_code"}}]},
                {"role": "tool", "tool_call_id": "c1", "name": "read_code", "content": "q" * 2000},
                {"role": "user", "content": "last"},
            ],
            "keep_recent_turns": 1,
        },
        "tool_search": {"query": "search"},
    }
    method = getattr(tool, name)
    result = method(**calls[name])
    assert isinstance(result, dict), f"{name} returned non-dict"
    return result


def screader(tool: CodeEmbeddingTool, buf: str, file: str, lineno: int) -> str:
    lines = tool.tool_call(
        "read_code", {"buffer_id": buf, "file": file, "start_line": lineno, "end_line": lineno + 1}
    ).get("lines") or [""]
    return lines[0]


def file_digest(tool: CodeEmbeddingTool, buf: str, file: str) -> str:
    import hashlib
    import json as _json

    snapshot = tool._load_source_snapshot(buf) or {}
    return hashlib.sha256(
        _json.dumps(snapshot.get(file, []), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    "name",
    [
        "embed_codebase",
        "semantic_search",
        "hybrid_search",
        "search_for",
        "search_symbols",
        "code_search",
        "look_for_file",
        "read_code",
        "write_code",
        "commit",
        "diff",
        "semantic_search_streaming",
        "expand_match",
        "read_hashlines",
        "edit_hashlines",
        "compress_context",
        "tool_search",
    ],
)
def test_response_matches_declared_output_schema(tool: CodeEmbeddingTool, name: str) -> None:
    schema = get_schema(name)
    assert schema is not None, f"missing schema for {name}"
    output_schema = schema.get("output_schema")
    assert isinstance(output_schema, dict), f"{name} has no output_schema"
    response = _invoke(tool, name)
    validator = Draft202012Validator(output_schema)
    errors = sorted(validator.iter_errors(response), key=lambda e: e.json_path)
    assert not errors, [f"{e.json_path}: {e.message}" for e in errors[:5]]


def test_write_code_then_commit_status_flow(tool: CodeEmbeddingTool) -> None:
    buf = tool._contract_buffer_id
    write = _invoke(tool, "write_code")
    assert write["status"] == "ok"
    assert write["buffer_state"] == "dirty"
    commit = tool.commit(buf, dry_run=True)
    assert commit["status"] in ("ok", "conflict")
