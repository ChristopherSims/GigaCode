"""Read/write token_metadata.json files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_INDENT = 2


def save_metadata(
    path: str | Path, metadata_list: list[dict[str, Any]], *, compact: bool = True
) -> None:
    """Persist a list of per-line metadata records to JSON.

    Args:
        path: Destination file path.
        metadata_list: List of metadata dicts (typically from
            :func:`flatten_embeddings`).
        compact: If ``True`` (default) write minified JSON for speed and
            smaller files. Set to ``False`` for human-readable output.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "count": len(metadata_list),
        "lines": metadata_list,
    }
    kwargs = {"ensure_ascii": False}
    if compact:
        kwargs["separators"] = (",", ":")
    else:
        kwargs["indent"] = _DEFAULT_INDENT
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, **kwargs)
    logger.info("Wrote metadata (%d lines) to %s", len(metadata_list), p)


def load_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Load per-line metadata records from JSON.

    Args:
        path: Source file path.

    Returns:
        The list of line metadata dicts.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the JSON schema is unrecognised.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metadata file not found: {p}")

    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, list):
        # Legacy / simple format
        logger.debug("Loaded legacy metadata list from %s", p)
        return payload

    if isinstance(payload, dict):
        version = payload.get("version", 1)
        if version == 1:
            lines = payload.get("lines", [])
            logger.debug("Loaded metadata v1 (%d lines) from %s", len(lines), p)
            return lines
        raise ValueError(f"Unsupported metadata version: {version}")

    raise ValueError("Metadata JSON must be a list or a dict")
