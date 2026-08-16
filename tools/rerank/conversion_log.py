"""Conversion log helpers (Phase 3).

Schema source of truth: tools.rerank.phase3_hook.CONVERSION_LOG_SCHEMA
Also written to artifacts/rerank/conversion_log_schema_v1.json by scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.rerank.phase3_hook import CONVERSION_LOG_SCHEMA, to_conversion_log_row


def write_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(CONVERSION_LOG_SCHEMA, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_row(row: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    for k in CONVERSION_LOG_SCHEMA["required"]:
        if k not in row or row[k] in (None, ""):
            errs.append(f"missing:{k}")
    if "nbest" in row and not isinstance(row["nbest"], list):
        errs.append("nbest_not_list")
    return errs


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    errs = validate_row(row)
    if errs:
        raise ValueError(f"invalid conversion log row: {errs}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


__all__ = [
    "CONVERSION_LOG_SCHEMA",
    "write_schema",
    "validate_row",
    "append_jsonl",
    "to_conversion_log_row",
]
