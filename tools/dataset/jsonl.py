from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, TextIO


def _open_text(path: Path, mode: str) -> TextIO:
    """Open plain or gzip JSONL using the same UTF-8/newline contract."""
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    # utf-8-sig tolerates PowerShell Set-Content BOM on Windows.
    if path.suffix == ".gz":
        source_open = gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    else:
        source_open = path.open(encoding="utf-8-sig", newline="")
    with source_open as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open_text(path, "wt") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            destination.write("\n")
            count += 1
    return count


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(path, "at") as destination:
        destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        destination.write("\n")
