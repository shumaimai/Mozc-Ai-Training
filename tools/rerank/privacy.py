"""Privacy gates shared by cloud training launchers.

Modal inputs must be copied deliberately into data/public/rerank_ctx.  Paths
that look like personal usage data are rejected both before spawn and inside
the remote function.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

_SENSITIVE = re.compile(
    r"private|personal|usage|ime[-_]?usage|conversion[-_]?log|"
    r"rerank[-_]?requests|chat|secret",
    re.IGNORECASE,
)


def ensure_public_modal_paths(*paths: str, datasets: bool = False) -> None:
    """Reject paths that could expose personal data to a cloud worker."""

    for raw in paths:
        for value in str(raw or "").split(","):
            value = value.strip().replace("\\", "/")
            if not value:
                continue
            normalized = str(PurePosixPath(value))
            if _SENSITIVE.search(normalized) or normalized.lower().endswith(
                (".log", ".jsonl.log")
            ):
                raise ValueError(f"refusing sensitive Modal path: {value}")
            if datasets and not normalized.startswith("data/rerank_ctx/"):
                raise ValueError(
                    "Modal datasets must come from the staged public directory "
                    f"data/public/rerank_ctx (remote path data/rerank_ctx): {value}"
                )
