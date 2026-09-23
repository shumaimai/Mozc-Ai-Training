"""Canonical Phase 0 input contract shared by audit tools and fixtures.

This module deliberately exposes both the historical v1 formatter and the
v2/runtime formatter.  Callers must select one explicitly; no silent fallback
is allowed in audit output.
"""
from __future__ import annotations

from dataclasses import dataclass

FORMAT_VERSION = "contextual-ranking-v2-format-v1"


def format_v1_train(reading: str, context: str, candidate: str) -> str:
    parts = [f"読み: {reading}"]
    if context:
        parts.append(f"文脈: {context}")
    parts.append(f"候補: {candidate}")
    return " [SEP] ".join(parts)


def format_v1_runtime(reading: str, context: str, candidate: str) -> str:
    return f"読み: {reading}\n文脈: {context}\n候補: {candidate}"


def format_v2(reading: str, context: str, candidate: str) -> str:
    """Canonical v2 wire format; versioned for train/eval/runtime parity."""
    return format_v1_runtime(reading, context, candidate)


@dataclass(frozen=True)
class ContractCase:
    case_id: str
    reading: str
    raw_preceding_text: str
    cleaned_context: str
    candidates: tuple[str, ...]
    target_segment: int = 0
