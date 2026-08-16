"""Inference-time rerank guards (NEXT_TASK_USAGE_GUARD).

Skip the model and keep Mozc order when any of:
  1. reading not in the context-sensitive whitelist (strict mode only)
  2. reading length (Unicode code points) <= 2
  3. cleaned context is empty or digits/symbols only
  4. (post-score) overwrite target is junk (halfwidth kana / katakana-only / kyujitai)

Python is the source of truth. Keep mozc_compat/rerank_guard.cc in parity.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet

from tools.rerank.context_clip import clean_context, normalize_reading

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ELIGIBLE_PATH = ROOT / "artifacts/rerank/rerank_eligible_readings.json"
FALLBACK_ELIGIBLE_INC = ROOT / "mozc_compat/rerank_eligible_readings.inc"

# Reason codes — must match C++ rerank_guard.cc
REASON_READING_TOO_SHORT = "reading_too_short"
REASON_CONTEXT_EMPTY_OR_SYMBOL = "context_empty_or_symbol"
REASON_READING_NOT_ELIGIBLE = "reading_not_eligible"
REASON_JUNK_CANDIDATE = "junk_candidate"

MAX_SHORT_READING_CP = 2
GUARD_MODE_STRICT = "strict"
GUARD_MODE_SAFETY = "safety"

# Kyujitai / rare forms that the 30m model promoted on the usage log.
_KYUJITAI_CHARS = frozenset("實舊讃與學體廣應藝縣擧瀧靜")


def guards_enabled() -> bool:
    v = (os.environ.get("MOZC_RERANK_GUARD") or "1").strip().lower()
    return v not in {"0", "false", "no", "off"}


def guard_mode() -> str:
    """Return strict (legacy allowlist) or safety (general personalized model)."""
    value = (os.environ.get("MOZC_RERANK_GUARD_MODE") or GUARD_MODE_STRICT).strip().lower()
    return GUARD_MODE_SAFETY if value == GUARD_MODE_SAFETY else GUARD_MODE_STRICT


def _codepoint_len(text: str) -> int:
    return len(text or "")


def has_linguistic_content(text: str) -> bool:
    """True if any hiragana, katakana, CJK ideograph, or Latin letter is present."""
    for c in text or "":
        o = ord(c)
        if 0x3040 <= o <= 0x309F:  # hiragana
            return True
        if 0x30A0 <= o <= 0x30FF:  # katakana
            return True
        if 0x31F0 <= o <= 0x31FF:  # katakana phonetic extensions
            return True
        if 0x3400 <= o <= 0x9FFF:  # CJK
            return True
        if 0xF900 <= o <= 0xFAFF:  # CJK compatibility ideographs
            return True
        if 0xFF66 <= o <= 0xFF9D:  # halfwidth katakana letters
            return True
        if c.isalpha() and not c.isdigit():
            return True
    return False


def context_empty_or_symbol(context_prev: str, *, already_cleaned: bool = False) -> bool:
    ctx = context_prev if already_cleaned else clean_context(context_prev or "")
    if not ctx or not ctx.strip():
        return True
    return not has_linguistic_content(ctx)


def is_junk_surface(surface: str) -> bool:
    s = surface or ""
    if not s:
        return True
    if any(c in _KYUJITAI_CHARS for c in s):
        return True
    if all(0xFF61 <= ord(c) <= 0xFF9F for c in s):
        return True
    kana = 0
    other = 0
    for c in s:
        o = ord(c)
        if c == "ー" or 0x30A0 <= o <= 0x30FF or 0xFF66 <= o <= 0xFF9D:
            kana += 1
        elif c.isspace():
            continue
        else:
            other += 1
    return kana > 0 and other == 0


@lru_cache(maxsize=4)
def load_eligible_readings(path: str | None = None) -> FrozenSet[str]:
    env = os.environ.get("MOZC_RERANK_ELIGIBLE")
    p = Path(path) if path else (Path(env) if env else DEFAULT_ELIGIBLE_PATH)
    if not p.is_file():
        if path or env or not FALLBACK_ELIGIBLE_INC.is_file():
            return frozenset()
        text = FALLBACK_ELIGIBLE_INC.read_text(encoding="utf-8")
        return frozenset(re.findall(r'^\s+"([^"]+)",$', text, flags=re.MULTILINE))
    blob = json.loads(p.read_text(encoding="utf-8"))
    readings = blob.get("readings") or blob.get("eligible") or []
    return frozenset(str(r) for r in readings if r)


def is_eligible_reading(reading: str, eligible: FrozenSet[str] | None = None) -> bool:
    s = normalize_reading(reading or "")
    pool = eligible if eligible is not None else load_eligible_readings()
    return s in pool


def skip_reason(
    reading: str,
    context_prev: str,
    *,
    eligible: FrozenSet[str] | None = None,
    already_cleaned: bool = False,
    enabled: bool | None = None,
    mode: str | None = None,
) -> str | None:
    """Return a reason code to skip scoring, or None to call the model."""
    if enabled is None:
        enabled = guards_enabled()
    if not enabled:
        return None
    r = normalize_reading(reading or "")
    if not r:
        return REASON_READING_TOO_SHORT
    if _codepoint_len(r) <= MAX_SHORT_READING_CP:
        return REASON_READING_TOO_SHORT
    if context_empty_or_symbol(context_prev or "", already_cleaned=already_cleaned):
        return REASON_CONTEXT_EMPTY_OR_SYMBOL
    active_mode = (mode or guard_mode()).strip().lower()
    if active_mode != GUARD_MODE_SAFETY and not is_eligible_reading(r, eligible):
        return REASON_READING_NOT_ELIGIBLE
    return None


def apply_post_score_guard(
    *,
    overwritten: bool,
    final_top1: str,
    mozc_top1: str,
    enabled: bool | None = None,
) -> tuple[bool, str, str | None]:
    """If overwrite target is junk, revert to Mozc.

    Returns (overwritten, final_top1, extra_reason_or_None).
    """
    if enabled is None:
        enabled = guards_enabled()
    if not enabled or not overwritten:
        return overwritten, final_top1, None
    if is_junk_surface(final_top1) and final_top1 != mozc_top1:
        return False, mozc_top1, REASON_JUNK_CANDIDATE
    return overwritten, final_top1, None
