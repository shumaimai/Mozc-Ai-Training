"""Shared context_prev clipping/cleaning for train-time and inference-time.

PLAN_CONTEXTUAL_RERANKER.md §2.2 + NEXT_TASK_CTX_SUBSET_FIX.md:
  Normalize whitespace, strip wiki markup, cut at previous sentence end
  (。！？), keep only the in-progress sentence, then clip to last max_chars.

`clean_context` is the single shared implementation used by dataset build
and the inference hook. Do not reimplement elsewhere.
"""

from __future__ import annotations

import re
import unicodedata

_SENT_END = re.compile(r"[。！？!?]")
_WIKI_EDIT = re.compile(r"\[\s*(?:edit|編集)\s*\]", re.IGNORECASE)
_WIKI_HEADING = re.compile(r"={2,}[^=\n]*={2,}")
_WIKI_LINK = re.compile(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]")
_WIKI_REF = re.compile(r"\[\d+\]|<ref\b[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
_BULLET_MARK = re.compile(r"(?:^|\s)[*＊#＃・]+(?=\s|$)")
_EQ_RUN = re.compile(r"={2,}")
_MULTI_SPACE = re.compile(r"[ \t\u3000]+")
_KANJI = re.compile(r"[\u4e00-\u9fff]")
# emoji / symbol blocks + Greek / Cyrillic leftovers often seen in Mozc garbage
_NBEST_GARBAGE = re.compile(
    r"["
    r"\U0001F300-\U0001FAFF"  # emoji / symbols
    r"\u2600-\u27BF"  # misc symbols
    r"\u2190-\u21FF"  # arrows
    r"\u0400-\u04FF"  # Cyrillic
    r"\u0370-\u03FF"  # Greek
    r"]"
)


def clean_context(text: str, max_chars: int = 50) -> str:
    """Normalize + strip markup + sentence-boundary clip for IME left-context.

    Shared by dataset assembly and inference. Empty string means "no usable
    context" (exclude from context_sensitive aggregation; ok as anchor).
    """
    if not text:
        return ""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", " ").replace("\t", " ")
    s = _WIKI_EDIT.sub("", s)
    s = _WIKI_HEADING.sub(" ", s)
    s = _WIKI_LINK.sub(r"\1", s)
    s = _WIKI_REF.sub("", s)
    s = _EQ_RUN.sub(" ", s)
    s = _BULLET_MARK.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip(" \u3000")
    if not s:
        return ""
    last = -1
    for m in _SENT_END.finditer(s):
        last = m.end()
    sentence = s[last:] if last >= 0 else s
    sentence = sentence.lstrip(" \u3000")
    if not sentence:
        return ""
    if len(sentence) <= max_chars:
        return sentence
    return sentence[-max_chars:]


def katakana_to_hiragana(text: str) -> str:
    out: list[str] = []
    for ch in text:
        o = ord(ch)
        if 0x30A1 <= o <= 0x30F6:
            out.append(chr(o - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def normalize_reading(text: str) -> str:
    """Hiragana + NFKC for Mozc keys (NOT applied to context_prev)."""
    if not text:
        return ""
    return katakana_to_hiragana(unicodedata.normalize("NFKC", str(text)))


def clip_context_prev(full_text: str, token_char_start: int, max_chars: int = 50) -> str:
    """Return left-context for IME-style conversion at token_char_start.

    Only *preceding* text is used (no future tokens). Delegates final
    normalization / sentence clip / length limit to :func:`clean_context`.
    """
    if token_char_start <= 0 or not full_text:
        return ""
    left = full_text[:token_char_start]
    return clean_context(left, max_chars=max_chars)


def has_kanji(text: str) -> bool:
    return bool(_KANJI.search(text or ""))


def sanitize_nbest(cands: list[str]) -> list[str]:
    """Drop emoji / obvious non-Japanese garbage from Mozc N-best (hygiene)."""
    out: list[str] = []
    seen: set[str] = set()
    for c in cands or []:
        if not c or c in seen:
            continue
        if _NBEST_GARBAGE.search(c):
            continue
        # pure punctuation / symbol-only
        if re.fullmatch(r"[\W_]+", c, flags=re.UNICODE) and not has_kanji(c):
            if not re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", c):
                continue
        seen.add(c)
        out.append(c)
    return out
