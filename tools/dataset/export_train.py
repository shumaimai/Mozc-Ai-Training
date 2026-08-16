"""Export DeepSeek/Qwen accept reviews into IME LoRA training JSONL."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from .jsonl import write_jsonl

ACCEPT = "accept"
_KANJI_RE = re.compile(r"[\u4e00-\u9fff]")


def build_ime_prompt(
    reading: str,
    candidates: list[str],
    context: list[str],
) -> str:
    """Mirror Mozc-Ai ollama_backend.cc BuildPrompt()."""
    parts = ["日本語入力の変換候補を提案してください。", ""]
    if context:
        ctx = ", ".join(context[:3])
        parts.append(f"直前の入力: {ctx}")
        parts.append("")
    parts.append(f"現在の入力: {reading}")
    parts.append("")
    existing = []
    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        existing.append(cand)
        if len(existing) >= 5:
            break
    if existing:
        parts.append(f"既存候補（これら以外を提案）: {', '.join(existing)}")
        parts.append("")
    parts.append("3つの候補を改行区切りで出力（説明不要）:")
    parts.append("")
    return "\n".join(parts)


def _dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def review_to_example(row: dict[str, Any], *, top_k: int = 5) -> dict[str, Any] | None:
    review = row.get("review") or {}
    if review.get("decision") != ACCEPT:
        return None
    comparison = row.get("comparison") or {}
    record = comparison.get("record") or {}
    reading = str(record.get("reading") or "").strip()
    surface = str(record.get("surface") or "").strip()
    if not reading or not surface:
        return None
    if len(reading) < 2:
        return None
    if not _KANJI_RE.search(surface):
        return None
    candidates = [str(c) for c in (comparison.get("candidates") or []) if c]
    top = _dedupe_preserve(candidates)[:top_k]
    if surface in top:
        # Mozc already had gold in top-K — skip (should be rare for generation_gap).
        return None
    context = [str(c) for c in (comparison.get("context") or []) if c]
    # Prefer short context snippets when metadata stored a long sentence.
    short_context = [c for c in context if len(c) <= 40][:3]
    prompt = build_ime_prompt(reading, candidates, short_context)
    provenance = record.get("provenance") or {}
    return {
        "instruction": "日本語IMEの変換候補を提案してください。",
        "input": prompt,
        "output": surface,
        "text": f"{prompt}{surface}",
        "meta": {
            "source_id": provenance.get("source_id"),
            "reading": reading,
            "surface": surface,
            "category": record.get("category"),
            "reason_code": review.get("reason_code"),
            "confidence": review.get("confidence"),
        },
    }


def iter_examples(paths: list[Path], *, top_k: int = 5) -> Iterator[dict[str, Any]]:
    seen_keys: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                example = review_to_example(row, top_k=top_k)
                if example is None:
                    continue
                key = f"{example['meta']['source_id']}|{example['meta']['reading']}|{example['meta']['surface']}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                yield example


def export_train_jsonl(
    review_paths: list[Path],
    out_path: Path,
    *,
    top_k: int = 5,
    limit: int = 0,
) -> int:
    examples = list(iter_examples(review_paths, top_k=top_k))
    if limit and limit > 0:
        examples = examples[:limit]
    return write_jsonl(out_path, examples)
