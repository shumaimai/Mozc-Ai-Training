"""Dataset helpers for 5-candidate page-wise IME reranking."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PageExample:
    reading: str
    context: str
    candidates: tuple[str, ...]
    target: int
    weight: float
    is_gold_page: bool
    source: str = ""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_to_pages(
    row: dict[str, Any],
    *,
    page_size: int = 5,
    max_pages: int = 6,
    anchor_weight: float = 0.25,
) -> list[PageExample]:
    """Create gold-containing pages plus one first-page Mozc-preservation anchor."""
    reading = str(row.get("reading") or "")
    context = str(row.get("context_prev") or row.get("context") or "")
    gold = str(row.get("gold") or "")
    nbest = [str(x) for x in (row.get("mozc_nbest") or []) if x]
    source = str(row.get("source") or "")
    if not reading or not nbest:
        return []

    pages: list[PageExample] = []
    for page_index in range(min(max_pages, (len(nbest) + page_size - 1) // page_size)):
        start = page_index * page_size
        page = nbest[start : start + page_size]
        if not page:
            break
        if gold and gold in page:
            pages.append(
                PageExample(
                    reading=reading,
                    context=context,
                    candidates=tuple(page),
                    target=page.index(gold),
                    weight=1.0,
                    is_gold_page=True,
                    source=source,
                )
            )
        elif page_index == 0 and anchor_weight > 0:
            pages.append(
                PageExample(
                    reading=reading,
                    context=context,
                    candidates=tuple(page),
                    target=0,
                    weight=float(anchor_weight),
                    is_gold_page=False,
                    source=source,
                )
            )
    return pages


def build_pages(
    rows: Iterable[dict[str, Any]],
    *,
    page_size: int = 5,
    max_pages: int = 6,
    anchor_weight: float = 0.25,
) -> list[PageExample]:
    out: list[PageExample] = []
    for row in rows:
        out.extend(
            row_to_pages(
                row,
                page_size=page_size,
                max_pages=max_pages,
                anchor_weight=anchor_weight,
            )
        )
    return out


def build_candidate_text(reading: str, context: str, candidate: str) -> str:
    parts: list[str] = []
    if context:
        parts.append(f"文脈: {context}")
    parts.append(f"読み: {reading}")
    parts.append(f"候補: {candidate}")
    return "\n".join(parts)


def shuffle_gold_page(
    item: PageExample,
    *,
    seed: int,
    epoch: int,
    index: int,
) -> PageExample:
    """Deterministically shuffle a supervised gold page and remap its target.

    Anchor pages intentionally keep the original Mozc order.
    """
    if not item.is_gold_page or len(item.candidates) <= 1:
        return item
    order = list(range(len(item.candidates)))
    rng = random.Random(seed + epoch * 1_000_003 + index * 97)
    rng.shuffle(order)
    candidates = tuple(item.candidates[i] for i in order)
    target = order.index(item.target)
    return PageExample(
        reading=item.reading,
        context=item.context,
        candidates=candidates,
        target=target,
        weight=item.weight,
        is_gold_page=item.is_gold_page,
        source=item.source,
    )


class ListwisePageDataset(Dataset):
    def __init__(
        self,
        pages: list[PageExample],
        tokenizer,
        *,
        page_size: int = 5,
        max_length: int = 128,
        shuffle_gold_candidates: bool = False,
        shuffle_seed: int = 42,
    ):
        self.pages = pages
        self.tokenizer = tokenizer
        self.page_size = page_size
        self.max_length = max_length
        self.shuffle_gold_candidates = shuffle_gold_candidates
        self.shuffle_seed = shuffle_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def target_histogram(self) -> list[int]:
        hist = [0 for _ in range(self.page_size)]
        for index, original in enumerate(self.pages):
            item = (
                shuffle_gold_page(
                    original,
                    seed=self.shuffle_seed,
                    epoch=self.epoch,
                    index=index,
                )
                if self.shuffle_gold_candidates
                else original
            )
            if 0 <= item.target < self.page_size:
                hist[item.target] += 1
        return hist

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.pages[index]
        if self.shuffle_gold_candidates:
            item = shuffle_gold_page(
                item,
                seed=self.shuffle_seed,
                epoch=self.epoch,
                index=index,
            )
        candidates = list(item.candidates)
        valid = [True] * len(candidates)
        while len(candidates) < self.page_size:
            candidates.append(candidates[0])
            valid.append(False)

        texts = [
            build_candidate_text(item.reading, item.context, candidate)
            for candidate in candidates[: self.page_size]
        ]
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "candidate_mask": torch.tensor(valid[: self.page_size], dtype=torch.bool),
            "target": torch.tensor(item.target, dtype=torch.long),
            "weight": torch.tensor(item.weight, dtype=torch.float32),
            "is_gold_page": torch.tensor(item.is_gold_page, dtype=torch.bool),
        }
