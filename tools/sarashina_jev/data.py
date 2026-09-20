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
    """Human-readable form of the causal scoring prompt."""
    return (
        f"候補: {candidate}\n"
        f"読み: {reading}\n"
        f"文脈: {context}\n"
        "判定:"
    )


def encode_candidate_sequence(
    tokenizer,
    *,
    reading: str,
    context_ids: list[int],
    candidate: str,
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Encode a causal scoring prompt without truncating candidate/reading/decision.

    Layout:
        BOS + 候補 + 読み + 文脈(tail only) + 判定:

    Only the *oldest* context tokens are dropped when the sequence is too long.
    The final non-padding token is always part of the decision marker, which is
    where the decoder backbone is pooled by SarashinaJevScorer.
    """
    prefix_ids = tokenizer.encode(
        f"候補: {candidate}\n読み: {reading}\n文脈: ",
        add_special_tokens=False,
    )
    suffix_ids = tokenizer.encode("\n判定:", add_special_tokens=False)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    bos_ids = [int(bos_id)] if bos_id is not None else []

    fixed = bos_ids + list(prefix_ids) + list(suffix_ids)
    if len(fixed) > max_length:
        raise ValueError(
            "max_length is too small to preserve candidate/reading/decision "
            f"(need at least {len(fixed)}, got {max_length})"
        )

    context_budget = max_length - len(fixed)
    context_tail = list(context_ids[-context_budget:]) if context_budget > 0 else []
    input_ids = bos_ids + list(prefix_ids) + context_tail + list(suffix_ids)
    attention_mask = [1] * len(input_ids)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    pad_count = max_length - len(input_ids)
    if pad_count:
        input_ids.extend([int(pad_id)] * pad_count)
        attention_mask.extend([0] * pad_count)

    return input_ids, attention_mask


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

        context_ids = self.tokenizer.encode(
            item.context,
            add_special_tokens=False,
        )
        encoded = [
            encode_candidate_sequence(
                self.tokenizer,
                reading=item.reading,
                context_ids=context_ids,
                candidate=candidate,
                max_length=self.max_length,
            )
            for candidate in candidates[: self.page_size]
        ]
        input_ids = torch.tensor([ids for ids, _ in encoded], dtype=torch.long)
        attention_mask = torch.tensor([mask for _, mask in encoded], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "candidate_mask": torch.tensor(valid[: self.page_size], dtype=torch.bool),
            "target": torch.tensor(item.target, dtype=torch.long),
            "weight": torch.tensor(item.weight, dtype=torch.float32),
            "is_gold_page": torch.tensor(item.is_gold_page, dtype=torch.bool),
        }
