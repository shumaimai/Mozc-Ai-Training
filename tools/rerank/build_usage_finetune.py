"""Build a privacy-local fine-tuning set from IME usage outcomes.

The source log and every generated row can contain user text.  Outputs therefore
default to ``data/private`` (gitignored) and this module never prints row content.
No network access is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from tools.dataset.jsonl import read_jsonl
from tools.rerank.context_clip import clean_context, normalize_reading, sanitize_nbest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USAGE = ROOT.parent / "artifacts/rerank_ctx/eval/ime_usage_pairs.jsonl"
FALLBACK_USAGE = ROOT / "artifacts/rerank_ctx/eval/ime_usage_pairs.jsonl"


def _stable_fraction(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value or "").strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def usage_to_group(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    """Convert one local outcome to the common cross-encoder group schema."""
    reading = normalize_reading(row.get("reading") or "")
    gold = str(row.get("wanted") or "").strip()
    mozc = str(row.get("mozc_top1") or "").strip()
    if not reading or not gold or not mozc:
        return None
    candidates = sanitize_nbest(
        _dedupe(
            [
                mozc,
                row.get("rerank_top1") or "",
                row.get("shown") or "",
                gold,
            ]
        )
    )
    if gold not in candidates:
        candidates.append(gold)
    if mozc not in candidates:
        candidates.insert(0, mozc)
    # Preserve Mozc ordering at the front; the remaining candidates are only
    # locally observed hard negatives, not claimed to be a complete N-best.
    candidates = [mozc, *[c for c in candidates if c != mozc]]
    return {
        "reading": reading,
        "gold": gold,
        "context_prev": clean_context(row.get("context") or "", max_chars=50),
        "mozc_nbest": candidates,
        "mozc_top1": mozc,
        "gold_in_nbest": True,
        "mozc_hit1": mozc == gold,
        "source": "ime_usage_local",
        "category": "usage_anchor" if mozc == gold else "usage_correction",
        "usage_index": index,
    }


def split_groups(
    groups: list[dict[str, Any]], holdout_frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chronological-within-reading split, with singleton deterministic holdout."""
    by_reading: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_reading[group["reading"]].append(group)
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for reading, rows in sorted(by_reading.items()):
        rows.sort(key=lambda x: int(x["usage_index"]))
        if len(rows) == 1:
            (holdout if _stable_fraction(reading, seed) < holdout_frac else train).extend(rows)
            continue
        n_holdout = max(1, round(len(rows) * holdout_frac))
        n_holdout = min(len(rows) - 1, n_holdout)
        train.extend(rows[:-n_holdout])
        holdout.extend(rows[-n_holdout:])
    train.sort(key=lambda x: int(x["usage_index"]))
    holdout.sort(key=lambda x: int(x["usage_index"]))
    return train, holdout


def _sample_public(
    rows: list[dict[str, Any]], n: int, rng: random.Random
) -> list[dict[str, Any]]:
    usable = [
        row
        for row in rows
        if row.get("reading") and row.get("gold") and row.get("gold_in_nbest")
    ]
    anchors = [row for row in usable if row.get("mozc_top1") == row.get("gold")]
    residual = [row for row in usable if row.get("mozc_top1") != row.get("gold")]
    rng.shuffle(anchors)
    rng.shuffle(residual)
    # Keep public anchors dominant: they are the anti-regression backbone.
    n_anchor = min(len(anchors), round(n * 0.7))
    chosen = anchors[:n_anchor] + residual[: max(0, n - n_anchor)]
    if len(chosen) < n:
        used = {id(x) for x in chosen}
        rest = [x for x in usable if id(x) not in used]
        rng.shuffle(rest)
        chosen.extend(rest[: n - len(chosen)])
    return chosen[:n]


def build_mix(
    usage_train: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
    target_groups: int,
    usage_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    if not usage_train:
        raise ValueError("usage training split is empty")
    rng = random.Random(seed)
    n_usage = max(len(usage_train), round(target_groups * usage_fraction))
    n_public = max(0, target_groups - n_usage)
    public = _sample_public(public_rows, n_public, rng)
    # Balanced cycling prevents a few repeated raw rows from dominating solely
    # because they occurred many times in one typing session.
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usage_train:
        buckets[row["reading"]].append(row)
    readings = sorted(buckets)
    usage: list[dict[str, Any]] = []
    cursors = Counter()
    while len(usage) < n_usage:
        order = list(readings)
        rng.shuffle(order)
        for reading in order:
            rows = buckets[reading]
            row = rows[cursors[reading] % len(rows)]
            cursors[reading] += 1
            usage.append(dict(row))
            if len(usage) >= n_usage:
                break
    mixed = [*public, *usage]
    rng.shuffle(mixed)
    return mixed


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build local-only usage fine-tune data")
    parser.add_argument("--usage", default=str(DEFAULT_USAGE if DEFAULT_USAGE.exists() else FALLBACK_USAGE))
    parser.add_argument("--public-train", default=str(ROOT / "data/rerank_ctx/train_v2_clean.jsonl"))
    parser.add_argument("--out-dir", default=str(ROOT / "data/private/usage_finetune_v1"))
    parser.add_argument("--holdout-frac", type=float, default=0.25)
    parser.add_argument("--target-groups", type=int, default=12000)
    parser.add_argument("--usage-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args(argv)
    if not 0.05 <= args.holdout_frac <= 0.5:
        raise SystemExit("--holdout-frac must be in [0.05, 0.5]")
    if not 0.1 <= args.usage_fraction <= 0.9:
        raise SystemExit("--usage-fraction must be in [0.1, 0.9]")

    raw = list(read_jsonl(Path(args.usage)))
    groups = [g for i, row in enumerate(raw) if (g := usage_to_group(row, i)) is not None]
    usage_train, usage_holdout = split_groups(groups, args.holdout_frac, args.seed)
    public_rows = list(read_jsonl(Path(args.public_train)))
    mixed = build_mix(
        usage_train,
        public_rows,
        args.target_groups,
        args.usage_fraction,
        args.seed,
    )

    out = Path(args.out_dir)
    _write_jsonl(out / "train.jsonl", mixed)
    _write_jsonl(out / "usage_train.jsonl", usage_train)
    _write_jsonl(out / "usage_holdout.jsonl", usage_holdout)
    report = {
        "privacy": "local_only_no_network",
        "raw_usage_rows": len(raw),
        "usable_usage_rows": len(groups),
        "usage_train_rows": len(usage_train),
        "usage_holdout_rows": len(usage_holdout),
        "usage_train_readings": len({r["reading"] for r in usage_train}),
        "usage_holdout_readings": len({r["reading"] for r in usage_holdout}),
        "mixed_train_groups": len(mixed),
        "mixed_usage_rows": sum(r.get("source") == "ime_usage_local" for r in mixed),
        "mixed_public_rows": sum(r.get("source") != "ime_usage_local" for r in mixed),
        "seed": args.seed,
    }
    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
