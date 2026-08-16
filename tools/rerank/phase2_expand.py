"""Phase 2: A/B miss analysis + train expansion from corpus mozc_batch.

Keeps holdout fixed (rerank_v2) for fair before/after comparison.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from tools.dataset.jsonl import read_jsonl, write_jsonl
from tools.rerank.prepare import baseline_report, classify_to_rerank, gate_decision


def group_key(row: dict[str, Any]) -> str:
    return f"{row.get('source','')}\t{row.get('reading','')}\t{row.get('gold','')}"


def is_plausible_reading(reading: str) -> bool:
    r = (reading or "").strip()
    if not r or len(r) > 40:
        return False
    # hiragana / katakana / prolonged sound / middot only
    return bool(re.fullmatch(r"[\u3041-\u3096\u30A1-\u30FAー・･]+", r))


def is_plausible_surface(surface: str) -> bool:
    s = unicodedata.normalize("NFKC", surface or "").strip()
    if not s or len(s) > 48:
        return False
    return True


def qc_row(row: dict[str, Any]) -> bool:
    if not is_plausible_reading(row.get("reading") or ""):
        return False
    if not is_plausible_surface(row.get("gold") or ""):
        return False
    cands = row.get("mozc_nbest") or []
    if not cands:
        return False
    return True


def slice_tag(row: dict[str, Any]) -> str:
    """A = recoverable by rerank, B = need candidate sourcing, hit = mozc already ok."""
    if row.get("mozc_hit1"):
        return "hit"
    if row.get("gold_in_nbest"):
        return "A"
    return "B"


def analyze_ab(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    tags = Counter(slice_tag(r) for r in rows)
    b_rows = [r for r in rows if slice_tag(r) == "B"]
    a_rows = [r for r in rows if slice_tag(r) == "A"]
    by_cat = Counter((r.get("category") or "unknown") for r in b_rows)
    by_src = Counter((r.get("source") or "unknown") for r in b_rows)
    # crude type hints for B golds
    type_hints = Counter()
    for r in b_rows:
        g = r.get("gold") or ""
        cat = r.get("category") or ""
        if cat:
            type_hints[cat] += 1
        elif re.search(r"(市|区|町|村|駅|県|府|道)$", g):
            type_hints["place_suffix"] += 1
        elif re.search(r"(株式会社|有限会社|合同会社)", g):
            type_hints["company"] += 1
        else:
            type_hints["other"] += 1
    miss = tags["A"] + tags["B"]
    return {
        "n": n,
        "slices": {
            "hit": tags["hit"],
            "A_gold_in_nbest_miss": tags["A"],
            "B_gold_out_of_nbest": tags["B"],
        },
        "frac": {
            "hit": round(tags["hit"] / n, 4) if n else 0,
            "A": round(tags["A"] / n, 4) if n else 0,
            "B": round(tags["B"] / n, 4) if n else 0,
            "B_among_miss": round(tags["B"] / miss, 4) if miss else 0,
        },
        "B_by_category": dict(by_cat.most_common()),
        "B_by_source": dict(by_src.most_common()),
        "B_type_hints": dict(type_hints.most_common()),
        "sourcing_gate": (
            "GO_try_local_dict"
            if (tags["B"] / n if n else 0) >= 0.25
            else "DEFER_sourcing"
        ),
        "note": (
            "B>=25% of holdout → candidate sourcing is high value. "
            "A is residual rerank headroom."
        ),
    }


def load_corpus_classify(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        raw = list(read_jsonl(p))
        converted = classify_to_rerank(raw)
        for r in converted:
            r["phase2_origin"] = str(p)
        rows.extend(converted)
        print(f"loaded {p}: raw={len(raw)} converted={len(converted)}")
    return rows


def sample_balanced(
    pool: list[dict[str, Any]],
    *,
    target_total: int,
    residual_frac: float,
    anchor_frac: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample gold_in_nbest rows toward residual/anchor mix. Drops B for CE train."""
    rng = random.Random(seed)
    usable = [r for r in pool if r.get("gold_in_nbest") and qc_row(r)]
    residual = [r for r in usable if not r.get("mozc_hit1")]  # A
    anchors = [r for r in usable if r.get("mozc_hit1")]  # hit
    rng.shuffle(residual)
    rng.shuffle(anchors)
    n_res = int(target_total * residual_frac)
    n_anc = int(target_total * anchor_frac)
    # fill remainder from whichever has more leftover
    take_res = residual[:n_res]
    take_anc = anchors[:n_anc]
    out = take_res + take_anc
    need = target_total - len(out)
    if need > 0:
        leftover = residual[n_res:] + anchors[n_anc:]
        rng.shuffle(leftover)
        out.extend(leftover[:need])
    rng.shuffle(out)
    return out


def command_ab(args: argparse.Namespace) -> int:
    rows = list(read_jsonl(Path(args.input)))
    report = analyze_ab(rows)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


def command_expand(args: argparse.Namespace) -> int:
    holdout = list(read_jsonl(Path(args.holdout)))
    hold_keys = {group_key(r) for r in holdout}
    # also block reading+gold regardless of source to reduce near-leak
    hold_rg = {(r.get("reading"), r.get("gold")) for r in holdout}

    base_train = list(read_jsonl(Path(args.base_train)))
    corpus_paths = [Path(p) for p in args.corpus_classify]
    corpus = load_corpus_classify(corpus_paths)

    # drop holdout collisions + QC
    corpus_f = []
    dropped = Counter()
    for r in corpus:
        if group_key(r) in hold_keys or (r.get("reading"), r.get("gold")) in hold_rg:
            dropped["holdout_collision"] += 1
            continue
        if not qc_row(r):
            dropped["qc"] += 1
            continue
        corpus_f.append(r)

    # stats before sample
    pre = Counter(slice_tag(r) for r in corpus_f)
    print("corpus_after_qc_slices", dict(pre), "dropped", dict(dropped))

    # Prefer gold_in_nbest for CE; sample toward target
    extra = sample_balanced(
        corpus_f,
        target_total=args.extra_groups,
        residual_frac=args.residual_frac,
        anchor_frac=args.anchor_frac,
        seed=args.seed,
    )

    # merge with base train (dedupe)
    seen = {group_key(r) for r in base_train}
    merged = list(base_train)
    added = 0
    for r in extra:
        k = group_key(r)
        if k in seen:
            continue
        if (r.get("reading"), r.get("gold")) in hold_rg:
            continue
        seen.add(k)
        merged.append(r)
        added += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    hold_path = out_dir / "holdout.jsonl"
    write_jsonl(train_path, merged)
    write_jsonl(hold_path, holdout)  # identical holdout

    train_rep = baseline_report(merged)
    hold_rep = baseline_report(holdout)
    mix = Counter(slice_tag(r) for r in merged)
    summary = {
        "seed": args.seed,
        "base_train": len(base_train),
        "corpus_loaded": len(corpus),
        "corpus_after_qc": len(corpus_f),
        "extra_sampled": len(extra),
        "added_new": added,
        "train_total": len(merged),
        "holdout_total": len(holdout),
        "holdout_fixed_from": str(args.holdout),
        "dropped": dict(dropped),
        "train_slices": dict(mix),
        "train_baseline": train_rep,
        "holdout_baseline": hold_rep,
        "holdout_gate": gate_decision(hold_rep),
        "ab_holdout": analyze_ab(holdout),
    }
    (out_dir / "expand_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {train_path} ({len(merged)})")
    print(f"wrote {hold_path} ({len(holdout)}) [fixed holdout]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2 expand / A-B analysis")
    sub = p.add_subparsers(dest="command", required=True)

    ab = sub.add_parser("ab", help="A/B miss analysis on rerank JSONL")
    ab.add_argument("--input", required=True)
    ab.add_argument("--out", default="")
    ab.set_defaults(func=command_ab)

    ex = sub.add_parser("expand", help="expand train from corpus classify_in")
    ex.add_argument("--base-train", default="data/rerank_v2/train.jsonl")
    ex.add_argument("--holdout", default="data/rerank_v2/holdout.jsonl")
    ex.add_argument(
        "--corpus-classify",
        nargs="+",
        default=[
            "data/interim/mozc_batch/aozora/classify_in.jsonl",
            "data/interim/mozc_batch/wikidata/classify_in.jsonl",
            "data/interim/mozc_batch/japanpost/classify_in.jsonl",
        ],
    )
    ex.add_argument("--out-dir", default="data/rerank_v3")
    ex.add_argument("--extra-groups", type=int, default=25000)
    ex.add_argument("--residual-frac", type=float, default=0.55)
    ex.add_argument("--anchor-frac", type=float, default=0.45)
    ex.add_argument("--seed", type=int, default=42)
    ex.set_defaults(func=command_expand)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
