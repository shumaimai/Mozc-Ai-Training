"""Build Mozc N-best reranker datasets from accept-derived train_mixed."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from tools.dataset.jsonl import read_jsonl, write_jsonl
from tools.dataset.mozc_batch import (
    merge,
    parse_candidates_tsv,
    readings_from_records,
    resolve_batch_config,
    run_mozc_batch,
)


def normalize_surface(text: str) -> str:
    """NFKC + strip + collapse spaces/middle-dots for gold↔candidate compare."""
    s = unicodedata.normalize("NFKC", text or "")
    s = s.strip()
    s = s.replace("\u3000", "").replace(" ", "")
    s = s.replace("・", "").replace("･", "")
    s = re.sub(r"\s+", "", s)
    return s


def find_gold_rank(gold: str, cands: list[str]) -> int | None:
    """1-based rank under exact then normalized equality."""
    if gold in cands:
        return cands.index(gold) + 1
    ng = normalize_surface(gold)
    if not ng:
        return None
    for i, c in enumerate(cands):
        if normalize_surface(c) == ng:
            return i + 1
    return None


def train_mixed_to_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten LoRA export rows into TermRecord-like dicts for mozc-run."""
    out: list[dict[str, Any]] = []
    for row in rows:
        meta = row.get("meta") or {}
        reading = (meta.get("reading") or "").strip()
        surface = (meta.get("surface") or row.get("output") or "").strip()
        if not reading or not surface:
            continue
        out.append(
            {
                "reading": reading,
                "surface": surface,
                "category": meta.get("category") or "",
                "provenance": {
                    "source_id": meta.get("source_id") or "train_mixed",
                    "source_url": "",
                    "license_id": "",
                    "retrieved_at": "",
                },
                "reading_source": "train_mixed",
                "reading_confidence": "reviewed",
                "metadata": {
                    "reason_code": meta.get("reason_code"),
                    "confidence": meta.get("confidence"),
                },
            }
        )
    return out


def classify_to_rerank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert mozc-merge classify_in rows to PLAN_RERANKER §3.1 records."""
    out: list[dict[str, Any]] = []
    for row in rows:
        rec = row.get("record") or row
        cands = list(row.get("candidates") or [])
        # de-dupe while preserving order
        deduped: list[str] = []
        seen: set[str] = set()
        for c in cands:
            if not c or c in seen:
                continue
            seen.add(c)
            deduped.append(c)
        gold = rec.get("surface") or ""
        ctx = row.get("context") or []
        context_prev = ""
        if isinstance(ctx, list) and ctx:
            context_prev = str(ctx[0] or "")
        elif isinstance(ctx, str):
            context_prev = ctx
        rank = find_gold_rank(gold, deduped)
        top1 = deduped[0] if deduped else ""
        hit1 = bool(top1) and (
            top1 == gold or normalize_surface(top1) == normalize_surface(gold)
        )
        out.append(
            {
                "reading": rec.get("reading") or "",
                "context_prev": context_prev,
                "mozc_nbest": deduped,
                "gold": gold,
                "gold_in_nbest": rank is not None,
                "mozc_top1": top1,
                "mozc_hit1": hit1,
                "gold_rank": rank,
                "source": (rec.get("provenance") or {}).get("source_id") or "train_mixed",
                "category": rec.get("category") or "",
            }
        )
    return out


def _group_key(row: dict[str, Any]) -> str:
    # No full sentence in train_mixed; isolate by source + reading + gold.
    return f"{row.get('source','')}\t{row.get('reading','')}\t{row.get('gold','')}"


def split_holdout(
    rows: list[dict[str, Any]],
    *,
    holdout_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out by group key so the same reading/gold never leaks across splits."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)
    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_hold = max(1, int(round(len(keys) * holdout_frac)))
    hold_keys = set(keys[:n_hold])
    train: list[dict[str, Any]] = []
    hold: list[dict[str, Any]] = []
    for key, items in groups.items():
        (hold if key in hold_keys else train).extend(items)
    return train, hold


def baseline_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    hit1 = sum(1 for r in rows if r.get("mozc_hit1"))
    in_nbest = sum(1 for r in rows if r.get("gold_in_nbest"))
    ranks = [int(r["gold_rank"]) for r in rows if r.get("gold_rank") is not None]
    empty = sum(1 for r in rows if not r.get("mozc_nbest"))
    by_cat = Counter(r.get("category") or "unknown" for r in rows)
    miss = [r for r in rows if not r.get("mozc_hit1")]
    miss_recoverable = sum(1 for r in miss if r.get("gold_in_nbest"))
    prefix = 0
    for r in rows:
        top = r.get("mozc_top1") or ""
        gold = r.get("gold") or ""
        if top and gold.startswith(top):
            prefix += 1
    rank_hist = Counter()
    for rank in ranks:
        if rank == 1:
            rank_hist["rank1"] += 1
        elif rank <= 3:
            rank_hist["rank2_3"] += 1
        elif rank <= 10:
            rank_hist["rank4_10"] += 1
        else:
            rank_hist["rank11_plus"] += 1
    in_not_top1 = sum(1 for r in rows if r.get("gold_in_nbest") and not r.get("mozc_hit1"))
    return {
        "n": n,
        "mozc_hit1": hit1 / n,
        "gold_in_nbest": in_nbest / n,
        "gold_in_nbest_not_top1": in_not_top1 / n,
        "gold_startswith_mozc_top1": prefix / n,
        "empty_nbest": empty / n,
        "gold_rank_mean_when_present": (
            statistics.mean(ranks) if ranks else None
        ),
        "gold_rank_median_when_present": (
            statistics.median(ranks) if ranks else None
        ),
        "gold_rank_histogram_when_present": dict(rank_hist),
        "miss_n": len(miss),
        "miss_recoverable_by_rerank": miss_recoverable / max(len(miss), 1),
        "headroom_hit1_if_perfect_rerank": in_nbest / n - hit1 / n,
        "categories": dict(by_cat),
        "note": (
            "Full-path mozc_batch (best-path + segment alts + ResizeSegment "
            "single-segment) with normalized gold compare."
        ),
    }


def gate_decision(report: dict[str, Any]) -> dict[str, Any]:
    """GO/NO-GO for GPU reranker training."""
    gold_in = float(report.get("gold_in_nbest") or 0.0)
    not_top1 = float(report.get("gold_in_nbest_not_top1") or 0.0)
    hit1 = float(report.get("mozc_hit1") or 0.0)
    go = gold_in >= 0.50 and not_top1 > 0.05
    if go:
        decision = "GO"
        reason = (
            f"gold_in_nbest={gold_in:.3f} (>=0.50) and "
            f"gold_in_nbest_not_top1={not_top1:.3f} (>0.05): rerank headroom exists."
        )
    elif gold_in < 0.50:
        decision = "NO-GO"
        reason = (
            f"gold_in_nbest={gold_in:.3f} (<0.50): accept set still looks like "
            "generation/addition distribution; do not spend GPU on pure rerank."
        )
    else:
        decision = "NO-GO"
        reason = (
            f"gold_in_nbest={gold_in:.3f} but almost always already top1 "
            f"(hit1={hit1:.3f}, not_top1={not_top1:.3f}): little rerank headroom."
        )
    return {
        "decision": decision,
        "thresholds": {
            "gold_in_nbest_min": 0.50,
            "gold_in_nbest_not_top1_min": 0.05,
        },
        "reason": reason,
        "metrics": {
            "mozc_hit1": hit1,
            "gold_in_nbest": gold_in,
            "gold_in_nbest_not_top1": not_top1,
            "headroom_hit1_if_perfect_rerank": report.get(
                "headroom_hit1_if_perfect_rerank"
            ),
            "gold_rank_histogram_when_present": report.get(
                "gold_rank_histogram_when_present"
            ),
        },
    }


def command_prepare(args: argparse.Namespace) -> int:
    src = Path(args.train_mixed)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    records_path = work / "records.jsonl"
    keys_path = work / "keys.txt"
    cand_path = work / "candidates.tsv"
    classify_path = work / "classify_in.jsonl"
    all_path = work / "rerank_all.jsonl"

    raw = list(read_jsonl(src))
    if args.limit and args.limit > 0:
        raw = raw[: args.limit]
    records = train_mixed_to_records(raw)
    write_jsonl(records_path, records)
    print(f"records={len(records)} -> {records_path}")

    keys = readings_from_records(records)
    keys_path.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")
    print(f"unique_readings={len(keys)} -> {keys_path}")

    exe, engine, max_cands = resolve_batch_config(
        env_file=Path(args.env_file) if args.env_file else None,
        max_candidates=args.max_candidates,
    )
    print(f"mozc_batch exe={exe} data={engine} max={max_cands}")
    run_mozc_batch(exe, engine, keys_path, cand_path, max_cands)
    key_to_cands = parse_candidates_tsv(
        cand_path.read_text(encoding="utf-8").splitlines()
    )
    print(f"tsv_keys={len(key_to_cands)} -> {cand_path}")

    merged = merge(records, key_to_cands)
    write_jsonl(classify_path, merged)
    rerank_rows = classify_to_rerank(merged)
    write_jsonl(all_path, rerank_rows)
    print(f"rerank_all={len(rerank_rows)} -> {all_path}")

    report = baseline_report(rerank_rows)
    gate = gate_decision(report)
    (work / "baseline_all.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (work / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return 0


def command_split(args: argparse.Namespace) -> int:
    rows = list(read_jsonl(Path(args.input)))
    train, hold = split_holdout(
        rows, holdout_frac=args.holdout_frac, seed=args.seed
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    hold_path = out_dir / "holdout.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(hold_path, hold)

    train_rep = baseline_report(train)
    hold_rep = baseline_report(hold)
    summary = {
        "seed": args.seed,
        "holdout_frac": args.holdout_frac,
        "split_unit": "source+reading+gold",
        "train": train_rep,
        "holdout": hold_rep,
        "holdout_gate": gate_decision(hold_rep),
    }
    (out_dir / "baseline_split.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"train={len(train)} -> {train_path}")
    print(f"holdout={len(hold)} -> {hold_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_baseline(args: argparse.Namespace) -> int:
    rows = list(read_jsonl(Path(args.input)))
    report = baseline_report(rows)
    gate = gate_decision(report)
    payload = {"baseline": report, "gate": gate}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


def command_gate(args: argparse.Namespace) -> int:
    rows = list(read_jsonl(Path(args.input)))
    report = baseline_report(rows)
    gate = gate_decision(report)
    text = json.dumps({"baseline": report, "gate": gate}, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if gate["decision"] == "GO" else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reranker Phase0 data tools")
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="train_mixed -> Mozc N-best rerank JSONL")
    prep.add_argument("--train-mixed", default="data/train/train_mixed.jsonl")
    prep.add_argument("--work-dir", default="data/interim/rerank_phase0")
    prep.add_argument("--env-file", default="config/mozc_batch.env")
    prep.add_argument("--max-candidates", type=int, default=None)
    prep.add_argument("--limit", type=int, default=0)
    prep.set_defaults(func=command_prepare)

    spl = sub.add_parser("split", help="split rerank_all into train/holdout")
    spl.add_argument("--input", default="data/interim/rerank_phase0/rerank_all.jsonl")
    spl.add_argument("--out-dir", default="data/rerank")
    spl.add_argument("--holdout-frac", type=float, default=0.15)
    spl.add_argument("--seed", type=int, default=42)
    spl.set_defaults(func=command_split)

    base = sub.add_parser("baseline", help="print Mozc baseline metrics + gate")
    base.add_argument("--input", required=True)
    base.add_argument("--out", default="")
    base.set_defaults(func=command_baseline)

    gate = sub.add_parser("gate", help="GO/NO-GO for GPU rerank training")
    gate.add_argument("--input", required=True)
    gate.add_argument("--out", default="")
    gate.set_defaults(func=command_gate)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
