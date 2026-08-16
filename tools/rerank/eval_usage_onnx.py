"""Aggregate-only ONNX evaluation on private IME usage groups.

This command never prints or stores text examples.  It is intended for local
execution because both its input and the personalized model may contain user
data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.dataset.jsonl import read_jsonl
from tools.rerank.eval_cross_encoder import cap_groups, prepare_groups
from tools.rerank.latency_pack import OrtRunner, _score_groups_batched
from tools.rerank.usage_guard import (
    apply_post_score_guard,
    context_empty_or_symbol,
    normalize_reading,
    skip_reason,
)


def _pre_skip(policy: str, reading: str, context: str) -> str | None:
    if policy == "none":
        return None
    normalized = normalize_reading(reading)
    if len(normalized) <= 2:
        return "reading_too_short"
    if context_empty_or_symbol(context):
        return "context_empty_or_symbol"
    if policy == "strict":
        return skip_reason(reading, context, enabled=True)
    return None


def evaluate_scored(
    groups: list[dict[str, Any]],
    scores: list[list[float]],
    *,
    tau: float,
    policy: str,
    train_readings: set[str] | None = None,
) -> dict[str, Any]:
    n = len(groups)
    correct_mozc = 0
    correct_final = 0
    helped = 0
    hurt = 0
    overwritten = 0
    skipped = 0
    overlap_n = overlap_correct = unseen_n = unseen_correct = 0
    reasons: dict[str, int] = {}
    train_readings = train_readings or set()
    for group, row_scores in zip(groups, scores):
        candidates = group["candidates"]
        mozc = group["mozc_top1"]
        gold = group["gold"]
        mozc_idx = candidates.index(mozc)
        best_idx = max(range(len(row_scores)), key=lambda i: row_scores[i])
        best = candidates[best_idx]
        final = mozc
        reason = _pre_skip(policy, group["reading"], group["context_prev"])
        if reason:
            skipped += 1
            reasons[reason] = reasons.get(reason, 0) + 1
        elif best != mozc and row_scores[best_idx] - row_scores[mozc_idx] >= tau:
            final = best
            if policy != "none":
                _, final, post_reason = apply_post_score_guard(
                    overwritten=True,
                    final_top1=final,
                    mozc_top1=mozc,
                    enabled=True,
                )
                if post_reason:
                    skipped += 1
                    reasons[post_reason] = reasons.get(post_reason, 0) + 1
        did_overwrite = final != mozc
        correct_mozc += int(mozc == gold)
        correct_final += int(final == gold)
        overwritten += int(did_overwrite)
        helped += int(did_overwrite and final == gold and mozc != gold)
        hurt += int(did_overwrite and final != gold and mozc == gold)
        is_overlap = normalize_reading(group["reading"]) in train_readings
        if is_overlap:
            overlap_n += 1
            overlap_correct += int(final == gold)
        else:
            unseen_n += 1
            unseen_correct += int(final == gold)
    mozc_hit = correct_mozc / n if n else 0.0
    final_hit = correct_final / n if n else 0.0
    return {
        "n": n,
        "policy": policy,
        "tau": tau,
        "mozc_hit1": round(mozc_hit, 6),
        "final_hit1": round(final_hit, 6),
        "net_pt_vs_mozc": round((final_hit - mozc_hit) * 100, 3),
        "n_overwrite": overwritten,
        "n_helped": helped,
        "n_hurt": hurt,
        "n_skip": skipped,
        "skip_reasons": reasons,
        "reading_overlap": {
            "n": overlap_n,
            "hit1": round(overlap_correct / overlap_n, 6) if overlap_n else None,
        },
        "reading_unseen": {
            "n": unseen_n,
            "hit1": round(unseen_correct / unseen_n, 6) if unseen_n else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local aggregate usage ONNX evaluation")
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-data", default="")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--cand-cap", type=int, default=20)
    parser.add_argument("--intra-op", type=int, default=12)
    parser.add_argument("--tau-sweep", default="0,0.5,1,1.5,2,2.5,3,4,5")
    args = parser.parse_args(argv)

    rows = list(read_jsonl(Path(args.data)))
    groups = cap_groups(prepare_groups(rows), int(args.cand_cap))
    train_readings: set[str] = set()
    if args.train_data:
        train_readings = {
            normalize_reading(row.get("reading") or "")
            for row in read_jsonl(Path(args.train_data))
        }
    runner = OrtRunner(
        Path(args.onnx),
        Path(args.tokenizer),
        max_len=int(args.max_len),
        intra=int(args.intra_op),
        inter=1,
        opt_all=True,
        padding="longest",
    )
    scores = _score_groups_batched(runner, groups)
    taus = [float(x.strip()) for x in args.tau_sweep.split(",") if x.strip()]
    results = [
        evaluate_scored(
            groups,
            scores,
            tau=tau,
            policy=policy,
            train_readings=train_readings,
        )
        for policy in ("none", "safety", "strict")
        for tau in taus
    ]
    report = {
        "privacy": "aggregate_only_local_evaluation",
        "n_groups": len(groups),
        "cand_cap": int(args.cand_cap),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in results:
        print(json.dumps(row, ensure_ascii=False))
    print(f"DONE wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
