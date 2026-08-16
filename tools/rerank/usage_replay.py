#!/usr/bin/env python3
"""Replay ime_usage_pairs.jsonl with/without usage guards (NEXT_TASK_USAGE_GUARD)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.rerank.usage_guard import apply_post_score_guard, skip_reason

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = ROOT / "artifacts/rerank_ctx/eval/ime_usage_pairs.jsonl"
FALLBACK_PAIRS = DEFAULT_PAIRS
OUT_JSON = ROOT / "artifacts/rerank_ctx/eval/usage_replay_report.json"
WORKSPACE_JSON = OUT_JSON


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def classify(wanted: str, shown: str, mozc: str, overwritten: bool) -> str:
    if overwritten and shown == wanted and mozc != wanted:
        return "rerank_helped"
    if overwritten and shown != wanted and mozc == wanted:
        return "rerank_rejected"
    if shown == wanted:
        return "accepted_shown"
    return "user_picked_other"


def apply_guard(row: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    reading = str(row.get("reading") or "")
    ctx = str(row.get("context") or row.get("context_prev") or "")
    mozc = str(row.get("mozc_top1") or "")
    shown_hist = str(row.get("shown") or "")
    rerank_top1 = str(row.get("rerank_top1") or "")
    overwritten_hist = bool(row.get("overwritten"))
    wanted = str(row.get("wanted") or "")

    reason = None
    overwritten = overwritten_hist
    shown = shown_hist
    if enabled:
        reason = skip_reason(reading, ctx, enabled=True)
        if reason:
            overwritten = False
            shown = mozc
        elif overwritten_hist:
            overwritten, shown, junk = apply_post_score_guard(
                overwritten=True,
                final_top1=shown_hist,
                mozc_top1=mozc,
                enabled=True,
            )
            if junk:
                reason = junk
            else:
                shown = shown_hist
    kind = classify(wanted, shown, mozc, overwritten)
    return {
        "reading": reading,
        "context": ctx,
        "wanted": wanted,
        "shown": shown,
        "shown_hist": shown_hist,
        "mozc_top1": mozc,
        "rerank_top1": rerank_top1,
        "overwritten": overwritten,
        "overwritten_hist": overwritten_hist,
        "guard_reason": reason,
        "kind": kind,
        "match_shown": wanted == shown,
        "match_mozc": wanted == mozc,
    }


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(events)
    helped = [e for e in events if e["kind"] == "rerank_helped"]
    rejected = [e for e in events if e["kind"] == "rerank_rejected"]
    shown_hit = 100.0 * sum(1 for e in events if e["match_shown"]) / n if n else 0.0
    mozc_hit = 100.0 * sum(1 for e in events if e["match_mozc"]) / n if n else 0.0
    return {
        "n": n,
        "shown_hit": round(shown_hit, 1),
        "mozc_hit": round(mozc_hit, 1),
        "net_pt_vs_mozc": round(shown_hit - mozc_hit, 1),
        "n_overwrite": sum(1 for e in events if e["overwritten"]),
        "n_helped": len(helped),
        "n_hurt": len(rejected),
        "counts": dict(Counter(e["kind"] for e in events)),
        "helped": [
            {"reading": e["reading"], "context": e["context"], "wanted": e["wanted"],
             "shown": e["shown"], "mozc": e["mozc_top1"]}
            for e in helped
        ],
        "hurt": [
            {"reading": e["reading"], "context": e["context"], "wanted": e["wanted"],
             "shown": e["shown"], "mozc": e["mozc_top1"], "guard_reason": e.get("guard_reason")}
            for e in rejected
        ],
        "guard_reasons": dict(Counter(e["guard_reason"] for e in events if e.get("guard_reason"))),
        "n_guard_skip": sum(1 for e in events if e.get("guard_reason")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()
    pairs_path = args.pairs
    if pairs_path is None:
        pairs_path = DEFAULT_PAIRS if DEFAULT_PAIRS.is_file() else FALLBACK_PAIRS
    rows = load_jsonl(pairs_path)
    ung = [apply_guard(r, enabled=False) for r in rows]
    grd = [apply_guard(r, enabled=True) for r in rows]
    s_off = summarize(ung)
    s_on = summarize(grd)
    remaining_hurt = s_on["hurt"]
    report = {
        "source": str(pairs_path),
        "n_pairs": len(rows),
        "unguarded": {k: s_off[k] for k in (
            "n", "shown_hit", "mozc_hit", "net_pt_vs_mozc", "n_overwrite",
            "n_helped", "n_hurt", "counts",
        )},
        "guarded": {k: s_on[k] for k in (
            "n", "shown_hit", "mozc_hit", "net_pt_vs_mozc", "n_overwrite",
            "n_helped", "n_hurt", "counts", "guard_reasons", "n_guard_skip",
        )},
        "unguarded_helped": s_off["helped"],
        "unguarded_hurt": s_off["hurt"],
        "guarded_helped": s_on["helped"],
        "guarded_hurt": remaining_hurt,
        "success_net_ge_zero": s_on["net_pt_vs_mozc"] >= 0,
        "retrain_go": False,
        "retrain_note": (
            "Guards passed usage net >= 0. Task 3 retraining is the next stage "
            "and is not started automatically."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    WORKSPACE_JSON.parent.mkdir(parents=True, exist_ok=True)
    WORKSPACE_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "unguarded": report["unguarded"],
        "guarded": report["guarded"],
        "success_net_ge_zero": report["success_net_ge_zero"],
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
