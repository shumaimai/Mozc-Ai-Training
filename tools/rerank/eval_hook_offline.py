"""Offline hook vs clean eval at ship τ=2.5 / max_len=128 / context ON.

Compares OrtRunner (or PT) + margin.py to published clean_eval_trackB_tau2.5 numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.dataset.jsonl import read_jsonl
from tools.rerank.eval_cross_encoder import prepare_groups
from tools.rerank.eval_contextual import _restore_flags, _subset_metrics
from tools.rerank.latency_pack import OrtRunner, _score_groups_batched
from tools.rerank.margin import metrics_at_tau
from tools.rerank.phase3_hook import DEFAULT_CAND_CAP, DEFAULT_MAX_LEN, DEFAULT_TAU, cap_candidates


EXPECTED = {
    "seen": {"hit1": 0.9150, "cs_delta_pt": 32.57, "reg": 0.0104},
    "unseen": {"hit1": 0.9146, "cs_delta_pt": 29.31, "reg": 0.0134},
    "fresh": {"hit1": 0.9285, "cs_delta_pt": 49.14, "reg": 0.0066},
}


def _cap_groups(groups: list[dict[str, Any]], cand_cap: int) -> list[dict[str, Any]]:
    if cand_cap <= 0:
        return groups
    out = []
    for g in groups:
        cands = cap_candidates(list(g["candidates"]), cand_cap)
        if g["mozc_top1"] not in cands:
            cands = [g["mozc_top1"], *[c for c in cands if c != g["mozc_top1"]]][:cand_cap]
        out.append({**g, "candidates": cands})
    return out


def _blank(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**g, "context_prev": ""} for g in groups]


def eval_one(
    runner: OrtRunner,
    groups: list[dict[str, Any]],
    tau: float,
) -> dict[str, Any]:
    scores = _score_groups_batched(runner, groups)
    scores_off = _score_groups_batched(runner, _blank(groups))
    all_m = metrics_at_tau(groups, scores, tau)
    cs_on = _subset_metrics(groups, scores, tau, lambda g: bool(g.get("context_sensitive")))
    cs_off = _subset_metrics(
        _blank(groups), scores_off, tau, lambda g: bool(g.get("context_sensitive"))
    )
    cs_delta = None
    if cs_on.get("n") and cs_off.get("n"):
        cs_delta = round((cs_on["final_hit1"] - cs_off["final_hit1"]) * 100, 3)
    return {
        "all": {
            "n": len(groups),
            "final_hit1": all_m["final_hit1"],
            "mozc_hit1": all_m["mozc_hit1"],
            "delta_vs_mozc_pt": all_m["delta_vs_mozc_pt"],
            "regression_rate_on_mozc_hit": all_m["regression_rate_on_mozc_hit"],
            "overwrite_rate": all_m["overwrite_rate"],
        },
        "context_sensitive_on": cs_on,
        "context_sensitive_off": cs_off,
        "cs_delta_pt": cs_delta,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data-dir", default="data/rerank_ctx")
    p.add_argument("--out", default="artifacts/rerank_ctx/eval/hook_offline_tau2.5.json")
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--cand-cap", type=int, default=DEFAULT_CAND_CAP)
    p.add_argument("--intra-op", type=int, default=1)
    args = p.parse_args()

    runner = OrtRunner(
        Path(args.onnx),
        Path(args.tokenizer),
        max_len=int(args.max_len),
        intra=int(args.intra_op),
        inter=1,
        opt_all=True,
    )
    data_dir = Path(args.data_dir)
    report: dict[str, Any] = {
        "tau": args.tau,
        "max_len": args.max_len,
        "cand_cap": args.cand_cap,
        "onnx": args.onnx,
        "sets": {},
        "match": {},
    }
    ok = True
    for name in ("seen", "unseen", "fresh"):
        path = data_dir / f"eval_{name}_v2_clean.jsonl"
        rows = list(read_jsonl(path))
        groups = prepare_groups(rows)
        _restore_flags(groups, rows)
        groups = _cap_groups(groups, int(args.cand_cap))
        rec = eval_one(runner, groups, float(args.tau))
        report["sets"][name] = rec
        exp = EXPECTED[name]
        hit = rec["all"]["final_hit1"]
        cs = rec["cs_delta_pt"]
        reg = rec["all"]["regression_rate_on_mozc_hit"]
        hit_ok = abs(hit - exp["hit1"]) <= 0.005
        cs_ok = cs is not None and abs(cs - exp["cs_delta_pt"]) <= 0.8
        reg_ok = abs(reg - exp["reg"]) <= 0.005
        report["match"][name] = {
            "hit1_ok": hit_ok,
            "cs_delta_ok": cs_ok,
            "reg_ok": reg_ok,
            "hit1": hit,
            "expected_hit1": exp["hit1"],
            "cs_delta_pt": cs,
            "expected_cs_delta_pt": exp["cs_delta_pt"],
            "reg": reg,
            "expected_reg": exp["reg"],
        }
        if not (hit_ok and cs_ok and reg_ok):
            ok = False
        print(
            f"{name} hit1={hit:.4f} (exp {exp['hit1']:.4f}) "
            f"csΔ={cs} (exp {exp['cs_delta_pt']}) reg={reg:.4f}",
            flush=True,
        )

    report["ok"] = ok
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE wrote {out} ok={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
