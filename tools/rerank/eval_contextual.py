"""Contextual eval: context ON vs OFF + context_sensitive subset (PLAN §5 / CTX v2).

Primary subset flag: context_sensitive (NOT old ambiguous).
Ambiguous is retained as an auxiliary comparison only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.dataset.jsonl import read_jsonl
from tools.rerank.eval_cross_encoder import (
    evaluate_groups,
    load_model,
    prepare_groups,
)
from tools.rerank.margin import metrics_at_tau


def _subset_metrics(
    groups: list[dict[str, Any]],
    scores: list[list[float]],
    tau: float,
    predicate,
) -> dict[str, Any]:
    idx = [i for i, g in enumerate(groups) if predicate(g)]
    if not idx:
        return {"n": 0}
    gs = [groups[i] for i in idx]
    sc = [scores[i] for i in idx]
    m = metrics_at_tau(gs, sc, tau)
    return {
        "n": len(gs),
        "final_hit1": m["final_hit1"],
        "mozc_hit1": m["mozc_hit1"],
        "delta_vs_mozc_pt": m["delta_vs_mozc_pt"],
        "regression_rate_on_mozc_hit": m["regression_rate_on_mozc_hit"],
        "recovery_rate_on_mozc_miss": m["recovery_rate_on_mozc_miss"],
        "overwrite_rate": m["overwrite_rate"],
    }


def _restore_flags(groups: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Restore subset flags dropped by prepare_groups (via original row idx)."""
    for g in groups:
        i = int(g.get("idx", -1))
        if 0 <= i < len(rows):
            r = rows[i]
            g["ambiguous"] = bool(r.get("ambiguous"))
            g["context_sensitive"] = bool(r.get("context_sensitive"))
        else:
            g["ambiguous"] = False
            g["context_sensitive"] = False


def run_one(
    groups: list[dict[str, Any]],
    *,
    blank_context: bool,
    tokenizer,
    model,
    device: str,
    max_len: int,
    batch_size: int,
    use_amp: bool,
    tau_sweep: list[float],
) -> dict[str, Any]:
    gs = []
    for g in groups:
        g2 = dict(g)
        if blank_context:
            g2["context_prev"] = ""
        gs.append(g2)

    report = evaluate_groups(
        gs,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=max_len,
        batch_size=batch_size,
        use_amp=use_amp,
        measure_latency=False,
        tau=0.0,
        tau_sweep=tau_sweep,
    )
    rec = report.get("recommended_tau") or {}
    tau = float(rec.get("tau") if rec else 0.0)

    from tools.rerank.eval_cross_encoder import score_groups

    scores, _ = score_groups(
        gs,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=max_len,
        batch_size=batch_size,
        use_amp=use_amp,
        measure_latency=False,
    )
    report["context_mode"] = "blank" if blank_context else "with_context"
    report["at_recommended_tau"] = {
        "tau": tau,
        "all": _subset_metrics(gs, scores, tau, lambda g: True),
        "context_sensitive": _subset_metrics(
            gs, scores, tau, lambda g: bool(g.get("context_sensitive"))
        ),
        "non_context_sensitive": _subset_metrics(
            gs, scores, tau, lambda g: not bool(g.get("context_sensitive"))
        ),
        # Auxiliary (legacy ambiguous); not primary gate.
        "ambiguous": _subset_metrics(gs, scores, tau, lambda g: bool(g.get("ambiguous"))),
        "non_ambiguous": _subset_metrics(
            gs, scores, tau, lambda g: not bool(g.get("ambiguous"))
        ),
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--tau-sweep", default="0,0.5,1,1.5,2,2.5,3,4,5")
    p.add_argument("--model-name", default="")
    args = p.parse_args()

    import torch

    rows = list(read_jsonl(Path(args.data)))
    groups = prepare_groups(rows)
    _restore_flags(groups, rows)

    if args.device == "cuda":
        assert torch.cuda.is_available()
        _ = torch.zeros(8, device="cuda")
        torch.cuda.synchronize()

    tokenizer, model, base, use_amp = load_model(Path(args.ckpt), args.device, True)
    sweep = [float(x) for x in args.tau_sweep.split(",") if x.strip()]
    n_cs = sum(1 for g in groups if g.get("context_sensitive"))
    n_amb = sum(1 for g in groups if g.get("ambiguous"))
    print(
        f"groups={len(groups)} base={base} context_sensitive={n_cs} ambiguous={n_amb}",
        flush=True,
    )

    with_ctx = run_one(
        groups,
        blank_context=False,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        max_len=args.max_len,
        batch_size=args.batch_size,
        use_amp=use_amp,
        tau_sweep=sweep,
    )
    no_ctx = run_one(
        groups,
        blank_context=True,
        tokenizer=tokenizer,
        model=model,
        device=args.device,
        max_len=args.max_len,
        batch_size=args.batch_size,
        use_amp=use_amp,
        tau_sweep=sweep,
    )

    def delta(a: dict, b: dict) -> float | None:
        if not a or not b or a.get("n", 0) == 0:
            return None
        return round((a["final_hit1"] - b["final_hit1"]) * 100, 3)

    cs_with = with_ctx["at_recommended_tau"]["context_sensitive"]
    cs_blank = no_ctx["at_recommended_tau"]["context_sensitive"]
    amb_with = with_ctx["at_recommended_tau"]["ambiguous"]
    amb_blank = no_ctx["at_recommended_tau"]["ambiguous"]
    summary = {
        "model_name": args.model_name or base,
        "ckpt": str(args.ckpt),
        "data": str(args.data),
        "max_len": args.max_len,
        "n_groups": len(groups),
        "n_context_sensitive": n_cs,
        "n_ambiguous": n_amb,
        "primary_subset": "context_sensitive",
        "with_context": with_ctx["at_recommended_tau"],
        "blank_context": no_ctx["at_recommended_tau"],
        "context_delta_pt": {
            "all": delta(
                with_ctx["at_recommended_tau"]["all"],
                no_ctx["at_recommended_tau"]["all"],
            ),
            "context_sensitive": delta(cs_with, cs_blank),
            "non_context_sensitive": delta(
                with_ctx["at_recommended_tau"]["non_context_sensitive"],
                no_ctx["at_recommended_tau"]["non_context_sensitive"],
            ),
            "ambiguous": delta(amb_with, amb_blank),
            "non_ambiguous": delta(
                with_ctx["at_recommended_tau"]["non_ambiguous"],
                no_ctx["at_recommended_tau"]["non_ambiguous"],
            ),
        },
        "full_with_context": with_ctx,
        "full_blank_context": no_ctx,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "context_delta_pt": summary["context_delta_pt"],
                "with": summary["with_context"],
                "blank": summary["blank_context"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
