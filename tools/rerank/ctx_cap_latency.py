#!/usr/bin/env python3
"""WSL CPU: ship-profile latency at context caps 50/30/20 for 70m and 30m."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dataset.jsonl import read_jsonl
from tools.rerank.eval_cross_encoder import prepare_groups
from tools.rerank.latency_pack import OrtRunner, eval_quality_latency
from tools.rerank.train_cross_encoder import build_pair_text  # noqa: F401


def apply_ctx_cap(text: str, cap: int) -> str:
    ctx = text or ""
    if cap <= 0:
        return ""
    return ctx[-cap:] if len(ctx) > cap else ctx


def clip_groups(groups: list[dict], cap: int, cand_cap: int) -> list[dict]:
    out = []
    for g in groups:
        cands = g["candidates"][:cand_cap] if cand_cap > 0 else g["candidates"]
        if g["mozc_top1"] not in cands:
            cands = [g["mozc_top1"], *[c for c in cands if c != g["mozc_top1"]]][:cand_cap]
        out.append(
            {
                **g,
                "candidates": cands,
                "context_prev": apply_ctx_cap(g.get("context_prev") or "", cap),
            }
        )
    return out


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-json",
        default="artifacts/rerank_ctx/eval/ctx_cap_latency.json",
    )
    p.add_argument("--intra-op", type=int, default=0)
    args = p.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    data = Path("data/rerank_ctx/eval_unseen_v2_clean.jsonl")
    groups = prepare_groups(list(read_jsonl(data)))
    cpu = os.cpu_count() or 8
    intra = cpu if int(args.intra_op) <= 0 else int(args.intra_op)
    tau = 2.5
    cand_cap = 30
    caps = [50, 30, 20]
    models = [
        {
            "tag": "70m",
            "onnx": Path("artifacts/rerank_ctx/trackB_v2_continue/onnx/cross_encoder_fp32.onnx"),
            "tokenizer": Path("artifacts/rerank_ctx/trackB_v2_continue/onnx/tokenizer"),
        },
        {
            "tag": "30m",
            "onnx": Path("artifacts/rerank_ctx/track30m_ctx/onnx/cross_encoder_fp32.onnx"),
            "tokenizer": Path("artifacts/rerank_ctx/track30m_ctx/onnx/tokenizer"),
        },
    ]
    out_dir = Path("artifacts/rerank_ctx/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in models:
        if not m["onnx"].is_file():
            print(f"SKIP missing onnx {m['onnx']}", flush=True)
            continue
        print(f"=== load {m['tag']} intra={intra} ===", flush=True)
        for cap in caps:
            # Fresh ORT session per cell so cap=50 matches the ship-profile baseline.
            runner = OrtRunner(
                m["onnx"],
                m["tokenizer"],
                max_len=128,
                intra=intra,
                inter=1,
                opt_all=True,
                padding="longest",
            )
            g2 = clip_groups(groups, cap, cand_cap)
            rec = eval_quality_latency(
                runner,
                g2,
                tau=tau,
                latency_n=80,
                warmup=8,
                full_metrics=False,
            )
            row = {
                "tag": m["tag"],
                "cap": cap,
                "intra": intra,
                "padding": "longest",
                "p50": rec["latency"].get("p50"),
                "p95": rec["latency"].get("p95"),
                "mean": rec["latency"].get("mean"),
                "seq": rec["forward_check"].get("seq"),
                "forward_check": {
                    "one_session_run_per_group": rec["forward_check"].get(
                        "one_session_run_per_group"
                    ),
                    "frac_seq_lt_128": rec["forward_check"].get("frac_seq_lt_128"),
                },
                "p95_lt_200": bool(float(rec["latency"].get("p95") or 0) < 200),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {
        "profile": "ctx_cap_latency",
        "tau": tau,
        "cand_cap": cand_cap,
        "cpu_count": cpu,
        "n_timed": 80,
        "rows": rows,
    }
    path = Path(args.out_json)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
