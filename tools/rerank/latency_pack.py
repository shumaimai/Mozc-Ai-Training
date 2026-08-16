"""Latency packing for ONNX fp32 (no int8, no topK).

Steps from docs/NEXT_TASK_LATENCY.md:
  1) token length distribution + max_length sweep (quality + latency)
  2) ORT SessionOptions tuning
  3) latency vs n_candidates / warmup isolation

Example:
  python -m tools.rerank.latency_pack \\
    --data data/rerank_v2/holdout.jsonl \\
    --onnx artifacts/rerank/modernbert70m_ce/onnx/cross_encoder_fp32.onnx \\
    --tokenizer artifacts/rerank/modernbert70m_ce/onnx/tokenizer \\
    --out-dir artifacts/rerank
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tools.dataset.jsonl import read_jsonl
from tools.rerank.eval_cross_encoder import prepare_groups
from tools.rerank.margin import metrics_at_tau
from tools.rerank.train_cross_encoder import build_pair_text


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return float(ys[f])
    return float(ys[f] + (ys[c] - ys[f]) * (k - f))


def _summary(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    return {
        "n": len(xs),
        "mean": round(statistics.mean(xs), 3),
        "p50": round(_pct(xs, 50), 3),
        "p90": round(_pct(xs, 90), 3),
        "p95": round(_pct(xs, 95), 3),
        "p99": round(_pct(xs, 99), 3),
        "max": round(max(xs), 3),
        "min": round(min(xs), 3),
    }


class OrtRunner:
    def __init__(
        self,
        onnx_path: Path,
        tokenizer_dir: Path,
        *,
        max_len: int,
        intra: int,
        inter: int,
        opt_all: bool = True,
        padding: str = "longest",
    ):
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.np = np
        self.max_len = max_len
        # longest = pad to batch max (IME). max_length = always pad to max_len
        # (training used max_length; wastes compute when seq ≪ 128).
        if padding not in ("longest", "max_length"):
            raise ValueError(f"padding must be longest|max_length, got {padding}")
        self.padding = padding
        self.n_session_runs = 0
        self.last_forward: dict[str, Any] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir), trust_remote_code=True
        )
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, intra)
        so.inter_op_num_threads = max(1, inter)
        if opt_all:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.cfg = {
            "max_len": max_len,
            "padding": padding,
            "intra_op_num_threads": intra,
            "inter_op_num_threads": inter,
            "graph_optimization_level": "ALL" if opt_all else "BASIC",
        }

    def score_texts(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        np = self.np
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_len,
            padding=self.padding,
            return_tensors="np",
        )
        ids = enc["input_ids"].astype(np.int64)
        mask = enc["attention_mask"].astype(np.int64)
        if ids.ndim != 2:
            raise RuntimeError(f"expected 2d input_ids, got {ids.shape}")
        self.last_forward = {
            "n_texts": len(texts),
            "batch": int(ids.shape[0]),
            "seq": int(ids.shape[1]),
            "one_forward": ids.shape[0] == len(texts),
        }
        feeds = {"input_ids": ids, "attention_mask": mask}
        out = self.session.run(None, feeds)[0]
        self.n_session_runs += 1
        scores = [float(x) for x in np.array(out).reshape(-1).tolist()]
        if len(scores) != len(texts):
            raise RuntimeError(
                f"ORT returned {len(scores)} scores for {len(texts)} texts"
            )
        return scores

    def token_lens(self, texts: list[str]) -> list[int]:
        # true lengths without pad (for distribution)
        enc = self.tokenizer(
            texts,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        return [len(x) for x in enc["input_ids"]]


def _score_groups_batched(
    runner: OrtRunner, groups: list[dict[str, Any]], *, chunk: int = 256
) -> list[list[float]]:
    """Score all groups; flatten into large ORT batches then re-slice."""
    flat_texts: list[str] = []
    sizes: list[int] = []
    for g in groups:
        texts = [build_pair_text(g["reading"], g["context_prev"], c) for c in g["candidates"]]
        sizes.append(len(texts))
        flat_texts.extend(texts)
    flat_scores: list[float] = []
    for i in range(0, len(flat_texts), chunk):
        flat_scores.extend(runner.score_texts(flat_texts[i : i + chunk]))
    out: list[list[float]] = []
    off = 0
    for n in sizes:
        out.append(flat_scores[off : off + n])
        off += n
    return out


def eval_quality_latency(
    runner: OrtRunner,
    groups: list[dict[str, Any]],
    *,
    tau: float,
    latency_n: int,
    warmup: int = 3,
    full_metrics: bool = True,
) -> dict[str, Any]:
    metric_groups = groups if full_metrics else groups[: max(latency_n, 80)]
    group_scores = _score_groups_batched(runner, metric_groups)
    metrics = metrics_at_tau(metric_groups, group_scores, tau)

    # Warmup then timed sample
    for i in range(min(warmup, len(groups))):
        g = groups[i]
        texts = [build_pair_text(g["reading"], g["context_prev"], c) for c in g["candidates"]]
        _ = runner.score_texts(texts)

    per_ms: list[float] = []
    rows: list[dict[str, Any]] = []
    n = min(latency_n, len(groups))
    runs_before = runner.n_session_runs
    seqs: list[int] = []
    batches: list[int] = []
    one_forward_ok = 0
    for i in range(n):
        g = groups[i]
        texts = [build_pair_text(g["reading"], g["context_prev"], c) for c in g["candidates"]]
        t0 = time.perf_counter()
        _ = runner.score_texts(texts)
        dt = (time.perf_counter() - t0) * 1000.0
        per_ms.append(dt)
        meta = dict(runner.last_forward)
        seqs.append(int(meta.get("seq", 0)))
        batches.append(int(meta.get("batch", 0)))
        if meta.get("one_forward") and meta.get("n_texts") == len(texts):
            one_forward_ok += 1
        rows.append(
            {
                "idx": i,
                "n_cand": len(g["candidates"]),
                "ms": round(dt, 3),
                "batch": meta.get("batch"),
                "seq": meta.get("seq"),
                "one_forward": meta.get("one_forward"),
                "reading": g["reading"][:40],
            }
        )
    timed_runs = runner.n_session_runs - runs_before

    # cold start (new session not here — approximate first call without warmup on a copy path)
    return {
        "metrics": {
            "final_hit1": metrics["final_hit1"],
            "mozc_hit1": metrics["mozc_hit1"],
            "delta_vs_mozc_pt": metrics["delta_vs_mozc_pt"],
            "regression_rate_on_mozc_hit": metrics["regression_rate_on_mozc_hit"],
            "recovery_rate_on_mozc_miss": metrics["recovery_rate_on_mozc_miss"],
            "overwrite_rate": metrics["overwrite_rate"],
        },
        "latency": {
            **_summary(per_ms),
            "n_timed": n,
            "warmup": warmup,
        },
        "forward_check": {
            "timed_groups": n,
            "session_runs_during_timed": timed_runs,
            "one_session_run_per_group": timed_runs == n,
            "one_forward_ok": one_forward_ok,
            "seq": _summary([float(x) for x in seqs if x]),
            "batch": _summary([float(x) for x in batches if x]),
            "frac_seq_lt_128": round(
                sum(1 for x in seqs if 0 < x < 128) / max(1, len(seqs)), 4
            ),
        },
        "per_group": rows,
        "cfg": runner.cfg,
    }


def token_length_dist(
    runner: OrtRunner, groups: list[dict[str, Any]], sample_groups: int
) -> dict[str, Any]:
    lens: list[int] = []
    n = min(sample_groups, len(groups))
    for g in groups[:n]:
        texts = [build_pair_text(g["reading"], g["context_prev"], c) for c in g["candidates"]]
        lens.extend(runner.token_lens(texts))
    hist = Counter()
    for L in lens:
        bucket = (L // 8) * 8
        hist[bucket] += 1
    return {
        "n_tokenized_pairs": len(lens),
        "n_groups_sampled": n,
        "summary": _summary([float(x) for x in lens]),
        "hist_by_8": {str(k): hist[k] for k in sorted(hist)},
        "frac_le": {
            str(t): round(sum(1 for x in lens if x <= t) / max(1, len(lens)), 6)
            for t in (32, 40, 48, 64, 80, 96, 128, 160, 256)
        },
    }


def correlate_ncand(per_group: list[dict[str, Any]]) -> dict[str, Any]:
    if len(per_group) < 3:
        return {}
    xs = [float(r["n_cand"]) for r in per_group]
    ys = [float(r["ms"]) for r in per_group]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    pearson = (num / den) if den else float("nan")
    # bucket means
    buckets: dict[str, list[float]] = {}
    for r in per_group:
        b = "1-5" if r["n_cand"] <= 5 else "6-10" if r["n_cand"] <= 10 else "11-15" if r["n_cand"] <= 15 else "16+"
        buckets.setdefault(b, []).append(float(r["ms"]))
    # top 5% slowest
    ordered = sorted(per_group, key=lambda r: r["ms"], reverse=True)
    top = ordered[: max(1, len(ordered) // 20)]
    return {
        "pearson_ncand_vs_ms": round(pearson, 6),
        "bucket_mean_ms": {k: round(statistics.mean(v), 3) for k, v in buckets.items()},
        "slowest_p5_mean_ncand": round(statistics.mean(r["n_cand"] for r in top), 3),
        "slowest_p5_mean_ms": round(statistics.mean(r["ms"] for r in top), 3),
        "overall_mean_ncand": round(statistics.mean(xs), 3),
    }


def main(argv: list[str] | None = None) -> int:
    import os

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/rerank_v2/holdout.jsonl")
    p.add_argument(
        "--onnx",
        default="artifacts/rerank/modernbert70m_ce/onnx/cross_encoder_fp32.onnx",
    )
    p.add_argument(
        "--tokenizer",
        default="artifacts/rerank/modernbert70m_ce/onnx/tokenizer",
    )
    p.add_argument("--out-dir", default="artifacts/rerank")
    p.add_argument("--tau", type=float, default=2.5)
    p.add_argument("--latency-groups", type=int, default=200)
    p.add_argument("--token-sample-groups", type=int, default=500)
    p.add_argument("--cand-cap", type=int, default=30)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument(
        "--intra-op",
        type=int,
        default=0,
        help="ORT intra threads; 0 = os.cpu_count() (PLAN §4.5). Prior ship-profile wrongly used 1.",
    )
    p.add_argument(
        "--padding",
        choices=["longest", "max_length"],
        default="longest",
        help="longest=pad to batch max (serve); max_length=always pad to --max-len (train)",
    )
    p.add_argument(
        "--ship-profile",
        action="store_true",
        help="ship config + ablation (intra=1 vs cpu, longest vs max_length)",
    )
    p.add_argument(
        "--ship-only",
        action="store_true",
        help="with --ship-profile, measure only the recommended serving config",
    )
    p.add_argument(
        "--out-json",
        default="",
        help="ship-profile JSON path (default: <out-dir>/latency_ship_profile.json)",
    )
    args = p.parse_args(argv)

    groups = prepare_groups(list(read_jsonl(Path(args.data))))
    print(f"groups={len(groups)}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ship_profile:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        cpu = os.cpu_count() or 8
        intra_cpu = cpu if int(args.intra_op) <= 0 else int(args.intra_op)
        cap = int(args.cand_cap)
        g2 = []
        for g in groups:
            cands = g["candidates"][:cap] if cap > 0 else g["candidates"]
            if g["mozc_top1"] not in cands:
                cands = [g["mozc_top1"], *[c for c in cands if c != g["mozc_top1"]]][:cap]
            g2.append({**g, "candidates": cands})
        n_cand = [len(g["candidates"]) for g in g2]
        cfgs = [
            {"name": "prev_intra1_pad_max128", "intra": 1, "padding": "max_length"},
            {"name": "intra_cpu_pad_max128", "intra": intra_cpu, "padding": "max_length"},
            {"name": "ship_intra_cpu_pad_longest", "intra": intra_cpu, "padding": "longest"},
        ]
        if args.ship_only:
            cfgs = [cfgs[-1]]
        variants = []
        for cfg in cfgs:
            print(f"=== {cfg['name']} intra={cfg['intra']} pad={cfg['padding']} ===", flush=True)
            runner = OrtRunner(
                Path(args.onnx),
                Path(args.tokenizer),
                max_len=int(args.max_len),
                intra=int(cfg["intra"]),
                inter=1,
                opt_all=True,
                padding=str(cfg["padding"]),
            )
            rec = eval_quality_latency(
                runner,
                g2,
                tau=float(args.tau),
                latency_n=args.latency_groups,
                warmup=5,
                full_metrics=False,
            )
            row = {
                **cfg,
                "metrics": rec["metrics"],
                "latency": rec["latency"],
                "forward_check": rec["forward_check"],
                "p95_vs_150ms": rec["latency"].get("p95", 0) - 150.0,
            }
            variants.append(row)
            print(
                json.dumps(
                    {
                        "name": cfg["name"],
                        "p50": rec["latency"].get("p50"),
                        "p95": rec["latency"].get("p95"),
                        "forward": rec["forward_check"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        ship = next(v for v in variants if v["name"] == "ship_intra_cpu_pad_longest")
        report = {
            "profile": "ship_remeasure",
            "note": (
                "PLAN §4.5: intra=cpu_count, inter=1, OMP_NUM_THREADS=1. "
                "Serve padding=longest (batch max), not train padding=max_length. "
                "One conversion = one ORT session.run over cand_cap texts."
            ),
            "tau": args.tau,
            "max_len": args.max_len,
            "cand_cap": cap,
            "cpu_count": cpu,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "n_groups": len(g2),
            "avg_n_cand": round(sum(n_cand) / max(1, len(n_cand)), 3),
            "max_n_cand": max(n_cand) if n_cand else 0,
            "variants": variants,
            "recommended": ship,
            "p95_vs_150ms": ship["latency"].get("p95", 0) - 150.0,
            "degrade_before_cpp": bool(ship["latency"].get("p95", 0) > 150.0),
        }
        path = Path(args.out_json) if args.out_json else out_dir / "latency_ship_profile.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print("DONE", flush=True)
        return 0

    cpu = os.cpu_count() or 8
    # Baseline runner for token dist (max_len=256 so we see true lengths up to that)
    base = OrtRunner(
        Path(args.onnx),
        Path(args.tokenizer),
        max_len=256,
        intra=max(1, cpu // 2),
        inter=1,
        opt_all=True,
    )
    print("token length dist...", flush=True)
    tdist = token_length_dist(base, groups, args.token_sample_groups)
    (out_dir / "token_len_dist.json").write_text(
        json.dumps(tdist, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("token_len", tdist["summary"], "frac_le", tdist["frac_le"], flush=True)

    # Real lengths are short (p99~36, max~46). Sweep near that, keep 128 as quality baseline.
    p99 = float(tdist["summary"].get("p99", 36))
    pmax = float(tdist["summary"].get("max", 48))
    near = sorted(
        {
            32,
            40,
            48,
            64,
            int(max(32, min(128, ((int(p99) + 7) // 8) * 8))),
            int(max(32, min(128, ((int(pmax) + 7) // 8) * 8))),
            128,  # baseline for -1pt gate
        }
    )
    max_len_list = [m for m in near if m <= 128]
    print(
        "max_len_sweep",
        max_len_list,
        "p99",
        p99,
        "max",
        pmax,
        "frac_le",
        {k: tdist["frac_le"].get(k) for k in ("32", "48", "64")},
        flush=True,
    )

    # Thread configs
    thread_cfgs = [
        {"intra": 1, "inter": 1, "name": "t1x1"},
        {"intra": max(1, cpu // 4), "inter": 1, "name": f"t{max(1,cpu//4)}x1"},
        {"intra": max(1, cpu // 2), "inter": 1, "name": f"t{max(1,cpu//2)}x1"},
        {"intra": cpu, "inter": 1, "name": f"t{cpu}x1"},
        {"intra": max(1, cpu // 2), "inter": 2, "name": f"t{max(1,cpu//2)}x2"},
    ]

    # 1) max_len sweep at best-guess threads
    default_intra = max(1, cpu // 2)
    max_len_rows = []
    baseline_hit = None
    for ml in max_len_list:
        print(f"=== max_len={ml} ===", flush=True)
        runner = OrtRunner(
            Path(args.onnx),
            Path(args.tokenizer),
            max_len=ml,
            intra=default_intra,
            inter=1,
            opt_all=True,
        )
        rec = eval_quality_latency(
            runner, groups, tau=args.tau, latency_n=args.latency_groups, warmup=5
        )
        if baseline_hit is None and ml == 128:
            baseline_hit = rec["metrics"]["final_hit1"]
        row = {
            "max_len": ml,
            "token_cover_est": tdist["frac_le"].get(str(ml)),
            **rec["metrics"],
            **{f"lat_{k}": v for k, v in rec["latency"].items() if k != "n"},
            "cfg": rec["cfg"],
        }
        max_len_rows.append(row)
        print(
            f"  hit={row['final_hit1']:.4f} reg={row['regression_rate_on_mozc_hit']:.4f} "
            f"p50={row['lat_p50']} p95={row['lat_p95']}",
            flush=True,
        )

    if baseline_hit is None:
        baseline_hit = next(
            (r["final_hit1"] for r in max_len_rows if r["max_len"] == 128),
            max_len_rows[-1]["final_hit1"],
        )

    # Pick best max_len: hit within -1pt of baseline(128), minimize p95 then p50
    viable = [
        r
        for r in max_len_rows
        if r["final_hit1"] >= baseline_hit - 0.01
        and r["regression_rate_on_mozc_hit"] <= 0.02 + 1e-9
    ]
    if not viable:
        viable = max_len_rows
    best_ml = min(viable, key=lambda r: (r["lat_p95"], r["lat_p50"], -r["final_hit1"]))
    print("best_max_len", best_ml["max_len"], flush=True)

    # 2) thread sweep at best max_len
    thread_rows = []
    for tc in thread_cfgs:
        print(f"=== threads {tc['name']} max_len={best_ml['max_len']} ===", flush=True)
        runner = OrtRunner(
            Path(args.onnx),
            Path(args.tokenizer),
            max_len=int(best_ml["max_len"]),
            intra=tc["intra"],
            inter=tc["inter"],
            opt_all=True,
        )
        rec = eval_quality_latency(
            runner, groups, tau=args.tau, latency_n=args.latency_groups, warmup=5
        )
        row = {
            "name": tc["name"],
            **tc,
            **rec["metrics"],
            **{f"lat_{k}": v for k, v in rec["latency"].items() if k != "n"},
        }
        thread_rows.append(row)
        print(
            f"  hit={row['final_hit1']:.4f} p50={row['lat_p50']} p95={row['lat_p95']}",
            flush=True,
        )

    viable_t = [
        r
        for r in thread_rows
        if r["final_hit1"] >= baseline_hit - 0.01
        and r["regression_rate_on_mozc_hit"] <= 0.02 + 1e-9
    ]
    if not viable_t:
        viable_t = thread_rows
    best_t = min(viable_t, key=lambda r: (r["lat_p95"], r["lat_p50"], -r["final_hit1"]))

    # Also compare BASIC vs ALL opt at best threads
    print("=== opt BASIC vs ALL ===", flush=True)
    opt_rows = []
    for opt_all, name in [(True, "ALL"), (False, "BASIC")]:
        runner = OrtRunner(
            Path(args.onnx),
            Path(args.tokenizer),
            max_len=int(best_ml["max_len"]),
            intra=int(best_t["intra"]),
            inter=int(best_t["inter"]),
            opt_all=opt_all,
        )
        rec = eval_quality_latency(
            runner, groups, tau=args.tau, latency_n=args.latency_groups, warmup=5
        )
        opt_rows.append(
            {
                "opt": name,
                **rec["metrics"],
                **{f"lat_{k}": v for k, v in rec["latency"].items() if k != "n"},
            }
        )

    # 3) p95 analysis on best config + cold start
    print("=== p95 / cold-start isolation ===", flush=True)
    runner = OrtRunner(
        Path(args.onnx),
        Path(args.tokenizer),
        max_len=int(best_ml["max_len"]),
        intra=int(best_t["intra"]),
        inter=int(best_t["inter"]),
        opt_all=True,
    )
    # cold: first call on fresh session already counted as first below without warmup
    g0 = groups[0]
    texts0 = [build_pair_text(g0["reading"], g0["context_prev"], c) for c in g0["candidates"]]
    t0 = time.perf_counter()
    _ = runner.score_texts(texts0)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    # more warmup
    for i in range(1, 6):
        g = groups[i]
        _ = runner.score_texts(
            [build_pair_text(g["reading"], g["context_prev"], c) for c in g["candidates"]]
        )
    rec = eval_quality_latency(
        runner, groups, tau=args.tau, latency_n=min(400, len(groups)), warmup=0
    )
    corr = correlate_ncand(rec["per_group"])
    # candidate-count cap proposals (accuracy check)
    cap_rows = []
    for cap in (0, 20, 30, 40, 50):
        if cap == 0:
            g2 = groups
            label = "no_cap"
        else:
            g2 = []
            for g in groups:
                cands = g["candidates"][:cap]
                if g["mozc_top1"] not in cands:
                    cands = [g["mozc_top1"], *[c for c in cands if c != g["mozc_top1"]]][:cap]
                g2.append({**g, "candidates": cands})
            label = f"cap_{cap}"
        gs = _score_groups_batched(runner, g2)
        m = metrics_at_tau(g2, gs, args.tau)
        # quick latency on 100
        per = []
        for g in g2[:100]:
            texts = [build_pair_text(g["reading"], g["context_prev"], c) for c in g["candidates"]]
            tt = time.perf_counter()
            _ = runner.score_texts(texts)
            per.append((time.perf_counter() - tt) * 1000.0)
        cap_rows.append(
            {
                "label": label,
                "cap": cap,
                "avg_cands": round(
                    sum(len(g["candidates"]) for g in g2) / max(1, len(g2)), 3
                ),
                "final_hit1": m["final_hit1"],
                "delta_vs_mozc_pt": m["delta_vs_mozc_pt"],
                "regression": m["regression_rate_on_mozc_hit"],
                "recovery": m["recovery_rate_on_mozc_miss"],
                "lat_p50": round(_pct(per, 50), 3),
                "lat_p95": round(_pct(per, 95), 3),
                "hit_drop_vs_baseline": round((m["final_hit1"] - baseline_hit) * 100, 3),
            }
        )
        print("cap", cap_rows[-1], flush=True)

    latency_vs = {
        "best_cfg": {
            "max_len": best_ml["max_len"],
            "intra": best_t["intra"],
            "inter": best_t["inter"],
            "tau": args.tau,
        },
        "cold_first_call_ms": round(cold_ms, 3),
        "warm_latency": rec["latency"],
        "warm_metrics": rec["metrics"],
        "correlation": corr,
        "slowest_examples": sorted(rec["per_group"], key=lambda r: r["ms"], reverse=True)[:15],
    }
    (out_dir / "latency_vs_ncand.json").write_text(
        json.dumps(latency_vs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "baseline_hit_at_max128": baseline_hit,
        "targets": {"p50_ms": 50, "p95_ms": 120, "hit_drop_max_pt": 1.0},
        "cpu_count": cpu,
        "token_len_dist_summary": tdist["summary"],
        "token_frac_le": tdist["frac_le"],
        "max_len_sweep": max_len_rows,
        "best_max_len": best_ml,
        "thread_sweep": thread_rows,
        "best_threads": best_t,
        "opt_sweep": opt_rows,
        "candidate_cap_sweep": cap_rows,
        "latency_vs_ncand_file": "latency_vs_ncand.json",
        "recommended": {
            "backend": "onnx_fp32",
            "max_len": best_ml["max_len"],
            "intra_op_num_threads": best_t["intra"],
            "inter_op_num_threads": best_t["inter"],
            "tau": args.tau,
            "topK": None,
            "int8": False,
            "final_hit1": best_t["final_hit1"],
            "p50_ms": best_t["lat_p50"],
            "p95_ms": best_t["lat_p95"],
            "meets_p50_50": best_t["lat_p50"] <= 50,
            "meets_p95_120": best_t["lat_p95"] <= 120,
            "candidate_cap": None,
        },
        "need_30m": None,
    }
    # Prefer a quality-safe candidate cap if it hits latency targets.
    viable_caps = [
        c
        for c in cap_rows
        if c["final_hit1"] >= baseline_hit - 0.01
        and c["regression"] <= 0.02 + 1e-9
        and c["lat_p50"] <= 50
        and c["lat_p95"] <= 120
    ]
    if viable_caps:
        # Prefer higher cap (less truncation) among those meeting latency.
        best_cap = max(viable_caps, key=lambda c: (c["cap"] if c["cap"] > 0 else 10**9, c["final_hit1"], -c["lat_p95"]))
        report["recommended"].update(
            {
                "candidate_cap": None if best_cap["cap"] == 0 else best_cap["cap"],
                "final_hit1": best_cap["final_hit1"],
                "p50_ms": best_cap["lat_p50"],
                "p95_ms": best_cap["lat_p95"],
                "meets_p50_50": True,
                "meets_p95_120": True,
                "cap_note": best_cap["label"],
            }
        )
    # Decision on 30m
    recm = report["recommended"]
    if recm["meets_p50_50"] and recm["meets_p95_120"]:
        report["need_30m"] = False
        report["decision"] = (
            "70m ONNX fp32 + tuned max_len/threads"
            + (f" + cand_cap={recm['candidate_cap']}" if recm.get("candidate_cap") else "")
            + " meets latency targets; no 30m needed"
        )
    elif recm["final_hit1"] >= baseline_hit - 0.01 and recm["p50_ms"] <= 80 and recm["p95_ms"] <= 180:
        report["need_30m"] = "optional"
        report["decision"] = (
            "Close but not at p50<=50/p95<=120; 30m optional if UX demands more, else ship 70m tuned"
        )
    else:
        report["need_30m"] = True
        report["decision"] = (
            "Tuned 70m still far from p50<=50/p95<=120 while keeping accuracy; consider 30m fp32 next"
        )

    # Update eval_cpu_latency.json nested
    lat_path = out_dir / "eval_cpu_latency.json"
    old: dict[str, Any] = {}
    if lat_path.exists():
        try:
            old = json.loads(lat_path.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    old["latency_pack_onnx_fp32"] = report
    # keep previous keys
    lat_path.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "latency_pack_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({"recommended": report["recommended"], "decision": report["decision"]}, ensure_ascii=False, indent=2), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
