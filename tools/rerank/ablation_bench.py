"""Ablation: batch / top-K / ONNX-int8 × GPU/CPU on holdout.

Patterns (7) + optional baseline (all-off):
  batch | topk | onnx | batch+topk | batch+onnx | topk+onnx | batch+topk+onnx

Each pattern runs on device=cuda and device=cpu.
Margin gate uses tau (default 2.0).

ONNX Runtime in this ROCm container only exposes CPUExecutionProvider;
onnx+cuda requests are executed on ORT CPU and flagged actual_device=cpu_ort.

Example:
  python -m tools.rerank.ablation_bench \\
    --data data/rerank_v2/holdout.jsonl \\
    --ckpt artifacts/rerank/modernbert70m_ce \\
    --onnx artifacts/rerank/modernbert70m_ce/onnx/cross_encoder_int8.onnx \\
    --tau 2 --topk 5 \\
    --out artifacts/rerank/ablation_report.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tools.dataset.jsonl import read_jsonl
from tools.rerank.eval_cross_encoder import prepare_groups
from tools.rerank.margin import metrics_at_tau
from tools.rerank.train_cross_encoder import build_pair_text


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _select_cands(
    cands: list[str], topk: int | None, mozc_top1: str | None = None
) -> list[str]:
    if not topk or topk <= 0 or topk >= len(cands):
        selected = list(cands)
    else:
        selected = list(cands[:topk])
    if mozc_top1 and mozc_top1 not in selected:
        # Always keep Mozc top-1 so margin gate remains defined.
        selected = [mozc_top1, *[c for c in selected if c != mozc_top1]]
        if topk and topk > 0:
            selected = selected[:topk]
            if mozc_top1 not in selected:
                selected = [mozc_top1, *selected[:-1]]
    return selected


@dataclass
class Pattern:
    name: str
    batch: bool
    topk: bool
    onnx: bool

    def label(self) -> str:
        flags = []
        if self.batch:
            flags.append("batch")
        if self.topk:
            flags.append("topk")
        if self.onnx:
            flags.append("onnx_int8")
        return "+".join(flags) if flags else "baseline_seq_full_pt"


def all_patterns(include_baseline: bool = True) -> list[Pattern]:
    out: list[Pattern] = []
    if include_baseline:
        out.append(Pattern("baseline", False, False, False))
    # Non-ONNX first (GPU-friendly), then ONNX (ORT-CPU).
    combos = [
        (True, False, False),   # batch
        (False, True, False),   # topk
        (True, True, False),    # batch+topk
        (False, False, True),   # onnx
        (True, False, True),    # batch+onnx
        (False, True, True),    # topk+onnx
        (True, True, True),     # all
    ]
    for batch, topk, onnx in combos:
        out.append(Pattern(Pattern("", batch, topk, onnx).label(), batch, topk, onnx))
    return out


class Scorer:
    def score_batch(self, texts: list[str]) -> list[float]:
        raise NotImplementedError

    def score_group(self, texts: list[str], batched: bool) -> list[float]:
        if not texts:
            return []
        if batched:
            return self.score_batch(texts)
        scores: list[float] = []
        for t in texts:
            scores.extend(self.score_batch([t]))
        return scores

    def score_many(self, texts: list[str], chunk: int = 64) -> list[float]:
        """Score a flat list in chunks (fast path for full-holdout quality)."""
        if not texts:
            return []
        out: list[float] = []
        for i in range(0, len(texts), max(1, chunk)):
            out.extend(self.score_batch(texts[i : i + chunk]))
        return out


class TorchScorer(Scorer):
    def __init__(self, ckpt: Path, device: str, max_len: int, fp16: bool):
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_len = max_len
        self.use_amp = bool(fp16 and device == "cuda")

        blob = torch.load(ckpt / "cross_encoder.pt", map_location="cpu", weights_only=False)
        base = blob.get("base_model")
        if not base:
            base = json.loads((ckpt / "train_meta.json").read_text(encoding="utf-8"))[
                "base_model"
            ]
        tok_dir = ckpt / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tok_dir if tok_dir.exists() else base), trust_remote_code=True
        )

        class CrossEncoder(nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(
                    name, trust_remote_code=True, torch_dtype=torch.float32
                )
                hidden = int(self.encoder.config.hidden_size)
                self.score = nn.Linear(hidden, 1)

            def forward(self, input_ids, attention_mask):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0]
                return self.score(cls).squeeze(-1)

        self.model = CrossEncoder(base)
        self.model.load_state_dict(blob["model"], strict=True)
        self.model.to(device)
        self.model.eval()
        self.base = base
        self.actual_device = device

    def score_batch(self, texts: list[str]) -> list[float]:
        torch = self.torch
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_len,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
                    logits = self.model(input_ids, attention_mask)
                torch.cuda.synchronize()
            else:
                logits = self.model(input_ids, attention_mask)
        return [float(x) for x in logits.detach().float().cpu().tolist()]


class OnnxScorer(Scorer):
    def __init__(self, onnx_path: Path, tokenizer_dir: Path, max_len: int, prefer_gpu: bool):
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.np = np
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
        available = ort.get_available_providers()
        providers: list[str] = []
        if prefer_gpu:
            for name in ("CUDAExecutionProvider", "ROCMExecutionProvider", "DmlExecutionProvider"):
                if name in available:
                    providers.append(name)
        providers.append("CPUExecutionProvider")
        # unique
        seen: set[str] = set()
        prov = []
        for x in providers:
            if x not in seen:
                seen.add(x)
                prov.append(x)
        self.session = ort.InferenceSession(str(onnx_path), providers=prov)
        self.active_providers = self.session.get_providers()
        self.actual_device = (
            "cuda"
            if any(p.startswith("CUDA") or p.startswith("ROCM") or p.startswith("Dml") for p in self.active_providers)
            else "cpu"
        )
        self.base = f"onnx:{onnx_path.name}"

    def score_batch(self, texts: list[str]) -> list[float]:
        np = self.np
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_len,
            padding=True,
            return_tensors="np",
        )
        feeds = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        out = self.session.run(None, feeds)[0]
        # shape (batch,) or (batch,1)
        arr = np.array(out).reshape(-1)
        return [float(x) for x in arr.tolist()]


def run_pattern(
    groups: list[dict[str, Any]],
    *,
    pattern: Pattern,
    scorer: Scorer,
    tau: float,
    topk_k: int,
    device_requested: str,
    latency_limit: int,
    quality_cache_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Latency honors batch/seq; quality is full holdout (optionally cached by onnx×topk)."""
    use_k = topk_k if pattern.topk else 0
    per_ms: list[float] = []
    n_timed = 0

    if groups:
        warm_c = _select_cands(groups[0]["candidates"], use_k or None, groups[0]["mozc_top1"])
        warm_t = [
            build_pair_text(groups[0]["reading"], groups[0]["context_prev"], c) for c in warm_c
        ]
        _ = scorer.score_group(warm_t, batched=True)

    lat_n = len(groups) if latency_limit <= 0 else min(latency_limit, len(groups))
    timed_cands = 0
    t_lat0 = time.perf_counter()
    for gi in range(lat_n):
        g = groups[gi]
        cands = _select_cands(g["candidates"], use_k or None, g["mozc_top1"])
        texts = [build_pair_text(g["reading"], g["context_prev"], c) for c in cands]
        timed_cands += len(cands)
        t0 = time.perf_counter()
        _ = scorer.score_group(texts, batched=pattern.batch)
        per_ms.append((time.perf_counter() - t0) * 1000.0)
        n_timed += 1
    lat_wall = time.perf_counter() - t_lat0

    quality_bundle: dict[str, Any]
    if quality_cache_entry is not None:
        metrics = quality_cache_entry["metrics"]
        n_cands_scored = quality_cache_entry["n_cands_scored"]
        quality_wall = 0.0
        quality_bundle = quality_cache_entry
        cached_quality = True
    else:
        metric_groups: list[dict[str, Any]] = []
        flat_texts: list[str] = []
        offsets: list[tuple[int, int]] = []
        for g in groups:
            cands = _select_cands(g["candidates"], use_k or None, g["mozc_top1"])
            start = len(flat_texts)
            for c in cands:
                flat_texts.append(build_pair_text(g["reading"], g["context_prev"], c))
            offsets.append((start, len(flat_texts)))
            metric_groups.append({**g, "candidates": cands})
        n_cands_scored = len(flat_texts)
        chunk = 64 if getattr(scorer, "actual_device", device_requested) == "cpu" else 256
        if pattern.onnx:
            chunk = 128
        t_q0 = time.perf_counter()
        flat_scores = scorer.score_many(flat_texts, chunk=chunk)
        quality_wall = time.perf_counter() - t_q0
        group_scores = [flat_scores[a:b] for a, b in offsets]
        metrics = metrics_at_tau(metric_groups, group_scores, tau)
        quality_bundle = {"metrics": metrics, "n_cands_scored": n_cands_scored}
        cached_quality = False

    lat = {
        "n_timed_groups": n_timed,
        "latency_wall_s": round(lat_wall, 3),
        "quality_wall_s": round(quality_wall, 3),
        "ms_per_group_mean": round(statistics.mean(per_ms), 3) if per_ms else None,
        "p50_ms_per_group": round(_percentile(per_ms, 50), 3) if per_ms else None,
        "p95_ms_per_group": round(_percentile(per_ms, 95), 3) if per_ms else None,
        "max_ms_per_group": round(max(per_ms), 3) if per_ms else None,
        "ms_per_candidate_mean": round(sum(per_ms) / timed_cands, 4)
        if per_ms and timed_cands
        else None,
        "candidates_scored_total_quality": n_cands_scored,
        "avg_cands_per_group": round(n_cands_scored / max(1, len(groups)), 3),
        "latency_mode": "batch" if pattern.batch else "sequential",
        "quality_scoring": "cached" if cached_quality else "flat_chunked_full_holdout",
    }

    return {
        "pattern": pattern.label(),
        "flags": {"batch": pattern.batch, "topk": pattern.topk, "onnx_int8": pattern.onnx},
        "device_requested": device_requested,
        "actual_device": getattr(scorer, "actual_device", device_requested),
        "topk_k": use_k if pattern.topk else None,
        "tau": tau,
        "n_groups": len(groups),
        "metrics": metrics,
        "latency": lat,
        "backend": getattr(scorer, "base", "unknown"),
        "ort_providers": getattr(scorer, "active_providers", None),
        "quality_bundle": quality_bundle,
    }


def _write_report(out: Path, args: argparse.Namespace, groups: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    table = []
    for r in results:
        m = r.get("metrics") or {}
        L = r.get("latency") or {}
        table.append(
            {
                "device_req": r.get("device_requested"),
                "actual": r.get("actual_device"),
                "pattern": r.get("pattern"),
                "final_hit1": m.get("final_hit1"),
                "mozc_hit1": m.get("mozc_hit1"),
                "delta_pt": m.get("delta_vs_mozc_pt"),
                "recovery": m.get("recovery_rate_on_mozc_miss"),
                "regression": m.get("regression_rate_on_mozc_hit"),
                "overwrite": m.get("overwrite_rate"),
                "p50_ms": L.get("p50_ms_per_group"),
                "p95_ms": L.get("p95_ms_per_group"),
                "mean_ms": L.get("ms_per_group_mean"),
                "ms_per_cand": L.get("ms_per_candidate_mean"),
                "avg_cands": L.get("avg_cands_per_group"),
                "n_timed": L.get("n_timed_groups"),
            }
        )
    report = {
        "data": str(args.data),
        "ckpt": str(args.ckpt),
        "onnx": str(args.onnx),
        "tau": args.tau,
        "topk_k": args.topk,
        "n_groups": len(groups),
        "note_onnx_gpu": (
            "onnxruntime in this image has CPUExecutionProvider only; "
            "onnx+cuda rows use ORT on CPU (actual_device=cpu)."
        ),
        "results": results,
        "table": table,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rerank inference ablation bench")
    p.add_argument("--data", default="data/rerank_v2/holdout.jsonl")
    p.add_argument("--ckpt", default="artifacts/rerank/modernbert70m_ce")
    p.add_argument(
        "--onnx",
        default="artifacts/rerank/modernbert70m_ce/onnx/cross_encoder_int8.onnx",
    )
    p.add_argument("--out", default="artifacts/rerank/ablation_report.json")
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--devices", default="cuda,cpu", help="comma list: cuda,cpu")
    p.add_argument("--quality-chunk", type=int, default=0, help="override flat quality chunk size")
    p.add_argument(
        "--resume-from",
        default="",
        help="optional partial JSON (table) to merge/skip completed device+pattern",
    )
    p.add_argument(
        "--patterns",
        default="all",
        help="all | or comma names like batch,topk,onnx_int8,batch+topk,...",
    )
    p.add_argument("--include-baseline", action="store_true", default=True)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument(
        "--latency-limit",
        type=int,
        default=0,
        help="if >0, only first N groups contribute to latency stats (quality still full)",
    )
    p.add_argument(
        "--cpu-latency-limit",
        type=int,
        default=100,
        help="CPU sequential-heavy patterns: latency sample size (quality still full)",
    )
    p.add_argument("--limit-groups", type=int, default=0, help="debug: cap groups")
    args = p.parse_args(argv)
    if args.no_baseline:
        args.include_baseline = False

    import torch

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    patterns = all_patterns(include_baseline=args.include_baseline)
    if args.patterns != "all":
        wanted = {x.strip() for x in args.patterns.split(",") if x.strip()}
        patterns = [x for x in patterns if x.label() in wanted or x.name in wanted]

    rows = list(read_jsonl(Path(args.data)))
    groups = prepare_groups(rows)
    if args.limit_groups and args.limit_groups > 0:
        groups = groups[: args.limit_groups]
    print(f"groups={len(groups)} patterns={len(patterns)} devices={devices}", flush=True)

    # Preload scorers per device/backend
    ckpt = Path(args.ckpt)
    onnx_path = Path(args.onnx)
    tok_onnx = onnx_path.parent / "tokenizer"
    if not tok_onnx.exists():
        tok_onnx = ckpt / "tokenizer"

    results: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str]] = set()
    if args.resume_from:
        prev = json.loads(Path(args.resume_from).read_text(encoding="utf-8"))
        for r in prev.get("results") or []:
            results.append(r)
            done_keys.add((r["device_requested"], r["pattern"]))
        for t in prev.get("table") or []:
            # reconstruct minimal result stubs only if results empty
            key = (t.get("device_req"), t.get("pattern"))
            if key in done_keys:
                continue
            # keep table-only resume as lightweight metrics
            results.append(
                {
                    "pattern": t["pattern"],
                    "device_requested": t["device_req"],
                    "actual_device": t.get("actual"),
                    "metrics": {
                        "final_hit1": t.get("final_hit1"),
                        "delta_vs_mozc_pt": t.get("delta_pt"),
                        "regression_rate_on_mozc_hit": t.get("regression"),
                        "mozc_hit1": None,
                        "recovery_rate_on_mozc_miss": None,
                        "overwrite_rate": None,
                    },
                    "latency": {
                        "p50_ms_per_group": t.get("p50_ms"),
                        "p95_ms_per_group": t.get("p95_ms"),
                        "ms_per_group_mean": t.get("mean_ms"),
                        "ms_per_candidate_mean": t.get("ms_per_cand"),
                        "avg_cands_per_group": t.get("avg_cands"),
                        "n_timed_groups": t.get("n_timed"),
                    },
                    "flags": {},
                    "tau": args.tau,
                    "n_groups": len(groups),
                    "resumed_from_table": True,
                }
            )
            done_keys.add(key)
        print(f"resume skip={len(done_keys)}", flush=True)

    # Shared across devices: ONNX int8 always runs on ORT-CPU in this image.
    onnx_quality_cache: dict[bool, dict[str, Any]] = {}

    for device in devices:
        if device == "cuda":
            if not torch.cuda.is_available():
                print("SKIP cuda: unavailable", flush=True)
                continue
            _ = torch.zeros(8, device="cuda")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        pt_scorer = None
        onnx_scorer = None
        quality_cache: dict[tuple[bool, bool], dict[str, Any]] = {}

        for pat in patterns:
            if (device, pat.label()) in done_keys:
                print(f"=== SKIP device={device} pattern={pat.label()} (resume) ===", flush=True)
                continue
            print(f"=== RUN device={device} pattern={pat.label()} ===", flush=True)
            if pat.onnx:
                if onnx_scorer is None:
                    if not onnx_path.exists():
                        raise SystemExit(f"ONNX model missing: {onnx_path}")
                    onnx_scorer = OnnxScorer(
                        onnx_path, tok_onnx, args.max_len, prefer_gpu=(device == "cuda")
                    )
                    print(
                        f"ORT providers={onnx_scorer.active_providers} actual={onnx_scorer.actual_device}",
                        flush=True,
                    )
                scorer: Scorer = onnx_scorer
            else:
                if pt_scorer is None or pt_scorer.device != device:
                    pt_scorer = TorchScorer(
                        ckpt, device=device, max_len=args.max_len, fp16=(device == "cuda")
                    )
                    quality_cache = {}
                scorer = pt_scorer

            lat_lim = args.latency_limit
            if device == "cpu" and lat_lim <= 0:
                lat_lim = args.cpu_latency_limit
            # GPU-requested ONNX still runs on ORT-CPU; keep latency sample small.
            if pat.onnx and device == "cuda" and args.latency_limit <= 0:
                lat_lim = args.cpu_latency_limit

            if pat.onnx:
                cached = onnx_quality_cache.get(pat.topk)
            else:
                cached = quality_cache.get((False, pat.topk))
            if cached is None:
                # Reuse metrics from already-finished rows with same backend×topk.
                for r in results:
                    flags = r.get("flags") or {}
                    same_topk = bool(flags.get("topk")) == bool(pat.topk)
                    same_onnx = bool(flags.get("onnx_int8")) == bool(pat.onnx)
                    m = (r.get("metrics") or {}).get("final_hit1")
                    if m is None or not same_topk or not same_onnx:
                        continue
                    if pat.onnx or r.get("device_requested") == device:
                        cached = {
                            "metrics": r["metrics"],
                            "n_cands_scored": (r.get("latency") or {}).get(
                                "candidates_scored_total_quality"
                            )
                            or 0,
                        }
                        print(
                            f"  quality_reuse from {r.get('device_requested')}/{r.get('pattern')}",
                            flush=True,
                        )
                        break

            rec = run_pattern(
                groups,
                pattern=pat,
                scorer=scorer,
                tau=args.tau,
                topk_k=args.topk,
                device_requested=device,
                latency_limit=lat_lim,
                quality_cache_entry=cached,
            )
            if cached is None and "quality_bundle" in rec:
                bundle = rec.pop("quality_bundle")
                if pat.onnx:
                    onnx_quality_cache[pat.topk] = bundle
                else:
                    quality_cache[(False, pat.topk)] = bundle
            elif "quality_bundle" in rec:
                rec.pop("quality_bundle", None)
            results.append(rec)
            done_keys.add((device, pat.label()))
            m = rec["metrics"]
            L = rec["latency"]
            print(
                f"  final_hit1={m['final_hit1']:.4f} dMozc={m['delta_vs_mozc_pt']:+.2f} "
                f"reg={m['regression_rate_on_mozc_hit']:.4f} "
                f"p50={L['p50_ms_per_group']} p95={L['p95_ms_per_group']} "
                f"actual={rec['actual_device']}",
                flush=True,
            )

            if device == "cuda" and not pat.onnx:
                torch.cuda.empty_cache()

            _write_report(Path(args.out), args, groups, results)

    _write_report(Path(args.out), args, groups, results)
    report = json.loads(Path(args.out).read_text(encoding="utf-8"))
    table = report["table"]
    print("TABLE", flush=True)
    print(
        f"{'dev':<5} {'pattern':<22} {'hit':>6} {'dPt':>6} {'reg':>6} {'p50ms':>8} {'p95ms':>8}",
        flush=True,
    )
    for t in table:
        hit = t.get("final_hit1")
        dpt = t.get("delta_pt")
        reg = t.get("regression")
        p50 = t.get("p50_ms")
        p95 = t.get("p95_ms")
        print(
            f"{t['device_req']:<5} {t['pattern']:<22} "
            f"{(hit if hit is not None else -1):.4f} "
            f"{(dpt if dpt is not None else 0):+6.2f} "
            f"{(reg if reg is not None else -1):.4f} "
            f"{(p50 if p50 is not None else -1):8.1f} "
            f"{(p95 if p95 is not None else -1):8.1f}",
            flush=True,
        )
    print(f"DONE wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
