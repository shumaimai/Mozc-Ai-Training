"""Backend-wise holdout eval + tau sweep + gold-rank dist + latency.

Backends: pytorch_fp32, onnx_fp32, onnx_int8
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


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _recommend(sweep: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    rule = "max final_hit1 among tau with regression<=2%"
    for m in sweep:
        if m["regression_rate_on_mozc_hit"] <= 0.02:
            if best is None or m["final_hit1"] > best["final_hit1"]:
                best = m
    if best is None:
        rule = "max final_hit1 among tau with regression<=5%"
        for m in sweep:
            if m["regression_rate_on_mozc_hit"] <= 0.05:
                if best is None or m["final_hit1"] > best["final_hit1"]:
                    best = m
    if best is None:
        rule = "max final_hit1 unconstrained"
        best = max(sweep, key=lambda m: m["final_hit1"])
    return {
        "tau": best["tau"],
        "final_hit1": best["final_hit1"],
        "delta_vs_mozc_pt": best["delta_vs_mozc_pt"],
        "regression_rate_on_mozc_hit": best["regression_rate_on_mozc_hit"],
        "recovery_rate_on_mozc_miss": best["recovery_rate_on_mozc_miss"],
        "overwrite_rate": best["overwrite_rate"],
        "rule": rule,
    }


class Backend:
    name: str

    def score_many(self, texts: list[str], chunk: int = 64) -> list[float]:
        raise NotImplementedError


class PtBackend(Backend):
    def __init__(self, ckpt: Path, max_len: int):
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_len = max_len
        self.name = "pytorch_fp32"
        blob = torch.load(ckpt / "cross_encoder.pt", map_location="cpu", weights_only=False)
        base = blob.get("base_model") or json.loads((ckpt / "train_meta.json").read_text(encoding="utf-8"))["base_model"]
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
                self.score = nn.Linear(int(self.encoder.config.hidden_size), 1)

            def forward(self, input_ids, attention_mask):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                return self.score(out.last_hidden_state[:, 0]).squeeze(-1)

        self.model = CrossEncoder(base)
        self.model.load_state_dict(blob["model"], strict=True)
        self.model.eval()

    def score_many(self, texts: list[str], chunk: int = 64) -> list[float]:
        torch = self.torch
        out: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(texts), chunk):
                batch = texts[i : i + chunk]
                enc = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=self.max_len,
                    padding=True,
                    return_tensors="pt",
                )
                logits = self.model(enc["input_ids"], enc["attention_mask"])
                out.extend(float(x) for x in logits.detach().float().cpu().tolist())
        return out


class OnnxBackend(Backend):
    def __init__(self, onnx_path: Path, tokenizer_dir: Path, max_len: int, name: str):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.np = __import__("numpy")
        self.max_len = max_len
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def score_many(self, texts: list[str], chunk: int = 64) -> list[float]:
        np = self.np
        out: list[float] = []
        for i in range(0, len(texts), chunk):
            batch = texts[i : i + chunk]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_len,
                padding=True,
                return_tensors="np",
            )
            feeds = {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }
            arr = np.array(self.session.run(None, feeds)[0]).reshape(-1)
            out.extend(float(x) for x in arr.tolist())
        return out


def score_groups(backend: Backend, groups: list[dict[str, Any]], chunk: int) -> list[list[float]]:
    flat: list[str] = []
    offsets: list[tuple[int, int]] = []
    for g in groups:
        start = len(flat)
        for c in g["candidates"]:
            flat.append(build_pair_text(g["reading"], g["context_prev"], c))
        offsets.append((start, len(flat)))
    scores = backend.score_many(flat, chunk=chunk)
    return [scores[a:b] for a, b in offsets]


def gold_rank_distribution(groups: list[dict[str, Any]]) -> dict[str, Any]:
    ranks: list[int] = []
    missing = 0
    for g in groups:
        gold = g["gold"]
        cands = g["candidates"]
        if gold in cands:
            ranks.append(cands.index(gold) + 1)  # 1-based Mozc order
        else:
            missing += 1
    ctr = Counter(ranks)
    n = len(groups)
    in_nbest = len(ranks)
    # recovery-relevant: among mozc miss & gold in nbest
    miss_in = []
    for g in groups:
        if g["mozc_top1"] == g["gold"]:
            continue
        if g["gold"] in g["candidates"]:
            miss_in.append(g["candidates"].index(g["gold"]) + 1)
    miss_ctr = Counter(miss_in)

    def cover(ks: list[int], pool: list[int]) -> dict[str, float]:
        out = {}
        for k in ks:
            out[f"K={k}"] = round(sum(1 for r in pool if r <= k) / max(1, len(pool)), 6)
        return out

    return {
        "n_groups": n,
        "gold_in_nbest": in_nbest,
        "gold_missing": missing,
        "rank_hist_all_in_nbest": {str(k): ctr[k] for k in sorted(ctr)},
        "rank_hist_mozc_miss_in_nbest": {str(k): miss_ctr[k] for k in sorted(miss_ctr)},
        "coverage_all_in_nbest": cover([3, 5, 8, 10, 15, 20], ranks),
        "coverage_mozc_miss_in_nbest": cover([3, 5, 8, 10, 15, 20], miss_in),
        "k_for_95pct_miss_in_nbest": next(
            (
                k
                for k in range(1, 101)
                if (sum(1 for r in miss_in if r <= k) / max(1, len(miss_in))) >= 0.95
            ),
            None,
        ),
    }


def latency_batch(
    backend: Backend,
    groups: list[dict[str, Any]],
    n_timed: int,
) -> dict[str, Any]:
    # warmup
    g0 = groups[0]
    _ = backend.score_many(
        [build_pair_text(g0["reading"], g0["context_prev"], c) for c in g0["candidates"]],
        chunk=64,
    )
    per = []
    n = min(n_timed, len(groups))
    for i in range(n):
        g = groups[i]
        texts = [build_pair_text(g["reading"], g["context_prev"], c) for c in g["candidates"]]
        t0 = time.perf_counter()
        _ = backend.score_many(texts, chunk=max(8, len(texts)))
        per.append((time.perf_counter() - t0) * 1000.0)
    return {
        "n_timed": n,
        "p50_ms": round(_percentile(per, 50), 3),
        "p95_ms": round(_percentile(per, 95), 3),
        "mean_ms": round(statistics.mean(per), 3),
        "max_ms": round(max(per), 3),
        "ms_per_cand": round(sum(per) / max(1, sum(len(groups[i]["candidates"]) for i in range(n))), 4),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/rerank_v2/holdout.jsonl")
    p.add_argument("--ckpt", default="artifacts/rerank/modernbert70m_ce")
    p.add_argument("--onnx-dir", default="artifacts/rerank/modernbert70m_ce/onnx")
    p.add_argument("--tau-sweep", default="0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4,5")
    p.add_argument("--latency-groups", type=int, default=100)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--out-dir", default="artifacts/rerank")
    args = p.parse_args(argv)

    groups = prepare_groups(list(read_jsonl(Path(args.data))))
    print(f"groups={len(groups)}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = Path(args.onnx_dir)
    tok = onnx_dir / "tokenizer"
    if not tok.exists():
        tok = Path(args.ckpt) / "tokenizer"

    backends: list[Backend] = [
        PtBackend(Path(args.ckpt), 128),
        OnnxBackend(onnx_dir / "cross_encoder_fp32.onnx", tok, 128, "onnx_fp32"),
        OnnxBackend(onnx_dir / "cross_encoder_int8.onnx", tok, 128, "onnx_int8"),
    ]
    # PyTorch CPU batch latency is known-slow; keep a small sample, focus timing on ONNX.
    latency_n = {
        "pytorch_fp32": min(20, args.latency_groups),
        "onnx_fp32": args.latency_groups,
        "onnx_int8": args.latency_groups,
    }

    taus = _parse_floats(args.tau_sweep)
    policy: dict[str, Any] = {
        "policy": "margin_gate",
        "rule": "final=rerank iff score_r-score_m>=tau else mozc_top1",
        "backends": {},
    }
    speed: dict[str, Any] = {"note": "batch, no topK; accuracy at recommended tau", "rows": []}

    gold = gold_rank_distribution(groups)
    (out_dir / "gold_rank_dist.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("gold_rank k95_miss", gold.get("k_for_95pct_miss_in_nbest"), flush=True)

    for be in backends:
        print(f"=== backend {be.name} scoring ===", flush=True)
        gscores = score_groups(be, groups, chunk=args.chunk)
        sweep = [metrics_at_tau(groups, gscores, t) for t in taus]
        rec = _recommend(sweep)
        # strip flip samples for compact policy
        sweep_compact = [
            {
                k: m[k]
                for k in (
                    "tau",
                    "final_hit1",
                    "mozc_hit1",
                    "delta_vs_mozc_pt",
                    "recovery_rate_on_mozc_miss",
                    "regression_rate_on_mozc_hit",
                    "overwrite_rate",
                    "counts",
                )
            }
            for m in sweep
        ]
        policy["backends"][be.name] = {
            "recommended": rec,
            "tau_sweep": sweep_compact,
        }
        print("recommended", rec, flush=True)

        # latency at batch
        print(f"=== latency {be.name} ===", flush=True)
        lat = latency_batch(be, groups, latency_n.get(be.name, args.latency_groups))
        primary = metrics_at_tau(groups, gscores, rec["tau"])
        speed["rows"].append(
            {
                "backend": be.name,
                "tau": rec["tau"],
                "final_hit1": primary["final_hit1"],
                "delta_vs_mozc_pt": primary["delta_vs_mozc_pt"],
                "regression": primary["regression_rate_on_mozc_hit"],
                "recovery": primary["recovery_rate_on_mozc_miss"],
                "latency": lat,
                "pattern": "batch_full_nbest",
            }
        )
        print("latency", lat, flush=True)

    (out_dir / "margin_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Keep old cpu latency file; write new matched-accuracy bench alongside.
    (out_dir / "eval_cpu_latency_matched.json").write_text(
        json.dumps(speed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Also update eval_cpu_latency.json with a nested key preserving old if present.
    lat_path = out_dir / "eval_cpu_latency.json"
    old: dict[str, Any] = {}
    if lat_path.exists():
        try:
            old = json.loads(lat_path.read_text(encoding="utf-8"))
        except Exception:
            old = {"_raw_previous": lat_path.read_text(encoding="utf-8")[:2000]}
    merged = {
        "previous_ablation_or_legacy": old,
        "matched_accuracy_batch_no_topk": speed,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    lat_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"policy_backends": {k: v["recommended"] for k, v in policy["backends"].items()}, "speed": speed["rows"]}, ensure_ascii=False, indent=2), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
