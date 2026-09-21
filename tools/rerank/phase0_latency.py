"""Controlled Phase 0-D latency benchmark for the shipped 30M ONNX.

The benchmark uses deterministic synthetic Mozc-shaped requests, one persistent
ORT session, warmup, and one CPU process.  It never writes model weights or
user text.  Use the same container/CPU for all configurations.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path


def pct(xs: list[float], p: float) -> float:
    ys = sorted(xs); k = (len(ys) - 1) * p / 100
    a, b = int(k), min(int(k) + 1, len(ys) - 1)
    return ys[a] + (ys[b] - ys[a]) * (k - a)


def summarize(xs: list[float]) -> dict[str, float]:
    return {"n": len(xs), "p50_ms": pct(xs, 50), "p95_ms": pct(xs, 95),
            "max_ms": max(xs), "mean_ms": statistics.fmean(xs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(args.onnx, sess_options=so,
                                   providers=["CPUExecutionProvider"])
    readings = ["きしゃ", "とうきょう", "まーじ", "かんじ", "へんかん"]
    contexts = ["駅に停まった列車のそばで", "新聞の一面を読んで", "mainに追加して", "漢字の説明を", "入力を続けて"]
    def run(cand_n: int, padding: str, max_len: int) -> dict:
        texts = []
        for i in range(args.n + args.warmup):
            r, c = readings[i % len(readings)], contexts[i % len(contexts)]
            cs = [f"候補{i % 17}_{j}" for j in range(cand_n)]
            texts.append([f"読み: {r}\n文脈: {c}\n候補: {x}" for x in cs])
        times, seqs = [], []
        for batch in texts[:args.warmup]:
            enc = tok(batch, truncation=True, max_length=max_len, padding=padding, return_tensors="np")
            session.run(None, {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)})
        for batch in texts[args.warmup:]:
            enc = tok(batch, truncation=True, max_length=max_len, padding=padding, return_tensors="np")
            seqs.append(int(enc["input_ids"].shape[1]))
            t0 = time.perf_counter()
            session.run(None, {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)})
            times.append((time.perf_counter() - t0) * 1000)
        return {"candidate_count": cand_n, "padding": padding, "max_len": max_len,
                "effective_seq": {"p50": pct(seqs, 50), "p95": pct(seqs, 95), "max": max(seqs)},
                "latency": summarize(times)}
    cpu_model = platform.processor()
    if not cpu_model:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    report = {"phase": "0-D", "status": "complete", "model": "sbintuitions/modernbert-ja-30m",
              "cpu_model": cpu_model, "platform": platform.platform(),
              "logical_cores": __import__("os").cpu_count(), "ort_version": ort.__version__,
              "threads": {"intra": args.threads, "inter": 1}, "warmup": args.warmup,
              "n": args.n, "model_size_bytes": Path(args.onnx).stat().st_size,
              "configurations": [run(5, "longest", 128), run(30, "longest", 128),
                                  run(5, "max_length", 128), run(30, "max_length", 128)]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
