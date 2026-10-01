"""Measure uncensored scoring latency; guard skips never enter percentiles.

Runs configurations sequentially on an identical seeded validation sample.
No cross-request cache, candidate pruning, or context clipping is used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np

from tools.dataset.jsonl import read_jsonl


def load_runtime(path: str):
    spec = importlib.util.spec_from_file_location("optimization_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentiles(values: list[float]) -> dict:
    return {**{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)},
            "mean": statistics.fmean(values), "max": max(values),
            "over_200ms": sum(v > 200 for v in values), "n": len(values)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=240)
    p.add_argument("--threads", default="8,4,2,1")
    p.add_argument("--dedup", default="0,1")
    args = p.parse_args()
    runtime = load_runtime(args.runtime)
    requests = []
    for row in read_jsonl(Path(args.data)):
        r = runtime.normalize_reading(row["reading"])
        c = runtime.clean_context(row["context_prev"])
        if runtime.skip_reason(r, c):
            continue
        candidates = [x["surface"] for x in row["candidates"] if x.get("surface")][:30]
        requests.append([runtime.build_pair_text(r, c, x) for x in candidates])
    random.Random(20260930).shuffle(requests)
    if args.limit:
        requests = requests[:args.limit]
    reports = []
    for threads in map(int, args.threads.split(",")):
        for dedup in map(int, args.dedup.split(",")):
            scorer = runtime.OrtScorer(Path(args.model), Path(args.tokenizer), 128, threads)
            batches = [list(dict.fromkeys(t)) if dedup else t for t in requests]
            for texts in batches[:20]:
                scorer.score(texts)
            latencies = []
            for texts in batches:
                start = time.perf_counter()
                scorer.score(texts)
                latencies.append(1000 * (time.perf_counter() - start))
            report = {"threads": threads, "dedup": bool(dedup), "ms": percentiles(latencies),
                      "mean_batch": statistics.fmean(map(len, batches))}
            print(json.dumps(report), flush=True)
            reports.append(report)
            del scorer
    result = {"platform": platform.platform(), "model": args.model,
              "runtime": args.runtime, "seed": 20260930, "reports": reports}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
