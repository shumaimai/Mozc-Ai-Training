#!/usr/bin/env python3
"""Measure daemon IPC+inference latency (warm). Requires a running daemon or --spawn."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.rerank.rerank_daemon import send_request


def _summary(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    def pct(p: float) -> float:
        i = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
        return round(s[i], 3)
    return {
        "n": len(s),
        "mean": round(statistics.fmean(s), 3),
        "p50": pct(50),
        "p95": pct(95),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=17890)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timeout", type=float, default=2.0)
    p.add_argument(
        "--data",
        default=str(ROOT / "data/rerank_ctx/eval_unseen_v2_clean.jsonl"),
    )
    p.add_argument("--out", default="")
    args = p.parse_args()

    ping, sock = send_request(args.host, args.port, {"op": "ping"}, timeout=args.timeout)
    if not ping.get("ok"):
        print("daemon ping failed", ping, file=sys.stderr)
        return 2

    rows = []
    with Path(args.data).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.n + args.warmup:
                break

    times: list[float] = []
    daemon_ms: list[float] = []
    for i, row in enumerate(rows):
        payload = {
            "reading": row.get("reading") or "",
            "context_prev": row.get("context_prev") or row.get("context") or "",
            "nbest": row.get("mozc_nbest") or row.get("nbest") or row.get("candidates") or [],
        }
        t0 = time.perf_counter()
        resp, sock = send_request(
            args.host, args.port, payload, timeout=args.timeout, sock=sock
        )
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= args.warmup:
            times.append(dt)
            if isinstance(resp.get("daemon_ms"), (int, float)):
                daemon_ms.append(float(resp["daemon_ms"]))

    sock.close()
    report = {
        "e2e_ms": _summary(times),
        "daemon_infer_ms": _summary(daemon_ms),
        "ipc_approx_ms": round(
            (_summary(times).get("p50", 0) - _summary(daemon_ms).get("p50", 0)), 3
        )
        if times and daemon_ms
        else None,
        "host": args.host,
        "port": args.port,
        "n": len(times),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
