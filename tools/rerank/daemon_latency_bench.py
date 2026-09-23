"""Drive the shipped loopback rerank daemon with validation conversions.

The daemon (Mozc-Ai/runtime/rerank_daemon.py) must already be running with the
frozen Phase 2 ONNX model.  This bench replicates the C++ CallDaemon TCP
pattern exactly: a new connection per conversion, one JSON line out, one JSON
line back, a hard 200 ms deadline (the RerankRewriter default), sequential
requests (one conversion = one candidate batch, single-user IME).

Reports round-trip and daemon-side scoring percentiles, timeout rate, guard
skips, and overwrite decisions at the shipped policy tau.
"""
from __future__ import annotations

import argparse
import json
import socket
import statistics
import time
from pathlib import Path

from tools.dataset.jsonl import read_jsonl


def one_request(host: str, port: int, payload: dict, timeout_ms: int) -> tuple[bool, float, dict | None]:
    """New connection per request, mirroring C++ TcpExchange."""
    started = time.perf_counter()
    sock = None
    try:
        sock = socket.create_connection((host, port), timeout=timeout_ms / 1000.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(timeout_ms / 1000.0)
        sock.sendall((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        buffer = bytearray()
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                return False, (time.perf_counter() - started) * 1000.0, None
            buffer.extend(chunk)
            if b"\n" in buffer:
                line = bytes(buffer).split(b"\n", 1)[0]
                elapsed = (time.perf_counter() - started) * 1000.0
                return True, elapsed, json.loads(line.decode("utf-8"))
    except (socket.timeout, TimeoutError):
        return False, (time.perf_counter() - started) * 1000.0, None
    except OSError:
        return False, (time.perf_counter() - started) * 1000.0, None
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17890)
    parser.add_argument("--timeout-ms", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="0 = all validation rows")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.data)))
    requests = []
    for row in rows:
        candidates = [c["surface"] for c in row["candidates"] if c.get("surface")]
        if not candidates:
            continue
        requests.append({
            "reading": row["reading"],
            "context_prev": row["context_prev"] or "",
            "nbest": candidates,
            "_gold": row["gold"],
        })
    if args.limit > 0:
        requests = requests[: args.limit]
    print(f"requests={len(requests)} timeout_ms={args.timeout_ms}", flush=True)

    # Warm the resident ORT session so percentiles exclude lazy first-touch
    # allocations. Warmup results are discarded.
    for req in requests[: max(0, args.warmup)]:
        warm = dict(req)
        warm.pop("_gold", None)
        one_request(args.host, args.port, warm, args.timeout_ms)

    ok = fail = timeouts = skipped = overwritten = guard_skipped = mozc_final_hit = final_hit = 0
    roundtrip_ms: list[float] = []
    daemon_ms: list[float] = []
    for i, req in enumerate(requests):
        gold = req.pop("_gold")
        success, elapsed, response = one_request(args.host, args.port, req, args.timeout_ms)
        if not success:
            fail += 1
            if elapsed >= args.timeout_ms:
                timeouts += 1
            continue
        ok += 1
        roundtrip_ms.append(elapsed)
        if isinstance(response, dict):
            daemon_ms.append(float(response.get("daemon_ms") or 0.0))
            if response.get("guard_skip"):
                guard_skipped += 1
                skipped += 1
            if response.get("overwritten"):
                overwritten += 1
            final_top1 = response.get("final_top1")
            mozc_top1 = req["nbest"][0]
            if mozc_top1 == gold:
                mozc_final_hit += 1
            if final_top1 == gold:
                final_hit += 1
        if (i + 1) % 1000 == 0:
            print(f"progress={i + 1}/{len(requests)} ok={ok} timeouts={timeouts}", flush=True)

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        return ys[min(len(ys) - 1, int(len(ys) * p))]

    report = {
        "daemon": {"host": args.host, "port": args.port, "timeout_ms": args.timeout_ms},
        "requests": len(requests),
        "ok": ok,
        "failed": fail,
        "timeouts": timeouts,
        "timeout_rate": timeouts / len(requests) if requests else 0.0,
        "roundtrip_ms": {
            "p50": round(pct(roundtrip_ms, 0.50), 3),
            "p95": round(pct(roundtrip_ms, 0.95), 3),
            "p99": round(pct(roundtrip_ms, 0.99), 3),
            "max": round(max(roundtrip_ms), 3) if roundtrip_ms else 0.0,
            "mean": round(statistics.fmean(roundtrip_ms), 3) if roundtrip_ms else 0.0,
        },
        "daemon_ms": {
            "p50": round(pct(daemon_ms, 0.50), 3),
            "p95": round(pct(daemon_ms, 0.95), 3),
            "p99": round(pct(daemon_ms, 0.99), 3),
            "mean": round(statistics.fmean(daemon_ms), 3) if daemon_ms else 0.0,
        },
        "guard_skipped": guard_skipped,
        "overwritten": overwritten,
        "mozc_hit1": round(mozc_final_hit / len(requests), 6) if requests else 0.0,
        "daemon_final_hit1": round(final_hit / len(requests), 6) if requests else 0.0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"DONE wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
