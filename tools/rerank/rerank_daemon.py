"""Resident rerank daemon: load ONNX+tokenizer once, serve over localhost TCP.

Protocol (NDJSON, one request/response per line, connection reused):
  request:  {"reading","context_prev","nbest"|"candidates"}  or  {"op":"ping"}
  response: phase3_hook.rerank_one JSON (ranked_surfaces, overwritten, scores, ...)

Same context_clip / pair text / margin as the one-shot hook.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.rerank.phase3_hook import (  # noqa: E402
    DEFAULT_CAND_CAP,
    DEFAULT_MAX_LEN,
    DEFAULT_TAU,
    OnnxScorer,
    PtScorer,
    rerank_one,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 17890


def load_policy(path: Path) -> tuple[float, int, int]:
    tau, max_len, cand_cap = DEFAULT_TAU, DEFAULT_MAX_LEN, DEFAULT_CAND_CAP
    if path.is_file():
        blob = json.loads(path.read_text(encoding="utf-8"))
        tau = float(blob.get("tau", tau))
        max_len = int(blob.get("max_len", max_len))
        cand_cap = int(blob.get("cand_cap", cand_cap))
    return tau, max_len, cand_cap


def build_onnx_scorer(onnx: Path, tokenizer: Path, max_len: int, intra: int):
    return OnnxScorer(onnx, tokenizer, max_len, intra)


def _append_daemon_log(row: dict[str, Any]) -> None:
    # Request rows contain readings, candidates, and context.  Persist them
    # only when a local operator explicitly opts in with a path.
    path = os.environ.get("MOZC_RERANK_DAEMON_LOG", "").strip()
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        return


def handle_payload(req: dict[str, Any], scorer, *, tau: float, cand_cap: int) -> dict[str, Any]:
    if str(req.get("op") or "").lower() == "ping":
        return {"ok": True, "op": "pong"}
    if "nbest" not in req and "candidates" in req:
        req = dict(req)
        req["nbest"] = req.get("candidates") or []
    t0 = time.perf_counter()
    out = rerank_one(req, scorer, tau=tau, cand_cap=cand_cap)
    out["daemon_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    _append_daemon_log(
        {
            "reading": out.get("reading"),
            "context_prev": out.get("context_prev"),
            "mozc_top1": out.get("mozc_top1"),
            "rerank_top1": out.get("rerank_top1"),
            "final_top1": out.get("final_top1"),
            "overwritten": out.get("overwritten"),
            "margin": out.get("margin"),
            "reason": out.get("reason"),
            "guard_skip": out.get("guard_skip"),
            "daemon_ms": out.get("daemon_ms"),
        }
    )
    return out


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: "RerankTCPServer" = self.server  # type: ignore[assignment]
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise ValueError("request must be a JSON object")
                resp = handle_payload(
                    req, server.scorer, tau=server.tau, cand_cap=server.cand_cap
                )
            except Exception as exc:  # noqa: BLE001 — fail-safe JSON for IME
                resp = {"ok": False, "error": str(exc), "ranked_surfaces": []}
            blob = (json.dumps(resp, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            self.wfile.write(blob)
            self.wfile.flush()


class RerankTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr, scorer, *, tau: float, cand_cap: int):
        super().__init__(addr, _Handler)
        self.scorer = scorer
        self.tau = float(tau)
        self.cand_cap = int(cand_cap)


def send_request(
    host: str,
    port: int,
    payload: dict[str, Any],
    *,
    timeout: float = 0.2,
    sock: socket.socket | None = None,
) -> tuple[dict[str, Any], socket.socket]:
    """Send one NDJSON request. Reuse sock if given. Returns (response, sock)."""
    owned = sock is None
    if sock is None:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(timeout)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    try:
        sock.sendall(line.encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("daemon closed connection")
            buf += chunk
        raw, _rest = buf.split(b"\n", 1)
        return json.loads(raw.decode("utf-8")), sock
    except Exception:
        if owned:
            try:
                sock.close()
            except OSError:
                pass
        raise


def serve(
    scorer,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tau: float = DEFAULT_TAU,
    cand_cap: int = DEFAULT_CAND_CAP,
    ready_file: Path | None = None,
) -> None:
    server = RerankTCPServer((host, port), scorer, tau=tau, cand_cap=cand_cap)
    bound_host, bound_port = server.server_address[:2]
    print(
        f"rerank_daemon listening {bound_host}:{bound_port} tau={tau} cand_cap={cand_cap}",
        flush=True,
    )
    if ready_file is not None:
        ready_file.parent.mkdir(parents=True, exist_ok=True)
        ready_file.write_text(
            json.dumps({"host": bound_host, "port": bound_port, "pid": os.getpid()}),
            encoding="utf-8",
        )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if ready_file is not None:
            try:
                ready_file.unlink()
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Resident Mozc rerank daemon (TCP)")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument(
        "--onnx",
        default=str(ROOT / "artifacts/rerank_ctx/track30m_ctx/onnx/cross_encoder_fp32.onnx"),
    )
    p.add_argument(
        "--tokenizer",
        default=str(ROOT / "artifacts/rerank_ctx/track30m_ctx/onnx/tokenizer"),
    )
    p.add_argument("--ckpt", default="")
    p.add_argument(
        "--policy",
        default=str(ROOT / "artifacts/rerank_ctx/track30m_ctx/onnx/margin_policy.json"),
    )
    p.add_argument("--tau", type=float, default=0.0, help="override policy tau; 0=use policy")
    p.add_argument("--cand-cap", type=int, default=0, help="0=use policy")
    p.add_argument("--max-len", type=int, default=0, help="0=use policy")
    p.add_argument("--intra-op", type=int, default=0, help="0=cpu_count")
    p.add_argument("--ready-file", default="")
    p.add_argument(
        "--ping",
        action="store_true",
        help="send ping to an already-running daemon and exit",
    )
    args = p.parse_args(argv)

    if args.ping:
        resp, sock = send_request(args.host, int(args.port), {"op": "ping"}, timeout=1.0)
        sock.close()
        print(json.dumps(resp, ensure_ascii=False), flush=True)
        return 0 if resp.get("ok") else 2

    tau, max_len, cand_cap = load_policy(Path(args.policy))
    if args.tau > 0:
        tau = float(args.tau)
    if args.max_len > 0:
        max_len = int(args.max_len)
    if args.cand_cap > 0:
        cand_cap = int(args.cand_cap)
    intra = int(args.intra_op) if int(args.intra_op) > 0 else max(1, os.cpu_count() or 1)

    onnx = Path(args.onnx)
    tok = Path(args.tokenizer)
    ckpt = Path(args.ckpt) if args.ckpt else onnx.parent.parent
    if onnx.is_file():
        tok_dir = tok if tok.is_dir() else onnx.parent / "tokenizer"
        print(f"loading onnx={onnx} tokenizer={tok_dir} intra={intra}", flush=True)
        scorer = build_onnx_scorer(onnx, tok_dir, max_len, intra)
    elif (ckpt / "cross_encoder.pt").is_file():
        print(f"loading pt ckpt={ckpt}", flush=True)
        scorer = PtScorer(ckpt, "cpu", max_len, False)
    else:
        print("no onnx or ckpt for rerank daemon", file=sys.stderr)
        return 2

    ready = Path(args.ready_file) if args.ready_file else None
    serve(
        scorer,
        host=args.host,
        port=int(args.port),
        tau=tau,
        cand_cap=cand_cap,
        ready_file=ready,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
