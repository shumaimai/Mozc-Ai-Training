"""Phase 3 offline/online rerank hook (shippable path scaffold).

Flow (matches docs/RERANK_HOOK.md):
  Mozc N-best → score (ONNX fp32 or PyTorch) → margin gate → ranked list

Defaults (contextual Track B / NEXT_TASK_PHASE3_CTX):
  tau=2.5, cand_cap=30, max_len=128, int8=OFF, topK default=OFF
  context_prev is required for quality (empty OK at sentence start)

CLI:
  # single request JSON on stdin
  echo '{"reading":"とうきょう","nbest":["東京","東響"],"context_prev":""}' \\
    | python -m tools.rerank.phase3_hook rerank \\
        --onnx artifacts/rerank/modernbert70m_ce/onnx/cross_encoder_fp32.onnx \\
        --tokenizer artifacts/rerank/modernbert70m_ce/onnx/tokenizer

  # batch file + optional conversion log append
  python -m tools.rerank.phase3_hook rerank \\
    --input artifacts/rerank/phase3_smoke_in.jsonl \\
    --out artifacts/rerank/phase3_smoke_out.jsonl \\
    --log artifacts/rerank/conversion_logs/conv.jsonl \\
    --ckpt artifacts/rerank/modernbert70m_ce_v3 --device cpu

Schema helper:
  python -m tools.rerank.phase3_hook schema
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.rerank.context_clip import clean_context, normalize_reading
from tools.rerank.margin import apply_margin, pick_rerank_top1
from tools.rerank.train_cross_encoder import build_pair_text
from tools.rerank.usage_guard import apply_post_score_guard, skip_reason

# Ship defaults (contextual Track B)
DEFAULT_TAU = 2.5
DEFAULT_CAND_CAP = 30
DEFAULT_MAX_LEN = 128

CONVERSION_LOG_SCHEMA: dict[str, Any] = {
    "name": "mozc_ai_conversion_log_v1",
    "description": (
        "One JSONL row per conversion commit (or offline smoke). "
        "Used for honest B eval + targeted dict — not accept-holdout grinding."
    ),
    "required": ["ts", "reading", "nbest", "chosen"],
    "properties": {
        "ts": {"type": "string", "format": "iso8601", "example": "2026-08-12T04:00:00+00:00"},
        "reading": {"type": "string", "description": "hiragana/katakana key as typed"},
        "nbest": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Mozc candidate list before/after rerank (store pre-rerank native list)",
        },
        "chosen": {"type": "string", "description": "surface the user committed"},
        "context_prev": {"type": "string", "optional": True, "description": "left context if available"},
        "rerank_top1": {"type": "string", "optional": True},
        "final_top1": {"type": "string", "optional": True},
        "overwritten": {"type": "boolean", "optional": True},
        "tau": {"type": "number", "optional": True},
        "scores": {
            "type": "array",
            "items": {"type": "number"},
            "optional": True,
            "description": "parallel to nbest_scored; omit in privacy-tight builds",
        },
        "session_id": {"type": "string", "optional": True},
        "source": {
            "type": "string",
            "optional": True,
            "description": "ime_online | offline_smoke | mozc_batch",
        },
    },
    "privacy": "No raw keystrokes beyond reading; no clipboard; user opt-in required for online collection.",
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def cap_candidates(nbest: list[str], cand_cap: int) -> list[str]:
    if cand_cap <= 0 or len(nbest) <= cand_cap:
        return list(nbest)
    return list(nbest[:cand_cap])


def rank_with_scores(
    candidates: list[str],
    scores: list[float],
) -> list[dict[str, Any]]:
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [
        {"surface": candidates[i], "score": float(scores[i]), "orig_index": i}
        for i in order
    ]


class PtScorer:
    def __init__(self, ckpt: Path, device: str, max_len: int, fp16: bool):
        from tools.rerank.eval_cross_encoder import load_model, score_texts

        self.tokenizer, self.model, _, self.use_amp = load_model(ckpt, device, fp16)
        self.device = device
        self.max_len = max_len
        self._score_texts = score_texts

    def score(self, texts: list[str]) -> list[float]:
        return self._score_texts(
            texts,
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device,
            max_len=self.max_len,
            batch_size=64,
            use_amp=self.use_amp,
        )


class OnnxScorer:
    def __init__(self, onnx: Path, tokenizer: Path, max_len: int, intra: int):
        from tools.rerank.latency_pack import OrtRunner

        self.runner = OrtRunner(
            onnx, tokenizer, max_len=max_len, intra=intra, inter=1, opt_all=True
        )

    def score(self, texts: list[str]) -> list[float]:
        return self.runner.score_texts(texts)


def build_scorer(args: argparse.Namespace):
    if args.onnx:
        tok = Path(args.tokenizer) if args.tokenizer else Path(args.onnx).parent / "tokenizer"
        return OnnxScorer(Path(args.onnx), tok, int(args.max_len), int(args.intra_op))
    if args.ckpt:
        return PtScorer(Path(args.ckpt), args.device, int(args.max_len), bool(args.fp16))
    raise SystemExit("need --onnx or --ckpt")


def rerank_one(
    req: dict[str, Any],
    scorer,
    *,
    tau: float,
    cand_cap: int,
) -> dict[str, Any]:
    reading = normalize_reading(req.get("reading") or "")
    # Same clean_context as dataset build (NEXT_TASK_CTX_SUBSET_FIX).
    context_prev = clean_context(req.get("context_prev") or req.get("context") or "")
    nbest_raw = [c for c in (req.get("nbest") or req.get("mozc_nbest") or []) if c]
    if not reading:
        raise ValueError("reading required")
    if not nbest_raw:
        raise ValueError("nbest required")

    cands = cap_candidates(nbest_raw, cand_cap)
    mozc_top1 = cands[0]
    guard = skip_reason(reading, context_prev, already_cleaned=True)
    if guard:
        return {
            "reading": reading,
            "context_prev": context_prev,
            "nbest_in": nbest_raw,
            "nbest_scored": cands,
            "scores": [],
            "ranked": [{"surface": c, "score": None} for c in cands],
            "ranked_surfaces": list(cands),
            "mozc_top1": mozc_top1,
            "rerank_top1": mozc_top1,
            "final_top1": mozc_top1,
            "overwritten": False,
            "margin": None,
            "tau": float(tau),
            "cand_cap": int(cand_cap),
            "reason": guard,
            "guard_skip": True,
        }

    texts = [build_pair_text(reading, context_prev, c) for c in cands]
    scores = scorer.score(texts)
    decision = apply_margin(cands, scores, mozc_top1, tau)
    overwritten, final, junk_reason = apply_post_score_guard(
        overwritten=bool(decision["overwritten"]),
        final_top1=str(decision["final_top1"]),
        mozc_top1=mozc_top1,
    )
    if junk_reason:
        decision = dict(decision)
        decision["overwritten"] = overwritten
        decision["final_top1"] = final
        decision["reason"] = junk_reason
    ranked = rank_with_scores(cands, scores)
    # Put final_top1 first for IME candidate window convenience
    surfaces_ranked = [final] + [r["surface"] for r in ranked if r["surface"] != final]

    return {
        "reading": reading,
        "context_prev": context_prev,
        "nbest_in": nbest_raw,
        "nbest_scored": cands,
        "scores": scores,
        "ranked": ranked,
        "ranked_surfaces": surfaces_ranked,
        "mozc_top1": mozc_top1,
        "rerank_top1": decision["rerank_top1"],
        "final_top1": final,
        "overwritten": overwritten,
        "margin": decision.get("margin"),
        "tau": float(tau),
        "cand_cap": int(cand_cap),
        "reason": decision.get("reason"),
        "guard_skip": False,
    }


def to_conversion_log_row(
    req: dict[str, Any],
    result: dict[str, Any],
    *,
    chosen: str | None = None,
    source: str = "offline_smoke",
    include_scores: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ts": req.get("ts") or _iso_now(),
        "reading": result["reading"],
        "nbest": result["nbest_in"],
        "chosen": chosen if chosen is not None else result["final_top1"],
        "context_prev": result.get("context_prev") or "",
        "rerank_top1": result.get("rerank_top1"),
        "final_top1": result.get("final_top1"),
        "overwritten": result.get("overwritten"),
        "tau": result.get("tau"),
        "source": source,
    }
    if include_scores:
        row["scores"] = result.get("scores")
    if req.get("session_id"):
        row["session_id"] = req["session_id"]
    return row


def _iter_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input:
        path = Path(args.input)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json" and text.lstrip().startswith("{"):
            obj = json.loads(text)
            return obj if isinstance(obj, list) else [obj]
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    # stdin
    raw = sys.stdin.read().strip()
    if not raw:
        raise SystemExit("empty stdin; pass --input or JSON on stdin")
    obj = json.loads(raw)
    return obj if isinstance(obj, list) else [obj]


def cmd_schema(_: argparse.Namespace) -> int:
    print(json.dumps(CONVERSION_LOG_SCHEMA, ensure_ascii=False, indent=2))
    return 0


def cmd_rerank(args: argparse.Namespace) -> int:
    if args.int8:
        raise SystemExit("int8 is disabled (ranking collapse). Use ONNX/PT fp32 only.")
    scorer = build_scorer(args)
    reqs = _iter_requests(args)
    outs: list[dict[str, Any]] = []
    log_path = Path(args.log) if args.log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    for req in reqs:
        result = rerank_one(
            req,
            scorer,
            tau=float(args.tau),
            cand_cap=int(args.cand_cap),
        )
        outs.append(result)
        if log_path:
            row = to_conversion_log_row(
                req,
                result,
                chosen=req.get("chosen"),
                source=str(args.log_source),
                include_scores=bool(args.log_scores),
            )
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for o in outs:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
    else:
        # single-object pretty for interactive use
        payload = outs[0] if len(outs) == 1 else outs
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 3 Mozc N-best rerank hook")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("schema", help="print conversion log JSON schema")
    s.set_defaults(func=cmd_schema)

    r = sub.add_parser("rerank", help="rerank Mozc N-best (JSON/JSONL)")
    r.add_argument("--input", default="", help="JSON or JSONL file; else stdin")
    r.add_argument("--out", default="", help="JSONL results path (optional)")
    r.add_argument("--log", default="", help="append conversion log JSONL")
    r.add_argument("--log-source", default="offline_smoke")
    r.add_argument("--log-scores", action="store_true")
    r.add_argument("--onnx", default="", help="ONNX fp32 model path")
    r.add_argument("--tokenizer", default="", help="tokenizer dir for ONNX")
    r.add_argument("--ckpt", default="", help="PyTorch ckpt dir (fallback)")
    r.add_argument("--device", default="cpu")
    r.add_argument("--fp16", action="store_true")
    r.add_argument("--tau", type=float, default=DEFAULT_TAU)
    r.add_argument("--cand-cap", type=int, default=DEFAULT_CAND_CAP)
    r.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    r.add_argument("--intra-op", type=int, default=12)
    r.add_argument(
        "--int8",
        action="store_true",
        help="rejected; kept only to fail loudly if someone passes it",
    )
    r.set_defaults(func=cmd_rerank)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
