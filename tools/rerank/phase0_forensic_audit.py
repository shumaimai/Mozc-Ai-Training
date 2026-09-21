"""Phase 0-A: quantify v1 train/runtime formatter drift.

No training is performed.  With tokenizer/model artifacts, this writes raw
per-case text, token IDs, lengths, scores, and ranking deltas.  Without them,
it still writes a reproducible fixture and an explicit blocked status rather
than fabricating scores.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
from pathlib import Path
from typing import Any

from tools.rerank.contextual_ranking_v2_contract import format_v1_runtime, format_v1_train


CASES = [
    ("kishya_station", "きしゃ", "駅に", ("記者", "汽車", "貴社")),
    ("kishya_newspaper", "きしゃ", "新聞の", ("記者", "汽車", "貴社")),
    ("maaji_main", "まーじ", "mainに", ("マージ", "まーじ", "麻痺")),
    ("short_reading", "に", "駅", ("に", "二", "荷")),
    ("numbers", "さん", "3月", ("三", "さん", "3")),
    ("punctuation", "てん", "文末", ("点", "・", "、")),
    ("multisegment", "きしゃ", "駅に電車で", ("記者", "汽車", "貴社")),
]


def _ids(tokenizer: Any, text: str, max_len: int) -> list[int]:
    enc = tokenizer(text, truncation=True, max_length=max_len, padding=False,
                    add_special_tokens=True, return_attention_mask=False)
    return [int(x) for x in enc["input_ids"]]


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="", help="v1 HF tokenizer directory")
    ap.add_argument("--onnx", default="", help="v1 fp32 ONNX model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for cid, reading, context, candidates in CASES:
        for rank, candidate in enumerate(candidates):
            train_text = format_v1_train(reading, context, candidate)
            runtime_text = format_v1_runtime(reading, context, candidate)
            rows.append({
                "case_id": cid, "candidate_rank": rank, "reading": reading,
                "context": context, "candidate": candidate,
                "training_raw_text": train_text,
                "runtime_raw_text": runtime_text,
                "raw_text_equal": train_text == runtime_text,
            })
    status = "blocked_missing_artifacts"
    tokenizer_name = None
    try:
        if not args.tokenizer:
            raise FileNotFoundError("--tokenizer is required for token IDs")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        tokenizer_name = getattr(tokenizer, "name_or_path", args.tokenizer)
        for row in rows:
            a = _ids(tokenizer, row["training_raw_text"], args.max_len)
            b = _ids(tokenizer, row["runtime_raw_text"], args.max_len)
            row["training_token_ids"] = a
            row["runtime_token_ids"] = b
            row["training_length"] = len(a)
            row["runtime_length"] = len(b)
            row["token_ids_equal"] = a == b
        if args.onnx:
            import numpy as np
            import onnxruntime as ort
            session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
            for row in rows:
                scores: dict[str, float] = {}
                for label in ("training", "runtime"):
                    ids = np.asarray([row[f"{label}_token_ids"]], dtype=np.int64)
                    mask = np.ones_like(ids, dtype=np.int64)
                    value = session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
                    scores[label] = float(np.asarray(value).reshape(-1)[0])
                row["training_score"] = scores["training"]
                row["runtime_score"] = scores["runtime"]
                row["score_abs_diff"] = abs(scores["training"] - scores["runtime"])
            status = "complete"
        else:
            status = "partial_missing_onnx"
    except Exception as exc:
        for row in rows:
            row.setdefault("training_token_ids", None)
            row.setdefault("runtime_token_ids", None)
        error = {"type": type(exc).__name__, "message": str(exc)}
    else:
        error = None
    ids_rows = [r for r in rows if isinstance(r.get("training_token_ids"), list)]
    token_equal = sum(r["token_ids_equal"] for r in ids_rows)
    score_rows = [r for r in rows if "score_abs_diff" in r]
    rank_changes = 0
    for cid in {r["case_id"] for r in rows}:
        group = [r for r in score_rows if r["case_id"] == cid]
        if group and max(group, key=lambda r: r["training_score"])["candidate"] != max(group, key=lambda r: r["runtime_score"])["candidate"]:
            rank_changes += 1
    report = {
        "phase": "0-A",
        "status": status,
        "v1_tag": "v1.0.0",
        "fixture_cases": len(CASES),
        "fixture_candidate_rows": len(rows),
        "max_len": args.max_len,
        "tokenizer": tokenizer_name,
        "platform": platform.platform(),
        "error": error,
        "metrics": {
            "token_id_exact_match_rate": token_equal / len(ids_rows) if ids_rows else None,
            "token_id_changed_rows": sum(not r.get("token_ids_equal", False) for r in ids_rows),
            "sequence_length_mae": statistics.fmean(abs(r["training_length"] - r["runtime_length"]) for r in ids_rows) if ids_rows else None,
            "score_mae": statistics.fmean(r["score_abs_diff"] for r in score_rows) if score_rows else None,
            "top1_changed_count": rank_changes if score_rows else None,
        },
        "rows": rows,
    }
    # Candidate-level pairs are intentionally not enough to infer ranking; the
    # raw report records the required score inputs and leaves this explicit.
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("phase", "status", "metrics", "error")}, ensure_ascii=False, indent=2))
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
