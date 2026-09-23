"""Production-policy replay of validation through the shipped rerank policy.

Replays every validation group through the exact chain the live system uses:

  C++ RerankRewriter guard (strict allowlist by default, or safety mode)
    → loopback daemon guard (short reading / empty-or-symbol context)
    → ONNX fp32 scoring of every candidate (one batch per conversion)
    → margin gate ``final = rerank_top1 if margin >= tau else mozc_top1``
    → daemon junk-candidate post guard
    → C++ hard-protect revert when the Mozc top-1 candidate carries
      USER_SEGMENT_HISTORY_REWRITER / RERANKED / NUMBER / NO_MODIFICATION /
      NO_DELETABLE or is punctuation/symbol (dataset ``protection`` field,
      frozen by the same C++ classifier)

Outputs helped / hurt / overwrite / skip counts per mode, a tau sweep on the
retained margins, and one-conversion-one-batch latency percentiles.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

from tools.dataset.jsonl import read_jsonl
from tools.rerank.context_clip import normalize_reading
from tools.rerank.usage_guard import is_junk_surface, load_eligible_readings, skip_reason as cpp_skip_reason


def daemon_skip_reason(reading: str, context: str) -> str | None:
    """Mozc-Ai/runtime/rerank_daemon.py skip_reason (loopback daemon)."""
    if len(reading) <= 2:
        return "reading_too_short"
    if not context:
        return "context_empty_or_symbol"
    has_ling = False
    for char in context:
        code = ord(char)
        if (
            0x3040 <= code <= 0x30FF
            or 0x31F0 <= code <= 0x31FF
            or 0x3400 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or 0xFF66 <= code <= 0xFF9D
            or (char.isalpha() and not char.isdigit())
        ):
            has_ling = True
            break
    if not has_ling:
        return "context_empty_or_symbol"
    return None


class OrtScorer:
    """Same session options and tokenization as the shipped daemon."""

    def __init__(self, model: Path, tokenizer_dir: Path, max_len: int, intra: int):
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_dir / "tokenizer.model"))
        self.max_len = max_len
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, intra)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_mem_pattern = True
        options.enable_cpu_mem_arena = True
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])

    def score(self, texts: list[str]) -> list[float]:
        rows = [[1] + self.tokenizer.encode(t, out_type=int)[: self.max_len - 2] + [2] for t in texts]
        width = max(len(r) for r in rows)
        input_ids = np.full((len(rows), width), 3, dtype=np.int64)
        attention_mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = r
            attention_mask[i, : len(r)] = 1
        result = self.session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
        return [float(v) for v in np.asarray(result).reshape(-1).tolist()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tau", type=float, default=2.5, help="shipped margin_policy.json tau")
    parser.add_argument("--cand-cap", type=int, default=30)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--intra-op", type=int, default=4)
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.data)))
    eligible = load_eligible_readings()
    scorer = OrtScorer(Path(args.onnx), Path(args.tokenizer_dir), args.max_len, args.intra_op)

    modes = ("strict", "safety")
    # margin is rerank_top1_score - mozc_top1_score for scored groups; retained
    # so the tau sweep costs no extra inference.
    margins: dict[str, list[float]] = {m: [] for m in modes}
    rerank_differs: dict[str, list[bool]] = {m: [] for m in modes}     # rerank argmax != rank 0
    base_hits: dict[str, list[bool]] = {m: [] for m in modes}          # mozc_top1 == gold
    rerank_improves: dict[str, list[bool]] = {m: [] for m in modes}    # rerank argmax fixes a mozc miss
    rerank_breaks: dict[str, list[bool]] = {m: [] for m in modes}      # rerank argmax breaks a mozc hit
    junk_rerank_top1: dict[str, list[bool]] = {m: [] for m in modes}
    hard_protect_mozc: dict[str, list[bool]] = {m: [] for m in modes}
    skipped_mozc_hit: dict[str, int] = {m: 0 for m in modes}
    counts: dict[str, dict[str, int]] = {
        m: {"groups": 0, "skipped": 0, "scored": 0, "overwritten": 0,
            "reverted_protect": 0, "reverted_junk": 0, "helped": 0, "hurt": 0,
            "mozc_hit": 0}
        for m in modes
    }
    skip_reasons: dict[str, dict[str, int]] = {m: {} for m in modes}
    latencies_ms: list[float] = []

    for row in rows:
        reading = normalize_reading(row["reading"])
        context = row["context_prev"] or ""
        candidates = [c["surface"] for c in row["candidates"] if c.get("surface")][: args.cand_cap]
        if not reading or not candidates:
            continue
        mozc_top1 = candidates[0]
        gold = row["gold"]
        hard_protect = row["candidates"][0].get("protection") == "HARD_PROTECT"
        mozc_hit = mozc_top1 == gold
        # Guards are score-independent: decide per mode first, then score the
        # group once and share it across modes (strict skip set ⊆ safety).
        mode_reasons: dict[str, str | None] = {}
        for mode in modes:
            reason = cpp_skip_reason(reading, context, eligible=eligible, already_cleaned=True, mode=mode)
            if reason is None:
                reason = daemon_skip_reason(reading, context)
            mode_reasons[mode] = reason
        scores: list[float] | None = None
        if any(reason is None for reason in mode_reasons.values()):
            started = time.perf_counter()
            scores = scorer.score(
                [f"読み: {reading}\n文脈: {context}\n候補: {cand}" for cand in candidates]
            )
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
        for mode in modes:
            c = counts[mode]
            c["groups"] += 1
            if mozc_hit:
                c["mozc_hit"] += 1
            reason = mode_reasons[mode]
            if reason is not None:
                c["skipped"] += 1
                if mozc_hit:
                    skipped_mozc_hit[mode] += 1
                skip_reasons[mode][reason] = skip_reasons[mode].get(reason, 0) + 1
                continue
            c["scored"] += 1
            assert scores is not None
            best = max(range(len(candidates)), key=lambda j: scores[j])
            margin = scores[best] - scores[0]
            margins[mode].append(margin)
            base_hits[mode].append(mozc_hit)
            rerank_differs[mode].append(best != 0)
            rerank_improves[mode].append((not mozc_hit) and candidates[best] == gold)
            rerank_breaks[mode].append(mozc_hit and best != 0)
            junk_rerank_top1[mode].append(is_junk_surface(candidates[best]))
            hard_protect_mozc[mode].append(hard_protect)

    taus = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    tau_tables: dict[str, list[dict[str, Any]]] = {}
    for mode in modes:
        table = []
        for tau in taus:
            helped = hurt = overwritten = reverted_junk = reverted_protect = 0
            final_hits = 0
            for margin, differs, mozc_hit, improves, breaks, junk, protect in zip(
                margins[mode], rerank_differs[mode], base_hits[mode],
                rerank_improves[mode], rerank_breaks[mode],
                junk_rerank_top1[mode], hard_protect_mozc[mode],
            ):
                # Exact ApplyMargin semantics (rerank_margin.h): overwrite iff
                # rerank argmax != mozc rank 0 and margin >= tau.
                will_overwrite = differs and margin >= tau
                if will_overwrite and junk:
                    will_overwrite = False
                    reverted_junk += 1
                if will_overwrite and protect:
                    will_overwrite = False
                    reverted_protect += 1
                if will_overwrite:
                    overwritten += 1
                    if improves:
                        helped += 1
                        final_hits += 1
                    elif breaks:
                        hurt += 1
                    # else: miss → miss, final stays a miss
                elif mozc_hit:
                    final_hits += 1
            # Skipped groups keep Mozc order, so their Mozc hits always count.
            final_hits += skipped_mozc_hit[mode]
            n = counts[mode]["groups"]
            table.append({
                "tau": tau,
                "overwritten": overwritten,
                "reverted_junk": reverted_junk,
                "reverted_protect": reverted_protect,
                "helped": helped,
                "hurt": hurt,
                "final_hit1": round(final_hits / n, 6) if n else 0.0,
            })
        tau_tables[mode] = table

    # Shipped-tau headline counts come straight from the tau table.
    for mode in modes:
        row = next(t for t in tau_tables[mode] if t["tau"] == args.tau)
        counts[mode]["overwritten"] = row["overwritten"]
        counts[mode]["reverted_junk"] = row["reverted_junk"]
        counts[mode]["reverted_protect"] = row["reverted_protect"]
        counts[mode]["helped"] = row["helped"]
        counts[mode]["hurt"] = row["hurt"]

    latencies_ms.sort()

    def pct(p: float) -> float:
        if not latencies_ms:
            return 0.0
        return latencies_ms[min(len(latencies_ms) - 1, int(len(latencies_ms) * p))]

    latency = {
        "n_scored_conversions": len(latencies_ms),
        "p50_ms": round(pct(0.50), 3),
        "p95_ms": round(pct(0.95), 3),
        "p99_ms": round(pct(0.99), 3),
        "max_ms": round(latencies_ms[-1], 3) if latencies_ms else 0.0,
        "mean_ms": round(statistics.fmean(latencies_ms), 3) if latencies_ms else 0.0,
        "timeout_over_200ms": sum(1 for x in latencies_ms if x > 200.0),
        "timeout_rate": (sum(1 for x in latencies_ms if x > 200.0) / len(latencies_ms)) if latencies_ms else 0.0,
    }

    report = {
        "policy": {
            "tau": args.tau,
            "cand_cap": args.cand_cap,
            "max_len": args.max_len,
            "source": "Mozc-Ai/runtime/model/margin_policy.json + rerank_rewriter.cc defaults",
        },
        "counts": counts,
        "skip_reasons": skip_reasons,
        "tau_sweep": tau_tables,
        "latency_one_batch_per_conversion": latency,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"DONE wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
