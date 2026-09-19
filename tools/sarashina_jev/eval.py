"""Evaluate page-wise accuracy and latency for a Sarashina-JEV artifact."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torch.utils.data import DataLoader

from tools.sarashina_jev.data import ListwisePageDataset, build_pages, read_jsonl
from tools.sarashina_jev.model import SarashinaJevScorer


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-pages", type=int, default=6)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device == "cuda" else torch.float32)
    )
    model, tokenizer, meta = SarashinaJevScorer.load_artifact(
        args.artifact,
        torch_dtype=dtype,
    )
    model.to(device).eval()
    page_size = int(meta.get("page_size", 5))

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    pages = build_pages(
        rows,
        page_size=page_size,
        max_pages=args.max_pages,
        anchor_weight=args.anchor_weight,
    )
    dataset = ListwisePageDataset(
        pages,
        tokenizer,
        page_size=page_size,
        max_length=args.max_length,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    latencies: list[float] = []
    total = correct = 0
    gold_total = gold_correct = gold_mozc = 0
    anchor_total = anchor_kept = 0
    changed = 0

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        for i, batch in enumerate(loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cand_mask = batch["candidate_mask"].to(device)
            target = batch["target"].to(device)

            sync()
            t0 = time.perf_counter()
            scores = model.score_pages(ids, mask, cand_mask)
            sync()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            pred = scores.argmax(dim=1)
            if i >= args.warmup:
                latencies.append(elapsed_ms)

            ok = bool(pred.eq(target).item())
            total += 1
            correct += int(ok)
            changed += int(pred.item() != 0)
            is_gold = bool(batch["is_gold_page"].item())
            if is_gold:
                gold_total += 1
                gold_correct += int(ok)
                gold_mozc += int(target.item() == 0)
            else:
                anchor_total += 1
                anchor_kept += int(pred.item() == 0)

    report = {
        "pages": total,
        "page_hit1": correct / total if total else 0.0,
        "gold_pages": gold_total,
        "gold_page_hit1": gold_correct / gold_total if gold_total else 0.0,
        "gold_page_mozc_hit1": gold_mozc / gold_total if gold_total else 0.0,
        "anchor_keep_rate": anchor_kept / anchor_total if anchor_total else 0.0,
        "changed_rate": changed / total if total else 0.0,
        "latency_ms": {
            "count": len(latencies),
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
        "artifact_meta": meta,
        "device": device,
        "dtype": str(dtype),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
