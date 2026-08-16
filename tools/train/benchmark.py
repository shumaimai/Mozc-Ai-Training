"""Run IME LoRA inference and a small benchmark report."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from tools.train.infer import generate_candidates, load_model


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Accept either benchmark fixture or train_mixed example."""
    if "key" in row and "gold" in row:
        return {
            "reading": row["key"],
            "gold": row["gold"],
            "context": row.get("context") or [],
            "mozc_candidates": row.get("mozc_candidates") or [],
            "category": row.get("category"),
            "id": row.get("source_id") or row.get("category") or row["key"],
        }
    meta = row.get("meta") or {}
    # Recover mozc candidates from the prompt's「既存候補」line when present.
    mozc: list[str] = []
    prompt = row.get("input") or ""
    for line in prompt.splitlines():
        if line.startswith("既存候補"):
            payload = line.split(":", 1)[-1].strip()
            mozc = [p.strip() for p in payload.split(",") if p.strip()]
            break
    return {
        "reading": meta.get("reading") or row.get("reading") or "",
        "gold": meta.get("surface") or row.get("output") or row.get("gold") or "",
        "context": [],
        "mozc_candidates": mozc,
        "category": meta.get("category") or row.get("category"),
        "id": f"{meta.get('source_id','')}|{meta.get('reading','')}|{meta.get('surface','')}",
    }


def score_one(gold: str, candidates: list[str]) -> dict[str, bool]:
    return {
        "hit1": bool(candidates) and candidates[0] == gold,
        "hit3": gold in candidates[:3],
        "any": gold in candidates,
    }


def run_benchmark(
    model,
    tokenizer,
    device: str,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    hits = Counter()
    for index, raw in enumerate(rows, start=1):
        row = normalize_row(raw)
        result = generate_candidates(
            model,
            tokenizer,
            reading=row["reading"],
            mozc_candidates=row["mozc_candidates"],
            context=row["context"],
            max_new_tokens=max_new_tokens,
            device=device,
        )
        scores = score_one(row["gold"], result["candidates"])
        for key, value in scores.items():
            hits[key] += int(value)
        detail = {
            "id": row["id"],
            "category": row["category"],
            "reading": row["reading"],
            "gold": row["gold"],
            "candidates": result["candidates"],
            "raw": result["raw"],
            **scores,
        }
        details.append(detail)
        mark = "OK" if scores["hit3"] else "NG"
        print(
            f"[{index}/{len(rows)}] {mark} reading={row['reading']} "
            f"gold={row['gold']} pred={result['candidates']}",
            flush=True,
        )
    n = max(len(rows), 1)
    summary = {
        "n": len(rows),
        "hit1": hits["hit1"] / n,
        "hit3": hits["hit3"] / n,
        "any": hits["any"] / n,
    }
    return {"summary": summary, "details": details}


def main() -> int:
    parser = argparse.ArgumentParser(description="IME LoRA infer / benchmark")
    parser.add_argument("--base-model", default="rinna/japanese-gpt2-medium")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--benchmark", default="data/benchmark/v1.jsonl")
    parser.add_argument("--holdout", default="", help="optional extra JSONL (e.g. train_mixed sample)")
    parser.add_argument("--holdout-limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--out", default="artifacts/benchmark/lora_poc_full.json")
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="also run the same cases without adapter",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="print a few hand-written demo generations and exit",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="pass trust_remote_code for custom models such as PLaMo",
    )
    args = parser.parse_args()

    adapter = args.adapter or None
    model, tokenizer, device = load_model(
        args.base_model,
        adapter,
        trust_remote_code=args.trust_remote_code or None,
    )
    print(f"device={device} adapter={adapter or '(none)'}", flush=True)

    if args.demo:
        demos = [
            {"reading": "とうきょうと", "mozc": ["東京と", "とうきょうと"], "ctx": []},
            {"reading": "じんこうこきゅうき", "mozc": ["人口呼吸器"], "ctx": ["患者に"]},
            {"reading": "まるばしら", "mozc": ["丸柱", "まるばしら"], "ctx": []},
        ]
        for demo in demos:
            result = generate_candidates(
                model,
                tokenizer,
                reading=demo["reading"],
                mozc_candidates=demo["mozc"],
                context=demo["ctx"],
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            print("---", flush=True)
            print("reading:", demo["reading"], flush=True)
            print("pred:", result["candidates"], flush=True)
            print("raw:", repr(result["raw"][:200]), flush=True)

    suites: list[tuple[str, list[dict[str, Any]]]] = []
    bench_path = Path(args.benchmark)
    # --demo alone: smoke only. Suites run when --holdout is set or --out is customized,
    # or when --demo is omitted.
    run_suites = (not args.demo) or bool(args.holdout) or (
        Path(args.out).name != "lora_poc_full.json"
    )
    if run_suites and bench_path.exists():
        suites.append(("v1", read_jsonl(bench_path)))
    if args.holdout:
        holdout_rows = read_jsonl(Path(args.holdout))
        random.Random(args.seed).shuffle(holdout_rows)
        suites.append(("holdout", holdout_rows[: args.holdout_limit]))
    if not suites:
        return 0

    report: dict[str, Any] = {
        "base_model": args.base_model,
        "adapter": adapter,
        "suites": {},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _save() -> None:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"checkpoint wrote {out}", flush=True)

    for name, rows in suites:
        print(f"\n=== suite={name} n={len(rows)} adapter={adapter or 'base'} ===", flush=True)
        report["suites"][name] = run_benchmark(
            model,
            tokenizer,
            device,
            rows,
            max_new_tokens=args.max_new_tokens,
        )
        summary = report["suites"][name]["summary"]
        print(
            f"summary {name}: hit1={summary['hit1']:.3f} "
            f"hit3={summary['hit3']:.3f} any={summary['any']:.3f}",
            flush=True,
        )
        _save()

    if args.compare_base and adapter:
        print("\n=== base model comparison ===", flush=True)
        del model
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
        base_model, base_tok, base_device = load_model(
            args.base_model,
            adapter=None,
            trust_remote_code=args.trust_remote_code or None,
        )
        report["base_suites"] = {}
        for name, rows in suites:
            print(f"\n=== suite={name} n={len(rows)} adapter=(none) ===", flush=True)
            report["base_suites"][name] = run_benchmark(
                base_model,
                base_tok,
                base_device,
                rows,
                max_new_tokens=args.max_new_tokens,
            )
            summary = report["base_suites"][name]["summary"]
            print(
                f"summary {name} BASE: hit1={summary['hit1']:.3f} "
                f"hit3={summary['hit3']:.3f} any={summary['any']:.3f}",
                flush=True,
            )
            _save()

    _save()
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
