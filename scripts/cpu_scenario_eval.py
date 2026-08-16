#!/usr/bin/env python3
"""CPU IME-scenario eval: latency + normal / garbled readings."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from tools.train.infer import generate_candidates, load_model

CASES = [
    # normal
    {"tag": "normal", "reading": "とうきょうと", "mozc": ["東京と"], "ctx": []},
    {"tag": "normal", "reading": "じんこうこきゅうき", "mozc": ["人口呼吸器"], "ctx": ["患者に"]},
    {"tag": "normal", "reading": "おきる", "mozc": ["起きる", "起る"], "ctx": []},
    # garbled / typo-ish readings
    {"tag": "garbled", "reading": "をきる", "mozc": [], "ctx": [], "note": "おきるの先頭がをに崩れた想定"},
    {"tag": "garbled", "reading": "をきる", "mozc": ["起きる", "切る"], "ctx": [], "note": "崩れた読み+既存候補あり"},
    {"tag": "garbled", "reading": "とうきようと", "mozc": ["東京と"], "ctx": [], "note": "長音欠落"},
    {"tag": "garbled", "reading": "とうきょうとと", "mozc": [], "ctx": [], "note": "末尾重複"},
    {"tag": "garbled", "reading": "きる", "mozc": ["切る", "着る"], "ctx": ["髪を"], "note": "短すぎる読み"},
    {"tag": "garbled", "reading": "ををきる", "mozc": [], "ctx": [], "note": "を重複"},
    {"tag": "garbled", "reading": "ｗをきる", "mozc": [], "ctx": [], "note": "全角英混入"},
    {"tag": "garbled", "reading": "", "mozc": ["東京"], "ctx": [], "note": "空入力"},
]


def main() -> int:
    base = "pfnet/plamo-2-1b"
    adapter = "/work/mozc-ai-training/artifacts/plamo2_1b_lora_full/adapter"
    out_path = Path("/work/mozc-ai-training/artifacts/benchmark/cpu_scenario_eval.json")

    print("loading model on CPU...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer, device = load_model(
        base, adapter, device="cpu", trust_remote_code=True
    )
    load_s = time.perf_counter() - t0
    print(f"loaded device={device} in {load_s:.1f}s", flush=True)

    rows = []
    latencies = []
    for i, case in enumerate(CASES, start=1):
        reading = case["reading"]
        print(f"\n[{i}/{len(CASES)}] tag={case['tag']} reading={reading!r}", flush=True)
        t1 = time.perf_counter()
        result = generate_candidates(
            model,
            tokenizer,
            reading=reading,
            mozc_candidates=case.get("mozc") or [],
            context=case.get("ctx") or [],
            max_new_tokens=16,
            device=device,
        )
        elapsed = time.perf_counter() - t1
        latencies.append(elapsed)
        row = {
            **case,
            "latency_s": round(elapsed, 3),
            "prompt": result["prompt"],
            "raw": result["raw"],
            "candidates": result["candidates"],
        }
        rows.append(row)
        print(f"  latency={elapsed:.2f}s pred={result['candidates']} raw={result['raw']!r}", flush=True)

    summary = {
        "device": device,
        "base_model": base,
        "adapter": adapter,
        "load_s": round(load_s, 3),
        "n": len(latencies),
        "latency_mean_s": round(statistics.mean(latencies), 3),
        "latency_median_s": round(statistics.median(latencies), 3),
        "latency_min_s": round(min(latencies), 3),
        "latency_max_s": round(max(latencies), 3),
        "by_tag": {},
    }
    for tag in sorted({c["tag"] for c in CASES}):
        vals = [r["latency_s"] for r in rows if r["tag"] == tag]
        summary["by_tag"][tag] = {
            "n": len(vals),
            "mean_s": round(statistics.mean(vals), 3) if vals else None,
        }

    report = {"summary": summary, "cases": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
