#!/usr/bin/env python3
"""Phase 1 production-parity pilots: SplitMode and prefix replay.

No training or large-scale dataset generation is performed.  This runner uses
the same CPU-resident converter workers as the v2 pilot and writes all
alignment failures explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase1_dataset_v2_pilot_v2 as v2


def tokenize(sentence: str, mode: str):
    tokenizer = v2.sudachi_dictionary.Dictionary().create()
    split_mode = getattr(v2.SplitMode, mode)
    out = []
    for morpheme in tokenizer.tokenize(sentence, split_mode):
        surface = v2.normalize_surface(morpheme.surface())
        reading = v2.normalize_reading(morpheme.reading_form() or surface)
        if not surface or not reading:
            continue
        out.append({
            "surface": surface, "reading": reading,
            "begin": morpheme.begin(), "end": morpheme.end(),
            "pos": morpheme.part_of_speech(),
        })
    return out


def _source_kind(morphemes: list[dict[str, Any]]) -> str:
    if any(len(m.get("pos", ())) > 1 and m["pos"][1] == "固有名詞" for m in morphemes):
        return "proper_noun"
    return ""


def prefix_replay(work: dict[str, Any], max_examples: int, split_mode: str):
    examples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    text = work["text"]
    for sentence_index, (sentence_start, sentence) in enumerate(v2.sentence_spans(text)):
        morphemes = tokenize(sentence, split_mode)
        for boundary in range(1, len(morphemes) + 1):
            prefix_morphemes = morphemes[:boundary]
            reading = "".join(m["reading"] for m in prefix_morphemes)
            if not reading:
                continue
            try:
                segments = v2.query_segments(reading, work["converter_exe"], work["mozc_cwd"], work["top_k"])
            except Exception as exc:
                failures.append({"source_id": work["source_id"], "sentence_index": sentence_index, "boundary": boundary, "reason": "CONVERTER_FAILURE", "detail": f"{type(exc).__name__}:{exc}"})
                continue
            target = segments[-1]
            target_reading = v2.normalize_reading(target["reading"])
            if not target_reading or not reading.endswith(target_reading):
                failures.append({"source_id": work["source_id"], "sentence_index": sentence_index, "boundary": boundary, "target_segment_index": len(segments) - 1, "reason": "ALIGNMENT_FAILURE", "detail": "last segment reading did not match prefix suffix"})
                continue
            starts = []
            cursor = 0
            for morpheme in prefix_morphemes:
                starts.append(cursor)
                cursor += len(morpheme["reading"])
            ends = starts[1:] + [len(reading)]
            target_start = len(reading) - len(target_reading)
            if target_start not in starts or len(reading) not in ends:
                failures.append({"source_id": work["source_id"], "sentence_index": sentence_index, "boundary": boundary, "target_segment_index": len(segments) - 1, "reason": "ALIGNMENT_FAILURE", "detail": "last segment start split a morpheme"})
                continue
            first = starts.index(target_start)
            last = len(prefix_morphemes) - 1
            source_start = sentence_start + prefix_morphemes[first]["begin"]
            source_end = sentence_start + prefix_morphemes[last]["end"]
            gold = "".join(m["surface"] for m in prefix_morphemes[first:last + 1])
            context = v2.clean_context(text[:source_start])
            key = (work["source_id"], target_reading, context, gold)
            if key in seen:
                continue
            seen.add(key)
            examples.append({
                "source_id": work["source_id"], "reading": target_reading,
                "context_prev": context, "gold": gold,
                "target_segment_index": len(segments) - 1,
                "candidates": target["candidates"], "_source_kind": _source_kind(prefix_morphemes[first:last + 1]),
            })
            if len(examples) >= max_examples:
                return examples, failures
    return examples, failures


def full_mode(work: dict[str, Any], max_examples: int, split_mode: str):
    return v2.aligned_sentences(work, max_examples, split_mode)


def run_mode(mode: str, docs: list[dict[str, Any]], args: argparse.Namespace):
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    def one(doc):
        work = {**doc, "converter_exe": args.converter_exe, "mozc_cwd": args.mozc_cwd, "top_k": args.top_k}
        if mode == "prefix_replay_last_segment":
            examples, local_failures = prefix_replay(work, args.max_examples_per_document, args.split_mode)
        else:
            examples, local_failures = full_mode(work, args.max_examples_per_document, args.split_mode)
        return examples, local_failures
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="mozc-parity") as pool:
        futures = [pool.submit(one, doc) for doc in docs]
        for future in as_completed(futures):
            examples, local_failures = future.result()
            rows.extend(v2.record_for(example) for example in examples)
            failures.extend(local_failures)
    rows.sort(key=lambda row: (row["source_id"], row["reading"], row["context_prev"], row["gold"], row["target_segment_index"]))
    elapsed = max(time.monotonic() - started, 1e-9)
    return rows, failures, v2.metric_report(rows, len(docs), failures, elapsed, args.workers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=("split-ablation", "prefix-compare"), required=True)
    ap.add_argument("--documents-input", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--converter-exe", required=True)
    ap.add_argument("--mozc-cwd", required=True)
    ap.add_argument("--documents", type=int, default=100)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--max-examples-per-document", type=int, default=50)
    ap.add_argument("--split-mode", default="C", choices=("A", "B", "C"))
    args = ap.parse_args()
    docs = [json.loads(line) for line in args.documents_input.read_text(encoding="utf-8").splitlines() if line][:args.documents]
    args.workers = max(1, min(args.workers or (os.cpu_count() or 1), len(docs)))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.experiment == "split-ablation":
        modes = {f"full_sentence_split_{mode}": mode for mode in ("A", "B", "C")}
        reports = {}
        for name, mode in modes.items():
            args.split_mode = mode
            rows, failures, report = run_mode("full_sentence_segment_aligned", docs, args)
            reports[name] = report
            (args.work_dir / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
            (args.work_dir / f"{name}.failures.jsonl").write_text("".join(json.dumps(f, ensure_ascii=False, sort_keys=True) + "\n" for f in failures), encoding="utf-8")
    else:
        reports = {}
        for name, mode in (("full_sentence_segment_aligned", "full_sentence_segment_aligned"), ("prefix_replay_last_segment", "prefix_replay_last_segment")):
            rows, failures, report = run_mode(mode, docs, args)
            reports[name] = report
            (args.work_dir / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
            (args.work_dir / f"{name}.failures.jsonl").write_text("".join(json.dumps(f, ensure_ascii=False, sort_keys=True) + "\n" for f in failures), encoding="utf-8")
    report = {"phase": "1-production-parity-pilot", "experiment": args.experiment, "status": "complete", "model_training_started": False, "documents": len(docs), "workers": args.workers, "split_mode": args.split_mode, "modes": reports, "provenance": {"source": "wikimedia/wikipedia supplied JSONL", "converter": args.converter_exe, "mozc_cwd": args.mozc_cwd, "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.lower().startswith("model name")), "unknown"), "logical_cpu_count": os.cpu_count()}}
    if args.experiment == "split-ablation":
        report["split_modes"] = ["A", "B", "C"]
    (args.work_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
