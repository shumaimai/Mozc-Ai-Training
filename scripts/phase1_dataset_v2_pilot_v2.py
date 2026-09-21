#!/usr/bin/env python3
"""Pilot v2: compare standalone-morpheme and Mozc-segment-aligned extraction.

This deliberately stops at dataset measurement.  It keeps one converter per
worker thread, writes document shards, and retains coverage/alignment failures
as explicit status or failure records instead of dropping them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import phase1_dataset_v2_pilot as v1
from tools.dataset.normalize import normalize_reading, normalize_surface
from tools.rerank.context_clip import clean_context
from tools.rerank.contextual_ranking_v2_schema import (
    FORMAT_VERSION,
    SCHEMA_VERSION,
    validate_record,
)
try:
    from sudachipy import SplitMode, dictionary as sudachi_dictionary
except ImportError:
    sudachi_dictionary = None
    SplitMode = None

SEGMENT_RE = re.compile(r"^-+ Segment (\d+)/(\d+) \[(.*?)\] -+")
CANDIDATE_RE = re.compile(r"^\s+(-?\d+)/(\d+) (.*)$")
INT_RE = re.compile(r"\((-?\d+)\)")
FUNCTION_READINGS = {
    "は", "が", "を", "に", "へ", "で", "と", "も", "の", "や", "ね", "よ", "か", "な", "ぞ", "さ", "し",
    "て", "ば", "から", "まで", "だけ", "ほど", "など", "って", "です", "ます", "ない", "ある", "いる", "する", "なる",
}
NUMBER_BIT = 1 << 23
HARD_BITS = (1 << 1) | (1 << 16) | (1 << 17) | (1 << 19)
LATIN_RE = re.compile(r"[A-Za-z]")


def parse_segments(lines: list[str], top_k: int) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        match = SEGMENT_RE.match(line)
        if match:
            current = {"index": int(match.group(1)), "reading": "", "candidates": []}
            segments.append(current)
            last = None
            continue
        if current is None:
            continue
        candidate_match = CANDIDATE_RE.match(line)
        if candidate_match:
            last = {
                "surface": candidate_match.group(3),
                "rank": int(candidate_match.group(1)),
                "cost": 0, "wcost": 0, "cost_delta": 0,
                "lid": 0, "rid": 0, "attributes": 0,
                "category": "DEFAULT", "converted_segment_count": 1,
                "protection": "NORMAL", "attributes_text": "",
            }
            current["candidates"].append(last)
            continue
        stripped = line.strip()
        if not current["candidates"] and stripped and not stripped.startswith("Engine type:"):
            current["reading"] = stripped
            continue
        if last is None:
            continue
        if stripped.startswith("cost:"):
            match = re.search(r"cost:\s*(-?\d+).*?wcost:\s*(-?\d+)", stripped)
            if match:
                last["cost"], last["wcost"] = int(match.group(1)), int(match.group(2))
        elif stripped.startswith("wcost:"):
            last["wcost"] = int(re.search(r"-?\d+", stripped).group(0))
        elif stripped.startswith("lid:"):
            match = INT_RE.search(stripped)
            if match:
                last["lid"] = int(match.group(1))
        elif stripped.startswith("rid:"):
            match = INT_RE.search(stripped)
            if match:
                last["rid"] = int(match.group(1))
        elif stripped.startswith("attributes:"):
            last["attributes"] = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("attr:"):
            last["attributes_text"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("category:"):
            last["category"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("converted_segment_count:"):
            last["converted_segment_count"] = int(stripped.rsplit(" ", 1)[1])
    if not segments:
        raise RuntimeError("converter returned no segments")
    for segment in segments:
        segment["candidates"] = segment["candidates"][:top_k]
        if not segment["candidates"]:
            raise RuntimeError(f"segment {segment['index']} returned no candidates")
        base = segment["candidates"][0]["cost"]
        for candidate in segment["candidates"]:
            candidate["cost_delta"] = candidate["cost"] - base
            candidate["protection"] = v1._protection(
                candidate.pop("attributes_text", ""), candidate["category"], candidate["surface"]
            )
    return segments


def query_segments(reading: str, exe: str, cwd: str, top_k: int) -> list[dict[str, Any]]:
    proc = getattr(v1._thread_state, "proc", None)
    if proc is None or proc.poll() is not None:
        proc = v1._converter(exe, cwd, top_k)
        v1._thread_state.proc = proc
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(f"startconversion {reading}\n")
    proc.stdin.flush()
    lines: list[str] = []
    while True:
        line = proc.stdout.readline()
        if line == "":
            raise RuntimeError(f"converter exited with {proc.poll()}")
        if line == "\n" and lines:
            break
        lines.append(line)
    return parse_segments(lines, top_k)


def sentence_spans(text: str):
    for match in re.finditer(r"[^。！？!?\n]+(?:[。！？!?]|$)|\n+", text):
        sentence = match.group(0)
        if sentence.strip():
            yield match.start(), sentence


def aligned_sentences(work: dict[str, Any], max_examples: int, split_mode: str = "C") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if sudachi_dictionary is None:
        raise RuntimeError("segment alignment requires SudachiPy")
    tokenizer = sudachi_dictionary.Dictionary().create()
    text = work["text"]
    examples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sentence_index, (sentence_start, sentence) in enumerate(sentence_spans(text)):
        morphemes = []
        mode = getattr(SplitMode, split_mode)
        for morpheme in tokenizer.tokenize(sentence, mode):
            surface = normalize_surface(morpheme.surface())
            reading = normalize_reading(morpheme.reading_form() or surface)
            if not surface or not reading:
                continue
            pos = morpheme.part_of_speech()
            morphemes.append({"surface": surface, "reading": reading, "begin": morpheme.begin(), "end": morpheme.end(), "pos": pos})
        if not morphemes:
            continue
        full_reading = "".join(m["reading"] for m in morphemes)
        starts = []
        cursor = 0
        for morpheme in morphemes:
            starts.append(cursor)
            cursor += len(morpheme["reading"])
        ends = starts[1:] + [len(full_reading)]
        try:
            segments = query_segments(full_reading, work["converter_exe"], work["mozc_cwd"], work["top_k"])
        except Exception as exc:
            failures.append({"source_id": work["source_id"], "sentence_index": sentence_index, "reason": "CONVERTER_FAILURE", "detail": f"{type(exc).__name__}:{exc}"})
            continue
        reading_cursor = 0
        for segment in segments:
            segment_reading = normalize_reading(segment["reading"])
            if not segment_reading or full_reading[reading_cursor:reading_cursor + len(segment_reading)] != segment_reading:
                failures.append({"source_id": work["source_id"], "sentence_index": sentence_index, "segment_index": segment["index"], "reason": "ALIGNMENT_FAILURE", "detail": "segment reading did not match Sudachi reading stream"})
                if segment_reading:
                    found = full_reading.find(segment_reading, reading_cursor + 1)
                    reading_cursor = found + len(segment_reading) if found >= 0 else min(len(full_reading), reading_cursor + len(segment_reading))
                continue
            end_cursor = reading_cursor + len(segment_reading)
            if reading_cursor not in starts or end_cursor not in ends:
                failures.append({"source_id": work["source_id"], "sentence_index": sentence_index, "segment_index": segment["index"], "reason": "ALIGNMENT_FAILURE", "detail": "segment boundary split a morpheme"})
                reading_cursor = end_cursor
                continue
            first = starts.index(reading_cursor)
            last = ends.index(end_cursor)
            source_start = sentence_start + morphemes[first]["begin"]
            source_end = sentence_start + morphemes[last]["end"]
            examples.append({
                "source_id": work["source_id"], "reading": segment_reading,
                "context_prev": clean_context(text[:source_start]),
                "gold": "".join(m["surface"] for m in morphemes[first:last + 1]),
                "target_segment_index": segment["index"], "candidates": segment["candidates"],
                "_source_kind": "proper_noun" if any(len(m.get("pos", ())) > 1 and m["pos"][1] == "固有名詞" for m in morphemes[first:last + 1]) else "",
                "_source_start": source_start, "_source_end": source_end,
                "_sentence_index": sentence_index,
            })
            reading_cursor = end_cursor
            if len(examples) >= max_examples:
                return examples, failures
    return examples, failures


def classify(example: dict[str, Any]) -> tuple[str, str, str]:
    candidates = example["candidates"]
    surfaces = [c["surface"] for c in candidates]
    gold = example["gold"]
    candidate = candidates[surfaces.index(gold)] if gold in surfaces else None
    if gold and all(not ch.isalnum() and not ch.isspace() for ch in gold):
        eligibility, protected_reason = "PROTECTED_EVAL_ONLY", "punctuation"
    elif any(ch.isdigit() for ch in gold) or (candidate and candidate.get("attributes", 0) & NUMBER_BIT):
        eligibility, protected_reason = "PROTECTED_EVAL_ONLY", "number"
    elif example.get("_source_kind") == "proper_noun":
        eligibility, protected_reason = "PROTECTED_EVAL_ONLY", "proper_noun"
    elif example["reading"] in FUNCTION_READINGS or len(example["reading"]) <= 1:
        eligibility, protected_reason = "PROTECTED_EVAL_ONLY", "short_function_word"
    elif candidate and (candidate.get("category") == "SYMBOL" or candidate.get("attributes", 0) & HARD_BITS or candidate.get("protection") == "HARD_PROTECT"):
        eligibility, protected_reason = "PROTECTED_EVAL_ONLY", "hard_protected_attribute"
    elif LATIN_RE.search(gold):
        eligibility, protected_reason = "NEURAL_ELIGIBLE", "latin_mixed"
    else:
        eligibility, protected_reason = "NEURAL_ELIGIBLE", "normal_contextual"
    if gold not in surfaces:
        if gold and all(not ch.isalnum() and not ch.isspace() for ch in gold):
            reason = "punctuation_or_symbol"
        elif any(ch.isdigit() for ch in gold):
            reason = "number"
        elif example.get("_source_kind") == "proper_noun":
            reason = "proper_noun"
        elif LATIN_RE.search(gold):
            reason = "latin_mixed"
        elif example["reading"] in FUNCTION_READINGS or len(example["reading"]) <= 1:
            reason = "function_word"
        else:
            reason = "normal_japanese_content"
        return "COVERAGE_FAILURE", reason, eligibility
    return eligibility, protected_reason, eligibility


def record_for(example: dict[str, Any]) -> dict[str, Any]:
    status, reason, eligibility = classify(example)
    record = {k: example[k] for k in ("source_id", "reading", "context_prev", "gold", "target_segment_index", "candidates")}
    record.update({"schema_version": SCHEMA_VERSION, "format_version": FORMAT_VERSION, "example_status": status, "example_reason": reason, "eligibility_status": eligibility})
    return record


def metric_report(rows: list[dict[str, Any]], docs: int, failures: list[dict[str, Any]], elapsed: float, workers: int) -> dict[str, Any]:
    total = len(rows)
    status_counts = {}
    for row in rows:
        status_counts[row["example_status"]] = status_counts.get(row["example_status"], 0) + 1
    def hits(k: int):
        return sum(row["gold"] in [c["surface"] for c in row["candidates"][:k]] for row in rows)
    coverage = {f"top{k}": hits(k) / total if total else 0 for k in (1, 5, 10, 30)}
    coverage_by_status = {}
    for status, group in [("ALL", rows), ("NEURAL_ELIGIBLE", [r for r in rows if r.get("eligibility_status", r["example_status"]) == "NEURAL_ELIGIBLE"]), ("PROTECTED_EVAL_ONLY", [r for r in rows if r.get("eligibility_status", r["example_status"]) == "PROTECTED_EVAL_ONLY"])]:
        coverage_by_status[status] = {f"top{k}": sum(row["gold"] in [c["surface"] for c in row["candidates"][:k]] for row in group) / len(group) if group else 0 for k in (1, 5, 10, 30)}
    conditional = [row for row in rows if row["gold"] in [c["surface"] for c in row["candidates"][:30]]]
    conditional_top = {f"top{k}": sum(row["gold"] in [c["surface"] for c in row["candidates"][:k]] for row in conditional) / len(conditional) if conditional else 0 for k in (1, 5, 10, 30)}
    lengths = [len(row["context_prev"]) for row in rows]
    counts = [len(row["candidates"]) for row in rows]
    per_doc = {}
    for row in rows:
        per_doc[row["source_id"]] = per_doc.get(row["source_id"], 0) + 1
    return {
        "examples": total, "documents": docs, "workers": workers,
        "rows_per_sec": total / elapsed if elapsed else 0, "docs_per_sec": docs / elapsed if elapsed else 0,
        "end_to_end_accuracy": {"top1": coverage["top1"], "top5": coverage["top5"]},
        "conditional_accuracy": conditional_top, "candidate_coverage": coverage, "candidate_coverage_by_status": coverage_by_status,
        "coverage_failure_rate": status_counts.get("COVERAGE_FAILURE", 0) / total if total else 0,
        "status_counts": status_counts,
        "reason_counts": {reason: sum(row["example_reason"] == reason for row in rows) for reason in sorted({row["example_reason"] for row in rows})},
        "reason_ratios": {reason: sum(row["example_reason"] == reason for row in rows) / total if total else 0 for reason in sorted({row["example_reason"] for row in rows})},
        "alignment_failures": len(failures),
        "alignment_success_rate": total / (total + len(failures)) if total + len(failures) else 0,
        "alignment_failure_reasons": {reason: sum(f.get("reason") == reason for f in failures) for reason in sorted({f.get("reason") for f in failures})},
        "alignment_failure_details": {
            "boundary_split_morpheme": sum("boundary split" in f.get("detail", "") or "start split" in f.get("detail", "") for f in failures),
            "reading_stream_mismatch": sum("reading" in f.get("detail", "") for f in failures),
            "other": sum("boundary split" not in f.get("detail", "") and "start split" not in f.get("detail", "") and "reading" not in f.get("detail", "") for f in failures),
        },
        "per_document_examples": {"min": min(per_doc.values()) if per_doc else 0, "median": median(per_doc.values()) if per_doc else 0, "max": max(per_doc.values()) if per_doc else 0, "mean": mean(per_doc.values()) if per_doc else 0},
        "context_length": {"min": min(lengths) if lengths else 0, "median": median(lengths) if lengths else 0, "max": max(lengths) if lengths else 0},
        "candidate_count": {"min": min(counts) if counts else 0, "median": median(counts) if counts else 0, "max": max(counts) if counts else 0},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents-input", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--converter-exe", required=True)
    ap.add_argument("--mozc-cwd", required=True)
    ap.add_argument("--documents", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--max-examples-per-document", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    source_docs = [json.loads(line) for line in args.documents_input.read_text(encoding="utf-8").splitlines() if line][:args.documents]
    workers = max(1, min(args.workers or (os.cpu_count() or 1), len(source_docs)))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    all_rows: dict[str, list[dict[str, Any]]] = {"old_standalone": [], "new_segment_aligned": []}
    all_failures: dict[str, list[dict[str, Any]]] = {"old_standalone": [], "new_segment_aligned": []}
    def one(work: dict[str, Any]):
        work = {**work, "converter_exe": args.converter_exe, "mozc_cwd": args.mozc_cwd, "top_k": args.top_k}
        old_examples = v1.extract_document(work, args.max_examples_per_document)
        old_rows = []
        for example in old_examples:
            try:
                example = {**example, "candidates": v1.query_mozc(example["reading"], args.converter_exe, args.mozc_cwd, args.top_k)}
                old_rows.append(record_for(example))
            except Exception as exc:
                all_failures_local = {"source_id": work["source_id"], "reason": "CONVERTER_FAILURE", "detail": f"{type(exc).__name__}:{exc}"}
                old_rows.append({"_failure": all_failures_local})
        new_examples, new_failures = aligned_sentences(work, args.max_examples_per_document)
        new_rows = [record_for(example) for example in new_examples]
        return work["source_id"], old_rows, new_rows, new_failures
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mozc-v2") as pool:
        futures = [pool.submit(one, doc) for doc in source_docs]
        for index, future in enumerate(as_completed(futures), 1):
            source_id, old_rows, new_rows, failures = future.result()
            all_rows["old_standalone"].extend(row for row in old_rows if "_failure" not in row)
            all_failures["old_standalone"].extend(row["_failure"] for row in old_rows if "_failure" in row)
            all_rows["new_segment_aligned"].extend(new_rows)
            all_failures["new_segment_aligned"].extend(failures)
            elapsed = max(time.monotonic() - started, 1e-9)
            print(f"progress docs={index}/{len(source_docs)} rows_old={len(all_rows['old_standalone'])} rows_new={len(all_rows['new_segment_aligned'])} rows/sec={sum(map(len, all_rows.values())) / elapsed:.2f} docs/sec={index / elapsed:.2f} ETA={(len(source_docs)-index) * elapsed / index if index else 0:.1f}s", flush=True)
    elapsed = max(time.monotonic() - started, 1e-9)
    for mode, rows in all_rows.items():
        rows.sort(key=lambda r: (r["source_id"], r["reading"], r["context_prev"], r["gold"], r["target_segment_index"]))
        (args.work_dir / f"{mode}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        (args.work_dir / f"{mode}.failures.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in all_failures[mode]), encoding="utf-8")
    report = {"phase": "1-pilot-v2", "status": "complete", "model_training_started": False, "documents": len(source_docs), "workers": workers, "seed": args.seed, "modes": {mode: metric_report(rows, len(source_docs), all_failures[mode], elapsed, workers) for mode, rows in all_rows.items()}, "provenance": {"source": "wikimedia/wikipedia supplied JSONL", "converter": args.converter_exe, "mozc_cwd": args.mozc_cwd, "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.lower().startswith("model name")), "unknown"), "logical_cpu_count": os.cpu_count(), "format_version": FORMAT_VERSION, "schema_version": SCHEMA_VERSION}}
    (args.work_dir / "pilot_v2_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
