#!/usr/bin/env python3
"""Deterministic, full-document prefix replay and candidate-cap pilot.

This runner scans every natural SplitMode-A prefix event before sampling.  It
then applies a SHA-256 priority within each document, so worker scheduling,
resume order, and the old first-N bias cannot affect the selected rows.  One
resident converter query requests top-100; lower caps are evaluated by slicing
the same raw candidate list.  This is a measurement tool only: it never trains
or starts production-scale generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import phase1_dataset_v2_pilot_v2 as v2
from tools.dataset.normalize import normalize_reading, normalize_surface
from tools.rerank.context_clip import clean_context

CAPS = (10, 20, 30, 50, 100)
COVERAGE_KS = (1, 5, 10, 20, 30, 50, 100)


def _tokenize(sentence: str) -> list[dict[str, Any]]:
    tokenizer = v2.sudachi_dictionary.Dictionary().create()
    out = []
    for morpheme in tokenizer.tokenize(sentence, v2.SplitMode.A):
        surface = normalize_surface(morpheme.surface())
        reading = normalize_reading(morpheme.reading_form() or surface)
        if surface and reading:
            out.append({"surface": surface, "reading": reading,
                        "begin": morpheme.begin(), "end": morpheme.end(),
                        "pos": morpheme.part_of_speech()})
    return out


def _source_kind(morphemes: list[dict[str, Any]]) -> str:
    if any(len(m.get("pos", ())) > 1 and m["pos"][1] == "固有名詞" for m in morphemes):
        return "proper_noun"
    return "normal_contextual"


def _priority(seed: str, source_id: str, sentence_index: int, boundary: int,
              reading: str, gold: str) -> str:
    payload = "|".join((seed, source_id, str(sentence_index), str(boundary), reading, gold))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scan_document(doc: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    work = {**doc, "converter_exe": args.converter_exe,
            "mozc_cwd": args.mozc_cwd, "top_k": max(CAPS)}
    text = work["text"]
    valid: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    query_latencies_ms: list[float] = []
    total_events = 0
    for sentence_index, (sentence_start, sentence) in enumerate(v2.sentence_spans(text)):
        morphemes = _tokenize(sentence)
        for boundary in range(1, len(morphemes) + 1):
            total_events += 1
            prefix = morphemes[:boundary]
            reading = "".join(m["reading"] for m in prefix)
            if not reading:
                failures.append({"source_id": work["source_id"], "sentence_index": sentence_index,
                                  "boundary": boundary, "reason": "ALIGNMENT_FAILURE",
                                  "detail": "empty prefix reading"})
                continue
            started = time.perf_counter()
            try:
                # Candidate payload is not needed to establish an alignment
                # event.  Enrichment re-queries only the deterministic sample
                # at top-100 below.
                segments = v2.query_segments(reading, work["converter_exe"], work["mozc_cwd"], args.scan_top_k)
            except Exception as exc:
                query_latencies_ms.append((time.perf_counter() - started) * 1000)
                failures.append({"source_id": work["source_id"], "sentence_index": sentence_index,
                                  "boundary": boundary, "reason": "CONVERTER_FAILURE",
                                  "detail": f"{type(exc).__name__}:{exc}"})
                continue
            query_latencies_ms.append((time.perf_counter() - started) * 1000)
            target = segments[-1]
            target_reading = normalize_reading(target["reading"])
            full_reading = reading
            target_start = len(full_reading) - len(target_reading)
            starts = []
            cursor = 0
            for m in prefix:
                starts.append(cursor)
                cursor += len(m["reading"])
            ends = starts[1:] + [len(full_reading)]
            if not target_reading or not full_reading.endswith(target_reading):
                failures.append({"source_id": work["source_id"], "sentence_index": sentence_index,
                                  "boundary": boundary, "target_segment_index": len(segments) - 1,
                                  "reason": "ALIGNMENT_FAILURE", "detail": "reading stream mismatch"})
                continue
            if target_start not in starts or len(full_reading) not in ends:
                failures.append({"source_id": work["source_id"], "sentence_index": sentence_index,
                                  "boundary": boundary, "target_segment_index": len(segments) - 1,
                                  "reason": "ALIGNMENT_FAILURE", "detail": "boundary split a morpheme"})
                continue
            first = starts.index(target_start)
            last = len(prefix) - 1
            source_start = sentence_start + prefix[first]["begin"]
            source_end = sentence_start + prefix[last]["end"]
            gold = "".join(m["surface"] for m in prefix[first:last + 1])
            context = clean_context(text[:source_start])
            key = (work["source_id"], target_reading, context, gold)
            valid.append({
                "source_id": work["source_id"], "reading": target_reading,
                "context_prev": context, "gold": gold,
                "target_segment_index": len(segments) - 1,
                "candidates": target["candidates"], "_prefix_reading": reading,
                "_source_kind": _source_kind(prefix[first:last + 1]),
                "_source_position_ratio": source_start / max(len(text), 1),
                "_sentence_index": sentence_index, "_boundary": boundary,
                "_source_start": source_start, "_source_end": source_end,
                "_dedupe_key": key,
            })
    # Dedupe after the complete scan, then priority-sample.  The raw event
    # counts above remain unchanged and are reported separately.
    deduped = {}
    for event in valid:
        deduped.setdefault(event["_dedupe_key"], event)
    for event in deduped.values():
        event["priority"] = _priority(args.seed, work["source_id"], event["_sentence_index"],
                                       event["_boundary"], event["reading"], event["gold"])
    sampled = sorted(deduped.values(), key=lambda e: (e["priority"], e["_dedupe_key"]))[:args.max_examples_per_document]
    for event in sampled:
        event.pop("_dedupe_key", None)
    return {"source_id": work["source_id"], "total_prefix_events": total_events,
            "alignment_successes": len(valid), "alignment_failures": failures,
            "query_latencies_ms": query_latencies_ms, "valid_events": sampled,
            "deduped_valid_events": len(deduped), "scanned_position_ratios": [
                e["_source_position_ratio"] for e in valid]}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]) if p < 100 else float(max(values))


def _coverage(rows: list[dict[str, Any]], ks: tuple[int, ...] = COVERAGE_KS) -> dict[str, float]:
    return {f"top{k}": sum(row["gold"] in {c["surface"] for c in row["candidates"][:k]} for row in rows) / len(rows) if rows else 0.0 for k in ks}


def _report(rows: list[dict[str, Any]], docs: int, scan: dict[str, Any], elapsed: float,
            cap: int, args: argparse.Namespace, all_query_latencies: list[float]) -> dict[str, Any]:
    groups = {
        "ALL": rows,
        "NEURAL_ELIGIBLE": [r for r in rows if r.get("eligibility_status") == "NEURAL_ELIGIBLE"],
        "PROTECTED_EVAL_ONLY": [r for r in rows if r.get("eligibility_status") == "PROTECTED_EVAL_ONLY"],
        "proper_noun": [r for r in rows if r.get("source_kind") == "proper_noun"],
        "normal_contextual": [r for r in rows if r.get("example_reason") == "normal_contextual"],
        "latin_mixed": [r for r in rows if r.get("example_reason") == "latin_mixed"],
    }
    counts = [len(r["candidates"]) for r in rows]
    positions = [r.get("source_position_ratio", 0.0) for r in rows]
    statuses = {}
    for r in rows:
        statuses[r["example_status"]] = statuses.get(r["example_status"], 0) + 1
    serialized_bytes = sum(len((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")) for r in rows)
    total_events = scan["total_prefix_events"]
    alignment_successes = scan["alignment_successes"]
    return {
        "candidate_cap": cap, "documents": docs, "sampled_rows": len(rows),
        "raw_prefix_events": total_events, "raw_alignment_successes": alignment_successes,
        "raw_alignment_failures": len(scan["alignment_failures"]),
        "raw_alignment_success_rate": alignment_successes / (alignment_successes + len(scan["alignment_failures"])) if total_events else 0.0,
        "deduped_valid_events": scan["deduped_valid_events"], "status_counts": statuses,
        "coverage_by_group": {name: _coverage(group) for name, group in groups.items()},
        "rows_per_sec": len(rows) / elapsed if elapsed else 0.0,
        "docs_per_sec": docs / elapsed if elapsed else 0.0,
        "query_latency_ms": {"p50": _percentile(all_query_latencies, 50), "p95": _percentile(all_query_latencies, 95), "max": max(all_query_latencies) if all_query_latencies else 0.0, "queries": len(all_query_latencies)},
        "candidate_count": {"median": statistics.median(counts) if counts else 0, "p95": _percentile([float(x) for x in counts], 95), "max": max(counts) if counts else 0},
        "serialized_dataset_bytes": serialized_bytes,
        "serialized_dataset_mib": serialized_bytes / (1024 * 1024),
        "context_length": {"median": statistics.median([len(r["context_prev"]) for r in rows]) if rows else 0, "max": max([len(r["context_prev"]) for r in rows], default=0)},
        "per_document_sampled": {"min": min([sum(r["source_id"] == d for r in rows) for d in {r["source_id"] for r in rows}], default=0), "median": statistics.median([sum(r["source_id"] == d for r in rows) for d in {r["source_id"] for r in rows}]) if rows else 0, "max": max([sum(r["source_id"] == d for r in rows) for d in {r["source_id"] for r in rows}], default=0)},
        "source_position_ratio_sampled": {"front_0_33": sum(x < 1/3 for x in positions), "middle_33_66": sum(1/3 <= x < 2/3 for x in positions), "back_66_100": sum(x >= 2/3 for x in positions)},
        "source_position_ratio_scanned_valid": {"front_0_33": sum(x < 1/3 for x in scan["scanned_position_ratios"]), "middle_33_66": sum(1/3 <= x < 2/3 for x in scan["scanned_position_ratios"]), "back_66_100": sum(x >= 2/3 for x in scan["scanned_position_ratios"])},
        "sampling": {"seed": args.seed, "algorithm": "sha256(seed|source_id|sentence_index|boundary|reading|gold), lowest N per document", "max_examples_per_document": args.max_examples_per_document},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents-input", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--converter-exe", required=True)
    ap.add_argument("--mozc-cwd", required=True)
    ap.add_argument("--documents", type=int, default=100)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-examples-per-document", type=int, default=50)
    ap.add_argument("--scan-top-k", type=int, default=30)
    ap.add_argument("--seed", default="contextual-ranking-v2-pilot-v3")
    args = ap.parse_args()
    docs = [json.loads(line) for line in args.documents_input.read_text(encoding="utf-8").splitlines() if line][:args.documents]
    args.workers = max(1, min(args.workers or (os.cpu_count() or 1), len(docs)))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.work_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    def shard_path(source_id: str) -> Path:
        return shard_dir / f"doc_{hashlib.sha256(source_id.encode('utf-8')).hexdigest()[:20]}.json"
    started = time.monotonic()
    scans = []
    pending = []
    resumed_docs = 0
    for doc in docs:
        path = shard_path(doc["source_id"])
        if path.exists():
            scans.append(json.loads(path.read_text(encoding="utf-8")))
            resumed_docs += 1
        else:
            pending.append(doc)
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="mozc-cap") as pool:
        futures = {pool.submit(scan_document, doc, args): doc for doc in pending}
        for i, future in enumerate(as_completed(futures), resumed_docs + 1):
            result = future.result()
            scans.append(result)
            path = shard_path(result["source_id"])
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            temp.replace(path)
            elapsed = max(time.monotonic() - started, 1e-9)
            events = sum(x["total_prefix_events"] for x in scans)
            print(f"progress docs={i}/{len(docs)} raw_events={events} docs/sec={i/elapsed:.2f} ETA={(len(docs)-i)*elapsed/i if i else 0:.1f}s", flush=True)
    scans.sort(key=lambda x: x["source_id"])
    elapsed = max(time.monotonic() - started, 1e-9)
    all_events = {"total_prefix_events": sum(x["total_prefix_events"] for x in scans), "alignment_successes": sum(x["alignment_successes"] for x in scans), "alignment_failures": [f for x in scans for f in x["alignment_failures"]], "deduped_valid_events": sum(x["deduped_valid_events"] for x in scans), "scanned_position_ratios": [p for x in scans for p in x["scanned_position_ratios"]]}
    all_query_latencies = [v for x in scans for v in x["query_latencies_ms"]]
    sampled = [e for x in scans for e in x["valid_events"]]
    # Enrich the selected, deterministic events with top-100 candidates.  The
    # scan remains complete, while expensive large-N output is not requested
    # for events that cannot enter the per-document sample.
    def enrich(event: dict[str, Any]) -> dict[str, Any]:
        started_query = time.perf_counter()
        try:
            segments = v2.query_segments(event["_prefix_reading"], args.converter_exe, args.mozc_cwd, max(CAPS))
            event = {**event, "candidates": segments[-1]["candidates"]}
        except Exception as exc:
            event = {**event, "_enrichment_failure": {"source_id": event["source_id"], "sentence_index": event["_sentence_index"], "boundary": event["_boundary"], "reason": "CONVERTER_TIMEOUT" if isinstance(exc, TimeoutError) else "CONVERTER_FAILURE", "detail": f"{type(exc).__name__}:{exc}"}}
        event["_query_latency_ms"] = (time.perf_counter() - started_query) * 1000
        return event
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="mozc-cap-enrich") as pool:
        sampled = list(pool.map(enrich, sampled))
    all_query_latencies.extend(e["_query_latency_ms"] for e in sampled)
    enrichment_failures = [e.pop("_enrichment_failure") for e in sampled if "_enrichment_failure" in e]
    sampled = [e for e in sampled if "_enrichment_failure" not in e]
    for event in sampled:
        event.pop("_prefix_reading", None)
        event.pop("_query_latency_ms", None)
    outputs = {}
    for cap in CAPS:
        rows = []
        for event in sampled:
            capped = {**event, "candidates": event["candidates"][:cap]}
            rows.append(v2.record_for(capped))
        rows.sort(key=lambda r: (r["source_id"], r["reading"], r["context_prev"], r["gold"]))
        name = f"prefix_replay_splitA_within_doc_top{cap}"
        (args.work_dir / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        outputs[str(cap)] = _report(rows, len(docs), all_events, elapsed, cap, args, all_query_latencies)
    (args.work_dir / "alignment_failures.jsonl").write_text("".join(json.dumps(f, ensure_ascii=False, sort_keys=True) + "\n" for f in all_events["alignment_failures"]), encoding="utf-8")
    (args.work_dir / "enrichment_failures.jsonl").write_text("".join(json.dumps(f, ensure_ascii=False, sort_keys=True) + "\n" for f in enrichment_failures), encoding="utf-8")
    report = {"phase": "1-dataset-design-pilot", "status": "complete", "model_training_started": False, "documents": len(docs), "workers": args.workers, "resumed_documents": resumed_docs, "split_mode": "A", "candidate_caps": outputs, "raw_scan": {"total_prefix_events": all_events["total_prefix_events"], "alignment_successes": all_events["alignment_successes"], "alignment_failures": len(all_events["alignment_failures"]), "alignment_success_rate": all_events["alignment_successes"] / all_events["total_prefix_events"] if all_events["total_prefix_events"] else 0, "deduped_valid_events": all_events["deduped_valid_events"], "enrichment_failures": len(enrichment_failures)}, "sampling_resume": {"shard_dir": str(shard_dir), "per_document_shards": len(scans), "resume_safe": True}, "provenance": {"source": "wikimedia/wikipedia supplied JSONL", "converter": args.converter_exe, "mozc_cwd": args.mozc_cwd, "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.lower().startswith("model name")), "unknown"), "logical_cpu_count": os.cpu_count(), "seed": args.seed}}
    (args.work_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
