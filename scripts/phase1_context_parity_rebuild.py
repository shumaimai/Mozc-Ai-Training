#!/usr/bin/env python3
"""Rebuild Dataset v2 contexts from selected prefix events only.

This intentionally does *not* tokenize or scan every natural prefix again.
It consumes the frozen source documents and the completed production scan
shards, replays only their deterministic selected events, and constructs the
same history that ``RerankRewriter`` sends to its scorer:

    committed preceding text + top-1 of earlier conversion segments

The output is a new, versioned dataset.  It never overwrites the production
dataset assembled with source-surface context.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import phase1_dataset_v2_pilot as v1
from scripts import phase1_dataset_v2_pilot_v2 as v2
from tools.rerank.context_clip import runtime_context_prev
from tools.rerank.contextual_ranking_v2_schema import validate_record

SEED = "contextual-ranking-v2-production-20231101-ja-v1"
CAP = 30
RETRIES = 3
SPLITS = ("train", "validation", "final_test")
CONTEXT_BUILDER = "runtime_preceding_text_plus_conversion_top1-v1"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shard_name(source_id: str) -> str:
    return hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]


def read_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fin:
        return json.load(fin)


def write_gz(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="wb", compresslevel=6, mtime=0) as fout:
        fout.write((json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="wb", compresslevel=6, mtime=0) as fout:
        for row in rows:
            fout.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dispose_converter() -> None:
    proc = getattr(v1._thread_state, "proc", None)
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    v1._thread_state.proc = None
    v1._thread_state.proc_top_k = None
    v1._thread_state.stdout_buffer = b""


def query_retry(reading: str, converter: str, runfiles: str) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], float]:
    """Replace a dead resident converter and retry the event at most 3 times."""
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for attempt in range(1, RETRIES + 1):
        try:
            return v2.query_segments(reading, converter, runfiles, CAP), attempt, failures, (time.perf_counter() - started) * 1000
        except Exception as exc:
            failures.append({"attempt": attempt, "error": f"{type(exc).__name__}:{exc}"})
            dispose_converter()
    raise RuntimeError(json.dumps({"reason": "CONVERTER_FAILURE", "reading": reading, "attempt_failures": failures}, ensure_ascii=False))


def split_map(documents: list[dict[str, Any]]) -> dict[str, str]:
    ordered = sorted(documents, key=lambda d: (sha(f"{SEED}|split|{d['source_id']}"), d["source_id"]))
    train_n, validation_n = int(len(ordered) * 0.8), int(len(ordered) * 0.1)
    return {
        doc["source_id"]: "train" if i < train_n else "validation" if i < train_n + validation_n else "final_test"
        for i, doc in enumerate(ordered)
    }


def original_row_key(
    *,
    source_id: str,
    reading: str,
    gold: str,
    target_segment_index: int,
    conversion_segments_size: int,
    split: str,
    source_position_ratio: float,
) -> tuple[Any, ...]:
    """Stable join key for the immutable production candidate payload.

    The original Dataset v2 predates ``sampling_identity``.  Its source
    position ratio disambiguates the otherwise repeated
    (source, reading, gold, target) events, and is unique across the frozen
    59,537-row production dataset.
    """
    return (
        source_id,
        reading,
        gold,
        target_segment_index,
        conversion_segments_size,
        split,
        source_position_ratio,
    )


def load_original_candidates(root: Path) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """Load frozen target N-best metadata without changing it during replay."""
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for split in SPLITS:
        path = root / f"{split}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = original_row_key(
                    source_id=row["source_id"],
                    reading=row["reading"],
                    gold=row["gold"],
                    target_segment_index=row["target_segment_index"],
                    conversion_segments_size=row["conversion_segments_size"],
                    split=row["split"],
                    source_position_ratio=row["source_position_ratio"],
                )
                if key in result:
                    raise RuntimeError(f"non-unique frozen production join key: {key}")
                result[key] = row["candidates"]
    return result


def sentence_starts(text: str) -> list[int]:
    return [start for start, _ in v2.sentence_spans(text)]


def replay_document(
    doc: dict[str, Any],
    scan: dict[str, Any],
    converter: str,
    runfiles: str,
    split_membership: dict[str, str],
    original_candidates: dict[tuple[Any, ...], list[dict[str, Any]]],
) -> dict[str, Any]:
    starts = sentence_starts(doc["text"])
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    diffs: list[dict[str, Any]] = []
    latencies: list[float] = []
    for event in scan["selected_events"]:
        try:
            segments, attempts, retry_failures, latency = query_retry(event["prefix_reading"], converter, runfiles)
            latencies.append(latency)
        except Exception as exc:
            failures.append({
                "source_id": doc["source_id"], "identity": event["identity"], "sentence_index": event["sentence_index"],
                "boundary": event["boundary"], "reason": "CONVERTER_FAILURE", "detail": str(exc), "retry_count": RETRIES,
            })
            continue
        target = len(segments) - 1
        if target != event["target_segment_index"] or len(segments) != event["conversion_segments_size"]:
            failures.append({
                "source_id": doc["source_id"], "identity": event["identity"], "sentence_index": event["sentence_index"],
                "boundary": event["boundary"], "reason": "SEGMENTATION_MISMATCH",
                "stored_target_segment_index": event["target_segment_index"], "replayed_target_segment_index": target,
                "stored_conversion_segments_size": event["conversion_segments_size"], "replayed_conversion_segments_size": len(segments),
            })
            continue
        if target < 0 or not segments[target]["candidates"]:
            failures.append({"source_id": doc["source_id"], "identity": event["identity"], "reason": "EMPTY_TARGET_CANDIDATES"})
            continue
        if event["sentence_index"] >= len(starts):
            failures.append({"source_id": doc["source_id"], "identity": event["identity"], "reason": "SENTENCE_INDEX_MISMATCH"})
            continue
        committed_text = doc["text"][:starts[event["sentence_index"]]]
        prefix_top1 = [segment["candidates"][0]["surface"] for segment in segments[:target] if segment["candidates"]]
        context_prev = runtime_context_prev(committed_text, prefix_top1, max_chars=50)
        frozen_key = original_row_key(
            source_id=event["source_id"],
            reading=event["reading"],
            gold=event["gold"],
            target_segment_index=target,
            conversion_segments_size=len(segments),
            split=split_membership[event["source_id"]],
            source_position_ratio=event["source_position_ratio"],
        )
        candidates = original_candidates.get(frozen_key)
        if candidates is None:
            failures.append({
                "source_id": doc["source_id"], "identity": event["identity"],
                "reason": "FROZEN_CANDIDATE_LOOKUP_FAILURE", "frozen_key": repr(frozen_key),
            })
            continue
        example = {
            "source_id": event["source_id"], "reading": event["reading"], "context_prev": context_prev,
            # Context construction consumes the freshly replayed pre-target
            # Mozc top-1.  The target candidate payload stays byte-for-byte
            # from the frozen production Dataset; requerying on a different
            # converter build must not alter the supervised candidate set.
            "gold": event["gold"], "target_segment_index": target, "candidates": candidates,
            "_source_kind": event["source_kind"], "_source_position_ratio": event["source_position_ratio"],
        }
        row = v2.record_for(example)
        if row.get("example_reason") == "latin_mixed":
            row["eligibility_status"] = "COVERAGE_LIMITED"
        row.update({
            "source_kind": event["source_kind"], "proper_noun": event["source_kind"] == "proper_noun",
            "conversion_segments_size": len(segments), "split": split_membership[event["source_id"]],
            "sampling_identity": event["identity"], "context_builder": CONTEXT_BUILDER,
            "context_retry_count": attempts - 1, "context_retry_failures": retry_failures,
        })
        rows.append(row)
        if event["context_prev"] != context_prev:
            diffs.append({
                "source_id": event["source_id"], "sampling_identity": event["identity"], "reading": event["reading"], "gold": event["gold"],
                "old_context_prev": event["context_prev"], "new_context_prev": context_prev,
                "committed_text_chars": len(committed_text), "conversion_prefix_top1": prefix_top1,
            })
    dispose_converter()
    return {
        "source_id": doc["source_id"], "selected_events": len(scan["selected_events"]), "rows": rows,
        "failures": failures, "context_diffs": diffs, "query_latencies_ms": latencies,
        "frozen_candidate_payload_reused": len(rows),
    }


def pct(values: list[float], p: int) -> float:
    return float(statistics.quantiles(values, n=100, method="inclusive")[p - 1]) if values else 0.0


def coverage(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        f"top{k}": sum(row["gold"] in {c["surface"] for c in row["candidates"][:k]} for row in rows) / len(rows) if rows else 0.0
        for k in (1, 5, 10, 20, 30)
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_eligibility: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_eligibility[row["eligibility_status"]].append(row)
    contexts, candidates = [len(row["context_prev"]) for row in rows], [len(row["candidates"]) for row in rows]
    return {
        "rows": len(rows), "schema_failures": sum(bool(validate_record(row)) for row in rows),
        "mozc_baseline": coverage(rows),
        "coverage_by_eligibility": {key: coverage(group) for key, group in sorted(by_eligibility.items())},
        "status_counts": dict(sorted(Counter(row["example_status"] for row in rows).items())),
        "eligibility_counts": dict(sorted((key, len(group)) for key, group in by_eligibility.items())),
        "context_length": {"min": min(contexts, default=0), "median": statistics.median(contexts) if contexts else 0, "p95": pct([float(v) for v in contexts], 95), "max": max(contexts, default=0)},
        "candidate_count": {"min": min(candidates, default=0), "median": statistics.median(candidates) if candidates else 0, "p95": pct([float(v) for v in candidates], 95), "max": max(candidates, default=0)},
    }


def canonical_hash(row: dict[str, Any], omit: Iterable[str]) -> str:
    value = {key: val for key, val in row.items() if key not in set(omit)}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def cross_split_duplicates(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    full = {split: {canonical_hash(row, {"split"}) for row in rows} for split, rows in split_rows.items()}
    model = {split: {canonical_hash(row, {"source_id", "split", "sampling_identity"}) for row in rows} for split, rows in split_rows.items()}
    pairs = (("train", "validation"), ("train", "final_test"), ("validation", "final_test"))
    return {
        "complete_record_excluding_split": {f"{a}_{b}": len(full[a] & full[b]) for a, b in pairs},
        "model_input_candidate_excluding_source_identity": {f"{a}_{b}": len(model[a] & model[b]) for a, b in pairs},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--scan-root", type=Path, required=True)
    ap.add_argument(
        "--original-dataset-root", type=Path, required=True,
        help="immutable production Dataset v2 split directory; target N-best is reused",
    )
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--converter", required=True)
    ap.add_argument("--runfiles", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    started = time.monotonic()
    manifest = json.loads((args.source_root / "selected_manifest.json").read_text(encoding="utf-8"))
    docs = [read_gz(args.source_root / "shards" / f"{shard_name(item['source_id'])}.json.gz") for item in manifest["documents"]]
    docs.sort(key=lambda doc: doc["source_id"])
    membership = split_map(docs)
    original_candidates = load_original_candidates(args.original_dataset_root)
    if len(original_candidates) != 59537:
        raise RuntimeError(f"expected 59537 frozen rows, got {len(original_candidates)}")
    source_map = {doc["source_id"]: doc for doc in docs}
    shard_root = args.out_root / "document_shards"
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    results: list[dict[str, Any]] = []
    for doc in docs:
        out = shard_root / f"{shard_name(doc['source_id'])}.json.gz"
        if out.exists():
            results.append(read_gz(out))
            continue
        scan = read_gz(args.scan_root / f"{shard_name(doc['source_id'])}.json.gz")
        if scan["source_id"] != doc["source_id"]:
            raise RuntimeError(f"scan/source mismatch {doc['source_id']}")
        pending.append((doc, scan))
    lock = threading.Lock()
    done = len(results)
    def run_one(doc: dict[str, Any], scan: dict[str, Any]) -> dict[str, Any]:
        result = replay_document(
            doc, scan, args.converter, args.runfiles, membership, original_candidates
        )
        write_gz(shard_root / f"{shard_name(doc['source_id'])}.json.gz", result)
        return result
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="context-parity") as pool:
        futures = [pool.submit(run_one, doc, scan) for doc, scan in pending]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with lock:
                done += 1
                elapsed = max(time.monotonic() - started, 1e-9)
                print(f"context replay docs={done}/{len(docs)} rows={sum(len(x['rows']) for x in results)} docs_sec={done / elapsed:.4f}", flush=True)
    results.sort(key=lambda value: value["source_id"])
    rows = [row for result in results for row in result["rows"]]
    failures = [failure for result in results for failure in result["failures"]]
    diffs = [diff for result in results for diff in result["context_diffs"]]
    selected = sum(result["selected_events"] for result in results)
    frozen_candidate_payload_reused = sum(
        result.get("frozen_candidate_payload_reused", 0) for result in results
    )
    if len({row["sampling_identity"] for row in rows}) != len(rows):
        raise RuntimeError("sampling identity collision or duplicate replay row")
    split_rows = {split: sorted([row for row in rows if row["split"] == split], key=lambda row: (row["source_id"], row["sampling_identity"])) for split in SPLITS}
    checksums = {f"{split}.jsonl.gz": write_jsonl_gz(args.out_root / "dataset" / f"{split}.jsonl.gz", group) for split, group in split_rows.items()}
    write_json(args.out_root / "failures.json", failures)
    duplicate_audit = cross_split_duplicates(split_rows)
    source_sets = {split: {row["source_id"] for row in group} for split, group in split_rows.items()}
    source_overlap = {"train_validation": len(source_sets["train"] & source_sets["validation"]), "train_final_test": len(source_sets["train"] & source_sets["final_test"]), "validation_final_test": len(source_sets["validation"] & source_sets["final_test"])}
    elapsed = time.monotonic() - started
    audit = {
        "phase": "1-dataset-v2-context-parity", "status": "complete", "model_training_started": False,
        "contract": {
            "context_builder": CONTEXT_BUILDER, "context_max_chars": 50,
            "candidate_cap": CAP, "reused_selected_events": True,
            "no_full_prefix_rescan": True,
            "target_candidate_payload": "reused_immutable_production_dataset",
        },
        "source": {
            "documents": len(docs), "selected_events": selected,
            "replayed_rows": len(rows), "sampling_identity_preserved": len(rows) == selected,
            "frozen_candidate_payload_reused": frozen_candidate_payload_reused,
        },
        "context_parity": {"old_new_context_exact_count": selected - len(diffs), "changed_count": len(diffs), "exact_rate": (selected - len(diffs)) / selected if selected else 1.0, "changed_rate": len(diffs) / selected if selected else 0.0, "examples": diffs[:20]},
        "generation": {"workers": args.workers, "elapsed_seconds": elapsed, "rows_per_sec": len(rows) / max(elapsed, 1e-9), "retries": sum(row["context_retry_count"] for row in rows), "unrecovered_failures": len(failures)},
        "total_rows": len(rows), "distinct_source_ids": len({row["source_id"] for row in rows}), "splits": {split: {"documents": len(source_sets[split]), **metrics(group)} for split, group in split_rows.items()},
        "all_rows": metrics(rows), "source_overlap": source_overlap, "cross_split_duplicates": duplicate_audit, "checksums": checksums,
        "gates": {"PHASE1_CONTEXT_PARITY_GATE": "PASS" if len(rows) == selected and frozen_candidate_payload_reused == selected and not failures and not any(source_overlap.values()) and not any(duplicate_audit["complete_record_excluding_split"].values()) and all(metrics(group)["schema_failures"] == 0 for group in split_rows.values()) else "FAIL"},
    }
    write_json(args.out_root / "audit_report.json", audit)
    write_json(args.out_root / "SHA256SUMS.json", checksums)
    print(json.dumps({"gate": audit["gates"]["PHASE1_CONTEXT_PARITY_GATE"], "rows": len(rows), "changed_contexts": len(diffs), "failures": len(failures)}, ensure_ascii=False), flush=True)
    return 0 if audit["gates"]["PHASE1_CONTEXT_PARITY_GATE"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
