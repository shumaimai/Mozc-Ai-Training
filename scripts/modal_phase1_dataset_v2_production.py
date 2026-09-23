#!/usr/bin/env python3
"""Modal CPU-only, production-faithful Dataset v2 generation.

This is deliberately a *generation-only* app.  It performs the required
preflight before it writes production rows, freezes a dump-derived source
manifest, and never imports or launches model-training code.

The app is resumable through the ``mozc-v2-dataset-production`` Modal Volume:
source documents, per-document scan shards, selected-event enrichment shards,
and final compressed JSONL files are all committed stage by stage.
"""
from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOCAL_RUNNER = os.environ.get("MOZC_V2_LOCAL_RUNNER") == "1"


if _LOCAL_RUNNER:
    # Keep the extraction core runnable on an operator-owned CPU host without
    # importing Modal credentials or mutating the Modal app/volume.  The stub
    # only supplies decorators for definitions that are not invoked locally.
    class _LocalApp:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def function(self, *args: Any, **kwargs: Any) -> Any:
            return lambda fn: fn

        def local_entrypoint(self, *args: Any, **kwargs: Any) -> Any:
            return lambda fn: fn

    class _LocalImage:
        def __getattr__(self, _name: str) -> Any:
            return lambda *args, **kwargs: self

    class _LocalVolume:
        def commit(self) -> None:
            return None

    app = _LocalApp("mozc-v2-dataset-production-local")
    image = _LocalImage()
    volume = _LocalVolume()
else:
    import modal

# Modal mounts repository directories but does not automatically put their
# parent on ``sys.path`` when a function is invoked by qualified name.
REPO_ROOT = Path(os.environ.get("MOZC_V2_REPO_ROOT", "/root/repo"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The converter was built from the pinned Mozc-Ai checkout.  ``copy=True``
# dereferences Bazel runfiles symlinks so Modal receives the actual bytes.
_BAZEL_BIN = Path(
    "/home/hashiguchishuhei/.cache/bazel/_bazel_hashiguchishuhei/"
    "580cd470e71d471880ff7374d0116a46/execroot/_main/bazel-out/"
    "k8-fastbuild/bin/converter"
)
_CONVERTER = _BAZEL_BIN / "converter_main"
_MOZC_DATA = _BAZEL_BIN / "converter_main.runfiles/_main/data_manager/oss/mozc.data"

if not _LOCAL_RUNNER:
    app = modal.App("mozc-v2-dataset-production")
    image = (
        # The pinned local Mozc converter needs GLIBC_2.38 / GLIBCXX_3.4.32.
        # Ubuntu 24.04 supplies both; Debian slim's older ABI does not.
        modal.Image.from_registry("ubuntu:24.04", add_python="3.11")
        .apt_install("ca-certificates")
        # Pin the published wheel: the newer source-distribution variant attempts
        # a second dictionary download during image build.
        .pip_install("datasets==3.6.0", "sudachipy", "sudachidict_core==20260723")
        .add_local_dir("scripts", "/root/repo/scripts", copy=True)
        .add_local_dir("tools", "/root/repo/tools", copy=True)
        .add_local_file(_CONVERTER, "/opt/mozc/converter_main", copy=True)
        .add_local_file(_MOZC_DATA, "/opt/mozc/runfiles/data_manager/oss/mozc.data", copy=True)
    )
    volume = modal.Volume.from_name("mozc-v2-dataset-production", create_if_missing=True)

ROOT = Path(os.environ.get("MOZC_V2_DATASET_ROOT", "/dataset-v2"))
SOURCE_ROOT = ROOT / "source_20231101_ja"
PREFLIGHT_ROOT = ROOT / "preflight"
PRODUCTION_ROOT = ROOT / "production"
SEED = "contextual-ranking-v2-production-20231101-ja-v1"
DOCS = 1200
MAX_PER_DOC = 50
CAP = 30
PREFLIGHT_DOCS = 20
RETRIES = 3
CONVERTER_PATH = os.environ.get("MOZC_V2_CONVERTER_PATH", "/opt/mozc/converter_main")
RUNFILES_ROOT = os.environ.get("MOZC_V2_RUNFILES_ROOT", "/opt/mozc/runfiles")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_gz_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="wb", compresslevel=6, mtime=0) as fout:
        fout.write(_json_bytes(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_gz_json(path: Path) -> Any:
    with gzip.open(path, "rb") as fin:
        return json.loads(fin.read().decode("utf-8"))


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="wb", compresslevel=6, mtime=0) as fout:
        for row in rows:
            fout.write(_json_bytes(row))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pct(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    return float(statistics.quantiles(values, n=100, method="inclusive")[p - 1]) if p < 100 else float(max(values))


def _source_id(row: dict[str, Any]) -> str:
    return f"wikipedia-ja:{row.get('id', '')}"


def _doc_shard_name(source_id: str) -> str:
    return hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]


def _freeze_sources() -> list[dict[str, Any]]:
    """Global bottom-k selection from the full dump-derived HF stream.

    A max heap holds only 1,200 complete selected docs.  All target-universe
    documents are inspected; neither stream order nor a local API extract can
    influence membership.
    """
    manifest_path = SOURCE_ROOT / "selected_manifest.json"
    if manifest_path.exists():
        return _read_json(manifest_path)["documents"]
    # Keep datasets/PyArrow native teardown isolated from the long-running
    # converter process.  The child exits with os._exit only after its manifest
    # and source shards have been atomically written.
    subprocess.check_call([sys.executable, "/root/repo/scripts/phase1_freeze_wikipedia_sources.py", "--out-dir", str(SOURCE_ROOT), "--seed", SEED, "--documents", str(DOCS)])
    return _read_json(manifest_path)["documents"]


def _load_frozen_docs() -> list[dict[str, Any]]:
    manifest = _read_json(SOURCE_ROOT / "selected_manifest.json")
    docs = []
    for item in manifest["documents"]:
        path = SOURCE_ROOT / "shards" / f"{_doc_shard_name(item['source_id'])}.json.gz"
        doc = _read_gz_json(path)
        if doc["source_id"] != item["source_id"] or doc["priority"] != item["priority"]:
            raise RuntimeError(f"source shard/manifest mismatch for {item['source_id']}")
        docs.append(doc)
    return docs


def _dispose_converter() -> None:
    from scripts import phase1_dataset_v2_pilot as v1

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


def _query_retry(reading: str, top_k: int, inject_exit_once: bool = False) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]], float]:
    """Query one resident converter, replacing it after every failed attempt."""
    from scripts import phase1_dataset_v2_pilot_v2 as v2
    from scripts import phase1_dataset_v2_pilot as v1

    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for attempt in range(1, RETRIES + 1):
        try:
            if inject_exit_once and attempt == 1:
                # Force the same process-exit recovery path as the historical
                # ``converter exited -15`` enrichment failures.
                _dispose_converter()
                v2.query_segments(reading, CONVERTER_PATH, RUNFILES_ROOT, top_k)
                proc = getattr(v1._thread_state, "proc", None)
                if proc is not None:
                    proc.terminate()
            segments = v2.query_segments(reading, CONVERTER_PATH, RUNFILES_ROOT, top_k)
            return segments, attempt, failures, (time.perf_counter() - started) * 1000
        except Exception as exc:
            failures.append({"attempt": attempt, "error": f"{type(exc).__name__}:{exc}"})
            _dispose_converter()
    raise RuntimeError(json.dumps({"reason": "CONVERTER_FAILURE", "reading": reading, "attempts": RETRIES, "attempt_failures": failures}, ensure_ascii=False))


def _tokenize(sentence: str) -> list[dict[str, Any]]:
    from scripts import phase1_dataset_v2_pilot_v2 as v2
    from tools.dataset.normalize import normalize_reading, normalize_surface

    # A function attribute is per-process but keys threads; avoids rebuilding
    # Sudachi dictionaries for every sentence while retaining thread safety.
    cache = getattr(_tokenize, "_cache", None)
    if cache is None:
        cache = threading.local()
        _tokenize._cache = cache  # type: ignore[attr-defined]
    tokenizer = getattr(cache, "tokenizer", None)
    if tokenizer is None:
        tokenizer = v2.sudachi_dictionary.Dictionary().create()
        cache.tokenizer = tokenizer
    tokens = []
    for morpheme in tokenizer.tokenize(sentence, v2.SplitMode.A):
        surface = normalize_surface(morpheme.surface())
        reading = normalize_reading(morpheme.reading_form() or surface)
        if surface and reading:
            tokens.append({"surface": surface, "reading": reading, "begin": morpheme.begin(), "end": morpheme.end(), "pos": morpheme.part_of_speech()})
    return tokens


def _source_kind(tokens: list[dict[str, Any]]) -> str:
    return "proper_noun" if any(len(t.get("pos", ())) > 1 and t["pos"][1] == "固有名詞" for t in tokens) else "normal_contextual"


def _event_priority(event: dict[str, Any]) -> str:
    return _sha("|".join((SEED, event["source_id"], str(event["sentence_index"]), str(event["boundary"]), event["reading"], event["gold"])))


def _scan_document(doc: dict[str, Any], top_k: int, capture_alignment_keys: bool = False) -> dict[str, Any]:
    from scripts import phase1_dataset_v2_pilot_v2 as v2
    from tools.rerank.context_clip import clean_context
    from tools.dataset.normalize import normalize_reading

    valid: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    latencies: list[float] = []
    raw_events = 0
    for sentence_index, (sentence_start, sentence) in enumerate(v2.sentence_spans(doc["text"])):
        morphemes = _tokenize(sentence)
        for boundary in range(1, len(morphemes) + 1):
            raw_events += 1
            prefix = morphemes[:boundary]
            full_reading = "".join(m["reading"] for m in prefix)
            if not full_reading:
                failures.append({"source_id": doc["source_id"], "sentence_index": sentence_index, "boundary": boundary, "reason": "ALIGNMENT_FAILURE", "detail": "empty_prefix_reading"})
                continue
            try:
                segments, attempts, retry_failures, latency = _query_retry(full_reading, top_k)
                latencies.append(latency)
            except Exception as exc:
                failures.append({"source_id": doc["source_id"], "sentence_index": sentence_index, "boundary": boundary, "reason": "CONVERTER_FAILURE", "detail": str(exc), "retry_count": RETRIES})
                continue
            target = segments[-1]
            target_reading = normalize_reading(target["reading"])
            starts, cursor = [], 0
            for m in prefix:
                starts.append(cursor)
                cursor += len(m["reading"])
            ends = starts[1:] + [len(full_reading)]
            target_start = len(full_reading) - len(target_reading)
            if not target_reading or not full_reading.endswith(target_reading):
                failures.append({"source_id": doc["source_id"], "sentence_index": sentence_index, "boundary": boundary, "target_segment_index": len(segments)-1, "reason": "ALIGNMENT_FAILURE", "detail": "reading_stream_mismatch", "retry_count": attempts - 1, "retry_failures": retry_failures})
                continue
            if target_start not in starts or len(full_reading) not in ends:
                failures.append({"source_id": doc["source_id"], "sentence_index": sentence_index, "boundary": boundary, "target_segment_index": len(segments)-1, "reason": "ALIGNMENT_FAILURE", "detail": "boundary_split_morpheme", "retry_count": attempts - 1, "retry_failures": retry_failures})
                continue
            first = starts.index(target_start)
            source_start = sentence_start + prefix[first]["begin"]
            source_end = sentence_start + prefix[-1]["end"]
            event = {
                "source_id": doc["source_id"], "reading": target_reading,
                "context_prev": clean_context(doc["text"][:source_start]),
                "gold": "".join(m["surface"] for m in prefix[first:]),
                "target_segment_index": len(segments) - 1,
                "conversion_segments_size": len(segments),
                "prefix_reading": full_reading, "source_kind": _source_kind(prefix[first:]),
                "source_position_ratio": source_start / max(1, len(doc["text"])),
                "sentence_index": sentence_index, "boundary": boundary,
                "source_start": source_start, "source_end": source_end,
                "retry_count": attempts - 1,
            }
            event["identity"] = _sha("|".join((event["source_id"], str(sentence_index), str(boundary), event["reading"], event["gold"], event["context_prev"])))
            valid.append(event)
    dedup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for event in valid:
        dedup.setdefault((event["source_id"], event["reading"], event["context_prev"], event["gold"]), event)
    for event in dedup.values():
        event["priority"] = _event_priority(event)
    selected = sorted(dedup.values(), key=lambda e: (e["priority"], e["identity"]))[:MAX_PER_DOC]
    result = {"source_id": doc["source_id"], "raw_prefix_events": raw_events, "alignment_successes": len(valid), "alignment_failures": failures, "deduped_valid_events": len(dedup), "selected_events": selected, "query_latencies_ms": latencies}
    if capture_alignment_keys:
        # Preflight compares *every* natural prefix boundary, not merely the
        # final priority sample.  Production keeps aggregate counts/failures
        # instead, avoiding needless source-shard bloat.
        result["alignment_success_keys"] = [[e["sentence_index"], e["boundary"]] for e in valid]
    return result


def _scan_docs(docs: list[dict[str, Any]], top_k: int, workers: int, root: Path, resume: bool, capture_alignment_keys: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=True)
    scans: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for doc in docs:
        path = root / "scan_shards" / f"{_doc_shard_name(doc['source_id'])}.json.gz"
        if resume and path.exists():
            scans.append(_read_gz_json(path))
        else:
            pending.append(doc)
    def run_one(doc: dict[str, Any]) -> dict[str, Any]:
        result = _scan_document(doc, top_k, capture_alignment_keys=capture_alignment_keys)
        _write_gz_json(root / "scan_shards" / f"{_doc_shard_name(doc['source_id'])}.json.gz", result)
        return result
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"scan-{top_k}") as pool:
        futures = [pool.submit(run_one, doc) for doc in pending]
        for index, future in enumerate(as_completed(futures), len(scans) + 1):
            scans.append(future.result())
            elapsed = max(time.monotonic() - started, 1e-9)
            print(f"scan top{top_k} docs={index}/{len(docs)} events={sum(s['raw_prefix_events'] for s in scans)} docs_sec={index/elapsed:.4f}", flush=True)
    scans.sort(key=lambda s: s["source_id"])
    elapsed = max(time.monotonic() - started, 1e-9)
    summary = {
        "documents": len(docs), "workers": workers, "top_k": top_k,
        "elapsed_seconds": elapsed,
        "raw_prefix_events": sum(s["raw_prefix_events"] for s in scans),
        "alignment_successes": sum(s["alignment_successes"] for s in scans),
        "alignment_failures": sum(len(s["alignment_failures"]) for s in scans),
        "selected_events": sum(len(s["selected_events"]) for s in scans),
        "rows_per_sec": sum(s["alignment_successes"] for s in scans) / elapsed,
        "docs_per_sec": len(docs) / elapsed,
        "query_latency_ms": {"p50": _pct([x for s in scans for x in s["query_latencies_ms"]], 50), "p95": _pct([x for s in scans for x in s["query_latencies_ms"]], 95)},
    }
    _write_json(root / "scan_summary.json", summary)
    return scans, summary


def _identity_set(scans: list[dict[str, Any]]) -> set[str]:
    return {event["identity"] for scan in scans for event in scan["selected_events"]}


def _alignment_map(scans: list[dict[str, Any]]) -> dict[tuple[str, int, int], str]:
    values = {}
    for scan in scans:
        for sentence_index, boundary in scan.get("alignment_success_keys", []):
            values[(scan["source_id"], sentence_index, boundary)] = "success"
        for failure in scan["alignment_failures"]:
            values[(failure["source_id"], failure.get("sentence_index", -1), failure.get("boundary", -1))] = failure["reason"] + ":" + failure.get("detail", "")
    return values


def _retry_probe(scans: list[dict[str, Any]]) -> dict[str, Any]:
    event = next(e for scan in scans for e in scan["selected_events"])
    try:
        segments, attempts, failures, _ = _query_retry(event["prefix_reading"], CAP, inject_exit_once=True)
        return {"status": "recovered", "source_id": event["source_id"], "identity": event["identity"], "attempts": attempts, "attempt_failures": failures, "candidate_count": len(segments[-1]["candidates"])}
    finally:
        _dispose_converter()


def _run_preflight(docs: list[dict[str, Any]]) -> dict[str, Any]:
    report_path = PREFLIGHT_ROOT / "report.json"
    if report_path.exists():
        return _read_json(report_path)
    subset = docs[:PREFLIGHT_DOCS]
    scan1, summary1 = _scan_docs(subset, 1, 4, PREFLIGHT_ROOT / "parity_top1_w4", resume=False, capture_alignment_keys=True)
    scan30, summary30 = _scan_docs(subset, 30, 4, PREFLIGHT_ROOT / "parity_top30_w4", resume=False, capture_alignment_keys=True)
    ids1, ids30 = _identity_set(scan1), _identity_set(scan30)
    align1, align30 = _alignment_map(scan1), _alignment_map(scan30)
    union = ids1 | ids30
    align_union = set(align1) | set(align30)
    worker_runs = {}
    for workers in (2, 4, 8):
        _, summary = _scan_docs(subset, 1, workers, PREFLIGHT_ROOT / f"workers_top1_w{workers}", resume=False)
        worker_runs[str(workers)] = summary
    chosen_workers = max((2, 4, 8), key=lambda w: (worker_runs[str(w)]["rows_per_sec"], worker_runs[str(w)]["docs_per_sec"], -w))
    probe = _retry_probe(scan1)
    identity_rate = len(ids1 & ids30) / len(union) if union else 1.0
    alignment_rate = sum(align1.get(key) == align30.get(key) for key in align_union) / len(align_union) if align_union else 1.0
    report = {
        "phase": "1-dataset-v2-production-preflight", "documents": len(subset),
        "scan_top_k_parity": {"top1": summary1, "top30": summary30, "selected_event_identity_exact_rate": identity_rate, "alignment_result_exact_rate": alignment_rate},
        "worker_count": {"runs": worker_runs, "chosen_workers": chosen_workers, "selection": "max alignment-success rows/sec, then docs/sec, then lower worker count"},
        "enrichment_retry_probe": probe,
        "gate": "PASS" if identity_rate == 1.0 and alignment_rate == 1.0 and probe["status"] == "recovered" else "FAIL",
        "production_scan_top_k": 1 if identity_rate == 1.0 and alignment_rate == 1.0 else None,
    }
    _write_json(report_path, report)
    if report["gate"] != "PASS":
        raise RuntimeError("preflight gate failed; production generation not started")
    return report


def _enrich_event(event: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        segments, attempts, retry_failures, latency = _query_retry(event["prefix_reading"], CAP)
        row = {**event, "candidates": segments[-1]["candidates"], "enrichment_retry_count": attempts - 1, "enrichment_latency_ms": latency, "enrichment_retry_failures": retry_failures}
        return row, None
    except Exception as exc:
        return None, {"source_id": event["source_id"], "sentence_index": event["sentence_index"], "boundary": event["boundary"], "identity": event["identity"], "reason": "CONVERTER_FAILURE", "detail": str(exc), "retry_count": RETRIES}


def _split_map(docs: list[dict[str, Any]]) -> dict[str, str]:
    order = sorted(docs, key=lambda d: (_sha(f"{SEED}|split|{d['source_id']}"), d["source_id"]))
    train_n, validation_n = int(len(order) * .8), int(len(order) * .1)
    return {d["source_id"]: "train" if i < train_n else "validation" if i < train_n + validation_n else "final_test" for i, d in enumerate(order)}


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from tools.rerank.contextual_ranking_v2_schema import validate_record
    by_eligibility: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_eligibility.setdefault(row["eligibility_status"], []).append(row)
    def coverage(group: list[dict[str, Any]]) -> dict[str, float]:
        return {f"top{k}": sum(r["gold"] in {c["surface"] for c in r["candidates"][:k]} for r in group) / len(group) if group else 0.0 for k in (1, 5, 10, 20, 30)}
    contexts = [len(r["context_prev"]) for r in rows]
    candidates = [len(r["candidates"]) for r in rows]
    statuses = Counter(r["example_status"] for r in rows)
    return {
        "rows": len(rows), "schema_failures": sum(bool(validate_record(r)) for r in rows),
        "mozc_baseline": coverage(rows), "coverage_by_eligibility": {key: coverage(group) for key, group in sorted(by_eligibility.items())},
        "status_counts": dict(sorted(statuses.items())),
        "context_length": {"min": min(contexts, default=0), "median": statistics.median(contexts) if contexts else 0, "p95": _pct([float(x) for x in contexts], 95), "max": max(contexts, default=0)},
        "candidate_count": {"min": min(candidates, default=0), "median": statistics.median(candidates) if candidates else 0, "p95": _pct([float(x) for x in candidates], 95), "max": max(candidates, default=0)},
    }


def _run_production(docs: list[dict[str, Any]], preflight: dict[str, Any]) -> dict[str, Any]:
    report_path = PRODUCTION_ROOT / "audit_report.json"
    if report_path.exists():
        return _read_json(report_path)
    started = time.monotonic()
    workers = int(preflight["worker_count"]["chosen_workers"])
    scans, scan_summary = _scan_docs(docs, 1, workers, PRODUCTION_ROOT, resume=True)
    selected = [event for scan in scans for event in scan["selected_events"]]
    enriched: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="enrich-top30") as pool:
        futures = [pool.submit(_enrich_event, event) for event in selected]
        for idx, future in enumerate(as_completed(futures), 1):
            row, failure = future.result()
            if row is not None:
                enriched.append(row)
            else:
                failures.append(failure or {})
            if idx % 500 == 0 or idx == len(selected):
                print(f"enrich selected={idx}/{len(selected)} successes={len(enriched)} failures={len(failures)}", flush=True)
    from scripts import phase1_dataset_v2_pilot_v2 as v2
    split_membership = _split_map(docs)
    rows = []
    for event in enriched:
        example = {"source_id": event["source_id"], "reading": event["reading"], "context_prev": event["context_prev"], "gold": event["gold"], "target_segment_index": event["target_segment_index"], "candidates": event["candidates"], "_source_kind": event["source_kind"], "_source_position_ratio": event["source_position_ratio"]}
        row = v2.record_for(example)
        # latin/mixed is retained, but categorically separated from normal
        # neural training eligibility; proper_noun remains explicit metadata.
        if row.get("example_reason") == "latin_mixed":
            row["eligibility_status"] = "COVERAGE_LIMITED"
        row["source_kind"] = event["source_kind"]
        row["proper_noun"] = event["source_kind"] == "proper_noun"
        row["conversion_segments_size"] = event["conversion_segments_size"]
        row["split"] = split_membership[row["source_id"]]
        rows.append(row)
    rows.sort(key=lambda r: (r["split"], r["source_id"], r["reading"], r["context_prev"], r["gold"]))
    split_rows = {split: [r for r in rows if r["split"] == split] for split in ("train", "validation", "final_test")}
    checksums = {}
    for split, group in split_rows.items():
        checksums[f"{split}.jsonl.gz"] = _write_jsonl_gz(PRODUCTION_ROOT / "dataset" / f"{split}.jsonl.gz", group)
    _write_json(PRODUCTION_ROOT / "failures" / "alignment_failures.json", [f for scan in scans for f in scan["alignment_failures"]])
    _write_json(PRODUCTION_ROOT / "failures" / "enrichment_failures.json", failures)
    positions = [r["source_position_ratio"] for r in rows]
    split_docs = {split: sorted({r["source_id"] for r in group}) for split, group in split_rows.items()}
    overlap = {"train_validation": len(set(split_docs["train"]) & set(split_docs["validation"])), "train_final_test": len(set(split_docs["train"]) & set(split_docs["final_test"])), "validation_final_test": len(set(split_docs["validation"]) & set(split_docs["final_test"]))}
    per_doc = Counter(r["source_id"] for r in rows)
    audit = {
        "phase": "1-dataset-v2-production", "status": "complete", "model_training_started": False,
        "contract": {"mode": "prefix_replay_last_conversion_segment", "split_mode": "A", "context_max_chars": 50, "sampling": "SHA256(seed|source_id|sentence_index|boundary|reading|gold) lowest 50/document", "candidate_cap": CAP, "scan_top_k": 1, "enrichment_top_k": CAP, "source": "wikimedia/wikipedia", "config": "20231101.ja"},
        "source": {"documents": len(docs), "manifest": str(SOURCE_ROOT / "selected_manifest.json"), "retrieval_config": str(SOURCE_ROOT / "retrieval_config.json")},
        "generation": {"workers": workers, "elapsed_seconds": time.monotonic()-started, "rows_per_sec": len(rows)/max(time.monotonic()-started, 1e-9), "scan": scan_summary, "enrichment_failures": len(failures), "enrichment_retries": sum(e.get("enrichment_retry_count", 0) for e in enriched)},
        "total_rows": len(rows), "distinct_source_ids": len(per_doc),
        "all_rows": _metrics(rows),
        "splits": {split: {"documents": len(split_docs[split]), **_metrics(group)} for split, group in split_rows.items()},
        "source_overlap": overlap,
        "rows_per_document": {"min": min(per_doc.values(), default=0), "median": statistics.median(per_doc.values()) if per_doc else 0, "max": max(per_doc.values(), default=0)},
        "source_position_distribution": {"front_0_33": sum(p < 1/3 for p in positions), "middle_33_66": sum(1/3 <= p < 2/3 for p in positions), "back_66_100": sum(p >= 2/3 for p in positions)},
        "alignment": {"raw_prefix_events": scan_summary["raw_prefix_events"], "successes": scan_summary["alignment_successes"], "failures": scan_summary["alignment_failures"], "success_rate": scan_summary["alignment_successes"] / max(scan_summary["raw_prefix_events"], 1)},
        "checksums": checksums,
        "gates": {"PHASE1_DATASET_DESIGN_GATE": "PASS", "PHASE1_DATASET_V2_GATE": "PASS" if not any(overlap.values()) and all(v["schema_failures"] == 0 for v in ( _metrics(split_rows["train"]), _metrics(split_rows["validation"]), _metrics(split_rows["final_test"]) )) else "FAIL"},
    }
    _write_json(PRODUCTION_ROOT / "audit_report.json", audit)
    _write_json(PRODUCTION_ROOT / "SHA256SUMS.json", checksums)
    return audit


@app.function(image=image, cpu=8, timeout=12 * 60 * 60, volumes={"/dataset-v2": volume})
def pipeline() -> dict[str, Any]:
    os.chdir(REPO_ROOT)
    docs = _freeze_sources()
    volume.commit()
    docs = _load_frozen_docs()
    preflight = _run_preflight(docs)
    volume.commit()
    audit = _run_production(docs, preflight)
    volume.commit()
    return {"preflight_gate": preflight["gate"], "dataset_gate": audit["gates"]["PHASE1_DATASET_V2_GATE"], "rows": audit["total_rows"], "workers": audit["generation"]["workers"]}


@app.function(image=image, cpu=2, timeout=10 * 60)
def converter_probe() -> dict[str, Any]:
    """Small remote proof that the copied converter/runfiles are usable."""
    os.chdir(REPO_ROOT)
    segments, attempts, failures, latency = _query_retry("とうきょう", 30)
    _dispose_converter()
    return {"segments": len(segments), "last_reading": segments[-1]["reading"], "candidates": len(segments[-1]["candidates"]), "attempts": attempts, "attempt_failures": failures, "latency_ms": latency, "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.lower().startswith("model name")), platform.processor())}


@app.function(image=image, cpu=2, timeout=10 * 60)
def converter_debug_probe() -> dict[str, Any]:
    """Temporary diagnostics for the immutable binary/runfiles mount."""
    result = subprocess.run([CONVERTER_PATH, "--engine_name=oss", "--max_conversion_candidates_size=1"], cwd=RUNFILES_ROOT, input="startconversion とうきょう\n", text=True, capture_output=True, timeout=30)
    data_path = Path(RUNFILES_ROOT) / "data_manager/oss/mozc.data"
    payload = {"returncode": result.returncode, "stdout": result.stdout[-2000:], "stderr": result.stderr[-4000:], "binary": {"exists": Path(CONVERTER_PATH).exists(), "mode": oct(Path(CONVERTER_PATH).stat().st_mode)}, "data": {"exists": data_path.exists(), "size": data_path.stat().st_size if data_path.exists() else 0}}
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return json.dumps(payload, ensure_ascii=False)


@app.function(image=image, cpu=2, timeout=10 * 60)
def source_probe() -> str:
    """Confirm remote access to the exact dump-derived source config."""
    from datasets import load_dataset

    row = next(iter(load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)))
    payload = {"source_id": _source_id(row), "title": str(row.get("title") or "")[:120], "text_chars": len(str(row.get("text") or "")), "fields": sorted(row)}
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return json.dumps(payload, ensure_ascii=False)


@app.function(image=image, cpu=2, timeout=10 * 60)
def extraction_smoke() -> str:
    """Exercise SplitMode-A prefix scan, retry, and top-30 enrichment remotely."""
    os.chdir(REPO_ROOT)
    doc = {"source_id": "smoke:1", "title": "smoke", "source_url": "", "text": "東京都で新しい電車を確認しました。昨日も同じ駅を利用しました。", "priority": "0"}
    scan = _scan_document(doc, 1, capture_alignment_keys=True)
    event = scan["selected_events"][0]
    enriched, failure = _enrich_event(event)
    retry = _retry_probe([scan])
    _dispose_converter()
    payload = {"events": scan["raw_prefix_events"], "alignment_successes": scan["alignment_successes"], "selected": len(scan["selected_events"]), "enriched": enriched is not None, "failure": failure, "retry": retry}
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return json.dumps(payload, ensure_ascii=False)


@app.local_entrypoint()
def main() -> None:
    call = pipeline.spawn()
    print(f"SPAWNED function_call_id={call.object_id} volume=mozc-v2-dataset-production", flush=True)


def local_main() -> None:
    """Run the frozen-source pipeline on an operator-owned CPU host."""
    if not _LOCAL_RUNNER:
        raise RuntimeError("local_main requires MOZC_V2_LOCAL_RUNNER=1")
    os.chdir(Path(__file__).resolve().parents[1])
    docs = _load_frozen_docs()
    preflight = _run_preflight(docs)
    audit = _run_production(docs, preflight)
    print(json.dumps({"preflight_gate": preflight["gate"], "dataset_gate": audit["gates"]["PHASE1_DATASET_V2_GATE"], "rows": audit["total_rows"], "workers": audit["generation"]["workers"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__" and _LOCAL_RUNNER:
    local_main()
