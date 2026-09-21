#!/usr/bin/env python3
"""CPU-parallel, resumable Phase 1 Dataset v2 pilot.

This is intentionally a pilot generator. It fetches only explicitly selected
public-domain Aozora works, keeps one persistent Mozc converter process per
Python worker thread, writes resumable shard outputs, and stops after the
requested document count. It does not train a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset.aozora import (
    EXPLICIT_RUBY,
    IMPLICIT_RUBY,
    download_index,
    download_work_text,
    public_domain_works,
)
from tools.dataset.normalize import normalize_reading, normalize_surface
from tools.rerank.context_clip import clean_context
from tools.rerank.contextual_ranking_v2_schema import (
    FORMAT_VERSION,
    SCHEMA_VERSION,
    validate_record,
)
try:
    from sudachipy import dictionary as sudachi_dictionary
except ImportError:  # Aozora mode does not require SudachiPy.
    sudachi_dictionary = None

SEGMENT_RE = re.compile(r"^-+ Segment (\d+)/(\d+) \[(.*?)\] -+")
CANDIDATE_RE = re.compile(r"^\s+(\d+)/(\d+) (.*)$")
INT_RE = re.compile(r"\((-?\d+)\)")
_thread_state = threading.local()
_progress_lock = threading.Lock()


def _protection(attrs: str, category: str, surface: str) -> str:
    hard = {"USER_SEGMENT_HISTORY_REWRITER", "RERANKED", "NUMBER", "NO_MODIFICATION", "NO_DELETABLE"}
    if category == "SYMBOL" or any(x in attrs for x in hard):
        return "HARD_PROTECT"
    if "USER_DICTIONARY" in attrs or "CONTEXT_SENSITIVE" in attrs:
        return "DELTA_ONLY"
    if surface and all(not ch.isalnum() and ch.isspace() is False for ch in surface):
        return "HARD_PROTECT"
    return "NORMAL"


def _parse_converter(lines: list[str], reading: str, top_k: int) -> list[dict[str, Any]]:
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    last: dict[str, Any] | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        segment_match = SEGMENT_RE.match(line)
        if segment_match:
            current = []
            segments.append(current)
            last = None
            continue
        candidate_match = CANDIDATE_RE.match(line)
        if candidate_match and current is not None:
            rank = int(candidate_match.group(1))
            last = {
                "surface": candidate_match.group(3),
                "rank": rank,
                "cost": 0,
                "cost_delta": 0,
                "lid": 0,
                "rid": 0,
                "attributes": 0,
                "category": "DEFAULT",
                "converted_segment_count": 1,
                "protection": "NORMAL",
            }
            current.append(last)
            continue
        if last is None:
            continue
        if line.strip().startswith("cost:"):
            values = re.findall(r"-?\d+", line)
            if values:
                last["cost"] = int(values[0])
        elif line.strip().startswith("lid:"):
            match = INT_RE.search(line)
            if match:
                last["lid"] = int(match.group(1))
        elif line.strip().startswith("rid:"):
            match = INT_RE.search(line)
            if match:
                last["rid"] = int(match.group(1))
        elif line.strip().startswith("attr:"):
            attrs = line.split(":", 1)[1].strip()
            last["attributes_text"] = attrs
            bits = {
                "RERANKED": 1 << 1,
                "CONTEXT_SENSITIVE": 1 << 4,
                "USER_DICTIONARY": 1 << 9,
                "NO_MODIFICATION": 1 << 16,
                "USER_SEGMENT_HISTORY_REWRITER": 1 << 17,
                "NUMBER": 1 << 23,
                "NO_DELETABLE": 1 << 19,
            }
            last["attributes"] = sum(bit for name, bit in bits.items() if name in attrs)
        elif line.strip().startswith("category:"):
            last["category"] = line.split(":", 1)[1].strip()
        elif line.strip().startswith("converted_segment_count:"):
            last["converted_segment_count"] = int(line.rsplit(" ", 1)[1])
    if not segments:
        raise RuntimeError(f"converter returned no segments for {reading!r}")
    candidates = segments[-1][:top_k]
    if not candidates:
        raise RuntimeError(f"converter returned no candidates for {reading!r}")
    base_cost = candidates[0]["cost"]
    for candidate in candidates:
        candidate["cost_delta"] = candidate["cost"] - base_cost
        candidate["protection"] = _protection(
            candidate.pop("attributes_text", ""), candidate["category"], candidate["surface"]
        )
    return candidates


def _converter(exe: str, cwd: str, max_candidates: int) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [exe, "--engine_name=oss", f"--max_conversion_candidates_size={max_candidates}"],
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        errors="replace",
        bufsize=1,
    )
    return proc


def query_mozc(reading: str, exe: str, cwd: str, max_candidates: int) -> list[dict[str, Any]]:
    proc = getattr(_thread_state, "proc", None)
    if proc is None or proc.poll() is not None:
        proc = _converter(exe, cwd, max_candidates)
        _thread_state.proc = proc
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(f"startconversion {reading}\n")
    proc.stdin.flush()
    lines: list[str] = []
    while True:
        line = proc.stdout.readline()
        if line == "":
            raise RuntimeError(f"converter exited with {proc.poll()} for {reading!r}")
        if line == "\n" and lines:
            break
        lines.append(line)
    return _parse_converter(lines, reading, max_candidates)


def extract_document(work: dict[str, str], max_examples: int) -> list[dict[str, Any]]:
    text = work.get("text") or download_work_text(work["text_url"], work.get("encoding", "ShiftJIS"))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    if "text" in work and work.get("source_id", "").startswith("wikipedia-ja:"):
        if sudachi_dictionary is None:
            raise RuntimeError("Wikipedia input requires SudachiPy")
        tokenizer = sudachi_dictionary.Dictionary().create()
        offset = 0
        for sentence in re.split(r"(?<=[。！？!?])|\n+", text):
            if not sentence.strip():
                offset += len(sentence)
                continue
            for morpheme in tokenizer.tokenize(sentence):
                surface = normalize_surface(morpheme.surface())
                reading = normalize_reading(morpheme.reading_form() or "")
                if not surface or not reading or reading == "*" or not any("\u3040" <= c <= "\u30ff" for c in reading):
                    continue
                prefix = sentence[: morpheme.begin()]
                context = clean_context(prefix)
                key = (reading, surface, context)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"source_id": work["source_id"], "reading": reading, "context_prev": context, "gold": surface, "target_segment_index": 0, "_source_title": work.get("title", ""), "_source_url": work.get("source_url", "")})
                if len(rows) >= max_examples:
                    return rows
            offset += len(sentence)
        return rows
    for line in text.splitlines():
        for pattern in (EXPLICIT_RUBY, IMPLICIT_RUBY):
            for match in pattern.finditer(line):
                surface = normalize_surface(match.group("surface"))
                reading = normalize_reading(match.group("reading"))
                context = clean_context(line[: match.start()])
                if not surface or not reading:
                    continue
                key = (reading, surface, context)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "source_id": f"aozora:{work['work_id']}",
                    "reading": reading,
                    "context_prev": context,
                    "gold": surface,
                    "target_segment_index": 0,
                    "_source_title": work.get("title", ""),
                    "_source_url": work["text_url"],
                })
                if len(rows) >= max_examples:
                    return rows
    return rows


def process_shard(shard_path: Path, output_path: Path, exe: str, cwd: str, top_k: int) -> dict[str, Any]:
    if output_path.with_suffix(".done.json").exists():
        return json.loads(output_path.with_suffix(".done.json").read_text(encoding="utf-8"))
    docs = [json.loads(line) for line in shard_path.read_text(encoding="utf-8").splitlines() if line]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    retryable_failures: list[dict[str, str]] = []
    for doc in docs:
        for example in doc["examples"]:
            try:
                candidates = query_mozc(example["reading"], exe, cwd, top_k)
                record = {k: example[k] for k in ("source_id", "reading", "context_prev", "gold", "target_segment_index")}
                record.update({"schema_version": SCHEMA_VERSION, "format_version": FORMAT_VERSION, "candidates": candidates})
                errors = validate_record(record)
                if errors:
                    failures.append({"source_id": example["source_id"], "reading": example["reading"], "error": ";".join(errors)})
                else:
                    rows.append(record)
            except Exception as exc:  # one bad reading must not lose a shard
                failure = {"source_id": example["source_id"], "reading": example["reading"], "error": type(exc).__name__ + ":" + str(exc)}
                failures.append(failure)
                retryable_failures.append(failure)
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(output_path)
    state = {"shard": shard_path.stem, "docs": len(docs), "rows": len(rows), "failures": failures, "retryable_failures": retryable_failures}
    if not retryable_failures:
        output_path.with_suffix(".done.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        output_path.with_suffix(".failed.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def split_name(source_id: str, seed: int) -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{source_id}".encode()).digest()[:4], "big") % 100
    return "train" if value < 80 else "validation" if value < 90 else "test"


def stats(rows: list[dict[str, Any]], docs: int, elapsed: float, workers: int, failures: list[dict[str, str]]) -> dict[str, Any]:
    per_doc: dict[str, int] = {}
    context_lengths: list[int] = []
    candidate_counts: list[int] = []
    top1 = top5 = 0
    for row in rows:
        per_doc[row["source_id"]] = per_doc.get(row["source_id"], 0) + 1
        context_lengths.append(len(row["context_prev"]))
        candidate_counts.append(len(row["candidates"]))
        surfaces = [c["surface"] for c in row["candidates"]]
        top1 += surfaces[0] == row["gold"]
        top5 += row["gold"] in surfaces[:5]
    values = list(per_doc.values())
    return {
        "rows": len(rows), "docs": docs, "workers": workers,
        "elapsed_sec": elapsed, "rows_per_sec": len(rows) / elapsed if elapsed else 0,
        "docs_per_sec": docs / elapsed if elapsed else 0, "eta_sec": 0,
        "mozc_top1_accuracy": top1 / len(rows) if rows else 0,
        "top5_oracle_coverage": top5 / len(rows) if rows else 0,
        "per_document_examples": {"min": min(values) if values else 0, "median": median(values) if values else 0, "max": max(values) if values else 0, "mean": mean(values) if values else 0},
        "context_length": {"min": min(context_lengths) if context_lengths else 0, "median": median(context_lengths) if context_lengths else 0, "max": max(context_lengths) if context_lengths else 0},
        "candidate_count": {"min": min(candidate_counts) if candidate_counts else 0, "median": median(candidate_counts) if candidate_counts else 0, "max": max(candidate_counts) if candidate_counts else 0},
        "schema_validation_failures": len(failures), "failure_examples": failures[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--converter-exe", required=True)
    parser.add_argument("--mozc-cwd", required=True)
    parser.add_argument("--documents", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0, help="0 means os.cpu_count()")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-examples-per-document", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--documents-input", type=Path, help="JSONL source documents prepared by fetch_wikipedia_documents.py")
    args = parser.parse_args()
    workers = max(1, min(args.workers or (os.cpu_count() or 1), args.documents))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.work_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    manifest_path = args.work_dir / "source_manifest.json"
    if manifest_path.exists():
        selected = json.loads(manifest_path.read_text(encoding="utf-8"))["documents"]
    elif args.documents_input:
        source_docs = [json.loads(line) for line in args.documents_input.read_text(encoding="utf-8").splitlines() if line]
        selected = []
        for index, work in enumerate(source_docs[: args.documents]):
            examples = extract_document(work, args.max_examples_per_document)
            selected.append({"doc_index": index, "source_id": work["source_id"], "title": work.get("title", ""), "url": work.get("source_url", ""), "examples": examples})
            print(f"source {index + 1}/{min(len(source_docs), args.documents)} docs={index + 1} rows={sum(len(x['examples']) for x in selected)}", flush=True)
        manifest_path.write_text(json.dumps({"source": "public documents supplied through --documents-input", "selection": "input order", "documents": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        works = sorted(public_domain_works(download_index()), key=lambda x: x["work_id"])
        selected_works = works[: args.documents]
        selected = []
        for index, work in enumerate(selected_works):
            try:
                examples = extract_document(work, args.max_examples_per_document)
                selected.append({"doc_index": index, "work_id": work["work_id"], "title": work["title"], "url": work["text_url"], "examples": examples})
                print(f"source {index + 1}/{len(selected_works)} docs={index + 1} rows={sum(len(x['examples']) for x in selected)}", flush=True)
            except Exception as exc:
                print(f"source_failed {work.get('work_id')}: {type(exc).__name__}: {exc}", flush=True)
        manifest_path.write_text(json.dumps({"source": "Aozora public-domain works", "license": "per_work_public_domain_or_explicit_license", "documents": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    shard_count = min(max(workers * 2, 1), max(len(selected), 1))
    shard_docs: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for index, doc in enumerate(selected):
        shard_docs[index % shard_count].append(doc)
    shards: list[tuple[Path, Path]] = []
    for index, docs in enumerate(shard_docs):
        input_path = shard_dir / f"shard-{index:04d}.input.jsonl"
        input_path.write_text("".join(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n" for doc in docs), encoding="utf-8")
        shards.append((input_path, shard_dir / f"shard-{index:04d}.jsonl"))
    started = time.monotonic()
    completed_docs = 0
    completed_rows = 0
    states: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mozc-worker") as pool:
        futures = {pool.submit(process_shard, source, output, args.converter_exe, args.mozc_cwd, args.top_k): (source, output) for source, output in shards}
        for future in as_completed(futures):
            state = future.result()
            states.append(state)
            completed_docs += state["docs"]
            completed_rows += state["rows"]
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = completed_docs / elapsed
            eta = (len(selected) - completed_docs) / rate if rate else 0
            print(f"progress docs={completed_docs}/{len(selected)} rows={completed_rows} rows/sec={completed_rows / elapsed:.2f} docs/sec={rate:.2f} ETA={eta:.1f}s", flush=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for _, output in sorted(shards, key=lambda x: x[0].name):
        rows.extend(json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line)
        state_path = output.with_suffix(".done.json")
        if not state_path.exists():
            state_path = output.with_suffix(".failed.json")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        failures.extend(state["failures"])
    rows.sort(key=lambda r: (r["source_id"], r["reading"], r["context_prev"], r["gold"]))
    (args.work_dir / "all.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    split_rows = {"train": [], "validation": [], "test": []}
    for row in rows:
        split_rows[split_name(row["source_id"], args.seed)].append(row)
    for name, split in split_rows.items():
        (args.work_dir / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in split), encoding="utf-8")
    elapsed = max(time.monotonic() - started, 1e-9)
    cpu_model = platform.processor()
    if not cpu_model:
        try:
            cpu_model = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.lower().startswith("model name"))
        except (OSError, StopIteration):
            cpu_model = "unknown"
    report = {"phase": "1-pilot", "status": "complete", "dataset_v2_generated": True, "model_training_started": False, "schema_version": SCHEMA_VERSION, "format_version": FORMAT_VERSION, "seed": args.seed, "split": {name: {"rows": len(split), "docs": len({r['source_id'] for r in split})} for name, split in split_rows.items()}, "metrics": stats(rows, len(selected), elapsed, workers, failures), "shards": {"count": len(shards), "completed": len(states), "failed": sum(1 for s in states if s.get("retryable_failures")), "resume": True}, "provenance": {"source": "public documents supplied through --documents-input" if args.documents_input else "Aozora public-domain works", "selection": "input order, then deterministic source_id merge/split" if args.documents_input else "sorted work_id prefix", "mozc_executable": args.converter_exe, "mozc_cwd": args.mozc_cwd, "cpu_model": cpu_model, "logical_cpu_count": os.cpu_count(), "worker_policy": "min(logical_cpu_count, documents)"}}
    (args.work_dir / "pilot_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
