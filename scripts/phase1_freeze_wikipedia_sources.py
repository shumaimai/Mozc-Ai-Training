#!/usr/bin/env python3
"""Freeze globally sampled, dump-derived Wikipedia source shards.

This runs as a subprocess in the Modal production app.  ``datasets``/PyArrow
can leave background native state during interpreter finalization in some
serverless images, so the process commits files, flushes stdio, and terminates
with ``os._exit`` after a successful freeze.  The parent validates the manifest
and shard checksums before any Dataset v2 extraction begins.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_gz_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="wb", compresslevel=6, mtime=0) as out:
        out.write((json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--documents", type=int, default=1200)
    args = ap.parse_args()
    manifest = args.out_dir / "selected_manifest.json"
    if manifest.exists():
        print(json.dumps({"status": "already_frozen", "manifest": str(manifest)}), flush=True)
        return 0
    from datasets import load_dataset

    started = time.monotonic()
    heap: list[tuple[int, str, dict[str, str]]] = []
    scanned = usable = 0
    stream = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
    for row in stream:
        scanned += 1
        source_id = f"wikipedia-ja:{row.get('id', '')}"
        text = str(row.get("text") or "").strip()
        if source_id != "wikipedia-ja:" and len(text) >= 100 and text.count("。") >= 2:
            usable += 1
            priority = sha(f"{args.seed}|{source_id}")
            doc = {"source_id": source_id, "title": str(row.get("title") or ""), "source_url": str(row.get("url") or ""), "text": text, "priority": priority}
            item = (-int(priority, 16), source_id, doc)
            if len(heap) < args.documents:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        if scanned % 100_000 == 0:
            print(f"source-freeze scanned={scanned} usable={usable} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    if len(heap) != args.documents:
        raise RuntimeError(f"only selected {len(heap)}/{args.documents} documents")
    docs = sorted((item[2] for item in heap), key=lambda d: (d["priority"], d["source_id"]))
    checksums = {}
    for doc in docs:
        name = hashlib.sha256(doc["source_id"].encode("utf-8")).hexdigest()[:24] + ".json.gz"
        checksums[name] = write_gz_json(args.out_dir / "shards" / name, doc)
    meta = {"source": "wikimedia/wikipedia", "config": "20231101.ja", "split": "train", "license": "CC BY-SA 4.0", "retrieved_at": datetime.now(UTC).isoformat(), "selection": "global bottom-k SHA256(seed|source_id) over full streaming target universe", "seed": args.seed, "requested_documents": args.documents, "streamed_documents": scanned, "usable_documents": usable, "source_text_truncation": "none", "source_shard_checksums": checksums}
    write_json(args.out_dir / "retrieval_config.json", meta)
    write_json(args.out_dir / "selected_manifest.json", {"metadata": meta, "documents": [{k: d[k] for k in ("source_id", "title", "source_url", "priority")} for d in docs]})
    print(json.dumps({"status": "frozen", "documents": len(docs), "scanned": scanned, "usable": usable, "elapsed_seconds": time.monotonic()-started}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException as exc:
        print(f"source-freeze-error:{type(exc).__name__}:{exc}", file=sys.stderr, flush=True)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
