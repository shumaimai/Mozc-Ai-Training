#!/usr/bin/env python3
"""Fetch a deterministic public Japanese Wikipedia document manifest.

This only stages public source documents for the Phase 1 pilot; it does not
create Dataset v2 rows or start training.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

API = "https://ja.wikipedia.org/w/api.php"
RANDOM_SUMMARY_API = "https://ja.wikipedia.org/api/rest_v1/page/random/summary"
UA = "Mozc-Ai-Training/phase1-pilot (research; contact via GitHub shumaimai/Mozc-Ai-Training)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--documents", type=int, default=100)
    ap.add_argument("--scan", type=int, default=500)
    ap.add_argument("--source", choices=("hf", "api"), default="hf")
    args = ap.parse_args()
    if args.source == "hf":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("HF mode requires: pip install datasets") from exc
        docs: list[dict[str, str]] = []
        dataset = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
        for page in dataset:
            text = str(page.get("text", "")).strip()
            if len(text) < 100 or text.count("。") < 2:
                continue
            page_id = str(page.get("id", ""))
            docs.append({"source_id": f"wikipedia-ja:{page_id}", "title": str(page.get("title", "")), "source_url": str(page.get("url", "")), "text": text})
            if len(docs) % 10 == 0:
                print(f"documents={len(docs)}/{args.documents}", flush=True)
            if len(docs) >= args.documents:
                break
        if len(docs) < args.documents:
            raise RuntimeError(f"only fetched {len(docs)} documents from HF stream")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in docs), encoding="utf-8")
        meta = {"source": "wikimedia/wikipedia", "config": "20231101.ja", "split": "train", "license": "CC BY-SA 4.0", "retrieved_at": datetime.now(UTC).isoformat(), "documents": len(docs), "selection": "deterministic streaming prefix filtered by text length and sentence count"}
        args.out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return 0
    session = requests.Session()
    session.headers["User-Agent"] = UA
    docs: list[dict[str, str]] = []
    seen: set[str] = set()
    attempts = 0
    while len(docs) < args.documents and attempts < args.scan:
        attempts += 1
        for retry in range(6):
            result = session.get(RANDOM_SUMMARY_API, timeout=60)
            if result.status_code != 429:
                break
            time.sleep(min(60, 2 ** retry))
        result.raise_for_status()
        page = result.json()
        page_id = str(page.get("pageid", ""))
        text = str(page.get("extract", ""))
        if page_id and page_id not in seen and len(text) >= 100:
            seen.add(page_id)
            docs.append({
                "source_id": f"wikipedia-ja:{page_id}",
                "title": str(page.get("title", "")),
                "source_url": f"https://ja.wikipedia.org/?curid={page_id}",
                "text": text,
            })
            if len(docs) % 10 == 0:
                print(f"documents={len(docs)}/{args.documents} attempts={attempts}", flush=True)
        time.sleep(0.2)
    if len(docs) < args.documents:
        raise RuntimeError(f"only fetched {len(docs)} documents; increase --scan")
    docs.sort(key=lambda row: row["source_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in docs), encoding="utf-8")
    meta = {"source": "Japanese Wikipedia API", "license": "CC BY-SA 4.0", "retrieved_at": datetime.now(UTC).isoformat(), "api": API, "user_agent": UA, "documents": len(docs)}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
