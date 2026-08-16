from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .aozora import (
    INDEX_URL,
    download_index,
    public_domain_works,
    ruby_records,
    work_records,
)
from .classify import classify
from .deepseek_review import ReviewBudget, comparison_key, execute_with_retries
from .export_train import export_train_jsonl
from .japanpost import DEFAULT_URL, download, records_from_zip
from .jsonl import append_jsonl, read_jsonl, write_jsonl
from .mozc_batch import merge as merge_candidates
from .mozc_batch import parse_candidates_tsv, readings_from_records
from .mozc_batch import resolve_batch_config, run_mozc_batch
from .records import TermRecord
from .wikidata import DEFAULT_ENDPOINT, fetch_places


def command_export_train(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.reviews]
    count = export_train_jsonl(
        paths,
        Path(args.out),
        top_k=args.top_k,
        limit=args.limit,
    )
    print(f"wrote {count} training examples to {args.out}")
    return 0


def command_japanpost(args: argparse.Namespace) -> int:
    archive = Path(args.archive)
    if args.download:
        download(args.url, archive)
    records = records_from_zip(archive, args.url)
    count = write_jsonl(Path(args.out), (record.to_dict() for record in records))
    print(f"wrote {count} records to {args.out}")
    return 0


def command_aozora(args: argparse.Namespace) -> int:
    text = Path(args.input).read_text(encoding="utf-8")
    records = ruby_records(text, args.source_id, args.source_url, args.source_version)
    count = write_jsonl(Path(args.out), (record.to_dict() for record in records))
    print(f"wrote {count} records to {args.out}")
    return 0


def command_aozora_index(args: argparse.Namespace) -> int:
    works = public_domain_works(download_index(args.url))
    if args.author:
        needle = args.author
        works = [work for work in works if needle in work["author"]]
    count = write_jsonl(Path(args.out), works)
    print(f"wrote {count} public-domain works to {args.out}")
    return 0


def command_aozora_fetch(args: argparse.Namespace) -> int:
    catalog = {work["work_id"]: work for work in read_jsonl(Path(args.index))}
    if args.work_ids:
        selected = [catalog[work_id] for work_id in args.work_ids if work_id in catalog]
        missing = [work_id for work_id in args.work_ids if work_id not in catalog]
        if missing:
            print(f"warning: work ids not in index: {', '.join(missing)}")
    else:
        selected = list(catalog.values())[: args.limit]
    records = []
    for work in selected:
        work_terms = work_records(work, capture_context=not args.no_context)
        records.extend(record.to_dict() for record in work_terms)
        print(f"  {work['work_id']} {work['author']}『{work['title']}』: {len(work_terms)} ruby terms")
    count = write_jsonl(Path(args.out), records)
    print(f"wrote {count} records from {len(selected)} works to {args.out}")
    return 0


def command_wikidata_places(args: argparse.Namespace) -> int:
    records = fetch_places(
        endpoint=args.endpoint,
        page_size=args.page_size,
        max_pages=args.max_pages,
    )
    count = write_jsonl(Path(args.out), (record.to_dict() for record in records))
    print(f"wrote {count} unverified place/facility records to {args.out}")
    return 0


def command_mozc_keys(args: argparse.Namespace) -> int:
    keys = readings_from_records(read_jsonl(Path(args.input)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(keys) + "\n", encoding="utf-8")
    print(f"wrote {len(keys)} unique readings to {args.out}")
    return 0


def command_mozc_merge(args: argparse.Namespace) -> int:
    with Path(args.candidates).open(encoding="utf-8") as source:
        key_to_candidates = parse_candidates_tsv(source)
    rows = merge_candidates(read_jsonl(Path(args.records)), key_to_candidates)
    count = write_jsonl(Path(args.out), rows)
    print(f"wrote {count} classify-ready rows to {args.out}")
    return 0


def command_mozc_run(args: argparse.Namespace) -> int:
    """Extract keys → C++ mozc_batch → merge into classify-ready JSONL."""
    records_path = Path(args.records)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    keys_path = Path(args.keys) if args.keys else work_dir / "keys.txt"
    candidates_path = Path(args.candidates) if args.candidates else work_dir / "candidates.tsv"
    out_path = Path(args.out)

    exe, engine_data, max_candidates = resolve_batch_config(
        env_file=Path(args.env_file) if args.env_file else None,
        exe=args.exe,
        engine_data=args.engine_data,
        max_candidates=args.max_candidates,
    )
    if args.max_candidates is not None:
        max_candidates = args.max_candidates

    rows = list(read_jsonl(records_path))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    keys = readings_from_records(rows)
    keys_path.parent.mkdir(parents=True, exist_ok=True)
    keys_path.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")
    print(f"wrote {len(keys)} unique readings to {keys_path}")

    print(
        f"running mozc_batch exe={exe} engine_data={engine_data} "
        f"max_candidates={max_candidates} keys={len(keys)}"
    )
    run_mozc_batch(exe, engine_data, keys_path, candidates_path, max_candidates)
    print(f"wrote candidates to {candidates_path}")

    with candidates_path.open(encoding="utf-8") as source:
        key_to_candidates = parse_candidates_tsv(source)
    merged = merge_candidates(rows, key_to_candidates)
    count = write_jsonl(out_path, merged)
    print(f"wrote {count} classify-ready rows to {out_path}")
    return 0


def command_classify(args: argparse.Namespace) -> int:
    rows = []
    for row in read_jsonl(Path(args.input)):
        record = TermRecord.from_dict(row["record"] if "record" in row else row)
        result = classify(record, row.get("candidates", []), row.get("context", []), args.top_k, args.max_rank)
        rows.append(result.to_dict())
    count = write_jsonl(Path(args.out), rows)
    print(f"wrote {count} comparisons to {args.out}")
    return 0


def command_review(args: argparse.Namespace) -> int:
    rows = list(read_jsonl(Path(args.input)))
    selected = [row for row in rows if row.get("action") in set(args.actions)]
    out_path = Path(args.out)
    done_keys: set[str] = set()
    spent = 0.0
    if out_path.exists():
        for existing in read_jsonl(out_path):
            comparison = existing.get("comparison") or {}
            done_keys.add(comparison_key(comparison))
            spent = float(existing.get("estimated_cost_usd", spent))

    pending = [row for row in selected if comparison_key(row) not in done_keys]
    # Deduplicate identical comparison keys within the input itself.
    seen_pending: set[str] = set()
    unique_pending: list[dict] = []
    for row in pending:
        key = comparison_key(row)
        if key in seen_pending:
            continue
        seen_pending.add(key)
        unique_pending.append(row)
    pending = unique_pending

    limit = args.limit if args.limit and args.limit > 0 else None
    batch = pending[:limit] if limit is not None else pending
    workers = max(1, int(args.workers))

    if not args.execute:
        print(
            f"dry run: selected={len(selected)} done={len(done_keys)} "
            f"pending={len(pending)} batch={len(batch)} workers={workers}; "
            f"no API request sent"
        )
        if spent:
            print(f"resume spend so far: ${spent:.6f}")
        return 0

    if done_keys:
        print(f"resume: {len(done_keys)} reviews already in {out_path}; spent=${spent:.6f}")
    print(
        f"batch: {len(batch)} of {len(pending)} pending "
        f"(limit={limit if limit is not None else 'none'} workers={workers})"
    )

    budget = ReviewBudget(
        args.max_cost_usd,
        args.input_price_per_million,
        args.output_price_per_million,
        spent_usd=spent,
    )
    written = 0
    index = 0
    wave_size = workers if workers > 1 else 1

    while index < len(batch):
        remaining = len(batch) - index
        # Shrink the wave until the reserved-token estimate fits the budget.
        take = min(wave_size, remaining)
        while take > 0 and not budget.can_afford(
            args.reserve_input_tokens * take,
            args.reserve_output_tokens * take,
        ):
            take -= 1
        if take <= 0:
            print(f"budget stop: {written} new reviews; spent=${budget.spent_usd:.6f}")
            break

        wave = batch[index : index + take]
        index += take

        try:
            if take == 1:
                results = [execute_with_retries(args.model, wave[0])]
            else:
                with ThreadPoolExecutor(max_workers=take) as pool:
                    futures = [pool.submit(execute_with_retries, args.model, row) for row in wave]
                    results = [future.result() for future in futures]
        except Exception as error:  # noqa: BLE001 - persist progress, fail the batch
            print(
                f"error after {written} new reviews (total done {len(done_keys) + written}): {error}"
            )
            print(f"resume with the same --out to continue; spent=${budget.spent_usd:.6f}")
            return 1

        for row, (verdict, usage) in zip(wave, results):
            budget = budget.charge(usage["input"], usage["output"])
            append_jsonl(
                out_path,
                {
                    "comparison": row,
                    "review": verdict,
                    "usage": usage,
                    "estimated_cost_usd": budget.spent_usd,
                },
            )
            written += 1
            done_so_far = len(done_keys) + written
            if written % 10 == 0 or written == len(batch):
                print(
                    f"progress {written}/{len(batch)} new "
                    f"(total {done_so_far}); spent=${budget.spent_usd:.6f}"
                )

    print(
        f"wrote {written} new reviews to {out_path}; "
        f"total={len(done_keys) + written}; estimated_spend=${budget.spent_usd:.6f}; "
        f"remaining_pending={max(0, len(pending) - written)}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mozc AI dataset pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    japanpost = subparsers.add_parser("japanpost")
    japanpost.add_argument("--archive", required=True)
    japanpost.add_argument("--out", required=True)
    japanpost.add_argument("--url", default=DEFAULT_URL)
    japanpost.add_argument("--download", action="store_true")
    japanpost.set_defaults(handler=command_japanpost)

    aozora = subparsers.add_parser("aozora-ruby")
    aozora.add_argument("--input", required=True)
    aozora.add_argument("--out", required=True)
    aozora.add_argument("--source-id", required=True)
    aozora.add_argument("--source-url", required=True)
    aozora.add_argument("--source-version", default="")
    aozora.set_defaults(handler=command_aozora)

    aozora_index = subparsers.add_parser("aozora-index")
    aozora_index.add_argument("--out", required=True)
    aozora_index.add_argument("--url", default=INDEX_URL)
    aozora_index.add_argument("--author", default="", help="keep only works whose author contains this string")
    aozora_index.set_defaults(handler=command_aozora_index)

    aozora_fetch = subparsers.add_parser("aozora-fetch")
    aozora_fetch.add_argument("--index", required=True, help="catalog JSONL from aozora-index")
    aozora_fetch.add_argument("--out", required=True)
    aozora_fetch.add_argument("--work-ids", nargs="+", default=[], help="specific work ids; default takes --limit works")
    aozora_fetch.add_argument("--limit", type=int, default=10)
    aozora_fetch.add_argument("--no-context", action="store_true", help="omit the containing sentence from metadata")
    aozora_fetch.set_defaults(handler=command_aozora_fetch)

    wikidata = subparsers.add_parser("wikidata-places")
    wikidata.add_argument("--out", required=True)
    wikidata.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="SPARQL endpoint; override if WQS is throttled")
    wikidata.add_argument("--page-size", type=int, default=10000)
    wikidata.add_argument("--max-pages", type=int, default=20)
    wikidata.set_defaults(handler=command_wikidata_places)

    mozc_keys = subparsers.add_parser("mozc-keys")
    mozc_keys.add_argument("--input", required=True, help="records JSONL (japanpost/aozora/wikidata)")
    mozc_keys.add_argument("--out", required=True, help="one reading per line for mozc_batch --input")
    mozc_keys.set_defaults(handler=command_mozc_keys)

    mozc_merge = subparsers.add_parser("mozc-merge")
    mozc_merge.add_argument("--records", required=True, help="the same records JSONL fed to mozc-keys")
    mozc_merge.add_argument("--candidates", required=True, help="TSV produced by mozc_batch")
    mozc_merge.add_argument("--out", required=True, help="classify-ready JSONL")
    mozc_merge.set_defaults(handler=command_mozc_merge)

    mozc_run = subparsers.add_parser(
        "mozc-run",
        help="keys + C++ mozc_batch + merge (uses config/mozc_batch.env)",
    )
    mozc_run.add_argument("--records", required=True, help="records JSONL (japanpost/aozora/wikidata)")
    mozc_run.add_argument("--out", required=True, help="classify-ready JSONL")
    mozc_run.add_argument(
        "--work-dir",
        default="data/interim/mozc_batch",
        help="directory for keys.txt and candidates.tsv",
    )
    mozc_run.add_argument("--keys", default="", help="optional keys.txt path override")
    mozc_run.add_argument("--candidates", default="", help="optional candidates.tsv path override")
    mozc_run.add_argument("--env-file", default="config/mozc_batch.env")
    mozc_run.add_argument("--exe", default="", help="override MOZC_BATCH_EXE")
    mozc_run.add_argument("--engine-data", default="", help="override MOZC_ENGINE_DATA_PATH")
    mozc_run.add_argument("--max-candidates", type=int, default=None)
    mozc_run.add_argument(
        "--limit",
        type=int,
        default=0,
        help="optional record cap before key extraction (0 = all)",
    )
    mozc_run.set_defaults(handler=command_mozc_run)

    classifier = subparsers.add_parser("classify")
    classifier.add_argument("--input", required=True)
    classifier.add_argument("--out", required=True)
    classifier.add_argument("--top-k", type=int, default=5)
    classifier.add_argument("--max-rank", type=int, default=50)
    classifier.set_defaults(handler=command_classify)

    review = subparsers.add_parser("deepseek-review")
    review.add_argument("--input", required=True)
    review.add_argument("--out", required=True)
    review.add_argument("--model", required=True)
    review.add_argument("--actions", nargs="+", default=["generation_gap"])
    review.add_argument("--max-cost-usd", type=float, default=10.0)
    review.add_argument("--input-price-per-million", type=float, required=True)
    review.add_argument("--output-price-per-million", type=float, required=True)
    review.add_argument("--reserve-input-tokens", type=int, default=800)
    review.add_argument("--reserve-output-tokens", type=int, default=3000)
    review.add_argument(
        "--limit",
        type=int,
        default=0,
        help="max NEW reviews this run (0 = all pending); use for chunked batches",
    )
    review.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel DeepSeek HTTP requests (wave size); 1 = sequential",
    )
    review.add_argument("--execute", action="store_true")
    review.set_defaults(handler=command_review)

    export_train = subparsers.add_parser(
        "export-train",
        help="Build IME LoRA JSONL from accept review files",
    )
    export_train.add_argument(
        "--reviews",
        nargs="+",
        required=True,
        help="one or more review JSONL paths (aozora/wikidata/...)",
    )
    export_train.add_argument("--out", required=True)
    export_train.add_argument("--top-k", type=int, default=5)
    export_train.add_argument("--limit", type=int, default=0)
    export_train.set_defaults(handler=command_export_train)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
