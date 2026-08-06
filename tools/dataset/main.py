from __future__ import annotations

import argparse
from pathlib import Path

from .aozora import ruby_records
from .classify import classify
from .deepseek_review import ReviewBudget, execute
from .japanpost import DEFAULT_URL, download, records_from_zip
from .jsonl import read_jsonl, write_jsonl
from .records import TermRecord


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
    if not args.execute:
        print(f"dry run: {len(selected)} records selected; no API request sent")
        return 0
    budget = ReviewBudget(args.max_cost_usd, args.input_price_per_million, args.output_price_per_million)
    output = []
    for row in selected:
        if not budget.can_afford(args.reserve_input_tokens, args.reserve_output_tokens):
            print(f"budget stop: {len(output)} records reviewed; spent=${budget.spent_usd:.6f}")
            break
        verdict, usage = execute(args.model, row)
        budget = budget.charge(usage["input"], usage["output"])
        output.append({"comparison": row, "review": verdict, "usage": usage, "estimated_cost_usd": budget.spent_usd})
    count = write_jsonl(Path(args.out), output)
    print(f"wrote {count} reviews to {args.out}; estimated_spend=${budget.spent_usd:.6f}")
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
    review.add_argument("--reserve-output-tokens", type=int, default=150)
    review.add_argument("--execute", action="store_true")
    review.set_defaults(handler=command_review)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
