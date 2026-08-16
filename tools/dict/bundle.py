"""Build a lean Mozc-diff dictionary bundle for IME candidate inject.

Pipeline:
  1. Load reading→surface rows from interim classify_in JSONL (preferred;
     already has Mozc N-best) or raw interim term JSONL (no Mozc-diff).
  2. Mozc-diff: drop surfaces already present in Mozc N-best for that reading.
  3. Optionally exclude holdout golds / (reading,gold) pairs (anti-leak).
  4. Emit onboardable artifacts under --out-dir.

Artifact format (lightweight; no FST framework):
  - mozc_diff_entries.jsonl
      one object per kept (reading, surface):
      {reading, surface, source, category, license_id}
  - reading_map.json
      {reading: [surface, ...]} — direct lookup for inject loaders
  - build_report.json
      before/after counts, sources, licenses, file sizes

Example:
  python -m tools.dict.bundle build \\
    --classify data/interim/mozc_batch/wikidata/classify_in.jsonl \\
               data/interim/mozc_batch/japanpost/classify_in.jsonl \\
    --exclude-holdout data/rerank_v3/holdout.jsonl \\
    --out-dir artifacts/dict/holdout_safe

  python -m tools.dict.bundle build \\
    --classify data/interim/mozc_batch/japanpost/classify_in.jsonl \\
    --out-dir artifacts/dict/japanpost_only
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from tools.dataset.jsonl import read_jsonl, write_jsonl


def normalize_surface(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    s = s.strip()
    s = s.replace("\u3000", "").replace(" ", "")
    s = s.replace("・", "").replace("･", "")
    s = re.sub(r"\s+", "", s)
    return s


def _provenance(rec: dict[str, Any]) -> dict[str, Any]:
    return rec.get("provenance") or {}


def _source_id(rec: dict[str, Any], fallback: str = "") -> str:
    return str(_provenance(rec).get("source_id") or fallback or "")


def _license_id(rec: dict[str, Any]) -> str:
    return str(_provenance(rec).get("license_id") or "")


def iter_classify_pairs(
    path: Path,
) -> Iterable[dict[str, Any]]:
    """Yield normalized candidate rows from classify_in JSONL."""
    for row in read_jsonl(path):
        rec = row.get("record") or row
        reading = (rec.get("reading") or "").strip()
        surface = (rec.get("surface") or "").strip()
        if not reading or not surface:
            continue
        cands = [c for c in (row.get("candidates") or []) if c]
        yield {
            "reading": reading,
            "surface": surface,
            "candidates": cands,
            "source": _source_id(rec, fallback=path.parent.name),
            "category": rec.get("category") or "",
            "license_id": _license_id(rec),
            "input_path": str(path),
        }


def iter_raw_term_pairs(path: Path) -> Iterable[dict[str, Any]]:
    """Yield rows from raw interim term JSONL (no Mozc candidates → no diff)."""
    for row in read_jsonl(path):
        reading = (row.get("reading") or "").strip()
        surface = (row.get("surface") or "").strip()
        if not reading or not surface:
            continue
        yield {
            "reading": reading,
            "surface": surface,
            "candidates": None,  # unknown; cannot Mozc-diff
            "source": _source_id(row, fallback=path.stem),
            "category": row.get("category") or "",
            "license_id": _license_id(row),
            "input_path": str(path),
        }


def load_holdout_exclusions(path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    """Return (gold_surfaces_norm, (reading, gold_norm) pairs) to exclude."""
    golds: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        reading = (row.get("reading") or "").strip()
        gold = normalize_surface(row.get("gold") or "")
        if gold:
            golds.add(gold)
        if reading and gold:
            pairs.add((reading, gold))
    return golds, pairs


def surface_in_mozc(surface: str, candidates: list[str] | None) -> bool | None:
    """True if Mozc already returns surface; None if candidates unknown."""
    if candidates is None:
        return None
    g = normalize_surface(surface)
    if not g:
        return True
    return any(normalize_surface(c) == g for c in candidates)


def build_mozc_diff(
    rows: Iterable[dict[str, Any]],
    *,
    exclude_gold_surfaces: set[str] | None = None,
    exclude_pairs: set[tuple[str, str]] | None = None,
    exclude_holdout_readings: bool = False,
    holdout_readings: set[str] | None = None,
    max_surfaces_per_reading: int = 0,
    require_mozc_candidates: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply Mozc-diff + anti-leak filters; return entries + report counters."""
    exclude_gold_surfaces = exclude_gold_surfaces or set()
    exclude_pairs = exclude_pairs or set()
    holdout_readings = holdout_readings or set()

    before = 0
    dropped_in_mozc = 0
    dropped_no_cands = 0
    dropped_holdout_gold = 0
    dropped_holdout_pair = 0
    dropped_holdout_reading = 0
    dropped_dup = 0
    source_before: Counter[str] = Counter()
    source_after: Counter[str] = Counter()
    license_after: Counter[str] = Counter()

    by_reading: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    entries: list[dict[str, Any]] = []

    for row in rows:
        before += 1
        src = row.get("source") or "unknown"
        source_before[src] += 1
        reading = row["reading"]
        surface = row["surface"]
        ns = normalize_surface(surface)
        if not ns:
            continue

        in_mozc = surface_in_mozc(surface, row.get("candidates"))
        if in_mozc is None:
            if require_mozc_candidates:
                dropped_no_cands += 1
                continue
        elif in_mozc:
            dropped_in_mozc += 1
            continue

        if exclude_holdout_readings and reading in holdout_readings:
            dropped_holdout_reading += 1
            continue
        if ns in exclude_gold_surfaces:
            dropped_holdout_gold += 1
            continue
        if (reading, ns) in exclude_pairs:
            dropped_holdout_pair += 1
            continue
        if ns in seen[reading]:
            dropped_dup += 1
            continue
        if max_surfaces_per_reading > 0 and len(by_reading[reading]) >= max_surfaces_per_reading:
            continue

        seen[reading].add(ns)
        by_reading[reading].append(surface)
        entry = {
            "reading": reading,
            "surface": surface,
            "source": src,
            "category": row.get("category") or "",
            "license_id": row.get("license_id") or "",
        }
        entries.append(entry)
        source_after[src] += 1
        license_after[entry["license_id"] or "unknown"] += 1

    report = {
        "rows_before": before,
        "entries_after": len(entries),
        "readings_after": len(by_reading),
        "dropped": {
            "already_in_mozc_nbest": dropped_in_mozc,
            "no_mozc_candidates": dropped_no_cands,
            "holdout_gold_surface": dropped_holdout_gold,
            "holdout_reading_gold_pair": dropped_holdout_pair,
            "holdout_reading": dropped_holdout_reading,
            "duplicate_surface_per_reading": dropped_dup,
        },
        "source_before": dict(source_before),
        "source_after": dict(source_after),
        "license_after": dict(license_after),
        "require_mozc_candidates": require_mozc_candidates,
        "max_surfaces_per_reading": max_surfaces_per_reading,
        "exclude_holdout_readings": exclude_holdout_readings,
        "exclude_gold_surface_count": len(exclude_gold_surfaces),
        "exclude_pair_count": len(exclude_pairs),
    }
    return entries, report


def entries_to_reading_map(entries: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_reading: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        r, s = e["reading"], e["surface"]
        ns = normalize_surface(s)
        if ns in seen[r]:
            continue
        seen[r].add(ns)
        by_reading[r].append(s)
    return dict(by_reading)


def load_reading_map_from_entries(path: Path) -> dict[str, list[str]]:
    """Loader API for eval / IME inject: entries JSONL → reading→surfaces."""
    return entries_to_reading_map(list(read_jsonl(path)))


def load_reading_map_json(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected object map")
    out: dict[str, list[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[str(k)] = [str(x) for x in v if x]
    return out


def write_bundle(
    out_dir: Path,
    entries: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    label: str = "",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries_path = out_dir / "mozc_diff_entries.jsonl"
    map_path = out_dir / "reading_map.json"
    report_path = out_dir / "build_report.json"

    n = write_jsonl(entries_path, entries)
    reading_map = entries_to_reading_map(entries)
    map_path.write_text(
        json.dumps(reading_map, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    def _sz(p: Path) -> int:
        return p.stat().st_size if p.is_file() else 0

    sizes = {
        "mozc_diff_entries_jsonl_bytes": _sz(entries_path),
        "reading_map_json_bytes": _sz(map_path),
        "total_bytes": _sz(entries_path) + _sz(map_path),
        "total_mb": round((_sz(entries_path) + _sz(map_path)) / (1024 * 1024), 3),
        "est_on_device_mb": round(_sz(map_path) / (1024 * 1024), 3),
        "note": "IME can ship reading_map.json alone; entries JSONL is audit/debug.",
    }
    full_report = {
        "label": label or out_dir.name,
        "format": {
            "entries": "jsonl reading/surface/source/category/license_id",
            "lookup": "reading_map.json: reading -> [surface,...]",
            "mozc_diff": "surface dropped if already in Mozc N-best for that reading",
            "fst": False,
            "prefix_trie": False,
            "rationale": "hashed/flat reading map is enough for N-best inject; no FST tooling in-repo",
        },
        "outputs": {
            "mozc_diff_entries": str(entries_path),
            "reading_map": str(map_path),
            "build_report": str(report_path),
        },
        "counts": report,
        "sizes": sizes,
        "neologd": {
            "included": False,
            "status": "follow_up",
            "reason": (
                "mecab-ipadic-NEologd is practical to fetch but distribution/license "
                "review is still required before bundling; proceeding with "
                "wikidata(CC0) + japanpost(terms-review-required) only."
            ),
        },
    }
    report_path.write_text(
        json.dumps(full_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote bundle {out_dir}: entries={n} readings={len(reading_map)} "
        f"size={sizes['total_mb']}MB",
        flush=True,
    )
    return full_report


def cmd_build(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    input_paths: list[str] = []
    for p in args.classify or []:
        path = Path(p)
        if not path.is_file():
            raise SystemExit(f"missing classify_in: {path}")
        input_paths.append(str(path))
        rows.extend(iter_classify_pairs(path))
    for p in args.raw_terms or []:
        path = Path(p)
        if not path.is_file():
            raise SystemExit(f"missing raw terms: {path}")
        input_paths.append(str(path))
        rows.extend(iter_raw_term_pairs(path))
    if not rows:
        raise SystemExit("No input rows. Pass --classify and/or --raw-terms")

    exclude_golds: set[str] = set()
    exclude_pairs: set[tuple[str, str]] = set()
    holdout_readings: set[str] = set()
    if args.exclude_holdout:
        hp = Path(args.exclude_holdout)
        if not hp.is_file():
            raise SystemExit(f"missing holdout: {hp}")
        exclude_golds, exclude_pairs = load_holdout_exclusions(hp)
        holdout_readings = {r for r, _ in exclude_pairs}
        if args.exclude_mode == "pair_only":
            exclude_golds = set()
        elif args.exclude_mode == "reading":
            # drop all surfaces for holdout readings (strongest anti-leak)
            pass
        else:
            # default: gold surface + pair (surface exclusion is the main lever)
            pass

    entries, counts = build_mozc_diff(
        rows,
        exclude_gold_surfaces=exclude_golds if args.exclude_mode != "pair_only" else set(),
        exclude_pairs=exclude_pairs,
        exclude_holdout_readings=args.exclude_mode == "reading",
        holdout_readings=holdout_readings,
        max_surfaces_per_reading=int(args.max_surfaces_per_reading),
        require_mozc_candidates=not bool(args.allow_no_mozc_diff),
    )
    counts["inputs"] = input_paths
    counts["exclude_holdout"] = args.exclude_holdout or ""
    counts["exclude_mode"] = args.exclude_mode
    report = write_bundle(
        Path(args.out_dir),
        entries,
        counts,
        label=args.label or Path(args.out_dir).name,
    )
    print(json.dumps({"entries": len(entries), "report": report["counts"]}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mozc-diff dictionary bundle builder")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build Mozc-diff onboardable dict artifacts")
    b.add_argument(
        "--classify",
        nargs="*",
        default=[],
        help="classify_in.jsonl paths (preferred; enables Mozc-diff)",
    )
    b.add_argument(
        "--raw-terms",
        nargs="*",
        default=[],
        help="raw interim term JSONL (no candidates; skipped unless --allow-no-mozc-diff)",
    )
    b.add_argument("--out-dir", required=True)
    b.add_argument("--label", default="")
    b.add_argument(
        "--exclude-holdout",
        default="",
        help="holdout.jsonl: exclude golds / pairs to prevent eval leak",
    )
    b.add_argument(
        "--exclude-mode",
        choices=["gold_and_pair", "pair_only", "reading"],
        default="gold_and_pair",
        help="anti-leak exclusion strength (default: drop holdout gold surfaces + pairs)",
    )
    b.add_argument("--max-surfaces-per-reading", type=int, default=0)
    b.add_argument(
        "--allow-no-mozc-diff",
        action="store_true",
        help="keep rows without Mozc candidates (NOT a true Mozc-diff)",
    )
    b.set_defaults(handler=cmd_build)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
