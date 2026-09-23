# Phase 1 Dataset v2 CPU pilot

Status: **pilot complete; large-scale generation paused for review**.

No model training, GPU use, Sarashina pruning, QAT, or vocabulary ablation was
started.

## Run

- Source: `wikimedia/wikipedia`, `20231101.ja`, streaming prefix filtered to
  100 documents, CC BY-SA 4.0.
- Mozc revision: `13c98988247aa711d99db9e348ec2a597d14b5cd`.
- Converter: pinned `converter_main` with candidate metadata output.
- CPU: Intel(R) Core(TM) i7-1060NG7 CPU @ 1.20GHz, 8 logical CPUs.
- Workers: 8 (`min(os.cpu_count(), documents)`). Each worker keeps one
  persistent converter process; it is not restarted per example.
- Shards: 16 document shards, intermediate JSONL plus `.done.json`/`.failed.json`
  state. Deterministic merge is by `source_id`, reading, context, and gold.
- Seed: `20260921`.
- Context: shared `clean_context`, maximum 50 Unicode characters.
- Candidate request: top 30; actual rows contain 3--30 candidates.

## Metrics

| metric | result |
|---|---:|
| documents | 100 |
| valid rows | 4,368 |
| schema validation failures | 632 |
| Mozc top-1 accuracy | 78.5256% |
| top-5 oracle coverage | 95.0321% |
| rows/sec | 441.15 |
| docs/sec | 10.10 |
| ETA at completion | 0 sec |
| examples/document (min / median / max / mean) | 32 / 44.5 / 50 / 43.68 |
| context length (min / median / max) | 0 / 19 / 50 |
| candidate count (min / median / max) | 3 / 25 / 30 |
| failed shards | 0 |

Accuracy and oracle coverage use the 4,368 schema-valid rows. The 632 rejected
examples were primarily `gold_not_in_candidates`: the article surface was not
present in Mozc's requested top-30, so they are retained as coverage failures
and not silently relabeled.

## Split and validation

Splitting is by `sha256(seed:source_id)`, never by row. The resulting source
sets are disjoint:

- train: 3,637 rows / 83 documents
- validation: 260 rows / 6 documents
- test: 471 rows / 11 documents

All committed rows pass `validate_record`; no source ID overlaps another split.

## Resume and provenance

The raw report, source manifest, source-fetch metadata, checksums, and split
JSONL are under `docs/contextual_ranking_v2/results/raw/phase1_pilot/` and
`data/public/contextual_ranking_v2_pilot/`. A completed shard is skipped on
resume; a shard with `.failed.json` is rerunnable without regenerating other
shards.

The raw report is [pilot_report.json](results/raw/phase1_pilot/pilot_report.json).
This is a pilot only. Do not start large-scale Dataset v2 generation until the
review explicitly accepts the 632 top-30 coverage failures and the observed
Mozc baseline metrics.
