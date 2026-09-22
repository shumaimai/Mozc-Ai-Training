# Phase 1 Dataset v2 Production Report

**Status:** complete — model training has not started.

> **Superseded for training/evaluation by runtime-context parity Dataset v2.**
> This immutable original is retained for provenance.  Use
> [`PHASE1_CONTEXT_PARITY_REPORT.md`](PHASE1_CONTEXT_PARITY_REPORT.md) and
> `data/public/contextual_ranking_v2_production_runtime_context/` for Phase 2.

## Frozen contract

- Source: `wikimedia/wikipedia`, `20231101.ja`, train split, CC BY-SA 4.0.
- Document selection: global bottom-1,200 by `SHA256(seed | source_id)` over
  the full streaming target universe; it is not a leading-stream sample.
- Extraction: prefix replay, **last conversion segment only**, Sudachi
  SplitMode A, and previous-surface-only `clean_context(max_chars=50)`.
- Sampling: lowest 50 deterministic priorities per document using
  `SHA256(seed | source_id | sentence_index | boundary | reading | gold)`.
- Candidate generation: top-1 during the full scan, then top-30 only for the
  selected events.  The candidate cap is 30.
- Splits: document-level deterministic 80/10/10 train/validation/final test.
  Final test is reserved for final evaluation, not model selection.

The source freeze streamed 1,389,467 records, found 1,325,613 usable
documents, and persisted the selected manifest, retrieval configuration, and
SHA-256 checksum for every source shard.  See the raw source metadata below.

## Preflight

The 20-document top-1 vs top-30 scan comparison had exact parity:

| scan cap | selected identity exact | alignment result exact | rows/sec | docs/sec | query p50 / p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 100% | 100% | 68.28 | 0.0401 | 18.98 / 164.34 ms |
| 30 | 100% | 100% | 45.33 | 0.0266 | 29.41 / 231.08 ms |

The production scan therefore used top-1.  The injected historical failure
(`converter exited -15`) recovered on its second attempt after the dead
converter was discarded and replaced.  Event-level retry budget is three.

| workers | rows/sec | docs/sec | query p50 / p95 |
|---:|---:|---:|---:|
| 2 | 36.58 | 0.0215 | 18.60 / 165.20 ms |
| 4 | 66.24 | 0.0389 | 19.54 / 168.68 ms |
| 8 | **67.86** | **0.0398** | 19.29 / 168.36 ms |

Eight CPU workers were selected by total successful-row throughput, then
document throughput.  No GPU was used.

## Generated dataset

| metric | value |
|---|---:|
| Documents / distinct `source_id`s | 1,200 / 1,200 |
| Rows | 59,537 |
| Train / validation / final test docs | 960 / 120 / 120 |
| Train / validation / final test rows | 47,589 / 5,974 / 5,974 |
| Source overlap | 0 for every split pair |
| Rows per document, min / median / max | 23 / 50 / 50 |
| Generation elapsed | 11,543.40 s |
| End-to-end rows/sec | 5.16 |
| Raw prefix events | 1,023,008 |
| Alignment successes / failures | 964,549 / 58,459 (94.29%) |
| Enrichment failures / retries | 0 / 0 |

Sampled target positions are `front/middle/back = 28,256 / 20,559 / 10,722`.
The reported distribution is retained as an audit measure; within-document
selection itself is priority-based across each full article, rather than
early-stop selection.

### Status and eligibility

| category | rows |
|---|---:|
| `NEURAL_ELIGIBLE` example status | 38,095 |
| `PROTECTED_EVAL_ONLY` example status | 15,526 |
| `COVERAGE_FAILURE` example status | 5,916 |
| `NEURAL_ELIGIBLE` eligibility | 39,523 |
| `PROTECTED_EVAL_ONLY` eligibility | 18,840 |
| `COVERAGE_LIMITED` eligibility | 1,174 |

`latin_mixed` rows remain in the records and evaluation subsets, but all 1,174
are marked `COVERAGE_LIMITED`.  `source_kind` and `proper_noun` are retained
for every row; proper nouns are not declared permanently hard-protected.

### Candidate coverage and Mozc baseline

All figures are oracle coverage / baseline accuracy in percent.

| subset | top1 | top5 | top10 | top20 | top30 |
|---|---:|---:|---:|---:|---:|
| ALL | 74.9551 | 86.1313 | 88.4307 | 89.5460 | 90.0633 |
| `NEURAL_ELIGIBLE` | 82.4027 | 93.4165 | 94.8056 | 95.7088 | **95.9897** |
| `PROTECTED_EVAL_ONLY` | 63.3599 | 75.3875 | 79.7399 | 81.3694 | 82.4098 |
| `COVERAGE_LIMITED` | 10.3066 | 13.2879 | 13.2879 | 13.2879 | 13.3731 |

| split | top1 | top5 | top10 | top20 | top30 |
|---|---:|---:|---:|---:|---:|
| train | 75.0510 | 86.3351 | 88.6276 | 89.7602 | 90.2667 |
| validation | 74.5899 | 85.0686 | 87.3452 | 88.3495 | 88.8684 |
| final test | 74.5564 | 85.5708 | 87.9478 | 89.0358 | 89.6384 |

Context length is min/median/p95/max = `0 / 26 / 50 / 50`; candidate count is
`1 / 13 / 30 / 30`.  All rows have a last-segment target index, candidate cap
at most 30, and context at most 50 characters.

## Integrity and gates

Independent local re-audit validated all 59,537 JSONL records against the
schema, recomputed all three compressed-file SHA-256 digests, verified exact
dataset/manifest source-id membership, verified zero split overlap, and found
zero contract violations.  The compressed dataset checksums are:

```text
train.jsonl.gz       927a34f961c8e0748d3db16fff43075d3045b7a91d7e459729cd23ba2865c986
validation.jsonl.gz  5b66b49cdd0a9c5154356da93b36b2604be2e61a4c4400842aa8547ffd73c81c
final_test.jsonl.gz  d5bb204b102a2551d34df4d9a281f6f2ef09fbc3bcd732814cb498b322650a7e
```

```text
PHASE1_DATASET_DESIGN_GATE = PASS
PHASE1_DATASET_V2_GATE = PASS
```

## Artifacts

- `data/public/contextual_ranking_v2_production/`: compressed train,
  validation, and final-test JSONL files.
- `docs/contextual_ranking_v2/results/raw/phase1_dataset_v2_production/`:
  audit report, preflight report, checksums, source manifest/configuration,
  scan summary, and retained alignment/enrichment failure records.

No large-scale model training, split regeneration, or model selection was
started as part of this generation.
