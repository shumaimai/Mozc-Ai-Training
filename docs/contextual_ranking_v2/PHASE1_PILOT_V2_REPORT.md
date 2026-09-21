# Phase 1 Dataset v2 pilot v2

Status: **complete; large-scale generation and training remain paused**.

The pilot compares the old Sudachi-morpheme/standalone-query extractor with a
Mozc-segment-aligned extractor on the same public Wikipedia documents. All
examples are retained with `example_status` and `example_reason`.

## Production-faithful changes

- `NEURAL_ELIGIBLE`, `PROTECTED_EVAL_ONLY`, and `COVERAGE_FAILURE` are stored
  per example.
- Coverage failures remain in the denominator. Reports contain both
  end-to-end metrics and conditional metrics restricted to gold in top-30.
- Candidate coverage is reported at top-1, top-5, top-10, and top-30.
- The aligned path converts a full sentence reading once, uses actual Mozc
  segment boundaries, maps each segment back to source surface, and uses only
  source text preceding that segment through `clean_context(max_chars=50)`.
- Alignment failures are retained in `*.failures.jsonl`, not silently dropped.
- Converter output now emits numeric `attributes` and explicit `wcost`; rows
  also retain cost, lid, rid, category, and converted segment count.

## 20-document PoC

| mode | examples | alignment success | end-to-end top1 | top5 | top10 | top30 | conditional top1 | rows/sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old standalone | 1,000 | 100.00% | 71.10% | 87.60% | 89.40% | 90.80% | 78.30% | 147.46 |
| new segment aligned | 1,000 | 94.34% | 79.30% | 88.50% | 89.90% | 90.70% | 87.42% | 147.46 |

The PoC produced 60 alignment failures, all recorded as
`ALIGNMENT_FAILURE`.

## 100-document pilot v2

| mode | examples | alignment success | end-to-end top1 | top5 | top10 | top30 | conditional top1 | rows/sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old standalone | 5,000 | 100.00% | 68.60% | 83.02% | 84.68% | 87.36% | 78.53% | 192.29 |
| new segment aligned | 4,924 | 89.89% | 78.76% | 86.31% | 87.33% | 88.04% | 89.46% | 189.37 |

The new path recorded 554 alignment failures. Status counts for the new path:

- `NEURAL_ELIGIBLE`: 3,148
- `PROTECTED_EVAL_ONLY`: 1,187
- `COVERAGE_FAILURE`: 589

Reason counts for the new path:

- `normal_contextual`: 3,148
- `punctuation`: 1,048
- `gold_not_in_top_k`: 589
- `short_function_word`: 84
- `number`: 52
- `symbol`: 3

All 4,924 new rows and all 5,000 comparison rows pass the Python record
validator. The source documents are the same deterministic 100-document
Wikipedia prefix, and processing used 8 CPU workers with persistent Mozc
processes. No GPU or model training was used.

## Interpretation and stop gate

Segment alignment improves end-to-end top-1 substantially over the standalone
baseline in this pilot, but alignment failure rate is still material and must
be investigated before production-scale generation. The 100-document result is
therefore a **pilot result, not approval for large-scale generation**.

Raw reports, rows, failure records, and checksums are under
`docs/contextual_ranking_v2/results/raw/phase1_pilot_v2/` and
`data/public/contextual_ranking_v2_pilot_v2/`.
