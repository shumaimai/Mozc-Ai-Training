# Phase 1 Dataset v2 — Runtime Context Parity

**Status:** complete.  This is the Phase 2 input dataset; the original
production Dataset v2 remains immutable and is not overwritten.

## Corrected contract

For each already-selected prefix event, `context_prev` now matches
`RerankRewriter` exactly:

```text
CleanContext(previous committed text + current-conversion earlier-segment top1, 50)
```

- Prefix replay, last conversion segment only, Sudachi SplitMode A, candidate
  cap 30, and document splits are unchanged.
- No raw-prefix scan was repeated.  The run reused the frozen 1,200 source
  documents and all 59,537 saved selected events.
- Mozc was replayed only to obtain the earlier current-conversion segment
  rank-0 surfaces used in the runtime context.
- Target candidate payloads were copied from the immutable production
  dataset.  This avoids converter-build-dependent N-best metadata drift while
  preserving the original actual candidate set.

## Context delta

| metric | value |
|---|---:|
| Selected events / replayed rows | 59,537 / 59,537 |
| Exact old/new `context_prev` | 14,132 (23.7365%) |
| Changed `context_prev` | 45,405 (76.2635%) |
| Replay failures / retries | 0 / 0 |
| Workers / elapsed / rows per second | 8 / 865.01 s / 68.83 |

Examples of the intended difference:

| reading → gold | old source-surface context | runtime-parity context |
|---|---|---|
| `すまーとふぉんの` → `スマートフォンの` | `インターネットの台頭や携帯電話・` | `インターネットの台頭や携帯電話` |
| `たよう` → `多様` | `…携帯電話・スマートフォンの普及により…` | `…携帯電話スマートフォンの普及により…` |
| `いっぱんこうぼし` → `一般公募し` | `昭和58年（1983年）、電話帳の愛称を` | `昭和後八年記号一級鉢三年記号、電話帳の相性を` |

The first two show source spelling/punctuation versus actual Mozc rank-0
history.  The third is deliberately more divergent: production parity means
using runtime's committed Mozc result, not preserving source text.

## Dataset and integrity

| split | documents | rows | Mozc top1 / top5 / top10 / top20 / top30 |
|---|---:|---:|---|
| train | 960 | 47,589 | 75.0510 / 86.3351 / 88.6276 / 89.7602 / 90.2667% |
| validation | 120 | 5,974 | 74.5899 / 85.0686 / 87.3452 / 88.3495 / 88.8684% |
| final test | 120 | 5,974 | 74.5564 / 85.5708 / 87.9478 / 89.0358 / 89.6384% |

- 1,200 distinct `source_id`s; all 59,537 `sampling_identity` values are
  unique and saved in the corrected records.
- Candidate payload is exactly equal to the original Dataset v2 for all
  59,537 rows.
- Source membership and document split are exactly equal to the original.
- Source overlap and complete record overlap across every split pair are 0.
- Source-independent identical model-input/candidate groups remain visible as
  an audit statistic: train/validation 2, train/final-test 2,
  validation/final-test 1.  They are distinct documents, not leakage by
  complete record or source identity.
- All records pass schema validation; target is always the last conversion
  segment; context length is at most 50; candidate count is at most 30.

Validation eligibility counts are `NEURAL_ELIGIBLE=4,021`,
`PROTECTED_EVAL_ONLY=1,773`, and `COVERAGE_LIMITED=180`.  Phase 2 training
uses only the first category with gold in the top-30; validation reports both
that primary subset and ALL rows.  `final_test` is excluded from the Modal
training image and is not a model-selection input.

## Runtime/train/eval fixture

`tests/fixtures/contextual_ranking_v2/runtime_context_parity.jsonl` covers
committed history, earlier-segment top-1 assembly, source-vs-Mozc divergence,
and 50-character clipping.  The shared Python builder and C++ standalone
runtime CLI agree on all 28 fixture checks.

## Checksums and gates

```text
train.jsonl.gz       b8ae9a55f2bb083fe0b15eb8f79be19d3bdfaeb16de85cbc00da64cc7c844d29
validation.jsonl.gz  087008d04e769006561721ec10306c7edb48811abf645bc9d2f7cbc1023a628b
final_test.jsonl.gz  4a197730ee4411d39f543fa7a0d258922e7aad907ae28f1b57547ef4fd24d145

PHASE1_DATASET_DESIGN_GATE = PASS
PHASE1_DATASET_V2_GATE = PASS
PHASE1_CONTEXT_PARITY_GATE = PASS
```

## Artifacts

- `data/public/contextual_ranking_v2_production_runtime_context/dataset/`
- `docs/contextual_ranking_v2/results/raw/phase1_context_parity/`
- `scripts/phase1_context_parity_rebuild.py`

Phase 2 may begin only from this corrected dataset.  Final-test evaluation
remains deferred until the validation-selected baseline is fixed.
