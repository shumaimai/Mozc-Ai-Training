# Phase 1 production-parity pilot v2

Status: **complete; large-scale generation and model training remain paused**.

All runs used the same 100 deterministic Wikipedia documents, 8 CPU workers,
resident Mozc converter processes, top-30 candidates, and `context_prev` from
source text before the target. No future text was passed to prefix replay.

## 1. Sudachi SplitMode ablation

Full-sentence segment alignment was run with the same documents and cap.

| mode | examples | alignment success | failures | boundary split | reading mismatch | rows/sec |
|---|---:|---:|---:|---:|---:|---:|
| A | 4,924 | 92.12% | 421 | 417 | 4 | 302.80 |
| B | 4,924 | 91.25% | 472 | 468 | 4 | 280.62 |
| C | 4,924 | 89.89% | 554 | 550 | 4 | 289.38 |

**SplitMode A is the adoption candidate.** It reduces boundary failures by
24% versus the current C mode. Split mode is used only as an alignment anchor;
the source gold and reading semantics are unchanged.

## 2. Full sentence vs prefix replay

| mode | examples | alignment success | E2E top1 | top5 | top10 | top30 | rows/sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| full sentence, SplitMode A | 4,924 | 92.12% | 78.80% | 86.43% | 87.45% | 88.18% | 307.50 |
| prefix replay, last segment | 4,988 | 96.41% | 74.18% | 84.12% | 85.63% | 86.93% | 46.45 |

Prefix replay uses each Sudachi morpheme boundary as a natural prefix boundary:

```text
source prefix -> reading prefix -> Mozc conversion
              -> last conversion segment
              -> source suffix alignment -> gold
```

Every prefix row sets `target_segment_index` to
`conversion_segments_size - 1`. Duplicate `(source_id, reading, context,
gold)` rows are removed. Prefix alignment failures are retained in raw JSON.

The prefix path has better alignment success but lower ranking coverage and is
much slower because it invokes Mozc for each natural prefix boundary. It is
production-faithful for the current runtime target, but requires a later
throughput optimization before large-scale extraction.

## 3. Coverage by status

### Full sentence / SplitMode A

| group | top1 | top5 | top10 | top30 |
|---|---:|---:|---:|---:|
| ALL | 78.80% | 86.43% | 87.45% | 88.18% |
| NEURAL_ELIGIBLE | 82.26% | 90.47% | 91.23% | 91.76% |
| PROTECTED_EVAL_ONLY | 73.35% | 80.09% | 81.50% | 82.55% |

### Prefix replay / last segment

| group | top1 | top5 | top10 | top30 |
|---|---:|---:|---:|---:|
| ALL | 74.18% | 84.12% | 85.63% | 86.93% |
| NEURAL_ELIGIBLE | 78.52% | 89.40% | 90.54% | 91.87% |
| PROTECTED_EVAL_ONLY | 65.53% | 73.62% | 75.84% | 77.10% |

The important `NEURAL_ELIGIBLE` top-30 coverage is 91.76% for full-sentence
alignment and 91.87% for prefix replay. These denominators retain rows whose
gold is absent from the candidate list; only `ALL` is the end-to-end total,
while the status groups split the full population by eligibility policy.

## 4. Coverage-failure categories

The extractor records `punctuation_or_symbol`, `number`, `proper_noun`,
`latin_mixed`, `function_word`, and `normal_japanese_content` where the
available source metadata supports the distinction. It does not collapse these
into a single `gold_not_in_top_k` bucket in the pilot raw records.

The full A and prefix raw reports contain exact counts and ratios. All rows also
carry `example_status` and `example_reason`; alignment failures are separate
failure records and are not silently converted into coverage failures.

For auditability, the status counts and coverage-failure category counts are:

| path | NEURAL_ELIGIBLE | PROTECTED_EVAL_ONLY | COVERAGE_FAILURE | coverage-failure categories (count) |
|---|---:|---:|---:|---|
| full sentence A | 2,762 | 1,580 | 582 | punctuation/symbol 298; number 60; proper noun 434; latin/mixed 72; normal Japanese 183 |
| prefix replay | 3,050 | 1,286 | 652 | punctuation/symbol 343; number 73; proper noun 529; latin/mixed 69; normal Japanese 207 |

`short_function_word` and `hard_protected_attribute` are retained as explicit
reasons in the raw records (they are not folded into the five coverage-failure
categories above). Median context lengths are 21 and 19 Unicode characters,
respectively; median candidate counts are 12 and 13. Prefix rows use
`target_segment_index = conversion_segments_size - 1` and never include text
to the right of the target.

## 5. Production sampling design

The 100-document pilot intentionally uses the fixed streaming prefix for
reproducibility. Production must not use that prefix. The planned selection is
one-pass deterministic priority sampling across the complete corpus:

```text
priority = SHA256(seed || source_id)
keep the N lowest (priority, source_id) pairs
```

This is bounded-memory reservoir/priority sampling, deterministic across worker
counts, and distributes documents across the corpus. The selected source
manifest and checksum become immutable inputs to the eventual Dataset v2 run.

## Gate

The parity pilot is complete and committed, but the alignment and throughput
results do **not** authorize large-scale Dataset v2 generation yet. Keep
train/validation production generation and model training paused pending review
of the prefix throughput/coverage tradeoff.
