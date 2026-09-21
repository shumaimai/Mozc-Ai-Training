# Contextual Ranking v2 — Phase 1 contract implementation

Status: **pilot v2 complete; large-scale generation paused**
Dataset v2 generation: **pilot only**
Model training: **not started**

## Scope of this increment

This increment makes the production record contract executable. It does not
collect new user data, query Mozc in bulk, create a train/validation split, or
run a model.

## Versioned identifiers

- Formatter: `contextual-ranking-v2-format-v1`
- Record schema: `contextual-ranking-v2-record-v1`
- Pinned Mozc source for metadata semantics:
  `13c98988247aa711d99db9e348ec2a597d14b5cd`

## Required record

Each future Dataset v2 row must contain:

```text
schema_version
format_version
source_id
reading
context_prev
gold
target_segment_index
candidates[]
```

Every candidate must preserve actual Mozc metadata:

```text
surface, rank, cost, cost_delta, lid, rid, attributes,
category, converted_segment_count, protection
```

`surface`-only records are rejected by
`tools.rerank.contextual_ranking_v2_schema.validate_record`.

Pilot v2 rows additionally carry `example_status` and `example_reason`:

- `NEURAL_ELIGIBLE`: normal contextual candidate suitable for future training.
- `PROTECTED_EVAL_ONLY`: retain for evaluation, but do not train over protected
  symbol/number/punctuation/function-word cases by default.
- `COVERAGE_FAILURE`: retain in the end-to-end denominator when gold is absent
  from top-K; reason is `gold_not_in_top_k`.

Ranking reports must publish both conditional metrics (gold present in top-30)
and end-to-end metrics over all extracted examples, plus top-1/top-5/top-10/
top-30 candidate coverage.

## Runtime parity

The Mozc runtime conversion log keeps the existing `nbest` field for backward
compatibility and now also emits `candidate_metadata` with the same fields.
Raw preceding text remains omitted by default. The C++ rewriter test checks
that metadata is emitted for the actual multi-segment path.

## Next contract-only steps

1. Build a Mozc-backed extractor that emits this schema for an explicitly
   approved public source fixture.
2. Add deterministic document-disjoint split manifests and concentration/leak
   audits.
3. Add Mozc-only top-1/top-K baseline evaluation and helped/hurt metrics.
4. Review the generated fixture and approve Dataset v2 generation separately.

No step above authorizes model training. Optimization remains blocked until the
unseen-document baseline and runtime replay gates pass.
