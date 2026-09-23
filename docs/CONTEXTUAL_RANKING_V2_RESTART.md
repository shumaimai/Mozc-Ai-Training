# Contextual Ranking v2 — Training Restart

Canonical architecture and safety principles live in:

- `shumaimai/Mozc-Ai/docs/MOZC_CONTEXTUAL_RANKING_V2_BEST_PRACTICES.md`

This branch is a clean restart for the training side.

## Frozen research

Do not continue production development from:

- `experiment/sarashina-jev-poc`
- the 3-article Sarashina proxy dataset
- the existing 64k/6L model-selection path

Keep them unchanged as research evidence.

## First milestone: no training

The first milestone is to create a production-faithful data/evaluation contract.

Implement:

1. actual Mozc top-K candidate extraction
2. candidate metadata capture
3. canonical production context cleaning/clipping
4. document-disjoint split creation
5. dataset concentration/leak reports
6. Mozc-only baseline evaluation
7. train/eval/runtime input-parity fixtures
8. controlled latency benchmark harness

Do not start a new neural training run until these checks pass.

## Dataset v2 minimum schema

Each example should contain enough information to reconstruct the production decision:

```json
{
  "source_id": "...",
  "reading": "...",
  "context_prev": "...",
  "gold": "...",
  "target_segment_index": 0,
  "candidates": [
    {
      "surface": "...",
      "rank": 0,
      "cost": 0,
      "cost_delta": 0,
      "lid": 0,
      "rid": 0,
      "attributes": 0
    }
  ]
}
```

Exact metadata may evolve, but actual Mozc rank/cost must not be discarded.

## Split requirements

- train/validation/final-test source overlap = 0
- deterministic seed/manifests
- per-source example cap
- final test untouched during model selection
- report source concentration
- report Mozc top-1 and top-K oracle coverage

## Baseline order

After the data contract passes:

1. ModernBERT-ja-30M cross encoder
2. page-wise small encoder
3. strong Japanese teacher only if needed
4. distillation only after teacher generalization is proven

## Optimization order

Do not optimize INT8/vocab/layers until:

- the small FP32/BF16 baseline beats Mozc on unseen validation
- real-use replay shows acceptable helped/hurt behavior
- runtime formatting parity is proven

## Required reports

Every experiment report must include:

- dataset revision/hash
- pinned Mozc revision
- model revision
- context policy
- Mozc Hit@1
- final Hit@1
- delta vs Mozc
- helped/hurt/overwrite counts
- p50/p95
- CPU/ORT/thread configuration
- model size
- raw JSON report path

## Immediate work queue

Phase 0:
- add forensic tests for old v1 `[SEP]` vs newline formatting
- add target-segment logging fixture
- reproduce 30M latency in one controlled environment

Phase 1:
- build Dataset v2 generator
- build leak/concentration auditor
- build baseline evaluator

Only then begin model training.
