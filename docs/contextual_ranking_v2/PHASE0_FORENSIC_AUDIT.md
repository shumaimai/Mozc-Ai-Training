# Mozc Contextual Ranking v2 — Phase 0 forensic audit

Date: 2026-09-21  
Branches: `v2/contextual-ranking-reset` in both repositories

## Scope and guardrails

This phase did not start model training, Dataset v2 generation, Sarashina/JEV
pruning, QAT, or vocabulary reduction. The v1 tag and the frozen JEV research
branch were not modified.

## 1. v1 train/serve formatter mismatch

`Mozc-Ai-Training v1.0.0/tools/rerank/train_cross_encoder.py` joins fields with
literal ` [SEP] ` delimiters. The shipped `Mozc-Ai v1.0.0/runtime/rerank_daemon.py`
uses literal newlines. The same SentencePiece tokenizer therefore sees
different inputs.

The audit fixture contains 7 cases × 3 candidates = 21 candidate rows and uses
the v1 tokenizer and v1 30M fp32 ONNX artifact from the Modal `track30m_ctx`
volume. Raw evidence: `results/raw/phase0a_formatter_audit.json`.

| metric | result |
|---|---:|
| token ID exact-match rate | **0/21 = 0.0%** |
| rows with changed token IDs | **21/21** |
| mean absolute sequence-length difference | **12.0 tokens** |
| score MAE | **1.028923** |
| top-1 changed groups | **1/7** |

This is a real model-score mismatch, not only a cosmetic text difference.
The runtime formatter must not be changed ad hoc; v2 must select and test one
canonical formatter across training/evaluation/runtime.

## 2. Target-segment logging

The audit confirmed the historical bug: `Rewrite` scores
`conversion_segments_size - 1`, while `Finish` read `conversion_segment(0)`.
The v2 branch now captures `conversion_id` and `target_segment_index`, reads the
committed candidate from that same index, and emits the required schema. Raw
context is removed from the default online log. The multi-segment C++ unit test
is `MultiSegmentLogUsesScoredTargetSegment`.

Raw evidence: `results/raw/phase0b_segment_logging_audit.json`.

## 3. Rewriter ordering and User History

Upstream `google/mozc` currently places `UserBoundaryHistoryRewriter` and
`UserSegmentHistoryRewriter` before the tail cleanup stages. v1 appended
`RerankRewriter` at the chain tail. This creates a real authority risk: a neural
reranker can overwrite an explicit user-history promotion after Mozc has made
that conservative decision.

Recommendation: treat user history as higher authority; run contextual ranking
as a correction/delta stage before user-history promotion, and add attribute-aware
hard protection for user-history, user-dictionary, numbers, punctuation, and
`CONTEXT_SENSITIVE` candidates. No broad chain reorder was done in Phase 0.

Raw evidence: `results/raw/phase0c_rewriter_order_audit.json`.

## 4. 30M latency baseline

`results/raw/phase0d_latency.json` is the controlled benchmark. It uses one
persistent ORT CPU session, warmup, one process/container, explicit intra/inter
threads, and the same v1 30M ONNX for all four configurations:

1. 5 candidates, dynamic longest padding, max length 128
2. 30 candidates, dynamic longest padding, max length 128
3. 5 candidates, fixed max length 128
4. 30 candidates, fixed max length 128

The report records CPU model, logical cores, ORT version, effective sequence
length, p50/p95/max, and model size. Results must only be compared within this
same report/container, never across Modal hosts.

Observed on Intel(R) Core(TM) i7-1060NG7, 8 logical cores, ORT 1.30.0,
intra/inter threads 1/1, model size 147,058,612 bytes:

| candidates | padding | seq p50/max | latency p50/p95/max |
|---:|---|---:|---:|
| 5 | dynamic longest | 33.5/37 | 43.44/47.52/48.67 ms |
| 30 | dynamic longest | 34.5/38 | 294.39/873.18/1177.39 ms |
| 5 | fixed 128 | 128/128 | 192.40/501.39/1031.28 ms |
| 30 | fixed 128 | 128/128 | 1241.12/2089.64/2109.98 ms |

The old ~51/128 ms claim is not reproduced for 30 candidates on this CPU and
thread configuration; the controlled result is the source of truth for this
environment. The 5-candidate dynamic result is within that order of magnitude.

## 5. Production contract

Canonical fixture: `tests/fixtures/contextual_ranking_v2/cases.jsonl`.

```text
reading: normalized Mozc reading
raw_preceding_text: input-only evidence, not persisted by default
cleaned_context: active preceding sentence, <= 50 Unicode characters
candidates: actual Mozc N-best, preserving rank and metadata in Dataset v2
target_segment: exact scored conversion segment
expected_format: 読み: X\n文脈: Y\n候補: Z
```

One implementation must be shared or parity-tested. The runtime already has
Python/C++ context parity coverage; CI must execute it with a real C++ CLI.

## 6. Phase 1 gate

**Do not start model training yet.** The formatter mismatch is quantified and
the segment logging fix is covered, but Phase 0 remains conditional until the
CI C++ parity test and the controlled latency artifact are green and the
attribute-aware user-history constraint is implemented in the production Mozc
integration. Dataset v2 may begin only after those contract checks pass.

## 7. Unresolved items

- Run the C++ context/tokenizer parity binaries in the repository's full Mozc CI image.
- Verify `RERANKED`, `USER_SEGMENT_HISTORY_REWRITER`, and user-dictionary attributes against the exact pinned Mozc revision.
- Add an actual Mozc N-best multi-segment replay, not only the compatibility fixture.
- Confirm the 30M benchmark on the intended production CPU; do not transfer Modal-host numbers to Windows.
- Re-evaluate score/top-1 mismatch on a larger, document-disjoint forensic fixture before selecting v2 formatter details.
