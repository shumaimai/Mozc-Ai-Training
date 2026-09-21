# Mozc Contextual Ranking v2 — Phase 0 forensic audit

Date: 2026-09-21  
Branches: `v2/contextual-ranking-reset` in both repositories

## Gate

**PHASE0_GATE = PASS**

The required production-contract audits are closed. No Dataset v2 generation,
model training, Sarashina/JEV pruning, QAT, or vocabulary reduction was run in
this closure. The old v1 tag and `experiment/sarashina-jev-poc` were not
modified.

## Closure matrix

| area | status | evidence |
|---|---|---|
| v1 formatter mismatch impact | closed | `raw/phase0a_formatter_audit.json` |
| target-segment logging | closed | `raw/phase0b_segment_logging_audit.json`, C++ compatibility test |
| context parity | green | `raw/phase0f_context_parity.json`, C++ CLI 23/23 |
| pending-log concurrency | closed | `raw/phase0h_concurrency_audit.json` |
| User History contract | defined | `raw/phase0g_user_history_contract.json` |
| production-faithful latency | complete | `raw/phase0d_latency_production.json` |
| canonical fixture/formatter | closed | `tests/fixtures/contextual_ranking_v2/cases.jsonl` |

## 1. v1 train/serve formatter mismatch

The v1 trainer used `読み: X [SEP] 文脈: Y [SEP] 候補: Z`; shipped runtime
used newline separators. With the v1 tokenizer/model, 7 cases × 3 candidates
produced:

- token-ID exact match: **0/21 (0.0%)**
- changed rows: **21/21**
- sequence-length MAE: **12.0 tokens**
- score MAE: **1.028923**
- changed top-1 groups: **1/7**

This was a real score-path mismatch. Runtime was not silently changed during
the forensic measurement. v2 now names the newline contract explicitly as
`contextual-ranking-v2-format-v1`; this is a shared contract, not a claim that
it is intrinsically better than v1.

## 2. Target-segment logging

The historical path scored `conversion_segments_size - 1` but read segment 0 at
Finish. The v2 branch stores `conversion_id` and `target_segment_index`, then
reads the committed candidate from that same index. The log schema includes
`conversion_id`, `target_segment_index`, `reading`, `mozc_top1`, `model_top1`,
`final_top1`, and `committed_candidate`; raw context is omitted by default.

`MultiSegmentLogUsesScoredTargetSegment` covers the two-segment compatibility
path, and the pinned Mozc converter was then built with Bazelisk. The actual
replay `startconversion 駅にきしゃ` produced two segments; target segment 1
had a 30-candidate N-best, and `commit 1 2` committed `汽車` from segment 1.
The replay evidence is in `tests/fixtures/contextual_ranking_v2/actual_mozc_multisegment_replay.json`
and `raw/phase0i_multisegment_replay.json`.

## 3. Context parity

The pinned upstream Mozc checkout was revision
`13c98988247aa711d99db9e348ec2a597d14b5cd`. A standalone C++ CLI compiled
against its real `src/rewriter/context_clip.cc` and was compared with Python
`clean_context`/reading normalization: **23/23 cases passed**. Reproducible
command:

```bash
g++ -std=c++17 -O2 -DMOZC_RERANK_STANDALONE \
  -I<mozc>/src/rewriter <mozc>/src/rewriter/context_clip.cc \
  <mozc>/src/rewriter/context_clip_cli.cc -o context_clip_cli
python -m tools.rerank.test_context_clip_parity --cli ./context_clip_cli
```

## 4. Rewriter ordering and User History contract

Pinned upstream order places `UserBoundaryHistoryRewriter` and
`UserSegmentHistoryRewriter` before tail cleanup; v1 appended RerankRewriter at
the tail. A tail neural reranker can therefore overwrite an explicit history
promotion. No broad reorder was performed.

Phase 1 candidate metadata must retain surface, rank, cost, wcost, lid, rid,
attributes, category, and converted-segment count. The policy is:

- **HARD_PROTECT:** `USER_SEGMENT_HISTORY_REWRITER`, `RERANKED`, `NUMBER`,
  punctuation/SYMBOL, `NO_MODIFICATION`, `NO_DELETABLE`.
- **DELTA_ONLY:** `USER_DICTIONARY`, `CONTEXT_SENSITIVE`.
- **NORMAL:** candidates without those protections.

The runtime guard `IsAiOverwriteProtected` implements the hard-protect subset;
the complete pinned-revision classification is in the raw contract report.

## 5. Pending-log concurrency

The old single `pending_log_`/`has_pending_log_` state could mix interleaved
Rewrite A, Rewrite B, and Finish A. It is now a mutex-protected map keyed by
calling thread, and hook temporary paths use a process-wide atomic suffix.
`conversion_counter_` is incremented under the same mutex. Because the pinned
Mozc API has no request/session ID, Rewrite and Finish for one lifecycle must
remain on the same calling thread. A future cross-thread embedding must add an
explicit ID rather than infer identity from segments.

## 6. Production-faithful 30M latency

The closure benchmark uses the 80-group old `track30m_ctx` parity fixture
(1,608 candidate scores), a persistent session, warmup, and one container for
ORT 1.22.1 and 1.30.0. It covers candidate counts 5/30, longest/fixed128
padding, and intra threads 1/8 (inter=1). The fixture reuses old reading and
candidate groups; contexts are controlled realistic Japanese preceding text
because the old score artifact does not contain source contexts. It is not
Dataset v2.

The Modal host exposed 24 logical cores but no CPU model string; therefore these
numbers are valid for same-container ORT comparison and must not be transferred
to the user's i7-1060NG7 or compared as if they were the old 51/128 benchmark.
Full p50/p95/max sequence and latency values are preserved in
`raw/phase0d_latency_production.json`. Representative dynamic-longest values:

| ORT | threads | candidates | seq p50/p95/max | latency p50/p95/max ms |
|---|---:|---:|---:|---:|
| 1.22.1 | 1 | 5 | 29/32.05/37 | 40.90/44.53/52.05 |
| 1.22.1 | 8 | 5 | 29/32.05/37 | 17.99/22.87/24.00 |
| 1.22.1 | 1 | 30 | 30/32.05/37 | 103.79/253.79/308.74 |
| 1.22.1 | 8 | 30 | 30/32.05/37 | 41.95/89.44/107.11 |
| 1.30.0 | 1 | 5 | 29/32.05/37 | 41.41/45.13/52.21 |
| 1.30.0 | 8 | 5 | 29/32.05/37 | 19.49/21.06/25.86 |
| 1.30.0 | 1 | 30 | 30/32.05/37 | 105.96/256.88/319.92 |
| 1.30.0 | 8 | 30 | 30/32.05/37 | 52.52/110.14/120.11 |

Fixed128 results and all means are raw-only to avoid hiding tail behavior.

## 7. Production contract

Canonical fixture: `tests/fixtures/contextual_ranking_v2/cases.jsonl` with 10
cases covering station/newspaper readings, short input, numbers, punctuation,
multi-segment, empty context, normalization, and sentence clipping. Every case
declares `format_version: contextual-ranking-v2-format-v1`.

The initial context policy is active preceding sentence, no future text, max 50
Unicode characters. Python and C++ implementations are parity-tested. Raw
preceding text is evidence only and is not persisted in the default online log.

## 8. Phase 1 decision

**Phase 1 may begin contract/schema implementation only.** Model training and
Dataset v2 generation remain explicitly out of scope until a separate Phase 1
approval. The formatter is provisional by design; no claim of superiority over
v1 is made.

## Remaining follow-ups

- Re-run the same latency matrix on the intended i7-1060NG7 when available;
  do not mix host results.
- Add a request/session ID if any future integration can move Finish across
  threads.
- Before Dataset v2, review the metadata schema against the exact production
  candidate protobuf and obtain separate Phase 1 approval.
