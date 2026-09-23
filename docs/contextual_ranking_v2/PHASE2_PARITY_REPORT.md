# Phase 2 Parity — ONNX export, tokenization, production policy, CPU latency

**Status:** complete.  The model is unchanged: the frozen Phase 2 `format_v2`
baseline ([`12b87f8`](https://github.com/shumaimai/Mozc-Ai-Training/commit/12b87f8),
report `PHASE2_BASELINE_REPORT.md`) is exported and verified, not retrained.
`final_test` is untouched.  Merge to `main` remains deferred.

## 1. Frozen inputs

Recorded in `artifacts/phase2_format_v2_frozen/FROZEN_RECORD.json` (key values
embedded here; the artifacts directory is not versioned):

| item | value |
|---|---|
| checkpoint | `cross_encoder.pt` sha256 `0b74f6f9cb68a1ddcc8cf8b09eecc52d25807ad5287d43deaa1dfae698a3b89b`, step 2228/2228, `complete: true`, `resumed_from: ""` |
| base model | `sbintuitions/modernbert-ja-30m` |
| tokenizer | `tokenizer.model` `00829302…`, `tokenizer.json` `0a94ac9a…`, `tokenizer_config.json` `fbab9c96…`, `special_tokens_map.json` `30bf8256…` |
| dataset train | `b8ae9a55f2bb083fe0b15eb8f79be19d3bdfaeb16de85cbc00da64cc7c844d29` (47,589 rows) |
| dataset validation | `087008d04e769006561721ec10306c7edb48811abf645bc9d2f7cbc1023a628b` (5,974 rows) |
| dataset final_test | present in the immutable directory, **not used anywhere in this verification** |
| formatter | canonical `format_v2` (`contextual-ranking-v2-format-v1`): `読み: {reading}\n文脈: {context}\n候補: {candidate}` |
| runtime policy | shipped `margin_policy.json`: tau 2.5, cand_cap 30, max_len 128, timeout 200 ms, context clip 50 |
| commits | training `777c085`, baseline fix `12b87f8`, runtime `dfbae07` |

## 2. ONNX fp32 export and PyTorch parity

Export (`tools/rerank/export_onnx_fp32.py`): opset 17, dynamic batch/seq,
inputs `input_ids:int64[batch,seq]` / `attention_mask:int64[batch,seq]`,
output `score:fp32[batch]`.  The graph is exactly the training forward pass
(Linear on `last_hidden_state[:, 0]`); no tokenizer is baked in.

> torch 2.14's dynamo exporter emits `Split.num_outputs`, which ONNX Runtime
> rejects (`INVALID_GRAPH`).  The export uses `dynamo=False` (TorchScript
> exporter); the rejected artifact was discarded.
> ONNX sha256 `00738ecfd6ee63e25cbc9cb6cfdb2976c381c930e423fc1401d356ef62a04a25`.

Parity on the first 800 validation groups (9,158 candidate texts,
`tools/rerank/onnx_parity_check.py`, Ryzen host, fp32):

| metric | value |
|---|---|
| max abs score diff (PyTorch vs ORT) | 4.15e-05 |
| mean abs diff / p99 | 3.55e-06 / 1.81e-05 |
| group argmax agreement | **800 / 800** |
| tau=2.5 final-selection agreement | **800 / 800** |

Every candidate ranking and every margin-gated decision is identical between
the training implementation and the exported runtime graph.

## 3. Token ID parity — train / Python serving / runtime

The live C++ IME engine does **not** tokenize: `RerankRewriter` ships
reading / context / candidates as JSON strings to the loopback daemon, which
encodes them with SentencePiece directly (BOS=1, EOS=2, PAD=3, truncate to
max_len−2).  Training and eval used the HF `AutoTokenizer` on the same
SentencePiece model.  The parity check compares both encoders on all 9,158
parity texts:

| metric | value |
|---|---|
| HF (train) vs SentencePiece (runtime daemon) exact matches | **9,158 / 9,158** |
| mismatches excluding truncated texts | 0 |
| texts exceeding the truncation window | 0 |
| max sequence length observed | 61 (limit 128) |

String-level runtime parity (the daemon never sees a different `context_prev`
than the dataset) is already enforced by the Phase 1 shared context builder
and its 28-check C++↔Python fixture suite (`test_runtime_context_parity.py`).
`mozc_compat/hf_tokenizer.cc` belongs to the retired WordPiece track and is
not part of the live path.

## 4. Production-policy replay (`tools/rerank/policy_replay.py`)

Every validation group is replayed through the exact shipped chain:
C++ guard (strict allowlist default, or safety mode) → daemon guard (short
reading / empty-or-symbol context) → ONNX fp32 scoring of the capped N-best
(one batch per conversion) → margin gate (`final = rerank_top1 if margin ≥
tau else mozc_top1`) → daemon junk-candidate revert → C++ hard-protect revert
(`protection == "HARD_PROTECT"` on the frozen rank-0 metadata, classified by
the same C++ rule: USER_SEGMENT_HISTORY_REWRITER / RERANKED / NUMBER /
NO_MODIFICATION / NO_DELETABLE or punctuation-symbol).

Counts at the shipped tau = 2.5 (5,974 groups, Mozc top1 = 74.590%):

| mode | skipped | scored | overwritten | helped | hurt | junk revert | hard-protect revert | final top1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strict (C++ default allowlist) | 5,884 | 90 | 21 | 11 | 1 | 0 | 0 | 74.757% |
| safety (daemon guards only) | 1,610 | 4,364 | 329 | 153 | 47 | 30 | 1 | **76.364%** |

Skip reasons — strict: `reading_not_eligible` 4,274, `reading_too_short` 1,388,
`context_empty_or_symbol` 222;  safety: 1,388 + 222 respectively.

Guarded tau sweep (safety mode; final top1 over all 5,974 groups):

| tau | overwritten | helped | hurt | final top1 |
|---:|---:|---:|---:|---:|
| 0.0 | 585 | 217 | 149 | 75.728% |
| 1.0 | 479 | 191 | 104 | 76.046% |
| 1.5 | 427 | 180 | 81 | 76.247% |
| 2.0 | 380 | 163 | 64 | 76.247% |
| 2.5 | 329 | 153 | 47 | **76.364%** |
| 3.0 | 292 | 140 | 35 | 76.348% |

Findings:

1. **The strict allowlist disables the new model.**  Only 90 of 5,974 groups
   reach scoring (final 74.757% vs Mozc 74.590%).  The shipped runtime must
   set `MOZC_RERANK_GUARD_MODE=safety` for the contextual model; strict mode
   remains the legacy-config default.
2. **The shipped tau = 2.5 is the best guarded operating point measured**
   (76.364%), confirming the margin policy chosen at export time.
3. End-to-end cross-check: driving the actual resident daemon over TCP with
   the same validation conversions yields final top1 76.381% (daemon applies
   the junk guard but not the C++ hard-protect layer).  The 0.017pt gap is
   exactly the one hard-protected group the C++ reverter restores by design.
   Mozc top1 matches the dataset baseline exactly (74.590%).

## 5. CPU latency — resident ONNX, one conversion = one batch

Environment: `shumain` — AMD Ryzen 7 5800X3D (8C/16T), Python 3.12, ORT
1.30.0, `intra_op=4`, model resident in the shipped loopback daemon
(`Mozc-Ai/runtime/rerank_daemon.py`), frozen Phase 2 fp32 ONNX.  The bench
(`tools/rerank/daemon_latency_bench.py`) mirrors the C++ `CallDaemon` pattern
exactly: **new TCP connection per conversion**, one JSON line each way,
TCP_NODELAY, hard 200 ms deadline, sequential requests (single-user IME),
20-request warmup discarded.

All 5,974 validation conversions:

| metric | value |
|---|---|
| round-trip p50 / p95 / p99 | **18.6 / 78.0 / 114.0 ms** |
| round-trip max / mean | 154.3 / 26.1 ms |
| daemon-side scoring p50 / p95 | 17.8 / 76.9 ms |
| timeouts (> 200 ms) | **0 / 5,974 (0%)** |
| failures | 0 |

A second, in-process measurement (`policy_replay.py`, no TCP) recorded p50
31.6 ms / p95 110.4 ms / 2 of 4,364 conversions over 200 ms (0.046%) while
sharing the CPU with the concurrent parity job; the daemon numbers above are
the deployment-relevant ones.  Both are far below the 200 ms C++ deadline.

This is the per-conversion latency regime the IME actually runs in; it is not
comparable to the L4 batched throughput (560 groups/s) quoted in the baseline
report, which scores whole validations at once on GPU.

## 6. C++↔Python guard parity

`tools/rerank/test_usage_guard.py` (compiles `rerank_guard_cli.cc` with g++
and compares decisions against Python): **11 / 11 OK**, including skip-reason
parity, safety-mode relaxation, junk-surface revert, and strict-allowlist
behavior.

## Gates

```text
PHASE2_TOKEN_PARITY_GATE  = PASS  (9,158/9,158 exact, 0 truncated)
PHASE2_ONNX_PARITY_GATE   = PASS  (max 4.15e-05, 800/800 argmax, 800/800 final)
PHASE2_POLICY_REPLAY_GATE = PASS  (strict-mode finding documented; safety tau=2.5 best)
PHASE2_CPU_LATENCY_GATE   = PASS  (p95 78.0 ms, 0 timeouts / 5,974)
```

## Artifacts

- `tools/rerank/export_onnx_fp32.py`, `onnx_parity_check.py`,
  `policy_replay.py`, `daemon_latency_bench.py` (versioned)
- `artifacts/phase2_format_v2_frozen/`: `FROZEN_RECORD.json`,
  `cross_encoder_fp32.onnx`, `parity_report.json`,
  `policy_replay_report.json`, `daemon_latency_report.json` (local, not
  versioned)

`final_test` was not mounted, not scored, and not consulted.  Merge to `main`
stays deferred pending review.
