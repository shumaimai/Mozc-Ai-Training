# Phase 2 Baseline — ModernBERT-ja-30M (format_v2, official)

**Status:** complete.  This is the official baseline produced by commit
[`12b87f8`](https://github.com/shumaimai/Mozc-Ai-Training/commit/12b87f8)
(canonical `format_v2` prompts).  The earlier `[SEP]`-formatter run is
**provisional only** and is kept solely as provenance at
`mozc-artifacts:/artifacts/phase2_dataset_v2_runtime_context_modernbert_ja_30m`;
it was never resumed and its numbers are not comparable.

`final_test` was neither mounted into the image nor scored
(`final_test_mounted: false`).

## Run identity

| item | value |
|---|---|
| Modal app | [`ap-SjFTiIAzsRUPPN5DgeMXBe`](https://modal.com/apps/syuhei2009/main/ap-SjFTiIAzsRUPPN5DgeMXBe) |
| function_call_id | `fc-01M34R8XX5RKBZW5K71Q3QA7QA` |
| run state | completed (ephemeral app, all functions finished) |
| output | `mozc-artifacts:/artifacts/phase2_dataset_v2_runtime_context_modernbert_ja_30m_format_v2` |
| GPU | NVIDIA L4 22 GB |
| manifest | `phase2_run_manifest.json` (`dataset_contract: phase1-runtime-context-parity-v1`) |

## Training

- Base model: `sbintuitions/modernbert-ja-30m`
- 2 epochs, batch 256, fp16, max_len 128, max_neg 15, lr 2e-5
- Eligibility: `NEURAL_ELIGIBLE` only, gold-in-N-best required
- 47,589 groups / 284,957 pairs → **total 2,228 steps** (1,114/epoch)
- `resumed_from: ""` (fresh start), elapsed 659.2 s, VRAM peak 1,408.6 MB

Loss trajectory (every-20-step log, 113 points):
0.6331 (step 1) → 0.7137 (step 20) → 0.2210 (step 100) → 0.1561 (step 400)
→ 0.0419 (step 1100, end of epoch 1) → 0.0724 (step 1120) → last-10-step mean
**0.0622** (first-10 mean 0.3604).  The final logged batch (step 2228) shows
0.1941, which is single-batch noise, not a trend change.

Checkpoint: `cross_encoder.pt` with `{"step": 2228, "total_steps": 2228,
"complete": true}` plus `checkpoint_latest.pt` (step 2200) and `tokenizer/`.

## Validation results

Counts are asserted in-run: `VALIDATION_COUNTS all=5974 neural_eligible=4021`.

### Subset summary (tau = 0, i.e. raw neural top-1 with no margin gate)

| subset | n | Mozc alone | neural alone | Δ | best gated |
|---|---:|---:|---:|---:|---|
| NEURAL_ELIGIBLE | 4,021 | 82.9396% (3,335) | **86.4710%** (3,477) | +3.531pt | **87.4409%** @ tau=1.5 (+4.501pt) |
| ALL | 5,974 | 74.5899% (4,456) | **76.2471%** (4,555) | +1.657pt | **77.2514%** @ tau=3.0 (+2.662pt) |

Recommended operating points (max final_hit1 with regression ≤ 2%):
tau=1.5 (NEURAL_ELIGIBLE, regression 1.7991%) and tau=3.0 (ALL, regression
1.3241%).

### Per-tau helped / hurt / overwrite counts (NEURAL_ELIGIBLE)

| tau | final top1 | helped (recovered) | hurt (regressed) | overwritten |
|---:|---:|---:|---:|---:|
| 0.00 | 0.864710 | 282 | 140 | 555 |
| 0.25 | 0.867446 | 277 | 124 | 530 |
| 0.50 | 0.870430 | 274 | 109 | 505 |
| 0.75 | 0.873415 | 267 | 90 | 467 |
| 1.00 | 0.872668 | 251 | 77 | 430 |
| 1.25 | 0.874161 | 247 | 67 | 406 |
| 1.50 | **0.874409** | 241 | 60 | 389 |
| 2.00 | 0.872171 | 224 | 52 | 356 |
| 2.50 | 0.871922 | 212 | 41 | 318 |
| 3.00 | 0.871176 | 196 | 28 | 278 |
| 4.00 | 0.866949 | 165 | 14 | 207 |
| 5.00 | 0.861726 | 135 | 5 | 157 |

### Per-tau helped / hurt / overwrite counts (ALL)

| tau | final top1 | helped (recovered) | hurt (regressed) | overwritten |
|---:|---:|---:|---:|---:|
| 0.00 | 0.762471 | 340 | 241 | 938 |
| 0.25 | 0.764145 | 331 | 222 | 899 |
| 0.50 | 0.766488 | 325 | 202 | 858 |
| 0.75 | 0.769166 | 315 | 176 | 801 |
| 1.00 | 0.769334 | 298 | 158 | 752 |
| 1.25 | 0.771510 | 289 | 136 | 703 |
| 1.50 | 0.771342 | 276 | 124 | 668 |
| 2.00 | 0.771677 | 253 | 99 | 599 |
| 2.50 | 0.772347 | 237 | 79 | 529 |
| 3.00 | **0.772514** | 218 | 59 | 465 |
| 4.00 | 0.771008 | 184 | 34 | 350 |
| 5.00 | 0.768329 | 150 | 16 | 259 |

Serving latency (ALL, L4, batch 1024): 1.785 ms/group, 560.3 groups/s.

## Inputs

| item | value |
|---|---|
| Dataset | runtime-context-parity Dataset v2 (`PHASE1_CONTEXT_PARITY_GATE = PASS`) |
| train.jsonl.gz | `b8ae9a55f2bb083fe0b15eb8f79be19d3bdfaeb16de85cbc00da64cc7c844d29` |
| validation.jsonl.gz | `087008d04e769006561721ec10306c7edb48811abf645bc9d2f7cbc1023a628b` |
| model | `sbintuitions/modernbert-ja-30m` |
| formatter | canonical `format_v2` (`contextual-ranking-v2-format-v1`), newline-separated `読み / 文脈 / 候補`; train, eval, and parity fixtures share `build_pair_text` |
| final_test | not mounted, not scored |

## Artifacts

- `validation_neural_eligible_margin.json`, `validation_all_margin.json`
- `train_meta.json`, `train.log`, `checkpoint_meta.json`, `phase2_run_manifest.json`

Final-test evaluation remains deferred until explicitly requested.
