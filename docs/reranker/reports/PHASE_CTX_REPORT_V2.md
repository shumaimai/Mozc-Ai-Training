# PHASE_CTX_REPORT_V2 — Contextual Reranker (v2 / Modal)

Date: 2026-08-12  
Environment: **Modal** (WSL CLI) / GPU train=`L4` eval=`T4` / `sbintuitions/modernbert-ja-70m` / max_len=128  
Local ROCm Track A v2 was **stopped** before Modal launch (no double-burn).

## Architecture choice

| item | choice |
|--|--|
| Track A init | HF `sbintuitions/modernbert-ja-70m` |
| Track B init | `artifacts/rerank/modernbert70m_ce_v3/cross_encoder.pt` → Volume `/current_ce/cross_encoder.pt` |
| Why not ruri | Current CE is ModernBERT; A/B must share one architecture (1-variable). AGENT doc ruri examples are illustrative only. |

## §5.1 Decision

**ship-candidate — prefer Track B**

1. **Context is working** (primary subset `context_sensitive`): CS Δ ≥ +5pt on seen/unseen/fresh for both tracks (A: +33 / +24 / +42; B: +35 / +26 / +42).
2. **A vs B winner** by fresh news hit@1 (with context, all groups): **B 95.27% > A 95.00%**.
3. Non-CS Δ ≈ 0 (slightly negative on some sets) — context gain is concentrated where expected.

Next: keep `.pt` on Volume / local `artifacts/` (git-ignore); optional gold audit before Mozc hook ship.

## Pipeline

1. Data: `data/rerank_ctx/train_v2.jsonl` + `eval_{seen,unseen,fresh}_v2.jsonl` (CS ≥500/eval)
2. Modal train scripts: `scripts/modal_train.py` (`--init-from` → `--init-ckpt`, `--max-len 128`, timeout 4h, `spawn()`, mid-run `save_every=200`)
3. Modal eval: `scripts/modal_eval.py` wraps `tools.rerank.eval_contextual` (ctx ON/OFF + τ sweep + CS subset)
4. Track A spawn app `ap-D4fh0brlnXDieb7fClMckk` / call `fc-01KZTX6QDSS9BTQSY8797AHGEB`
5. Track B spawn app `ap-YioDQi5FTxaUYEFeWwiWzI` / call `fc-01KZTX7BS288TPVMV3YDGVV3K8`
6. Eval apps `ap-XpWrHsMUvg9gzTy6DzUFx2` (A), `ap-OL6FbAFGPSCvqjzuMx1F2d` (B)

## Train metrics

| track | init | groups | pairs | epochs | bs | max_len | elapsed_s | complete |
|--|--|--|--|--|--|--|--|--|
| A | HF modernbert-ja-70m | 44 306 | 492 610 | 2 | 512 | 128 | 2298 | yes |
| B | modernbert70m_ce_v3 | 44 306 | 492 610 | 2 | 512 | 128 | 2378 | yes |

Checkpoints (Modal Volume `mozc-artifacts`):  
`/trackA_v2_modernbert70m/cross_encoder.pt`  
`/trackB_v2_continue/cross_encoder.pt`

## Eval (hit@1 @ recommended τ; primary = context_sensitive)

| model | set | n | n_CS | hit@1 all ON | CS ON | CS OFF | CS Δpt | τ |
|--|--|--|--|--|--|--|--|--|
| A | seen | 1500 | 525 | 91.00% | 76.76% | 43.81% | +32.95 | 1.5 |
| A | unseen | 1500 | 522 | 91.47% | 76.44% | 52.30% | +24.14 | 2.0 |
| A | fresh | 1500 | 525 | 95.00% | 88.00% | 46.48% | +41.52 | 0.0 |
| **B** | seen | 1500 | 525 | 91.53% | 77.33% | 42.67% | **+34.67** | 2.5 |
| **B** | unseen | 1500 | 522 | 91.53% | 76.82% | 50.96% | **+25.86** | 2.5 |
| **B** | fresh | 1500 | 525 | **95.27%** | 88.76% | 46.67% | **+42.10** | 0.5 |

Machine-readable: `artifacts/rerank_ctx/eval/summary_track{A,B}_v2.json`  
One-pager: `artifacts/rerank_ctx/summary_v2.md`

## Ops notes

- Early `modal run --detach` + `.remote()` jobs were cancelled when the local WSL client got SIGTERM; fixed by switching entrypoints to `.spawn()` and mid-run Volume commits (`save_every=200`).
- Track B `--init-from` must be **`/artifacts/current_ce/cross_encoder.pt`** (Volume is mounted at `/artifacts`).
- Do not commit `.pt` files to git.

## How to re-check logs / pull

```bash
export PATH="$HOME/.local/bin:$PATH"
cd <Mozc-Ai-Trainingのチェックアウト先>
modal app list
modal volume ls mozc-artifacts /trackB_v2_continue
modal volume get mozc-artifacts /eval ./artifacts/rerank_ctx/eval
```
