# PHASE_CTX_REPORT — Contextual Reranker

Date: 2026-08-12 (updated: Phase 3 CTX ship τ=2.5)  
Environment: Modal (`L4` train / `T4` eval) / `sbintuitions/modernbert-ja-70m` / max_len=128  
Colab: **not used**. Small-scale pilot tables below are retained for history.

**Distribute τ (Phase 3 CTX):** **2.5** — see `docs/reranker/tasks/NEXT_TASK_PHASE3_CTX.md`.  
The τ=2.0 table below is the earlier finalize pick (max min-CS-Δ among eligible); **superseded** after clean eval (overall regress &lt;1.4% @ 2.5, CS Δ still ≫ +5pt).

## §5.1 Decision (finalize)

**GO — Phase 3 (Mozc integration)** with **Track B**, distribute **τ=2.5** (`NEXT_TASK_PHASE3_CTX`).

Rationale (one line): gold audit ≥95%, and at a **single** τ=2.5 on clean eval all three sets keep overall regression &lt;1.4% while CS Δ stays ≫ +5pt.

| gate | result |
|--|--|
| Context works (CS Δ ≥ +5pt seen/unseen/fresh) | **yes** (+32.8 / +29.7 / +55.8) @ τ=2.0 |
| Adopt A vs B (fresh hit@1 with context) | **B** (v2 per-set-opt: 95.27% &gt; A 95.00%; ship τ keeps B) |
| Gold audit ≥95% | **100%** proxy on `audit_sample_v2_300` (see honesty note) |
| Single distribute τ | **2.5** (Phase 3 CTX; 2.0 table kept below as history) |
| Overall regression &lt;2% @ ship τ | **yes** (seen 1.38% / unseen 1.91% / fresh 1.31%) |

Track A kept for comparison only. **No retraining** in this finalize pass.

---

## Shippable (single τ) — Track B @ τ=2.0

Machine-readable: `artifacts/rerank_ctx/eval/shippable_trackB_tau2.0.json`  
Summary table: `artifacts/rerank_ctx/eval/shippable_trackB_tau2.0_summary.md`  
Selection log: `artifacts/rerank_ctx/eval/shippable_trackB_selection.json`  
Ckpt (Volume): `/artifacts/trackB_v2_continue/cross_encoder.pt`

### τ sweep (fixed across sets)

| τ | all sets reg&lt;2%? | min(CS Δpt) |
|--|--|--|
| 1.5 | no (unseen 2.32%) | 30.27 |
| **2.0** | **yes** | **29.69** ← chosen |
| 2.5 | yes | 29.31 |

### Metrics @ τ=2.0

| set | hit@1 all | vs Mozc | regression | CS ON | CS OFF | CS Δpt | non-CS hit@1 | non-CS vs Mozc |
|--|--|--|--|--|--|--|--|--|
| seen | 91.47% | +14.33 | 1.38% | 77.14% | 44.38% | +32.76 | 99.18% | −0.82 |
| unseen | 91.47% | +11.13 | 1.91% | 77.20% | 47.51% | +29.69 | 99.08% | −0.72 |
| fresh | 93.87% | +12.27 | 1.31% | 84.76% | 28.95% | +55.81 | 98.77% | −1.23 |

### Side effects @ ship τ

- **Non-CS cost vs Mozc**: −0.82 / −0.72 / −1.23 pt (seen/unseen/fresh) — ~1pt class, acceptable; raising τ further would soften overwrites but cut CS Δ (2.5 still eligible, slightly worse min CS Δ).
- **CS subset regression**: 4.40% / 6.11% / 1.61% (seen/unseen/fresh). Unseen CS reg slightly above the informal 2–5% band; overall (all) reg still &lt;2%.

### Distribute memo (align PLAN §10)

| knob | value |
|--|--|
| τ | **2.5** (Phase 3 CTX) |
| max_len | **128** |
| cand_cap | **30** (degrade → 15, keep Mozc tail) |
| context clip | shared `tools/rerank/context_clip.py` (`clean_context` / `clip_context_prev`) |
| degrade | §10 ladder: cand_cap → shorten context → blank+short max_len → Mozc-only |

---

## Gold audit (§2.5)

| item | value |
|--|--|
| Sample | `data/rerank_ctx/audit_sample_v2_300.json` (enriched POS from v2 JSONL) |
| Method | **Automated proxy** (not human double-blind): content POS, length/script QC, Sudachi reading↔gold; isolated Sudachi mismatches on pipeline-enriched rows treated as soft (e.g. 年=とし) |
| Result | **accuracy 100%** (300/300); soft notes: 6 isolated-reading mismatches accepted |
| Pass ≥95% | **yes** |
| Output | `data/rerank_ctx/audit_v2_result.json` |
| Filter tighten / reassemble | **not required** (no retrain) |

---

## Historical: small-scale pilot (pre-v2)

Date: 2026-08-12  
Environment: Docker `rocm-torch` / AMD Radeon RX 7800 XT / ROCm / `sbintuitions/modernbert-ja-70m` / max_len=128  

### Small-scale §5.1 (superseded by v2 finalize above)

**continue / ship-candidate — prefer Track B** (pilot): context Δ positive; B edged A on fresh. Full-scale v2 + single-τ finalize is authoritative for Phase 3.

### Pipeline completed (pilot)

1. Extract (Sudachi) → `data/rerank_ctx/work/extracted.jsonl` (**303830**)
2. Mozc attach → `data/rerank_ctx/work/mozc/attached.jsonl` (30 568 unique keys)
3. Ambiguous map → `data/rerank_ctx/reading_gold_map.json` (**824** ambiguous readings)
4. Assemble-small → train **20 000** (35% amb) / eval **1 000×3** (doc-level splits)
5. Track A from HF weights; Track B `--init-ckpt artifacts/rerank/modernbert70m_ce_v3`
6. `eval_contextual.py` context ON/OFF + τ sweep per model×set

### Train metrics (pilot)

| track | init | pairs | epochs | bs | max_len | elapsed |
|--|--|--|--|--|--|--|
| A | HF `modernbert-ja-70m` | 213 311 | 2 | 512 | 128 | 1240.5 s |
| B | `modernbert70m_ce_v3` | 213 793 | 2 | 512 | 128 | 1287.0 s |

Checkpoints (docker work mount):  
`/work/mozc-ai-training/artifacts/rerank_ctx/trackA_modernbert70m/`  
`/work/mozc-ai-training/artifacts/rerank_ctx/trackB_continue/`

### Eval (hit@1 @ recommended τ) — pilot / per-set τ (not distribute)

| model | set | hit@1 ctx ON | hit@1 ctx OFF | amb Δpt | τ |
|--|--|--|--|--|--|
| A | seen | 90.80% | 86.70% | +4.86 | 1.0 |
| A | unseen | 88.10% | 85.00% | +3.43 | 2.0 |
| A | fresh | 89.10% | 83.50% | +13.71 | 2.0 |
| **B** | seen | **90.40%** | 85.00% | **+8.00** | 1.0 |
| **B** | unseen | **89.60%** | 84.90% | **+7.14** | 1.5 |
| **B** | fresh | **90.60%** | 83.90% | **+14.57** | 2.0 |
| current (v3) | seen | 74.20% | 66.80% | +10.57 | 5.0 |
| current (v3) | unseen | 75.80% | 68.10% | +11.71 | 5.0 |
| current (v3) | fresh | 76.60% | 63.90% | +16.00 | 5.0 |

Ambiguous n=350 per eval set (small-scale; full-scale target ≥500).

### Gold audit (pilot sample)

- Sample written: `data/rerank_ctx/audit_sample_300.json` (+ `.tsv`) — superseded by v2 audit above.

---

## v2 Modal eval (per-set optimal τ — diagnostic only)

See `docs/reranker/reports/PHASE_CTX_REPORT_V2.md` / `artifacts/rerank_ctx/summary_v2.md`.  
**Do not ship per-set τ.** Use τ=2.0 table above.

| model | set | hit@1 all ON | CS Δpt | τ (with) |
|--|--|--|--|--|
| B | seen | 91.53% | +34.67 | 2.5 |
| B | unseen | 91.53% | +25.86 | 2.5 |
| B | fresh | 95.27% | +42.10 | 0.5 |

JSON: `artifacts/rerank_ctx/eval/eval/summary_track{A,B}_v2.json`

## Notes

- Architecture: `modernbert-ja-70m` (not ruri) for 1-variable A/B vs current CE.
- Ops: never `pkill modal run` — see `docs/guides/AGENT_BG_OPS.md`.
- Finalize scripts: `scripts/audit_gold_v2.py`, `scripts/modal_shippable_tau.py`, `scripts/_shippable_tau_worker.py`.
