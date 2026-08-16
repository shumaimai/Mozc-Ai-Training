# Phase 2 kickoff（2026-08-12）

前提: Phase 1 実証済み（hit@1 31.2%→44.6%、ONNX fp32、τ=2.0）。  
詳細タスク: `docs/reranker/tasks/NEXT_TASK_PHASE2.md`。

## ゲート結果（holdout = rerank_v2 固定）

| 指標 | 値 |
|--|--|
| holdout n | 1,669 |
| hit / A / B | 31.6% / 24.0% / **44.4%** |
| B among miss | 64.9% |
| B 内訳 | place_or_facility 673（wikidata） / literary_ruby 68（aozora） |
| sourcing_gate | **GO_try_local_dict** |
| holdout gold∈N-best | 0.556（ゲート GO） |

出典: `data/rerank_v3/expand_summary.json` → `ab_holdout`（AB 単体は `python -m tools.rerank.phase2_expand ab` でも可）。

## 学習データ v3

| 項目 | 値 |
|--|--|
| train | **34,348**（base 9,459 + added 24,889） |
| holdout | **1,669**（= `data/rerank_v2/holdout.jsonl` 固定） |
| train slices | A 15,746 / hit 14,371 / B 4,231（CE は `--require-gold-in-nbest` で B 除外） |
| パス | `data/rerank_v3/{train,holdout}.jsonl` + `expand_summary.json` |

## 学習（主経路 = ローカル ROCm / Colab deferred）

```bash
bash scripts/_run_phase2_train_rocm.sh 1280
# monitor: scripts/_monitor_phase2_train.sh
# log:     artifacts/rerank/train_v3_run.log
# out:     artifacts/rerank/modernbert70m_ce_v3
# eval:    scripts/_run_phase2_eval_v3.sh   # after train DONE
```

- docker `rocm-torch` / `sbintuitions/modernbert-ja-70m` / batch≈1280 + OOM backoff
- **do not** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
- Colab notebooks are deferred for this run

合格目安（同一 holdout）: hit@1 ≥ Phase1 **44.6%**、退行 ≤**2%**（Phase1 regress 1.15% 参考）。τ は新 ckpt で再スイープ。

## 候補ソーシング（B-inject）

### リーク上界（旧 trial・ターゲットではない）

ソース: **wikidata** のみ（`classify_in`、Mozc-diff なし）。成果物: `artifacts/rerank/phase2_b_inject_trial.json`。

| 指標 | no-inject (τ=2.5) | inject best (τ_inj=**3.0**) |
|--|--|--|
| B rescue（gold∈list） | — | **90.6%**（671/741） |
| hit@1 | 45.96% | **68.12%**（+22.2pt） |
| regress | 1.73% | **1.73%** |

**68.12% は same-family リーク上界であり目標値ではない。**

### 本番 Mozc-diff 辞書 + 正直再測定（DONE 2026-08-12）

詳細: `docs/reranker/tasks/NEXT_TASK_DICT_BUNDLE.md`。  
バンドル: `artifacts/dict/production/`（wiki+japanpost Mozc-diff、170,232 entries）。  
**リランクデータ scale-up は pause。辞書同梱が主レバー。**

| Protocol | B coverage | hit@1 | regress | 判定 |
|--|--|--|--|--|
| source-holdout（gold 除外） | **0.0%** | 45.84% | 1.73% | NO_SHIP（一般化） |
| cross-source japanpost→wiki holdout | **0.27%** | 45.96% | 1.73% | NO_SHIP |
| same-family Mozc-diff（参照） | 90.6% | 68.84% | 1.73% | リーク上界 |

- 推奨（更新）: accept-holdout 向け辞書研磨は **STOP**。audit+taxonomy 済み → **Phase 3**（`docs/reranker/tasks/NEXT_TASK_PHASE3.md`）。trial インフラは残置可。品質クレームに 68% を使わない。
- NEologd: 実ログ後・ライセンス後のみ。今は着手しない。
- 配布構成は従来どおり ONNX fp32・cand_cap=30・max_len=48・τ=2.5・int8 なし。

## 制約（再掲）

- holdout を作り直さない（v2 固定）
- int8 / topK なし
- 辞書を学習データに全部叩き込まない
