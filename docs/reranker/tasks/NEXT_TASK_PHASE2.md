# 次タスク指示: Phase 2 — 品質底上げ + 候補ソーシング

対象: リポジトリ作業エージェント  
前提: `docs/reranker/plans/PLAN_RERANKER.md` §3・§5.2・§6、`docs/reranker/reports/PHASE0_RERANK_REPORT.md`  
現状: Phase 1 実証済み（hit@1 31.2%→44.6%、τ=2.0、ONNX fp32 p50≈42/p95≈79）。学習実体は **`sbintuitions/modernbert-ja-70m`** → `artifacts/rerank/modernbert70m_ce`。

**学習環境（このラン）**: **ローカル ROCm が主**（docker `rocm-torch` / WSL `$HOME/work/mozc-ai-training`）。**Colab は一旦スキップ（deferred）**。起動スクリプト: `scripts/_run_phase2_train_rocm.sh` + `scripts/_train_rerank_gpu_v3.py`（`expandable_segments` は使わない）。

---

## ゴール

1. **品質**: 学習グループを 30k〜（第一目標）へ拡張し、**同一 holdout（rerank_v2）** で hit@1 / 退行を更新。退行 <2% を維持。
2. **候補ソーシング**: holdout の **B（gold∉N-best）** を測り、辞書注入の試験を 1 源だけ行う。
3. **GPU 再学習**: まずローカル ROCm。Colab 手順書はあるが **今は使わない**。

---

## 手順（この順）

### A. ゲート計測（ローカル・即時）

```bash
python -m tools.rerank.phase2_expand ab \
  --input data/rerank_v2/holdout.jsonl \
  --out artifacts/rerank/phase2_ab_analysis.json
```

- **A** = Mozc miss かつ gold∈N-best（リランクの仕事）
- **B** = gold∉N-best（辞書注入の仕事）
- B が holdout の ≥25% ならソーシング GO（現状見込み ~44%）

### B. 学習データ拡張（ローカル・Mozc 再実行不要）

既存 `data/interim/mozc_batch/{aozora,wikidata,japanpost}/classify_in.jsonl` を再利用:

```bash
python -m tools.rerank.phase2_expand expand \
  --base-train data/rerank_v2/train.jsonl \
  --holdout data/rerank_v2/holdout.jsonl \
  --out-dir data/rerank_v3 \
  --extra-groups 25000
```

- **holdout は v2 と同一のまま**（比較公正）
- CE 用は `gold_in_nbest` のみサンプリング（残差/アンカー比を調整）
- 出力: `data/rerank_v3/{train,holdout}.jsonl`, `expand_summary.json`
- 目視 QC: train から 30〜50 件サンプルし、読み↔表記が怪しければフィルタ強化

### C. ローカル ROCm 再学習（主経路・今ここ）

```bash
# WSL
bash scripts/_run_phase2_train_rocm.sh 1280
# log: $HOME/work/mozc-ai-training/artifacts/rerank/train_v3_run.log
# out: artifacts/rerank/modernbert70m_ce_v3
```

- モデル: `sbintuitions/modernbert-ja-70m`
- `--require-gold-in-nbest --fp16 --grad-checkpointing --max-len 128`
- batch は 1280 から OOM backoff（`_train_rerank_gpu_v3.py`）
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` は設定しない**（この ROCm では invalid）
- Colab（`COLAB_*`）は **deferred** — 触らない

### C'. Colab（deferred）

手順書は残置のみ。この Phase 2 ランでは使わない。

### D. 評価（ローカル CPU/GPU）

同一 holdout で:

```bash
python -m tools.rerank.eval_cross_encoder ...  # 既存手順 / scripts/_run_eval.sh 系
# margin τ は 70m 新 ckpt で再スイープ（流用禁止）
```

合格目安: hit@1 ≥ Phase1（44.6%）かつ退行 ≤2%。伸びなければデータ QC / 比率を見直して再学習。

### E. 候補ソーシング試験（B が GO のとき） — DONE（リーク上界）→ 正直再測定へ

1. B の category/source を見て、まず **wikidata places**（既にリポ内）を 1 源だけ。✓
2. Mozc N-best と diff し、**リスト末尾に注入**。注入候補には **より厳しい τ**。✓
3. 旧 trial（同系 wikidata）: hit@1 **68.12%** → **リーク上界。ターゲットではない。**
4. **本番 Mozc-diff バンドル + anti-leak 再測定** → `docs/reranker/tasks/NEXT_TASK_DICT_BUNDLE.md` ✓
   - production: `artifacts/dict/production/`（170k entries）
   - honest: source-holdout B coverage **0%** / japanpost cross **0.27%** / hit@1 ≈ no-inject **45.9%**
   - audit: honest≈0 は **mismatch でありバグではない**（`phase2_b_zero_audit.json`）
   - taxonomy: この holdout では **辞書深掘り STOP**（`phase2_b_manual_taxonomy.md`）；NEologd 即着手しない
5. **Phase 2 辞書 / CE scale-up は pause。主レバーは Phase 3** → `docs/reranker/tasks/NEXT_TASK_PHASE3.md`
   - 出荷勝ち筋 = no-inject rerank ≈45.9%（68% はリーク上界のみ）
   - 次: Mozc 経路へ rerank 統合 + 実変換ログ

---

## Colab 運用の鉄則（再掲）

- torch / torchvision / torchaudio は**入れ替えない**
- 依存追加後は**一度だけランタイム再起動**
- `/content` は消える → **Drive 保存**
- 赤文字でも Traceback が無ければ警告（`docs/guides/COLAB_TRAIN_GUIDE.md` §2.5）

---

## 成功基準

- [ ] `phase2_ab_analysis.json` があり、ソーシング GO/DEFER が明記
- [ ] `data/rerank_v3/train.jsonl` が ≥ ~30k グループ（第一到達）
- [ ] Colab ~~または~~ **ローカル ROCm** で v3 学習が完了し、holdout 比較表が `PHASE0` or 新 `PHASE2` 報告に追記（Colab deferred）
- [x] （GO 時）辞書注入のオフライン試験結果が数値で残る（`artifacts/rerank/phase2_b_inject_trial.json`）
- [ ] 配布構成は引き続き **ONNX fp32・cand_cap=30・max_len=48・τ 再スイープ・int8 なし**

## 制約

- holdout を勝手に作り直して「改善した」ことにしない（v2 holdout 固定）
- int8 / topK 既定に戻さない
- 辞書を全部学習データに叩き込まない（暗記化）
- B を CE に入れても `--require-gold-in-nbest` で落ちるだけ → 注入と分業
