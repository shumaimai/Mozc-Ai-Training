# Phase 0 報告：Mozc N-best リランカー（accept 再加工のみ）

日付: 2026-08-11  
方針: コーパス残差は使わず、既存 accept 由来 `train_mixed`（11,128）のみ。  
到達点: **データ再加工・holdout・Mozc ベースライン・学習 dry-run まで。GPU 学習は開始しない。**

---

## 結論（先に）

**このまま GPU 学習に進むべきではない。**

理由: 再取得した Mozc N-best（segment(0), max=50）に対して

- `mozc_hit1 = 0.0`
- `gold_in_nbest = 0.0`
- `headroom_hit1_if_perfect_rerank = 0.0`

つまり **並べ替えでは正解を 1 件も救えない**。リランカーを学習しても推論時に gold が候補集合に無い。

---

## 実施内容

| ステップ | 結果 |
|--|--|
| `train_mixed` → TermRecord 化 | 11,128 records / 10,818 unique readings |
| `mozc_batch` で N-best 再取得 | `data/interim/rerank_phase0/candidates.tsv` |
| rerank JSONL | `data/interim/rerank_phase0/rerank_all.jsonl` |
| split (15%, seed=42, unit=`source+reading+gold`) | train 9,459 / holdout 1,669 |
| 学習 dry-run（ペア展開のみ） | `artifacts/rerank/dry_run.json`（約 99k pairs） |
| GPU `train` | **未実行** |

コマンド:

```powershell
cd <Mozc-Ai-Trainingのチェックアウト先>
# AI 付き mozc_batch は毎キー API を叩くので、batch 中は ai_config の disable_ai/enabled を一時無効化推奨
python -m tools.rerank.prepare prepare --work-dir data\interim\rerank_phase0
python -m tools.rerank.prepare split --input data\interim\rerank_phase0\rerank_all.jsonl --out-dir data\rerank
python -m tools.rerank.train_cross_encoder dry-run --train data\rerank\train.jsonl
```

---

## ベースライン数値

全体 (`baseline_all.json`):

| 指標 | 値 |
|--|--|
| n | 11,128 |
| Mozc hit@1 | **0.0** |
| gold ∈ N-best | **0.0** |
| gold が Mozc top1 の接頭辞 | 0.534 |
| カテゴリ | place_or_facility 10,560 / literary_ruby 568 |

holdout も同様に hit@1=0 / gold_in_nbest=0。

---

## ブロッカー（2つが重なっている）

### 1. データが generation_gap 偏り
accept レビューはほぼ `action=generation_gap`。  
`export_train` も「gold が Mozc top-K に既にある行はスキップ」するため、`train_mixed` は **Mozc 候補外の正解** に寄っている。  
→ 純粋リランクの学習対象（「gold は N-best 内だが top1 ではない」）がほぼ無い。

### 2. `mozc_batch` が `conversion_segment(0)` のみ
長めの地名などは複数セグメントに割れ、TSV には先頭セグメント候補しか出ない。  
例: `朱雀大路` に対し top が `朱雀` など。  
`gold_startswith_mozc_top1 ≈ 53%` は「部分一致は多いがフル表記は候補に載らない」ことを示す。

加えて、現行ビルドの `mozc_batch.exe` は AIRewriter 付きで、AI 有効だと毎キー API 呼び出しになり極端に遅い。Phase 0 実行時は `ai_config.json` の `disable_ai/enabled` を一時オフにした。

---

## 作った成果物

- `tools/rerank/prepare.py` — prepare / split / baseline
- `tools/rerank/train_cross_encoder.py` — dry-run + train（train は未実行）
- `data/rerank/train.jsonl`, `holdout.jsonl`, `baseline_split.json`
- `docs/reranker/plans/RERANK_HOOK.md` — IME 接続点
- 本ファイル

---

## GPU 学習に進む前に必要なこと（次のゲート）

どれか（推奨順）:

1. **N-best 抽出を直す**（最優先）  
   - フルパス結合候補を出す / 単一セグメント強制 / 全 segment の候補を扱える形にする  
   - AI なしの `mozc_batch` をビルドし直す
2. **リランク可能なデータだけ残す**  
   - `gold_in_nbest=true` かつ `mozc_hit1=false` の集合が十分あること（目安: 数千グループ）
3. それでも足りなければ（accept のみ制約を緩める場合）  
   - 候補追加（generation）経路を Phase 1 に前倒し — ただし今回の「並べ替え専用」からは外れる

ゲート条件案:

- holdout 上で `gold_in_nbest >= 0.3` かつ `miss_recoverable_by_rerank` が意味のある値  
- その上で Mozc hit@1 ベースラインを再計測してから `train_cross_encoder train` を許可

---

## ONNX 精度崩壊の原因・修正・精度そろえ後の速度比較（2026-08-11）

前提: 学習済み `modernbert70m_ce`、holdout `data/rerank_v2`（1669）、マージンゲート。再学習はしていない。

### 1. パリティ結論（原因は1つ）

**崩壊原因 = post-training INT8 量子化（エクスポート自体ではない）。**

同一 200 グループ / 3037 スコア（`artifacts/rerank/parity_scores.json`）:

| 比較 | Spearman | Pearson | MAE | std(x)→std(y) |
|--|--|--|--|--|
| PT fp32 vs ONNX fp32 | **1.000** | **1.000** | **2e-6** | 1.44→1.44 |
| PT fp32 vs ONNX int8（dynamic 初期） | 0.837 | 0.776 | 1.39 | 1.44→**0.51** |
| PT fp32 vs ONNX int8（static QDQ） | **-0.23** | -0.13 | 2.71 | 1.44→**0.13** |
| PT fp32 vs ONNX int8（MatMul/Gemm-only） | 0.418 | 0.269 | 2.18 | 1.44→**0.11** |

- ONNX fp32 は PyTorch と数値一致 → pooling / logits / tokenizer は正しい。
- int8 はスコア分散が潰れ、候補間マージンが消える → `τ=2.0` では上書きがほぼ起きず hit が Mozc 付近へ落ちる。
- 試した修正（dynamic per-channel、static calib、quant_pre_process、MatMul限定）いずれも **±1pt 目標を満たすパリティは回復できず**。QAT なしの post-training int8 はこの ModernBERT-Ja 70m クロスエンコーダに不適、と切り分けた。

### 2. バックエンド別 τ（退行≤2%）

`artifacts/rerank/margin_policy.json`（backends 分離）:

| backend | 推奨 τ | hit@1 | vs Mozc | 退行 | 回復 |
|--|--|--|--|--|--|
| pytorch_fp32 | **2.0** | 44.5% | +13.4pt | 1.15% | 19.9% |
| onnx_fp32 | **2.0** | 44.5% | +13.4pt | 1.15% | 19.9% |
| onnx_int8 | 0.25 | **31.2%** | **+0.0pt** | 0.0% | 0.0% |

int8 は τ を下げてもランキングが壊れているため、Mozc と同着までしか戻らない。

### 3. 精度を揃えた速度比較（batch・topK無し）

`artifacts/rerank/eval_cpu_latency_matched.json`（CPU、推奨 τ）:

| backend | hit@1 | 退行 | p50 | p95 |
|--|--|--|--|--|
| pytorch_fp32 | 44.5% | 1.15% | 1910 ms | 2277 ms |
| **onnx_fp32** | **44.5%** | **1.15%** | **91 ms** | **300 ms** |
| onnx_int8 | 31.2% | 0.0% | 67 ms | 202 ms |

→ **精度を保ったまま速いのは ONNX fp32（batch）**。int8 は速いがリランカーとして死んでいる。

### 4. topK について（`gold_rank_dist.json`）

Mozc miss かつ gold∈N-best（回復対象）の順位カバー:

| K | カバー率 |
|--|--|
| 5 | **39.4%** |
| 8 | 66.1% |
| 10 | 79.3% |
| 15 | 94.5% |
| **16** | **95%**（95%点） |

K=5 は回復ケースの 6 割を切る。batch+ONNX fp32 が p50≈91 ms なので **topK は原則不採用**。速度不足時のみ K≈16 を検討。

### 5. 成果物

- `parity_scores.json` / `parity_scores_before_fix.json`
- `gold_rank_dist.json`
- `margin_policy.json`（backend 別 τ）
- `eval_cpu_latency_matched.json`（旧計測は `eval_cpu_latency.json` に `previous_*` として保持）
- ツール: `tools/rerank/parity_check.py`, `backend_eval.py`, 更新 `export_onnx.py`

### 6. 次の一手（データに基づく判断）

**IME 統合 / Phase 2 データ拡張へ進んでよい。** 推論本線は **batch + ONNX fp32 + τ=2.0（topKなし）**。  
int8 が必須なら QAT か、より量子化耐性のある小型モデル（xsmall/tiny）の再学習が必要で、現状の post-training int8 を本線にしない。p95≈300 ms が体感で足りなければ、次はモデル縮小より先に ORT 最適化（graph fusion / スレッド調整）と候補数の実測分布を見る。

---

## 速度短縮ワークストリーム（`docs/reranker/tasks/NEXT_TASK_LATENCY.md`・int8 なし）

前提: 上記 ONNX fp32 本線。再学習なし。計測は Docker `rocm-torch` CPU（12 論理コア）。旧基準は CPU **p50 91ms / p95 300ms**。

### 1. トークン長（`token_len_dist.json`）

holdout 500 グループ / 7317 ペア:

| | p50 | p95 | p99 | max |
|--|--|--|--|--|
| トークン長 | 21 | 32 | 36 | **46** |

`frac_le`: 32≈95.7% / 40≈99.8% / **48=100%**。学習・評価の `max_len=128` は既に過剰。動的パディングのため `max_len` 短縮だけではレイテンシ差はノイズ程度。**配布は品質安全側で `max_len=48`**（hit 不変）。

### 2. ORT スレッド（品質中立）

| 設定 | p50 | p95 |
|--|--|--|
| intra=1 | 189 | 465 |
| intra=3 | 79 | 180 |
| intra=6 | 59 | 142 |
| **intra=12** | **57** | **119** |
| intra=6 inter=2 | 62 | 134 |

**intra=CPU数・inter=1・`ORT_ENABLE_ALL`・`OMP_NUM_THREADS=1`**（ORT と OpenMP の二重化を避ける）。1 変換 = 候補を 1 バッチ 1 forward（確認済み）。

### 3. p95 の主因（`latency_vs_ncand.json`）

- 候補数↔ms の Pearson **0.90**。遅延上位 5% の平均候補数 ≈38（全体平均 ≈15）。
- ウォーム後 cold 初回は ~26–40ms 程度で、尾は主に **候補数**。
- holdout 候補数: mean 14.9 / p95≈29 / max 76。

### 4. 候補数上限（topK とは別・精度検証つき）

Mozc top1 を必ず残す cap。**静か再計測**（5×250 中央値、`latency_stable_bench_recheck.json`。前回はゲーム負荷で汚染の疑い）:

| 構成 | hit@1 | vs baseline | p50 med | p95 med |
|--|--|--|--|--|
| max_len=48 intra=12 **no cap** | 44.58% | +0.06pt | **42.7** | **90.3** |
| **max_len=48 intra=12 cap=30** | **44.58%** | **+0.06pt** | **42.1** | **78.6** |
| max_len=48 intra=8 cap=30 | 44.58% | +0.06pt | 49.4 | 94.6 |

3 構成とも目標（p50≤50 / p95≤120）達成。cap=30 は p95 尾をさらに切る（hit 不変）。

### 5. 最終配布構成（更新）

| 項目 | 値 |
|--|--|
| backend | **ONNX fp32**（int8 なし） |
| max_len | **48** |
| ORT threads | **intra=12 / inter=1**、OMP=1 |
| batch | 1 変換あたり全候補を 1 forward |
| topK | **なし** |
| candidate_cap | **30**（Mozc top1 保証・推奨。無しでも目標達成） |
| τ | **2.0** |
| hit@1 | **44.58%**（Mozc 31.2% → +13.4pt 前後） |
| CPU レイテンシ | **p50 ≈42ms / p95 ≈79ms**（旧 91 / 300） |

### 6. 目標達成と 30m 判断

- 目標 p50≤50 / p95≤120: **両方達成**（静か再計測）。
- **30m 再学習は不要。**
- **Phase 2（品質底上げ・候補ソーシング）へ進んでよい。**

成果物: `token_len_dist.json`, `latency_vs_ncand.json`, `latency_pack_report.json`, `latency_stable_bench.json`, `latency_stable_bench_recheck.json`, `tools/rerank/latency_pack.py`

---

## 一言

Phase 0 は「問題を可視化する」役割を果たした。  
**accept 再加工だけでは、現行 N-best 定義のままリランカーを学習しても改善余地が 0。**  
次はコーパス増やす前に、**Mozc 候補の取り方（フル表記が載る形）を直す**のが正しい。
（注: 上記「一言」は Phase0 初期ゲート時点の記述。N-best 抽出修正後の学習・ONNX・レイテンシ短縮は本節までの後続作業。）

---

## Phase 2 kickoff 追記（2026-08-12）

- AB: holdout B≈**44.4%** → `sourcing_gate=GO_try_local_dict`
- v3 train: **34,348** / holdout **1,669**（v2 固定）→ `data/rerank_v3/`
- Colab: `docs/guides/COLAB_TRAIN_GUIDE.md` + `docs/guides/COLAB_PHASE2_TRAIN.ipynb`
- 短い kickoff: `docs/reranker/reports/PHASE2_KICKOFF.md` / タスク: `docs/reranker/tasks/NEXT_TASK_PHASE2.md`
