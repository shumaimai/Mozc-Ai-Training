# 次タスク指示: ONNX int8 の精度崩壊の原因究明と修正

対象: リポジトリ作業エージェント
前提ドキュメント: `docs/reranker/plans/PLAN_RERANKER.md`, `docs/reranker/reports/PHASE0_RERANK_REPORT.md`, `docs/reranker/plans/RERANK_HOOK.md`

---

## 背景（現状の要約）

Mozc N-best を並べ替えるクロスエンコーダ・リランカー（base: `cl-nagoya/ruri-v3-pt-70m`）を学習済み。
holdout 1669 件で、fp32・全候補・マージン `τ=2.0` のとき **hit@1 44.6%（Mozc 比 +13.4pt / 退行 1.15%）** が出ている。これが本来の実力。

ところが速度最適化の計測で、**速い構成ほど精度が崩壊**した：

| 構成 | hit@1 | vs Mozc | 退行 | CPU p50 |
|--|--|--|--|--|
| batch のみ（fp32・全候補） | 44.5% | +13.4pt | 1.2% | 1820 ms |
| topK のみ | 35.7% | +4.5pt | 1.0% | 2970 ms |
| ONNX int8 のみ | 32.7% | +1.6pt | 0.2% | 75 ms |
| batch+topK+ONNX | 31.7% | +0.5pt | 0.0% | 25 ms |

問題は「速いが精度が消えた」こと。数十 ms に収まる ONNX 系はすべて +0.5〜+1.6pt しかなく、リランカーがほぼ機能していない。

## 仮説（検証すべき対象）

1. **ONNX int8 の 12pt 低下はバグ/設定ミスの可能性が高い。** 通常 int8 量子化の劣化は 1pt 未満。Mozc 素（31.2%）まで戻り、かつ退行率も 0.2% へ落ちている（＝上書きがほぼ起きていない）。これは「スコアがフラット化 or スケール変化で `τ=2.0` が効かなくなった」サイン。GPU 版 ONNX も同じ 32.7% なので、デバイス依存ではなく **ONNX/量子化パス自体**の問題。
   - 疑う順: (a) `τ` を int8 スコアで再スイープしていない → 尺度不一致、(b) ONNX エクスポートの出力ズレ（pooling / logits / sigmoid の取り違え、tokenizer 不一致）、(c) int8 キャリブレーション不良。
2. **topK の 9pt 低下は設計の衝突。** リランカーが直すのは gold が Mozc 下位に埋もれたケース。上位 K で切ると gold ごと捨てている。topK は本質的にリランクとケンカする。

## ゴール

- ONNX int8 パスで **fp32 と同等（±1pt 以内）の hit@1** を、数十 ms オーダーの CPU レイテンシで達成する。
- 目標構成は **batch + ONNX（topK 無し）**。原因が潰れれば CPU で概ね p50 ~65ms / p95 ~140ms が狙える見込み。
- 精度を回復できない場合は、その原因を切り分けて報告（推測で先に進めない）。

## タスク手順

### 1. 数値パリティ検証（最優先）
同一の holdout 入力（`data/rerank/holdout.jsonl` から代表 100〜300 例）に対し、次の3系統で **候補ごとの生スコアを出力して比較**する：
- PyTorch fp32（現行 `tools/rerank/eval_cross_encoder.py` の推論経路）
- ONNX fp32（量子化前）
- ONNX int8

出力: `artifacts/rerank/parity_scores.json`（例ごとに `{reading, candidate, score_pt_fp32, score_onnx_fp32, score_onnx_int8}`）。
判定:
- **ONNX fp32 が PyTorch fp32 と一致しない** → エクスポートのバグ。export スクリプトを修正（出力ノード・pooling・sigmoid 有無・tokenizer/特殊トークン・max_length を一致させる）。
- **ONNX fp32 は一致するが int8 だけズレる** → 量子化の問題。キャリブレーションデータ・量子化対象レイヤ・avx2/arm64 の選択を見直す。
- **スコアは概ね一致するが分布/スケールがずれている** → τ の再調整で解決（手順3）。

相関（Spearman）と平均絶対差を必ず数値で出すこと。

### 2. 原因の切り分けを1行で確定
パリティ結果から、崩壊が「エクスポート実装」「int8 量子化」「τ 尺度」のどれかを明記する。以降の修正はそこだけに絞る。

### 3. バックエンド別に τ を再スイープ
`τ=2.0` は fp32 で調整した値。**ONNX fp32 / ONNX int8 それぞれで τ を独立にスイープ**し直す（既存の `tools/rerank/margin.py` の `--tau-sweep` を利用）。
出力: バックエンド別の `recovery / regression / hit@1 vs τ` 表。退行 ≤2% 制約下での最良 τ を各バックエンドで選ぶ。
`artifacts/rerank/margin_policy.json` はバックエンド別に持てる形へ更新（fp32 用と int8 用を分離）。

### 4. 精度を揃えて再ベンチ
修正後、**同一 hit@1（≈44%）の条件**で速度を比較し直す。今の速度表は精度がバラバラでフェアでない。
計測構成:
- batch + ONNX fp32（topK 無し）
- batch + ONNX int8（topK 無し）
CPU / GPU の p50・p95・hit@1・退行を再取得し、`artifacts/rerank/eval_cpu_latency.json` を更新（旧結果は上書きせず別キー or 別ファイルで残す）。

### 5. topK の扱い
- 回復ケースにおける **gold の Mozc 順位分布**を出す（`artifacts/rerank/gold_rank_dist.json`）。
- batch+ONNX が数十 ms に収まるなら **topK は原則不採用**（精度を削るだけ）。どうしても速度が足りない場合のみ、順位分布を根拠に gold の 95% をカバーする K を提案する。勝手に小さい K を既定にしない。

## 使うファイル
- 学習/評価: `tools/rerank/train_cross_encoder.py`, `tools/rerank/eval_cross_encoder.py`, `tools/rerank/margin.py`
- データ: `data/rerank/{train,holdout}.jsonl`
- チェックポイント: `artifacts/rerank/`（`ruri70m_ce` または `modernbert70m_ce`。実在するものを確認して使う）
- 既存成果物: `artifacts/rerank/eval_holdout_gpu.json`, `eval_cpu_latency.json`, `margin_policy.json`

## 成功基準
- `parity_scores.json` に3系統のスコア比較（相関・平均絶対差つき）がある。
- 崩壊の原因が1つに特定され、修正されている。
- ONNX int8（batch, topK 無し）で **hit@1 が fp32 比 −1pt 以内**、かつ CPU p50 が数十 ms 台。
- バックエンド別 τ と、精度を揃えた速度比較表が `docs/reranker/reports/PHASE0_RERANK_REPORT.md` に追記されている。

## 制約・注意
- **再学習しない。** 今回は推論・エクスポート・量子化・τ の問題。学習コードやデータには触らない。
- **精度を揃えずに速度だけで勝敗を判断しない。** +0.5pt はリランカーが死んでいる状態であり「速くて良い」ではない。
- **τ はバックエンドごとに別物**として扱う。fp32 の τ を int8 に流用しない。
- モデルやベースを勝手に差し替えない。
- 旧計測結果は消さず、比較できる形で残す。

## 成果物
- `artifacts/rerank/parity_scores.json`, `gold_rank_dist.json`, 更新した `margin_policy.json`, `eval_cpu_latency.json`
- `docs/reranker/reports/PHASE0_RERANK_REPORT.md` に「ONNX 精度崩壊の原因・修正・精度そろえ後の速度比較」を追記
- 最後に、次に進める（Phase 2 データ拡張 / IME 統合）か、まだ速度が足りずモデル縮小（xsmall/tiny）を検討すべきか、データに基づく一言判断
