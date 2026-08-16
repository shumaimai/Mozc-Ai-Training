# 次タスク指示: リランカーのレイテンシ短縮（int8 なし）

対象: リポジトリ作業エージェント
前提: `docs/reranker/plans/PLAN_RERANKER.md`（§4.4, §4.5）, `docs/reranker/reports/PHASE0_RERANK_REPORT.md`, `docs/reranker/tasks/NEXT_TASK_ONNX_PARITY.md`

---

## 背景（確定事項）

Mozc N-best を並べ替えるクロスエンコーダ（base `cl-nagoya/ruri-v3-pt-70m`）は品質・速度の両面で本命構成が確定した：

- **配布構成: ONNX fp32・batch・topK なし・τ=2.0**
- holdout 1669: hit@1 **44.6%（Mozc 31.2% → +13.4pt）／退行 1.15%**
- CPU レイテンシ: **p50 91ms / p95 300ms**（Windows 実測）
- **int8 は不採用**（PTQ で順位崩壊。PT↔int8 Spearman 0.42〜0.84、QAT 無しでは回復不能）
- **topK は不採用**（回復対象の gold が Mozc 下位に多く、K=5 で 39%・K=16 で 95% しかカバーできず精度を削る）

品質は達成済み。**残る課題は p95=300ms を実 IME 体感（サクサク感）に近づけること**だけ。int8 が使えないので、別レバーで攻める。

## ゴール

CPU で **精度を fp32 と同等（hit@1 −1pt 以内）に保ったまま**、レイテンシを短縮する。
到達目標の目安: **p50 ≤ 50ms / p95 ≤ 120ms**（達成できない場合は 30m 縮小の要否をデータで判断）。

## タスク手順（安い順・すべて品質中立を優先）

### 1. max_length を実長に合わせて短縮（最優先・品質中立の可能性大）
- 現在の推論の `max_length` を確認（512 のままなら大きな無駄）。
- `data/rerank/holdout.jsonl` の実入力（読み＋文脈＋候補）のトークン長分布を出す（p50/p95/max）。
- p99 をカバーする最小の `max_length`（例: 64 / 96 / 128）に設定し、**hit@1 が変わらないこと**を確認しつつレイテンシを再計測。
- 出力: `artifacts/rerank/token_len_dist.json` と、max_length 別の (hit@1, p50, p95) 表。

### 2. ONNX Runtime の実行設定チューニング（品質中立）
- `intra_op_num_threads` / `inter_op_num_threads` を CPU コア数に合わせて調整。
- `graph_optimization_level=ORT_ENABLE_ALL`、`SessionOptions` の最適化を有効化。
- 可能なら 1 変換内の全候補を **1 バッチ 1 forward** にまとめているか再確認（逐次呼び出しが残っていないか）。
- 設定別の p50/p95 を記録。

### 3. p95 の尾の原因を特定
- 変換ごとの候補数とレイテンシの相関を出す（`artifacts/rerank/latency_vs_ncand.json`）。
- p95 を作っているのが「候補数の多い変換」か「初回ロード/アロケーション」かを切り分ける。
- 候補数依存なら、注入・N-best の**候補数上限**（精度を損なわない範囲）を提案。ウォームアップで初回コストを外す。

### 4. （1〜3 で目標未達のときのみ）30m へ縮小して再学習
- `cl-nagoya/ruri-v3-pt-30m` をベースに、同じ学習データ・手順でクロスエンコーダを再学習。
- **30m 用に τ を再スイープ**（バックエンド/モデルごとに τ は別物）。
- 70m fp32 と 30m fp32 を、**同一 holdout で hit@1・退行・p50・p95 を並べて比較**。
- 精度低下が許容範囲（例: hit@1 −2pt 以内）で速度が目標に届くなら 30m を配布候補にする。届かない/精度が落ちすぎるなら 70m 据え置きで p95 の現実的な上限を報告。

## 使うファイル
- 推論/評価: `tools/rerank/eval_cross_encoder.py`, `tools/rerank/margin.py`
- 学習（手順4のみ）: `tools/rerank/train_cross_encoder.py`
- データ: `data/rerank/holdout.jsonl`
- チェックポイント/ONNX: `artifacts/rerank/`（実在するものを確認）

## 成功基準
- max_length 短縮・runtime 設定・p95 調査の結果が数値で残り、`docs/reranker/reports/PHASE0_RERANK_REPORT.md` に追記されている。
- **精度を fp32 と同等（−1pt 以内）に保ったまま**の最速構成が確定している。
- 目標（p50 ≤50ms / p95 ≤120ms）に対する到達状況が明記され、未達なら 30m 縮小の要否がデータで結論づけられている。

## 制約・注意
- **int8 / QAT には戻らない**（§4.4 で不可と確定済み）。
- **topK を既定にしない**（gold を削る。手順3の候補数上限は精度検証つきで別物）。
- **品質を落とす変更（max_length 過小・候補数上限の過剰）を、速度だけ見て採用しない。** 必ず hit@1 と退行を同時に確認。
- τ はモデル/バックエンドごとに再スイープする。fp32-70m の τ=2.0 を 30m に流用しない。
- 旧計測結果は上書きせず残す。

## 成果物
- `artifacts/rerank/token_len_dist.json`, `latency_vs_ncand.json`, 更新した `eval_cpu_latency.json`（別キーで新旧比較）
- （手順4を実施した場合）30m チェックポイントと 70m/30m 比較表
- `docs/reranker/reports/PHASE0_RERANK_REPORT.md` に「速度短縮ワークストリームの結果と最終配布構成」を追記
- 最後に、Phase 2（品質底上げ・候補ソーシング）へ進んでよいか、モデル縮小が必要かの一言判断
