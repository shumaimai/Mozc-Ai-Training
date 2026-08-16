# Phase 0.5 報告：N-best 抽出修正後ゲート

日付: 2026-08-11  
前提: `docs/reranker/reports/PHASE0_RERANK_REPORT.md`（修正前は gold∈N-best=0）

## 何を直したか

`mozc_batch` をフルパス候補化:

1. 全 segment の top-1 連結（best path）
2. segment ごとの代替 × 他 segment top-1 固定
3. `ResizeSegment` で読み全体を単一 segment 化し、その候補を追加
4. `--max_candidates` 既定 100
5. gold 比較は exact + NFKC 正規化

成果物:

- ソース: `mozc_compat/mozc_batch.cc`（bazel ツリーへ同期・rebuild 済）
- データ: `data/interim/rerank_phase0_v2/`
- split: `data/rerank_v2/{train,holdout}.jsonl`

## 再計測（n=11,128）

| 指標 | 修正前 | 修正後 |
|--|--|--|
| Mozc hit@1 | 0.0 | **0.328** |
| gold ∈ N-best | 0.0 | **0.553** |
| gold ∈ N-best かつ非1位 | — | **0.225** |
| 完璧リランク時の hit@1 上積み | 0.0 | **+0.225** |
| gold 順位中央値（入っているとき） | — | 1 |
| gold 順位平均（入っているとき） | — | 3.68 |

順位ヒスト（gold が入っている 6,156 件）:

- rank1: 3646
- rank2–3: 663
- rank4–10: 1318
- rank11+: 529

holdout (n=1669) も同様: gold∈N-best **0.556**, not_top1 **0.240**。

## ゲート判定

**GO**

条件: `gold_in_nbest >= 0.50` かつ `gold_in_nbest_not_top1 > 0.05`  
→ 両方満たす。並べ替え学習に意味のある余地あり。

残 ~45% は依然 generation/追加側（N-best 外）。学習時は:

- `gold_in_nbest=true` のグループを主に使う
- Mozc hit 成功例をアンカーとして残す
- N-best 外は別経路（候補追加）として後段で扱う

## GPU 学習

ゲートは通過。**この報告時点ではまだ `train_cross_encoder train` は起動していない。**  
開始してよければ `data/rerank_v2` を使って回す。
