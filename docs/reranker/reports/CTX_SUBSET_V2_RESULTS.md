# context_sensitive 再定義＋文脈整形の結果

作成: 2026-08-12  
前提: `docs/reranker/tasks/NEXT_TASK_CTX_SUBSET_FIX.md` / `docs/reranker/plans/PLAN_CONTEXTUAL_RERANKER.md`  
Mozc: **再実行なし**（`data/rerank_ctx/work/mozc/attached.jsonl` 303,830 行を再利用。fresh 拡充分は既存 `candidates.tsv` を join）

## Track B base（1行）

`sbintuitions/modernbert-ja-70m` continued from `artifacts/rerank/modernbert70m_ce_v3`（Track A/B 同一アーキ）

## 実装

| 項目 | 場所 |
|--|--|
| 共有 `clean_context` | `tools/rerank/context_clip.py`（学習生成・`phase3_hook` 推論で同一） |
| `context_sensitive` 定義/付与 | `tools/rerank/build_ctx_dataset.py`（`build_context_sensitive_map` / `annotate_context_sensitive`） |
| 組立 | `python -m tools.rerank.build_ctx_dataset assemble-v2` |
| 旧 `ambiguous` | 残置（比較用）。主フラグは `context_sensitive` |

### context_sensitive 読み条件（AND）

- total ≥ 8（空 `clean_context` 行は集計除外）
- top gold share < 0.70
- 漢字 gold（`gold≠reading`）が ≥2、各頻度 ≥ 3
- content POS のみ（名詞/動詞/形容詞/形状詞/副詞/連体詞；助詞・助動詞・接続詞・記号・補助記号・フィラー・感動詞除外）

行フラグ: 上記読みに属し、かつ漢字 content・`gold≠reading`・整形後文脈非空。

## 成果物

- `data/rerank_ctx/train_v2.jsonl`
- `data/rerank_ctx/eval_seen_v2.jsonl`
- `data/rerank_ctx/eval_unseen_v2.jsonl`
- `data/rerank_ctx/eval_fresh_v2.jsonl`
- `data/rerank_ctx/assemble_summary_v2.json`
- `data/rerank_ctx/context_sensitive_map.json`
- 旧 `train.jsonl` / `eval_*.jsonl` は未上書き

CS 読み定義は **元 attached 303,830** に固定（`--cs-map-from`）。fresh は RSS＋本文取得で拡充し、既存 N-best を join（新規読みの Mozc 再実行なし）。

## 主要統計（v2）

| set | n | CS | CS% | CS distinct readings | median top_gold_share (CS) | newline residue | gold==reading (CS) |
|--|--|--|--|--|--|--|--|
| train | 44306 | **15203** | 34.3% | **488** | **0.550** | **0.0** | **0.0** |
| eval_seen | 1500 | **525** | 35.0% | 246 | 0.550 | 0.0 | 0.0 |
| eval_unseen | 1500 | **522** | 34.8% | 208 | 0.550 | 0.0 | 0.0 |
| eval_fresh | 1500 | **525** | 35.0% | 92 | 0.582 | 0.0 | 0.0 |

その他:

- 文書単位分割維持；fresh↔train リーク 0；seen 行重複 0
- CS 文脈非空率 100%；マークアップ残存 0
- train に少量 corrupted anchor（≈2%）
- 旧 `ambiguous` は train で約 48%（参考）。学習・評価の主役は CS

## 成功基準チェック

- [x] CS 非退化（median share < 0.6、漢字多択、機能語除外、gold≠reading）
- [x] train CS ≥ 15,000 / distinct ≥ 200
- [x] 各 eval CS ≥ 500
- [x] 改行/マークアップ残存 ≈ 0%、CS 文脈非空 100%
- [x] 文書分割・fresh 非混入

## Go / No-Go

**GO — Track A/B 学習に進んでよい。**

context_sensitive の量・非退化性・文脈整形・分割いずれも成功基準を満たした。評価の主指標は `context_sensitive` サブセットでの hit@1(ctx ON)−hit@1(ctx OFF) を使うこと（旧 `ambiguous` は補助比較のみ）。
