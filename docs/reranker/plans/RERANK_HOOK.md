# リランカー接続点（Phase 0 → Phase 3）

> **superseded by `docs/reranker/tasks/NEXT_TASK_PHASE3_CTX.md`** for deploy settings.
> This page’s **max_len=48 / no-context / `modernbert70m_ce`** numbers are the **context-less** era.
> Ship the contextual Track B rewriter (`trackB_v2_continue`, max_len=128, τ=2.5, context_prev required).

前提: `docs/reranker/plans/PLAN_RERANKER.md` / `docs/reranker/reports/PHASE0_RERANK_REPORT.md` / `docs/reranker/tasks/NEXT_TASK_PHASE3.md`

## 目的

生成系 `AIRewriter` とは別に、「Mozc が出した N-best を並べ替えて返す」経路を明示する。

## 現状の生成経路

`Mozc-Ai/src/rewriter/ai_rewriter.cc`（compat: `Mozc-Ai-Training/mozc_compat/ai_rewriter.*`）

1. `Rewrite(segments)` が呼ばれる
2. `GetExistingCandidates(segments)` で **conversion_segment(0)** の候補を読む
3. キャッシュミスなら AI 生成を非同期 enqueue
4. 結果は `InsertCandidates` で **末尾追加**（並べ替えではない）

## リランク経路

```
Mozc StartConversion
  → segments（各 segment の candidate list）
  → [rerank hook]  query=(reading, context) × candidates をスコア
  → margin gate（τ=2.5）で top1 上書き可否
  → candidate 順を並べ替え
  → 候補ウィンドウへ返却
  →（opt-in）conversion log: reading, nbest, chosen, …
```

### フック位置（推奨）

- **Rewriter chain の末尾**に `RerankRewriter` を追加する（`AIRewriter` と並立）。
- 操作対象は当面 `mutable_conversion_segment(0)`（現行 AIRewriter と同じ）。
- C++ スケルトン: `mozc_compat/rerank_rewriter.h|.cc`（現状 no-op / `enabled_=false`）。

### オフライン（今すぐ使える）

```bash
python -m tools.rerank.phase3_hook rerank \
  --onnx artifacts/rerank/modernbert70m_ce/onnx/cross_encoder_fp32.onnx \
  --tokenizer artifacts/rerank/modernbert70m_ce/onnx/tokenizer \
  --tau 2.5 --cand-cap 30 --max-len 48 \
  --input request.jsonl --out ranked.jsonl \
  --log artifacts/rerank/conversion_logs/conv.jsonl
```

Smoke: `scripts/_run_phase3_hook_smoke.sh`  
評価ハーネス: `tools/rerank/eval_cross_encoder.py`（holdout 計測）と共有ポリシー。

### 配布デフォルト

| 項目 | 値 |
|--|--|
| backend | ONNX **fp32**（int8 禁止） |
| τ | **2.5**（v3 holdout；Phase1 単体は 2.0 も可だが v3 計測に合わせる） |
| cand_cap | **30** |
| max_len | **48** |
| topK | なし |

## 決定ログ

| 項目 | 決定 |
|--|--|
| 生成経路 | 残す（リランクは別経路） |
| リランク対象 | Mozc N-best |
| Phase 0 | IME 未組み込み（候補集合定義待ち） |
| Phase 3（今） | オフラインフック + ログスキーマ + C++ stub。IME 配線が次 |
| 辞書 vs この holdout | accept-holdout 向け辞書研磨は pause（honest B≈0 は mismatch） |
