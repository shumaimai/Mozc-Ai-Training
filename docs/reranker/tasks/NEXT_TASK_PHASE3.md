# 次タスク指示: Phase 3 — リランカー統合 + 実変換ログ

> **superseded by `docs/reranker/tasks/NEXT_TASK_PHASE3_CTX.md`.**
> Deploy knobs below (max_len=48, no context, `modernbert70m_ce`) are **obsolete**.
> Contextual ship: `trackB_v2_continue` / max_len=128 / τ=2.5 / context_prev ON / ONNX fp32.

対象: リポジトリ作業エージェント  
前提: `docs/reranker/plans/RERANK_HOOK.md`, `docs/reranker/plans/PLAN_RERANKER.md` §5.1–5.2 / §6, `docs/reranker/reports/PHASE0_RERANK_REPORT.md`  
**Phase 2 辞書深掘り（accept-holdout 向け）は pause。** 根拠:  
`artifacts/rerank/phase2_b_zero_audit.json`（honest≈0 は lookup バグではない）+  
`artifacts/rerank/phase2_b_manual_taxonomy.md`（この holdout では NEologd 無効）。

---

## 現状の勝ち筋（正直に）

| 指標 | 値 | 扱い |
|--|--|--|
| no-inject rerank hit@1 | **≈45.9%**（τ=2.5, regress≈1.73%） | **出荷ターゲット** |
| leaky B-inject | ≈68% | **上界のみ。クレーム禁止** |
| honest B-inject | ≈0% | この accept-holdout への辞書研磨は止める |

制約: **ONNX fp32・cand_cap=30・max_len=48・τ=2.5（ネイティブ）・int8 なし・topK 既定なし。**  
CE 学習データ拡張は **しない**（pause 継続）。Colab は不要なら触らない。

---

## ゴール

1. Mozc 経路に **並べ替え専用** フックを載せ、N-best → rerank → 候補返却を動かす（まずオフライン、次に IME）。
2. **実変換ログ**を集め始め、将来の正直 B 評価と辞書ターゲットを accept-holdout から切り離す。
3. `AIRewriter`（生成/末尾追加）は残す。リランクは別経路。

---

## 接続点（Phase 0 決定の再掲）

ソース: `Mozc-Ai/src/rewriter/ai_rewriter.cc`（compat: `mozc_compat/ai_rewriter.*`）  
文書: `docs/reranker/plans/RERANK_HOOK.md`

```
Mozc StartConversion
  → segments（candidate list）
  → [RerankRewriter / offline hook]
  → margin gate (τ=2.5)
  → 候補ウィンドウへ返却
```

推奨フック位置: **Rewriter chain 末尾**に `RerankRewriter`（`AIRewriter` と並立）。  
当面 `mutable_conversion_segment(0)`（現行 AIRewriter と同じ）。複合語フルパス問題は別チケット。

---

## 手順（この順）

### A. オフラインフック（即時・必須）

成果物済みスケルトン: `tools/rerank/phase3_hook.py`

```bash
# schema
python -m tools.rerank.phase3_hook schema

# smoke（WSL）
wsl -e bash <Mozc-Ai-TrainingのWSLパス>/scripts/_run_phase3_hook_smoke.sh
```

入出力:

```json
{"reading":"とうきょう","nbest":["東京","東響"],"context_prev":""}
```

→ `ranked_surfaces` / `final_top1` / `overwritten`。  
`--log` で conversion log JSONL を append。

ONNX 優先: `artifacts/rerank/modernbert70m_ce/onnx/cross_encoder_fp32.onnx`  
（v3 を載せるなら先に `export_onnx` してから差し替え。品質再測必須。）

### B. 変換ログ収集（必須）

スキーマ: `artifacts/rerank/conversion_log_schema_v1.json`  
（定義: `tools/rerank/phase3_hook.py` → `CONVERSION_LOG_SCHEMA`）

必須フィールド: `ts`, `reading`, `nbest`, `chosen`  
任意: `context_prev`, `rerank_top1`, `final_top1`, `overwritten`, `tau`, `session_id`, `source`

ルール:

- ユーザ opt-in。生キーストリーム全体は保存しない（reading + nbest + chosen のみ）。
- オンライン収集前にプライバシー文言を UI/設定に出す。
- ログ置き場: `artifacts/rerank/conversion_logs/`（gitignore 推奨・個人データ扱い）。

### C. IME 側スケルトン（次エージェント）

1. `mozc_compat/rerank_rewriter.h|.cc` を新設（または `Mozc-Ai/src/rewriter/`）。
2. `Rewrite()` で segment(0) 候補を読み、**外部プロセス / ORT 埋め込み**のどちらかでスコア。
   - 最短経路: オフラインと同じ JSON を stdin で `phase3_hook` に渡す（遅延はプロトタイプのみ許容）。
   - 出荷経路: ORT C++ API で同 ONNX をロード（int8 禁止）。
3. margin τ=2.5 で top1 上書き可否を決め、候補順を並べ替え（末尾追加ではない）。
4. `rewriter.cc` チェーン末尾に登録（`AIRewriter` の後）。
5. 確定時に conversion log 1 行 append（opt-in）。

### D. やらないこと

- accept-holdout 向け辞書カバレッジ研磨 / NEologd 即バンドル
- CE データ scale-up
- int8 / topK 既定
- 68% を一般性能として報告

---

## 成功基準

- [ ] `python -m tools.rerank.phase3_hook rerank` が ONNX または PT で JSON→ranked を返す
- [ ] `conversion_log_schema_v1.json` があり、smoke ログが validate される
- [ ] `docs/reranker/plans/RERANK_HOOK.md` が Phase 3 実装パスを指している
- [ ]（次）`RerankRewriter` が Mozc ビルドに載る、または明確な PR 差分
- [ ]（次）opt-in で実ログが数日分たまる → そこで初めて B / 辞書を再評価

---

## 次の人間アクション

1. smoke が通ることを確認（上の `_run_phase3_hook_smoke.sh`）。
2. IME 統合担当（または次エージェント）に `RerankRewriter` 実装を依頼。
3. 実機で opt-in ログを数日回し、**そのログ上の B** を見てから辞書/NEologd を再検討。
