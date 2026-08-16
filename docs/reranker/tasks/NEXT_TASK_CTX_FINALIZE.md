# 次タスク指示: 文脈リランカー v2 の締め（監査・単一τ・出荷判定）

対象: リポジトリ作業エージェント
前提: `docs/reranker/reports/PHASE_CTX_REPORT.md` / `artifacts/rerank_ctx/summary_v2.md`。**v2 学習・評価は完了し、文脈は明確に効いた**（context_sensitive デルタ +25〜42pt、Track B 僅差勝ち）。**再学習は不要。** ここは出荷前の確定作業。

チェックポイント（Modal Volume `mozc-artifacts`）:
- `/artifacts/trackA_v2_modernbert70m/`, `/artifacts/trackB_v2_continue/`
- ベース arch: `sbintuitions/modernbert-ja-70m`（A/B 同一）

---

## タスク1: gold 監査を埋める（数字を信じる前提）

- `data/rerank_ctx/audit_sample_300`（`.json`/`.tsv`）を人手基準で確認、または v2 データから 300 件を再サンプルして gold 正解率を出す。
- **合格 ≥95%。** 下回ったら §2.5 のフィルタ（POS 除外・読み一致・長さ）を締めて再アセンブルし、影響を記録。
- 出力: `data/rerank_ctx/audit_v2_result.json`（正解率・誤りの内訳）。

## タスク2: 単一 τ で「出荷値」を確定（最重要）

現状の表は**各セット最適 τ**で出ており、配布では使えない（配布は**単一 τ**）。

- Track B を対象に、**全セット共通の τ を 1 つ**選ぶ。候補 `τ ∈ {1.5, 2.0, 2.5}` を固定でスイープ。
- 選定基準: **3 セット全てで全体 regression < 2%** を満たす中で、**context_sensitive デルタの最小値（seen/unseen/fresh の最小）が最大**になる τ。
- その単一 τ で seen/unseen/fresh を再集計し、以下を出荷値として確定:
  - 全体 hit@1・Mozc比・regression
  - context_sensitive hit@1（ON/OFF）とデルタ
  - non_context_sensitive の hit@1 と対 Mozc 差（易しい語のコスト）
- 出力: `artifacts/rerank_ctx/eval/shippable_trackB_tau<τ>.json` と要約表。

## タスク3: 副作用の確認と締め

- **非曖昧語のコスト**を明示（v2 では非CSで対 Mozc 約 −1pt）。単一 τ でこれが悪化しないか確認。許容できなければ τ を上げる方向で再評価。
- **CS サブセットの regression（2〜5%）**を単一 τ で再確認し、記録。
- Track A は比較用に残す。**採用は Track B**（要確認）。

## タスク4: レポート更新と go/no-go

- `docs/reranker/reports/PHASE_CTX_REPORT.md` に「監査結果・単一 τ の出荷値・採用モデル（B）・配布 τ」を追記。
- §5.1 の決定ルールで **Phase 3（Mozc 統合）へ進む go/no-go を明記**。
- 併せて配布メモ（`PLAN_CONTEXTUAL_RERANKER` §10 のデグレード）と整合: 配布 τ、max_len=128、cand_cap、文脈切り出し関数の共有を再掲。

---

## 成功基準
- gold 正解率 ≥95%（または締め直し後に達成）。
- **単一 τ での出荷値**が seen/unseen/fresh で確定（全体 regression <2% を保ちつつ CS デルタ維持）。
- 非CSコストと CS regression が数値で記録されている。
- レポートに採用モデル・配布 τ・Phase 3 go/no-go が明記。

## 制約・注意
- **再学習しない**（v2 の結論は確定済み。ここは評価・確定作業）。
- **配布は単一 τ**。各セット最適 τ の数字を出荷値と呼ばない。
- 評価は既存 v2 チェックポイント（Modal Volume）を使う。GPU が要る再評価は Modal `T4` で。
- 文書単位分割・fresh 非混入は維持。
- 旧成果物は上書きせず、確定版は別名で。

## 成果物
- `data/rerank_ctx/audit_v2_result.json`
- `artifacts/rerank_ctx/eval/shippable_trackB_tau<τ>.json` ＋要約
- 更新 `docs/reranker/reports/PHASE_CTX_REPORT.md`（出荷値・採用・配布 τ・go/no-go）
