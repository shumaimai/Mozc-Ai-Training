# 次タスク指示: Phase 3 統合（文脈リランカー版）

対象: リポジトリ作業エージェント
前提: `docs/reranker/plans/PLAN_CONTEXTUAL_RERANKER.md` / `docs/reranker/reports/PHASE_CTX_REPORT.md` / `docs/reranker/plans/RERANK_HOOK.md`（**旧・文脈なし版**）/ `docs/guides/MOZC_INTEGRATION.md`

> **最重要の注意**: 既存の `docs/reranker/plans/RERANK_HOOK.md` と `docs/reranker/tasks/NEXT_TASK_PHASE3.md` は**文脈なし時代の設定**（max_len=48・旧 model `modernbert70m_ce`・文脈を渡さない）。**今回は文脈リランカー（Track B）を載せるので、これらの deploy 設定は無効。** 本書がそれを更新する。まず両ドキュメントの deploy 設定節に「superseded by NEXT_TASK_PHASE3_CTX」と明記すること。

---

## 0. 確定した出荷仕様（新・これに統一）

| 項目 | 旧（文脈なし・使わない） | **新（文脈・採用）** |
|--|--|--|
| モデル | `modernbert70m_ce` | **`trackB_v2_continue`**（arch `sbintuitions/modernbert-ja-70m`） |
| 文脈 | 渡さない | **必須**: `reading` ＋ `context_prev`（前方確定文、`context_clip` で整形） |
| max_len | 48 | **128**（文脈が入るので 48 では切れる） |
| τ（margin） | 2.5 | **2.5**（clean ctx eval で確定。全セット退行<1.4%） |
| cand_cap | 30 | **30** |
| backend | ONNX fp32 | **ONNX fp32**（int8 禁止・順位崩壊） |
| topK | なし | **なし** |
| 出荷数値 | — | seen 91.5% / unseen 91.5% / fresh 92.9%、CSΔ +29〜49pt、退行<1.4% |

**margin gate**: Mozc の top-1 は、reranker スコアが τ=2.5 **以上**で上回った時だけ上書き。それ以外は Mozc 順を維持（＝易しい語・自信のない曖昧語を壊さない安全弁）。

---

## 1. コード接続点（どこを・どのファイルで・どう）

| ファイル | 変更内容 |
|--|--|
| `mozc_compat/rerank_rewriter.h/.cc` | 本体実装。現在 no-op / `enabled_=false`。ONNX Runtime(C++) で読み→文脈→候補をスコアし、margin gate で並べ替え。fail-safe/timeout 内蔵 |
| `mozc_compat/rerank_rewriter_test.cc` | 単体テスト（並べ替え・top1保護・fail-safe・空文脈） |
| `rewriter/rewriter.cc`（統合先 Mozc） | Rewriter chain の**末尾**に `RerankRewriter` を登録（`AIRewriter` と並立、生成経路は残す） |
| `rewriter/BUILD.bazel` | `rerank_rewriter` ターゲット追加、**onnxruntime 依存**、モデル資産（onnx/tokenizer/margin_policy）を data として同梱 |
| `scripts/integrate_mozc.py` | `rerank_rewriter.*` のコピーと chain 登録を自動化に追加（`ai_rewriter` と同様） |
| `tools/rerank/context_clip.py` | **C++ 側へ移植**（同一ロジック）。train/serve で文脈整形を一致させる。parity テスト必須 |
| `tools/rerank/phase3_hook.py` | オフライン検証を**新 model・max_len=128・文脈ON**で回せるよう更新 |
| `tools/rerank/conversion_log.py` | 実変換ログの schema（下記） |
| `tools/rerank/export_onnx.py` | **`trackB_v2_continue` を ONNX fp32 でエクスポート**＋parity_check |

処理フロー（`rerank_rewriter.cc` 内）:
```
Rewrite(request, segments):
  if !enabled_ or model未ロード: return   // 何もしない=Mozc順のまま(fail-safe)
  seg = mutable_conversion_segment(0)      // 当面。複合語フルパスは別チケット
  reading   = seg.key (→ NFKC/ひらがな正規化, 学習と一致)
  context   = clip_context_prev(前方確定テキスト)   // ← §2 が肝
  cands     = seg.candidate[0..cand_cap)
  scores    = onnx(reading, context, cand) for each   // cand_cap 分を batch 1 forward
  // 方針は margin.py / phase3_hook.py を正とする（top-1 だけ gate。全体ソートはしない）:
  if score(rerank_top) - score(mozc_top1) >= τ(2.5):
      final = rerank_top を先頭、残りは reranker スコア順、cand_cap 外は Mozc 順で後ろ
  else:
      Mozc 順を維持
  (opt-in) conversion_log に記録
```

---

## 2. 文脈抽出の設計（統合で一番きわどい所・最優先で正す）

**文脈は学習と同じ作り方でないと性能が出ない。** ここを外すと、あれだけ苦労した +29〜49pt が消える。

- **`context_prev` の出どころ** = 変換対象 segment より**前の、確定済みテキスト**。Mozc の `Segments` の history segments（確定済み前方）から組む。**後続は絶対に使わない**（IMEは変換時に後続を知らない＝学習と同条件）。
- **segment(0) のみの MVP では**、context_prev = **history segments の確定テキストを連結 → `clip_context_prev` で整形**（直前の文終端で切るので、実質「進行中の一文の前方」だけ残る）。history が無い（文頭）なら空文脈でよい（学習も空文脈を含む）。
- **複数 segment を将来対応する時**は、segment を前から処理し、**前の segment（確定＋既に rerank 済み）の採用表記も文脈に連結**（左→右の順序依存が出る＝別チケットで扱う）。MVP では踏み込まない。
- **整形は学習と同一関数**: `context_clip.py`（`clean_context`/`clip_context_prev`）を**実装のまま** C++ 移植する。**実際の順序＝①改行/マークアップ除去 → ②直前の文終端 。！？ で切る → ③末尾50字**。**NFKC は文脈に掛けない**（NFKC/ひらがな化は**読み側**）。※本書内の他の括弧書き説明は不正確な場合があるので、**必ず `context_clip.py` のコードを正**とする。
- **reading 正規化も一致**（ひらがな・NFKC）。カタカナ/全半角ズレは空スコア化の原因（過去の gold_in_nbest=0 の同型リスク）。
- **parity テスト必須（2種類）**:
  1. **文脈整形 parity**: 同じ入力を Python(`context_clip.py`) と C++ 実装に通し、**出力文字列が完全一致**。
  2. **トークナイズ parity（同じくらい重要・見落とし注意）**: ModernBERT-Ja のトークナイザ（SentencePiece/`tokenizer.json`）を C++ 側でも使い、**同じ文字列→同じ token id 列**になることを確認。HF tokenizers の C++/Rust バインディングか sentencepiece C++ を使い、Python と**同一 vocab・正規化・特殊トークン**であること。ここがズレると、文脈整形が合っていてもスコアが崩れる。
  - どちらもズレたら**統合前に**直す。

---

## 3. 段階手順（この順で）

### A. モデルを ONNX fp32 でエクスポート（新 model）
**採用モデルは `trackB_v2_continue`。** clean 再学習版 `trackB_v2_clean_continue` は**破棄済み・使わない**（退行が悪化したため。§clean eval 結論）。
```bash
# 1) Modal Volume から採用チェックポイントを回収（ローカル/WSL・CPUで可）
modal volume get mozc-artifacts /trackB_v2_continue ./artifacts/rerank_ctx/trackB_v2_continue
# 2) ローカルで ONNX fp32 export（GPU不要）
python -m tools.rerank.export_onnx \
  --ckpt ./artifacts/rerank_ctx/trackB_v2_continue/cross_encoder.pt \
  --out  ./artifacts/rerank_ctx/trackB_v2_continue/onnx --fp32
python -m tools.rerank.parity_check   # PyTorch↔ONNX fp32 の Spearman=1.0/MAE≈0 を確認
```
tokenizer と `margin_policy(τ=2.5)` も同梱物として書き出す。

### B. オフラインフックを新設定で再検証 ＋ 早期レイテンシ確認
`phase3_hook.py` を **max_len=128・文脈ON・新ONNX**で回し、`request.jsonl`（reading, context_prev, nbest）→ `ranked.jsonl`。
- `eval_contextual.py` の clean eval と**数値一致**することを確認（オフラインとIMEで同じ結果になる土台）。
- **並べ替え方針は `tools/rerank/margin.py` / `eval_contextual.py` の実装を正**とし、C++ はそれを**完全再現**する（§1 の擬似コードは説明用。出荷数値を出したロジックと1bitでも違えば数値が合わなくなる。step B の一致確認が最終担保）。
- **ここで CPU レイテンシも測る（C++ 実装前に de-risk）**。max_len 48→128 で重くなるため、`latency_pack.py` 相当で ONNX fp32・cand_cap=30 の CPU p50/p95 を先に把握。**目安を大きく超える（例 p95 ≫ 150ms）なら、C++ を作り込む前に**デグレード方針・30m 縮小を先に検討する。作ってから遅い、を避ける。

### C. C++ `RerankRewriter` 実装
- ONNX Runtime C++ でモデル＋tokenizer ロード（起動時1回、Volume/同梱パス）。
- `context_clip` 移植＋parity。
- margin gate（τ=2.5）、cand_cap=30、溢れは Mozc 順で後ろ。
- **fail-safe**: モデル欠落・例外・**タイムアウト（既定200ms）**時は**何もしない＝Mozc順を返す**。絶対にブロックしない。
- 単体テスト（並べ替え・top1保護・空文脈・タイムアウト時Mozc順）。

### D. Rewriter chain 登録 & ビルド
- `rewriter.cc` 末尾に登録、`BUILD.bazel` に target/onnxruntime/data。
- `bazelisk build //server:mozc_server` が通ること。

### E. レイテンシ再計測（文脈込み）＋デグレード配線
- **max_len 48→128 で推論は重くなる。** `latency_pack.py` 相当で CPU p50/p95 を再測。
- `PLAN_CONTEXTUAL_RERANKER` §10 のデグレードを配線: 層1（毎回200msタイムアウト→Mozc順）、層2（5回連続超過でティアダウン: cand_cap↓→文脈短縮→**文脈空+max_len最小(同一model)**→Mozc単体）。
- 「文脈空でも実用か」は §5 の文脈OFF評価が根拠（CS OFFでも Mozc よりは上）。

### F. 実変換ログ（opt-in・ローカル）
schema（`conversion_log.py`）: `reading, context_prev, mozc_nbest, mozc_top1, reranker_scores, chosen, overridden(bool), tau, ts`。
用途: 将来の**正直な評価**と**辞書ターゲット**を accept-holdout から切り離す。**既定 off・ローカル保存・個人情報配慮**。

### G. テスト（headless + 実機）
- **headless 統合テスト（CI 用・GUI 不要）**: `rerank_rewriter_test.cc` で `Segments` をプログラムから構築（reading＋history＋candidate list）し、`RerankRewriter` を直接叩いて **①同音異義が文脈で正しく並ぶ ②Mozc top-1 が理由なく壊れない ③fail-safe/タイムアウト時は Mozc 順 ④空文脈でも動く**をアサート。**文脈の正しさをここで自動検証**する（mozc_batch は無文脈なのでスモークに使えない）。
- **実機スモーク（人手・最終確認）**: Windows の Mozc IME で実変換し、同音異義が文脈で正しく上がる（例 きしゃ→文脈で 記者/汽車）・崩れ入力で破綻しない・遅延が体感許容、を目視。**自動クリックはしない**（ビルドと手順チェックリストを用意し、タイピングは人手）。

---

## 4. 注意事項（片っ端から）

1. **旧 deploy 設定を使わない**（max_len=48・旧 model・文脈なしは全部NG）。RERANK_HOOK.md/NEXT_TASK_PHASE3.md に superseded 明記。
2. **train/serve の「文脈整形・reading 正規化・トークナイズ」を完全一致**（§2 の parity テスト2種必須）。ここが統合の最大リスク。特に**トークナイザの一致は見落としやすい**。
3. **fail-safe 徹底**: model 欠落・例外・タイムアウトで**必ず Mozc 順**。IME を絶対に止めない。
4. **生成 `AIRewriter` は残す**（別経路・末尾追加）。リランクは並べ替え専用の別 Rewriter。
5. **当面 `conversion_segment(0)` のみ**。複合語フルパス・多 segment 最適は別チケット（ただし文脈は前 segment から取る、は今やる）。
6. **int8 禁止・topK 既定なし**（順位崩壊/gold落ち）。
7. **文脈で重くなる前提**でレイテンシ再計測。届かなければ §10 デグレードで吸収、それでもなら 30m を別途検討（今は不要）。
8. **ONNX Runtime を Mozc(bazel) ビルドに組む依存管理**を丁寧に（ビルドが割れやすい。`docs/guides/BUILD_ERRORS.md` 参照）。
9. **ログは opt-in・ローカル・個人情報配慮**。既定 off。
10. **margin gate の τ はモデル同梱の policy から読む**（ハードコードしない。将来差し替え可能に）。
11. モデル資産（onnx/tokenizer/policy）の**同梱パスとロード失敗時の挙動**を明示（失敗＝リランク無効＝Mozc単体）。
12. 既存成果物・旧チェックポイントは残す。新規は別名。
13. **採用は `trackB_v2_continue`**。破棄した `trackB_v2_clean_continue`（clean 再学習・退行悪化）を誤って使わない。
14. **有効/無効の実行時トグル**を用意（config/env）。再ビルドなしで rerank を切れるように（ロールアウト/切り戻し用。既定は安全側で判断）。
15. （任意 de-risk）**最初の疎通は `RerankRewriter` から `phase3_hook.py` を subprocess 呼び出し**で通し、経路が生きるのを確認 → その後 C++ 埋め込み ONNX へ置換、でもよい（遅いので出荷はしない）。作り込む前に配線を確かめたい時の手。

---

## 5. 成功基準
- `trackB_v2_continue` の ONNX fp32 が parity 一致でエクスポート済み。
- オフラインフック（max_len128・文脈ON）が clean eval と数値一致。
- C++ `RerankRewriter` が chain 末尾で動き、**実機で文脈が効いた並べ替え**が出る。Mozc top-1 保護・fail-safe・タイムアウト時Mozc順が単体テストで担保。
- **Python↔C++ parity テスト2種が緑**（①文脈整形 ②トークナイズ token id 一致）。
- CPU レイテンシ再計測済み、デグレード配線済み。
- 変換ログ（opt-in）が schema 通りに出る。
- 実機スモークで同音異義が文脈で正しく、崩れ入力で破綻しない。

## 6. 成果物
- `artifacts/rerank_ctx/trackB_v2_continue/onnx/`（fp32 + tokenizer + margin_policy τ=2.5）
- 実装した `mozc_compat/rerank_rewriter.{h,cc}` ＋ test、`rewriter.cc`/`BUILD.bazel` 差分、`integrate_mozc.py` 更新
- `context_clip` の C++ 移植 ＋ parity テスト
- 更新 `phase3_hook.py` / `conversion_log.py`
- レイテンシ再計測レポート、デグレード配線
- `docs/reranker/plans/RERANK_HOOK.md` / `docs/reranker/tasks/NEXT_TASK_PHASE3.md` に superseded 明記、本書へのリンク
- `docs/reranker/reports/PHASE3_CTX_REPORT.md`（実機スモーク結果・レイテンシ・go/no-go）
