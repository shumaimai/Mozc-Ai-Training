# 次タスク指示: 「文脈が効く」サブセットの再定義＋文脈整形（学習前の必須修正）

対象: リポジトリ作業エージェント
前提: `docs/reranker/plans/PLAN_CONTEXTUAL_RERANKER.md` / `docs/reranker/reports/MOZC_WSL_WINDOWS_BOUNDARY.md`
位置づけ: 境界解消・データ生成は完了。だが**「曖昧」サブセットが実質文脈非依存で、このまま学習すると文脈の効果を測れない**。学習に進む前にここを直す。**Mozc の再実行は不要**（既存の抽出結果を選別・整形するだけ）。

---

## 背景（検証で判明した問題）

現行 `data/rerank_ctx/` は境界解消後に生成され、文脈は 96% 埋まった（大きな前進）。しかし `ambiguous` フラグの中身を測ると:

- `ambiguous`=35%(7000/20000) だが、**本当に文脈で割れる語は 6.2%(137読み/1233行)しかない**。
- 残りは機能語・文法語の一強（`ねん`→年 100%、`し`→し 99%、`ある`→ある 99% …）。
- ambiguous gold の **43% が非漢字**、**38% が gold==reading（変換不要）**。
- 文脈の **19% に改行/Wiki マークアップ混入**。

→ このまま学習・評価すると頻度が支配し、「文脈を使えているか」を綺麗に測れない（過去の不明瞭な結果の再来）。

既存資産（再利用する。Mozc は叩き直さない）:
- 全プール: `data/rerank_ctx/work/mozc/attached.jsonl`（**303,830 行**、gold_in_nbest≈0.97）
- 読み→gold 分布: `data/rerank_ctx/reading_gold_map.json`
- レコード field: `reading, context_prev, gold, mozc_nbest, gold_in_nbest, gold_rank, mozc_top1, mozc_hit1, pos, reading_gold_count, doc_id, sent_id, char_begin, char_end, source, ambiguous`

---

## ゴール

1. **文脈が本当に効く語だけを選ぶ新フラグ `context_sensitive` を定義**し、それを主対象にデータを組み直す。
2. **文脈を整形**（マークアップ/改行除去、文境界切り）。
3. 文脈依存語は自然文だと希少なので、**十分な量を確保**（スケールアップ＋狙い撃ち抽出）。
4. train/eval を作り直し、統計を再出力。

---

## タスク

### 1. `context_sensitive` の定義（全プール 303,830 から算出）
`attached.jsonl` 全体で `reading → {gold: 頻度}` を作り直し、各**読み**を判定:

**context_sensitive な読みの条件（AND）**
- `total ≥ 8`（頻度が薄い読みは除外）
- **最頻 gold の占有率 < 0.70**（非退化＝1つが支配していない）
- **漢字 gold が 2 つ以上**（`gold` に漢字を含み、かつ `gold != reading`）。各漢字 gold の頻度 ≥ 3
- **機能語の除外**: `pos` を見て、助詞・助動詞・接続詞・記号・補助記号・フィラー・感動詞は除外。**名詞・動詞・形容詞・副詞（content 語）のみ**採用

各レコードに付与:
- `context_sensitive`（bool、上記読みに属し、かつ当該レコードの `gold` が漢字 content 語）
- `top_gold_share`（その読みの最頻 gold 占有率）
- `n_kanji_golds`

> 既存 `ambiguous` は消さず残す（比較用）。判定の主役を `context_sensitive` に切り替える。

### 2. 文脈整形（`clean_context`）
`context_prev` に対して:
- 改行・タブを空白へ、連続空白を 1 つに。
- Wikipedia マークアップ/節見出し/箇条書き記号/参照記号を除去（`概説`, `\n`, `==`, `*`, `[edit]` 等の痕跡）。
- 整形後に **文終端（。！？）で切って"進行中の一文"だけ** にし、末尾から最大 50 文字。
- **この関数は推論フックと同一実装**（`build_ctx_dataset.py` と統合ヘッダに1つ）。
- 整形後に空になったレコードは、**context_sensitive 集計からは除外**（文脈を測れないため。anchor には回してよい）。

### 3. 量の確保（希少性対策）
context_sensitive は自然文で希少（≈6%）。**全プール 303,830 から選別**したうえで、足りなければ狙い撃ち:
- context_sensitive 読みリストを作り、**その読みの出現を全プールから全部拾う**（オーバーサンプリング）。
- 目標: **context_sensitive train ≥ 15,000 行**、**distinct 読み ≥ 200**、1 読みあたり複数文脈が入ること。
- それでも足りなければ、`data/rerank_ctx/raw/*.jsonl`（wiki/news/aozora）から context_sensitive 読みの追加抽出を検討（Mozc は既存 candidates を再利用、無い読みだけ追加変換）。

### 4. train/eval 再構成
- **train**: context_sensitive を **30〜40%**（上の 15k+）＋ anchor（非曖昧・Mozc 正解、忘却防止）＋（少量）崩れ入力。gold==reading・機能語は anchor 側に置き、context_sensitive には入れない。
- **eval seen / unseen / fresh**: 各セットに **context_sensitive サブセット ≥ 500**。文書単位分割は維持（リーク厳禁）。fresh は最新ニュースのまま。
- gold_in_nbest フィルタは維持。

### 5. N-best 衛生（任意）
- 絵文字・明らかな非日本語ゴミ候補を N-best から落とす（`🐚 Χ` 等）。軽微だが入力を綺麗に。

### 6. 統計の再出力
`data/rerank_ctx/assemble_summary.json` を更新し、最低限:
- 各セットの `n`, `context_sensitive` 数/割合, distinct 読み数
- `top_gold_share` の中央値（context_sensitive で **< 0.6** が目安）
- 文脈非空率、**改行残存率（≈0% を確認）**
- gold==reading 率（context_sensitive では 0%）
- mozc_hit1（参考）

---

## 成功基準
- `context_sensitive` サブセットが**非退化**（`top_gold_share` 中央値 < 0.6、漢字多択、機能語なし、gold≠reading）。
- context_sensitive: train ≥ 15,000 / distinct 読み ≥ 200、各 eval ≥ 500。
- 文脈整形後、**改行/マークアップ残存 ≈ 0%**、context_sensitive の文脈非空 100%。
- 文書単位分割が保たれ、fresh が学習と非重複。
- 統計 JSON と短い所見（`docs/` に追記）。

## 制約・注意
- **Mozc を叩き直さない。** `attached.jsonl` の既存 N-best を再利用。読み追加が要る時だけ差分変換。
- **文脈整形関数は学習生成と推論で同一**（ズレ厳禁）。
- **文書単位分割を厳守**（文/語単位はリーク）。fresh は一滴も学習へ混ぜない。
- gold==reading・機能語は context_sensitive に入れない（anchor 行き）。
- 既存の `train.jsonl` / `eval_*.jsonl` は上書きせず、新版は別名（例 `*_v2`）で出し、summary で新旧比較できるように。
- **ベースモデル確認**: Track B(現行から継続) 用に「現行 CE の実体（base とチェックポイント）」を1行で確定し記録。Track A/B は同一アーキで揃える。

## 成果物
- 更新した選別/整形コード（`build_ctx_dataset.py`＋共通の `clean_context`／`is_context_sensitive`）
- `data/rerank_ctx/train_v2.jsonl` / `eval_{seen,unseen,fresh}_v2.jsonl`
- 更新 `assemble_summary.json`（context_sensitive 統計込み）
- `docs/` に「context_sensitive 再定義＋文脈整形の結果」を追記
- 最後に、学習（Track A/B）へ進んでよいかの一言判断（context_sensitive の量・非退化性が満たせたか）
