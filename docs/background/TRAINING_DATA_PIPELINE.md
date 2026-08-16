# 学習データ生成パイプライン構想

**更新**: 2026年8月2日  
**ステータス**: 設計案（未実装）

## ユーザー案の要約

1. 青空文庫等から本文を大量取得
2. ランダムな位置で切り出し
3. CSV: `正解文 | ひらがな | Mozc候補1 | Mozc候補2 | ...`
4. Mozc 変換エンジンだけでひらがな→漢字候補を取得
5. **正解文と Mozc 候補が一致しない行** を学習セットにする

→ **「Mozc が取りこぼした正解」を教師データにする** という方針。

---

## 評価: 方向性は正しい

| 観点 | 評価 |
|------|------|
| タスク適合 | ◎ IME の AI は「Mozc にない候補を足す」役割と一致 |
| ラベルコスト | ◎ 正解はコーパス原文。人手ラベル不要 |
| データ量 | ◎ 青空文庫だけで数十万〜百万行いける |
| 実装難度 | ○ Mozc API 抽出 + 形態素境界の処理が必要 |
| そのままの精度 | △ ランダム切断・文脈なしだとノイズが多い |

**結論**: 核となるアイデアは採用価値が高い。以下の修正を入れると実用レベルになる。

---

## 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 0: コーパス取得                                            │
│  青空文庫 / Wikipedia JA / 独自テキスト                           │
│  → ルビ除去・旧字体正規化・段落→文分割                           │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: セグメント生成（★ランダム切断の改良版）                  │
│  MeCab で形態素解析 → 読み（ひらがな）を付与                      │
│  変換単位（語 / 句）ごとに「IME入力キー」を作る                    │
│  左文脈（直前 N 語の表記）を保持                                   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Mozc 変換（AI rewriter なし）                           │
│  key=ひらがな, context=左文脈 → candidates[]                     │
│  mozc_server / Converter API / gconverter 等                     │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: フィルタ（学習に使う行だけ残す）                         │
│  条件: gold ∉ candidates[:K] かつ gold が有効な表記              │
│  除外: 曖昧すぎるキー、同一読みの別表記、ノイズ行                 │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4（任意）: DeepSeek で拡張・検証                           │
│  難例のみ: 追加候補の妥当性チェック、言い換え生成                 │
│  ノイズ行の除去、文脈説明の付与                                   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5: 学習用 JSONL                                            │
│  → LoRA fine-tune (PLaMo 2 1B / llm-jp-3.1-1.8b)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## ユーザー案からの主な改良点

### 1. 「適当にランダム切断」→ 形態素境界 or IME 入力シミュレーション

| 方式 | 説明 | 推奨 |
|------|------|------|
| 完全ランダム | 文字位置で切る | ✗ 単語の途中になりやすく非現実的 |
| 文境界 + ランダム語 | 文を選び、その中の1語を変換対象に | ○ 簡単でまず使える |
| **IME シミュレーション** | 「き」→「きょ」→「きょう」と段階入力 | ◎ 実使用に最も近い |
| 形態素単位 | MeCab の語境界で切る | ◎ バランス良い |

**推奨**: まず **形態素単位** で PoC。余裕があれば **段階入力**（プレフィックス列）も追加。

### 2. 文脈列を必ず入れる

現在の AIRewriter は `key` だけでなく **直前の入力履歴（context）** をプロンプトに渡している。

学習データにも必須:

```json
{
  "context": ["昨日は", "晴れ"],
  "key": "きょう",
  "mozc_candidates": ["今日", "きょう", "九曜"],
  "gold": "今日"
}
```

文脈なしだと「きょう」→「今日」と「きょう」が区別できず、学習が壊れる。

### 3. 「合わなかった」の定義を厳密に

```python
def is_training_sample(gold: str, candidates: list[str], k: int = 5) -> bool:
    top_k = candidates[:k]
    if gold in top_k:
        return False  # Mozc が既に正解を出している → 学習不要
    # 正規化して同一読み・表記ゆれを許容するかは別フラグ
    return True
```

追加フィルタ案:

- キー長 1 文字は除外（「は」「の」などノイズ多い）
- gold がひらがなのみ（漢字変換タスクでない）→ 除外
- Mozc 候補が 0 件 → 別カテゴリ（辞書未登録）

### 4. ひらがな化の品質

青空文庫は **ルビ付き**・**旧字体**・**送り仮名の揺れ** がある。

```
原文: 今日《きょう》は良い天気だ。
  → gold: 今日
  → key:  きょう
```

処理パイプライン:

1. 青空文庫のルビ記法をパース（`《》` `｜` 等）
2. ルビがあればそれを key に使う（MeCab より正確）
3. ルビなし部分は MeCab + UniDic で読み取得
4. NFKC 正規化、旧字体→新字体（任意）

---

## CSV / JSONL スキーマ案

### 中間 CSV（デバッグ用）

| 列 | 例 |
|----|-----|
| `source_id` | aozora:12345 |
| `sentence` | 今日は良い天気だ。 |
| `context` | 昨日は\|晴れだった。 |
| `gold` | 今日 |
| `key` | きょう |
| `mozc_1` | 今日 |
| `mozc_2` | きょう |
| `mozc_3` | 九曜 |
| `is_hard` | false |

`is_hard = (gold not in mozc_1..mozc_K)`

### 学習用 JSONL（最終形）

```json
{
  "instruction": "日本語IMEの変換候補を提案してください。",
  "input": "文脈: 昨日は, 晴れだった。\n入力: きょう\n既存候補: きょう, 九曜",
  "output": "今日"
}
```

または構造化:

```json
{
  "context": ["昨日は", "晴れだった。"],
  "key": "きょう",
  "mozc_candidates": ["きょう", "九曜"],
  "target_candidates": ["今日"]
}
```

---

## Mozc 変換エンジンの取り出し方

### 方式 A: Mozc バイナリ（最短 PoC）

Mozc ツリーに統合済みなら、converter のテスト用バイナリや `gconverter` 相当を流用。

```bash
# 概念例（実際の CLI は Mozc 版による）
echo "きょう" | mozc_convert --context "昨日は 晴れだった"
```

### 方式 B: C++ 小ツール（推奨・本番）

Mozc の `Converter` + `ConversionRequest` を直接呼ぶ:

```
ConversionRequest:
  - key: ひらがな
  - preceding text / segments: 左文脈
Rewriter チェーン: AI rewriter を **登録しない**
  → 辞書 + 言語モデル + 各種 rewriter のみ
```

本リポジトリの `mozc_compat/ai_rewriter.*` と逆で、「AI なし Mozc」の候補だけ欲しい。

### 方式 C: HTTP / IPC（バッチ向け）

`mozc_server` をヘッドレスで起動し、リクエストを投げる。  
大量バッチでは A/B より遅いが、Windows でも動く。

**PoC 順序**: B（バッチ用ネイティブツール）→ 速度が足りなければ並列化。

---

## DeepSeek の役割（ユーザー案との関係）

ユーザー案の **コアパイプラインは DeepSeek 不要** で回る。DeepSeek は **Phase 4 のオプション**:

| 用途 | 説明 |
|------|------|
| **検証** | 「gold が本当にその文脈で自然か」をスコアリング |
| **拡張** | Mozc も AI も難しい例に代替候補を生成 |
| **ハードネガティブ除去** | ランダム切断で生じた不自然文を弾く |
| **データ拡張** | 同じ key+context で言い換え gold を追加 |

```
コーパス + Mozc diff  →  80〜90% のデータ（無料・大量）
        ↓
   難例 5〜10% だけ DeepSeek API で品質チェック
        ↓
   最終 JSONL
```

コスト目安: 10万行のうち難例 1万行 × 短プロンプト ≒ 数ドル〜数十ドル。

---

## 青空文庫を使う場合の注意

| 項目 | 内容 |
|------|------|
| ライセンス | 作品はパブリックドメイン（利用規約は青空文庫サイト要確認） |
| ルビ | 正解読みの宝庫。パースを優先 |
| 文体 | 文語・近代口語混在。現代 IME 向けには Wikipedia 等と混合推奨 |
| 外字・旧字 | 正規化しないと gold と Mozc が永遠に一致しない |

**混合コーパス推奨比率（案）**:

- 青空文庫 40%（文学・固有名詞）
- Wikipedia 日本語 40%（現代語・百科事典）
- 合成テンプレート 20%（カバレッジ補完）

---

## 実装フェーズ

### Phase PoC（1〜2週間相当）

1. 青空文庫 10作品だけダウンロード
2. ルビパース + 形態素単位セグメント生成（Python）
3. Mozc 変換バッチ（1000行）
4. `gold ∉ top5` の行を手動サンプリング 50 件で目視品質確認

### Phase 1（本番データ）

1. 全コーパスパイプライン
2. 50万〜200万行生成 → フィルタ後 5万〜30万行
3. JSONL エクスポート

### Phase 2（学習）

1. PLaMo 2 1B LoRA
2. ベンチマーク（手動 500 パターン + テストセット）
3. Ollama `ai-mozc-ime` 化

---

## リスクと対策

| リスク | 対策 |
|--------|------|
| ランダム切断で非現実的キー | 形態素境界 + 最小キー長 |
| 文脈なしで同音異義語が区別不能 | context 列必須 |
| Mozc が正しくても gold と表記ゆれで mismatch | NFKC + 表記ゆれ辞書 |
| 学習データが「難しすぎる」ばかり | easy/hard を 1:3 くらいで混合 |
| 青空の文語が現代入力に効かない | Wikipedia 混合 |

---

## ディレクトリ構成（実装予定）

```
tools/dataset/
  fetch_aozora.py       # 青空文庫 DL
  fetch_wikipedia.py    # Wikipedia 抽出（任意）
  segment.py            # 形態素 + キー生成
  mozc_batch.cc          # Mozc 候補取得（C++）
  filter.py             # mismatch フィルタ
  to_jsonl.py           # 学習形式変換
  deepseek_enrich.py    # 任意: 難例検証
  config.yaml           # K, context_len, corpus paths
```

---

## まとめ

| ユーザー案 | 判定 |
|-----------|------|
| コーパスから自動生成 | ◎ 採用 |
| ひらがな + Mozc 候補の CSV | ◎ 採用（+ context 列を追加） |
| mismatch のみ学習 | ◎ 採用（IME AI の役割と一致） |
| 完全ランダム切断 | △ → 形態素境界 or IME シミュレーションに変更 |
| DeepSeek | 必須ではない。難例の品質担保に使うと良い |

**次のアクション**: Phase PoC として `segment.py` + 青空 10 作品 + Mozc バッチ 1000 行のスパイクを切る。

---

## データの運び方・処理の流れ（実務版）

「どのファイルがどこに流れ、誰が何を読むか」を段階ごとに固定する。

### ストレージ構成

```
data/
  raw/                    # Phase 0 出力（再取得可能・Git 管理外）
    aozora/000123_title.txt
    wikipedia/shards/wiki_000.jsonl
  interim/                # Phase 1〜2（再計算コスト大・Parquet 推奨）
    sentences.parquet     # 文単位
    samples.parquet       # 1語=1行（key, gold, context）
    samples_mozc.parquet  # Mozc 候補付き
  train/                  # Phase 3〜5（学習に使う最終成果物）
    hard.jsonl            # mismatch のみ
    mixed.jsonl           # hard + easy 混合（任意）
    stats.json            # 件数・フィルタ率
```

**形式の選び方**

| 段階 | 形式 | 理由 |
|------|------|------|
| raw テキスト | `.txt` / `.jsonl` | 人間が読める、差分しやすい |
| 中間（百万行） | **Parquet** | 列指向・圧縮・pandas/polars 向き |
| デバッグ | `.csv` | 1000 行以下のサンプルだけ |
| 学習 | **JSONL** | Hugging Face / axolotl / LLaMA-Factory 標準 |

---

### 5 段パイプライン（入出力を固定）

```
[raw/*.txt]
    │  fetch_aozora.py
    ▼
[sentences.parquet]          列: id, source, text
    │  segment.py
    ▼
[samples.parquet]            列: sample_id, sentence_id, pos, context[], gold, key
    │  mozc_batch (C++)
    ▼
[samples_mozc.parquet]       上記 + mozc_candidates[], mozc_latency_ms
    │  filter.py
    ▼
[hard.jsonl]                 学習用
    │  to_jsonl.py（プロンプト整形）
    ▼
[train_mixed.jsonl]          LoRA 投入
```

各段階は **独立したコマンド** にし、失敗したらその段から再実行できるようにする（Mozc バッチだけやり直し、など）。

---

### Phase 0: コーパス取得 → `sentences.parquet`

**処理**: 青空文庫 ZIP → ルビ・注記をパース → 1 文 1 レコード

```json
{"id": "aozora:56535:42", "source": "aozora", "text": "今日は良い天気だ。"}
```

- 改行・段落は捨て、**文分割**（`。` `！` `？` + 改行）
- 5 文字未満の文は捨てる
- 1 作品あたり数千〜数万文

```bash
python tools/dataset/fetch_aozora.py --out data/raw/aozora
python tools/dataset/normalize_sentences.py \
  --in data/raw/aozora --out data/interim/sentences.parquet
```

---

### Phase 1: セグメント生成 → `samples.parquet`

**1 文から複数行** に膨らませる（学習用。推論時の形態素連結ではない）。

入力文: `今日は良い天気だ。`

| pos | context（確定済み表記の列） | key | gold |
|-----|---------------------------|-----|------|
| 0 | `[]` | きょう | 今日 |
| 1 | `["今日"]` | は | は |
| 2 | `["今日","は"]` | いいてんきだ | 良い天気だ |

**処理ロジック（segment.py）**:

```python
for sentence in load_parquet("sentences.parquet"):
    morphemes = parse(sentence.text)   # MeCab or 青空ルビ
    context = []
    for i, m in enumerate(morphemes):
        if len(m.key) < 2:               # 「は」「の」スキップ可
            context.append(m.gold)
            continue
        yield {
            "sample_id": f"{sentence.id}:{i}",
            "sentence_id": sentence.id,
            "pos": i,
            "context": context.copy(),   # 左側の確定表記
            "key": m.key,                # ひらがな
            "gold": m.gold,              # 正解表記
        }
        context.append(m.gold)           # 次の語の文脈に
```

**ポイント**

- `context` は **表記（漢字混じり）** のリスト。AIRewriter の `context_history` と同じ
- `key` は **いま変換中のセグメント** 相当（形態素 1 個＝Mozc の最小単位に近い）
- 1 文 10 語 → 最大 10 行（キー長フィルタ後は 5〜7 行）

```bash
python tools/dataset/segment.py \
  --sentences data/interim/sentences.parquet \
  --out data/interim/samples.parquet \
  --min-key-len 2
```

---

### Phase 2: Mozc バッチ → `samples_mozc.parquet`

**各行を 1 リクエスト** として Mozc に渡す。

```
入力（1行）:
  context = ["今日", "は"]
  key     = "いいてんきだ"

Mozc Converter（AI rewriter なし）:
  → candidates = ["良い天気だ", "いい天気だ", "胃いてんきだ", ...]
```

**運び方（推奨）**:

1. `samples.parquet` を **シャード分割**（例: 10 万行/ファイル）
2. C++ `mozc_batch` が shard を読み、並列ワーカーで処理
3. 結果を shard ごとに書き出し → 最後に結合

```bash
# シャード分割（Python）
python tools/dataset/shard.py --in samples.parquet --shard-size 100000

# Mozc バッチ（8 並列例）
parallel -j 8 mozc_batch ::: data/interim/shards/samples_*.parquet

# 結合
python tools/dataset/merge_parquet.py \
  --in 'data/interim/shards/*_mozc.parquet' \
  --out data/interim/samples_mozc.parquet
```

**mozc_batch の入出力（stdin/stdout ではなくファイル）**:

```
読む列: sample_id, context (JSON array), key
書く列: 同上 + mozc_candidates (JSON array), error (nullable)
```

IPC 方式: 1 プロセスで Converter を初期化し、行ループで `Convert(key, context)` を呼ぶ。  
プロセス起動コストを 1 回に抑える（100 万行で重要）。

---

### Phase 3: フィルタ → `hard.jsonl`

```python
def process(row):
    cands = row.mozc_candidates[:5]
    if row.gold in cands:
        return None                    # Mozc が既に正解 → 捨てる
    if not has_kanji(row.gold):
        return None                    # 変換タスクじゃない
    if row.error:
        return None
    return row
```

**件数の目安**（経験則・要 PoC 検証）:

| 段階 | 行数 |
|------|------|
| samples | 100万 |
| mismatch (hard) | 5万〜15万（5〜15%） |
| ノイズ除去後 | 3万〜10万 |

**easy 混合（任意）**: 学習が難しすぎる場合、mismatch 行に加えて  
「Mozc 正解行の 10%」を `easy.jsonl` として混ぜる（比率 1:3 など）。

```bash
python tools/dataset/filter.py \
  --in data/interim/samples_mozc.parquet \
  --out-hard data/train/hard.jsonl \
  --out-easy data/train/easy.jsonl \
  --top-k 5
```

---

### Phase 4: 学習形式に整形 → `train_mixed.jsonl`

**推論時のプロンプトと同じ形** に揃える（最重要）。

現在の Ollama バックエンドのプロンプト:

```
日本語入力の変換候補を提案してください。

直前の入力: 今日, は

現在の入力: いいてんきだ

既存候補（これら以外を提案）: いい天気だ, ...

3つの候補を改行区切りで出力（説明不要）:
```

学習 JSONL（1 行）:

```json
{
  "text": "<prompt>...上記と同じ...</prompt>\n良い天気だ"
}
```

または instruction 形式:

```json
{
  "instruction": "日本語IMEの変換候補を提案してください。",
  "input": "直前の入力: 今日, は\n現在の入力: いいてんきだ\n既存候補: いい天気だ",
  "output": "良い天気だ"
}
```

`to_jsonl.py` は **ollama_backend.cc の BuildPrompt() と同じテンプレート** を Python で再実装し、学習・推論のズレを防ぐ。

```bash
python tools/dataset/to_jsonl.py \
  --in data/train/hard.jsonl \
  --easy data/train/easy.jsonl --easy-ratio 0.25 \
  --out data/train/train_mixed.jsonl \
  --prompt-template configs/ime_prompt.txt
```

---

### Phase 5: 学習 → モデル配布

```bash
# Colab / ローカル GPU
llamafactory-cli train configs/lora_plamo2_1b.yaml \
  --dataset data/train/train_mixed.jsonl

# GGUF → Ollama
python tools/export/gguf_export.py --adapter out/lora
ollama create ai-mozc-ime -f Modelfile
```

学習は **このリポジトリ外**（GPU マシン）でよい。運ぶのは `train_mixed.jsonl` だけ（数十 MB〜数百 MB）。

---

### 具体例: 1 文がどう流れるか

```
青空テキスト
  「きのうははれだった。きょうはよいてんきだ。」
        ↓ normalize
sentences.parquet 1行
        ↓ segment（2文 × 各5語 ≒ 8サンプル）
samples.parquet 8行
        ↓ mozc_batch
  行4: ctx=[昨日,は,晴れ,だった] key=きょう gold=今日
       mozc=[今日,きょう,九曜]  → gold in top3 → 捨て
  行7: ctx=[今日,は] key=よいてんきだ gold=良い天気だ
       mozc=[よい天気だ]         → gold not in top5 → hard.jsonl へ
        ↓ to_jsonl
train_mixed.jsonl 1行（行7のみ）
```

---

### 推論との対応関係

| 学習データの列 | IME 推論時 |
|---------------|-----------|
| `context[]` | AIRewriter の `context_history_` |
| `key` | `segments.conversion_segment(0).key()` |
| `mozc_candidates` | `GetExistingCandidates()` |
| `gold` / `output` | AI が追加すべき候補 |

データ生成は **「ユーザーが今のセグメントを変換しようとしている瞬間」** を 1 行で再現する。  
文全体は再現しない（Mozc がセグメント分割するから）。

---

### 再実行・増分

| やり直したいこと | 再実行開始点 |
|-----------------|-------------|
| コーパス追加 | Phase 0 → 以下すべて |
| セグメントロジック変更 | Phase 1 から |
| Mozc 辞書更新 | Phase 2 から |
| フィルタ条件変更 | Phase 3 から |
| プロンプト変更 | Phase 4 から |

`sample_id` を安定キーにしておけば、差分マージ可能。

---

### PoC で最初に作る最小セット

```bash
# 1. 青空 1 作品だけ
# 2. 100 文 → 約 500 サンプル
# 3. mozc_batch 500 行
# 4. hard を CSV で 50 行目視
# 5. train.jsonl 10 行で LoRA が動くか確認
```

ここまで動けば、あとはシャード並列でスケールするだけ。

---

## 初期実装（2026-08-06）

`tools/dataset/` に、標準ライブラリだけで動く初期パイプラインを追加した。

```bash
python -m unittest discover -s tools/dataset/tests -v
python -m tools.dataset.main japanpost --download --archive data/raw/japanpost/ken_all.zip --out data/interim/japanpost_terms.jsonl
python -m tools.dataset.main classify --input samples_with_candidates.jsonl --out comparisons.jsonl
python -m tools.dataset.main deepseek-review --input comparisons.jsonl --out review.jsonl --model <model> --input-price-per-million <price> --output-price-per-million <price>
```

`deepseek-review` は既定で dry run であり、API 呼び出しには `--execute` が必要。入力は公開ソースに限定し、行ごとの出典・利用条件・取得日時を保持する。日本郵便データは利用条件を確認してから再配布物へ含める。

最初の全国郵便番号データ変換では、都道府県・市区町村・町域・市区町村+町域を別の IME 入力単位として生成する。町域のプレースホルダー、範囲指定、非かなの読みは除外する。

`//converter:converter_main` は本番バッチ候補取得の参考実装として調査済みだが、現在の Windows SDK と clang の組合せではテスト専用ターゲットが Abseil の `offsetof` コンパイルエラーで失敗する。専用 `mozc_batch` は、AI rewriter を登録しない本番構成の C++ バイナリとして別途追加する。

関連: [AI_BACKEND_STRATEGY.md](./AI_BACKEND_STRATEGY.md), [JAPANESE_MODELS.md](./JAPANESE_MODELS.md)
