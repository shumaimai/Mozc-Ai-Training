# AI バックエンド戦略メモ

**更新**: 2026年8月2日

## 現状

| 項目 | 内容 |
|------|------|
| デフォルト | DeepSeek API（`deepseek-chat`）— Ollama 不安定時の推奨 |
| 実装済み | `openai_compatible_backend`（DeepSeek 等）, `ollama_backend`, `mock_backend` |
| 予定のみ | Groq 専用バックエンド（OpenAI 互換 API で代替可） |

## 「自前 AI を作る」とは何を指すか

IME 向け AI は **汎用 LLM を丸ごと自作する** のではなく、次のどれかを選ぶのが現実的です。

### A. ローカル小モデル（現行路線の強化）— 推奨

Ollama 上でモデルを差し替えるだけ。コード変更は設定のみ。

| モデル | 特徴 | IME 向き |
|--------|------|----------|
| `7shi/ezo-gemma-2-jpn:2b-instruct-q8_0` | Google 日本語版 Gemma 2 2B（**推奨・即試し**） | 日本語 ◎、速度 ○ |
| `lucas2024/gemma-2-2b-jpn-it` | 同上（別 quant） | 日本語 ◎、速度 ○ |
| `gemma3:1b` | 軽量・高速（現デフォルト） | 応答速度 ◎、日本語 △ |
| `deepseek-r1:1.5b` | 推論寄り・日本語可 | 品質 ○、やや重い |
| `qwen2.5:0.5b` | 超軽量 | **日本語 ✗（非推奨）** |
| `phi4-mini` | MS 製小型 | バランス型（日本語は要検証） |

詳細比較: [JAPANESE_MODELS.md](./JAPANESE_MODELS.md)

```json
{
  "ollama_model": "deepseek-r1:1.5b"
}
```

**メリット**: プライバシー、オフライン、MSI に同梱不要  
**デメリット**: ユーザーが Ollama + モデル DL が必要

### B. DeepSeek API（クラウド）— **実装済み（v0.0.3）**

DeepSeek は OpenAI 互換 API を提供。新バックエンド `openai_compatible_backend` 1本で以下をまとめて対応可能:

- DeepSeek (`https://api.deepseek.com`)
- Groq
- OpenAI
- ローカル LM Studio

```json
{
  "backend_type": "openai_compatible",
  "api_endpoint": "https://api.deepseek.com/v1",
  "api_model": "deepseek-chat",
  "api_key_env": "DEEPSEEK_API_KEY"
}
```

**メリット**: 高品質、セットアップ簡単（API キーのみ）  
**デメリット**: ネット必須、課金、プライバシー、レイテンシ（タイムアウト設計が重要）

### C. DeepSeek で学習データを作り、専用モデルを作る — 有力な中長期案

**アイデア**: DeepSeek（API または大モデル）で **教師データを大量生成**し、小さいモデルを **ファインチューニング** して IME 専用 AI にする。

```
┌─────────────┐     オフライン生成      ┌──────────────────┐
│ DeepSeek API │ ──────────────────▶ │ 学習データセット    │
│ (教師モデル)  │   プロンプト + 辞書   │ JSONL / parquet  │
└─────────────┘                       └────────┬─────────┘
                                             │ LoRA / QLoRA
                                             ▼
                                    ┌──────────────────┐
                                    │ 専用小モデル       │
                                    │ (0.5B〜1.5B)      │
                                    └────────┬─────────┘
                                             │ GGUF export
                                             ▼
                                    ┌──────────────────┐
                                    │ Ollama で配布      │
                                    │ ai-mozc-ime モデル │
                                    └──────────────────┘
```

#### 1データポイントの形（例）

```json
{
  "input_key": "きょうのてんき",
  "context": ["おはよう", "今日は"],
  "mozc_candidates": ["今日の天気", "きょうのてんき"],
  "ai_candidates": ["今日の天気は", "きょうの天気"]
}
```

#### データソース

| ソース | 用途 | 取得方法 |
|--------|------|----------|
| Mozc 辞書 | 読み→表記の正解ペア | 既存辞書データ |
| Wikipedia 日本語 | 自然な文脈 | 公開コーパス |
| DeepSeek 生成 | 文脈付き候補の補完 | API でバッチ生成 |
| 合成テンプレート | カバレッジ拡大 | スクリプトで自動生成 |

#### DeepSeek の役割（教師）

オフラインで以下を実行（IME 実行時ではない）:

```
入力: きょう / 文脈: [昨日は晴れ]
既存候補: [今日, きょう, 九曜]
→ 追加すべき自然な候補を最大3つ、JSONで
```

生成結果を人間レビュー or 自動フィルタ（重複除去、長さ制限）して学習データに。

#### 学習（学生モデル）

| 項目 | 推奨 |
|------|------|
| ベースモデル | **`pfnet/plamo-2-1b`** または `llm-jp/llm-jp-3.1-1.8b-instruct4`（Qwen 小型は日本語品質のため非推奨） |
| 手法 | LoRA / QLoRA（GPU 1枚で可能） |
| タスク | 入力+文脈 → 候補文字列を生成（seq2seq または JSON 出力） |
| 出力 | GGUF → Ollama Modelfile で `ai-mozc-ime` として配布 |

#### メリット

- 推論は **ローカル小モデル** → プライバシー・速度◎
- IME タスクに特化 → 汎用 LLM より候補品質が上がる可能性
- MSI にはモデル同梱せず、Ollama `pull ai-mozc-ime` で配布可能

#### リスク・注意点

| リスク | 対策 |
|--------|------|
| DeepSeek 生成データの品質 | サンプリング検証、辞書との整合チェック |
| 学習データのライセンス | 生成データの利用規約確認、辞書は BSD 等を確認 |
| 評価基準がない | 「候補の適切さ」ベンチマークセットを先に作る |
| 学習パイプラインの工数 | Phase 1 は 1万〜5万サンプルで PoC |
| 推論速度 | 0.5B + 量子化で 500ms 以内を目標 |

#### 推奨 PoC 手順（2〜4週間相当の作業量）

1. **ベンチマーク** 100〜500 入力パターンを手動で定義
2. **データ生成** DeepSeek API で 5,000 件生成（オフライン）
3. **LoRA 学習** PLaMo 2 1B または llm-jp-3.1-1.8b を Colab / ローカル GPU で学習
4. **Ollama 化** GGUF エクスポート + Modelfile
5. **比較** ezo-gemma-2-jpn / gemma3:1b / 専用モデル でベンチマーク

### D. IME 特化の小型モデル（ランキング型）— 別アプローチ

生成ではなく **既存候補のリランキング** なら、データ量・学習コストがさらに小さい。

- 入力: ひらがな + 文脈 + Mozc 候補リスト
- 出力: 各候補のスコア
- モデル: 小さな MLP / 軽量 transformer
- 推論: ONNX Runtime（Ollama 不要、数 ms）

DeepSeek はここでも教師データ生成に使える。

### E. プロンプト＋ルールのみ — すぐ試せる

モデルはそのまま、プロンプトを日本語 IME 特化にする。

```
入力: {key}
既存候補: {candidates}
文脈: {history}
→ 既存にない自然な変換を最大3つ、JSON配列で返せ
```

**メリット**: 実装コスト最小  
**デメリット**: 根本的な品質上限はモデル依存

---

## 推奨ロードマップ

```
短期  DeepSeek を Ollama 経由で試す（設定変更のみ）
  ↓
中期  DeepSeek API で学習データ生成 → LoRA ファインチューニング PoC
  ↓
      OpenAI 互換 API バックエンド（DeepSeek API / Groq）
  ↓
長期  専用モデル `ai-mozc-ime` を Ollama で配布
  ↓
任意  ランキング型 ONNX モデル（Ollama 不要）
```

## 設計上の制約（変わらない）

- IME は絶対にブロックしない
- クラウド API はタイムアウト 500ms 以下が必須
- ローカルモデルは 1〜3B パラメータが現実的上限
- MSI にモデル本体は同梱しない（サイズ・更新頻度の問題）

## DeepSeek を試す最短手順（Ollama 経由）

```bash
ollama pull deepseek-r1:1.5b
```

`%LOCALAPPDATA%\Google\Mozc\ai_config.json`:

```json
{
  "ollama_model": "deepseek-r1:1.5b"
}
```

同じ入力を2回打って、2回目以降に AI 候補が出るか確認。
