# 日本語向けベースモデル選定メモ

**更新**: 2026年8月2日

IME 向け AI の素体（ベースモデル）として、Qwen 小型版のような多言語汎用モデルより **日本語特化・日本製** の方が適している。本ドキュメントは候補の比較と推奨方針をまとめる。

## 結論（先に）

| 用途 | 推奨モデル | 理由 |
|------|-----------|------|
| **今すぐ Ollama で試す** | `7shi/ezo-gemma-2-jpn:2b-instruct-q8_0` または `lucas2024/gemma-2-2b-jpn-it` | instruct 済み・Ollama コミュニティあり・日本語品質が gemma3:1b / Qwen 小型より上 |
| **LoRA 学習の素体（最優先）** | `pfnet/plamo-2-1b` | 日本語ベンチ（JMMLU）が 1B クラス最強級・Apache 2.0・PFN 製 |
| **LoRA 学習の素体（代替）** | `llm-jp/llm-jp-3.1-1.8b-instruct4` | 国立プロジェクト・日本語トークナイザー・instruct 済みで学習しやすい |
| **避ける** | `qwen2.5:0.5b`, `qwen2.5:1.5b` | 小型 Qwen は日本語が壊滅的に弱い（実運用フィードバック） |

> **注意**: PLaMo 2 1B は **チャット用 instruct ではない**（事前学習のみ）。そのまま IME に使うより、LoRA で IME タスクに合わせるか、instruct 版を選ぶ。

---

## 候補一覧

### Tier 1 — 推奨（1〜2B、IME 向き）

#### 1. PLaMo 2 1B（`pfnet/plamo-2-1b`）

| 項目 | 内容 |
|------|------|
| 開発 | Preferred Elements（PFN グループ） |
| サイズ | 1B |
| 言語 | 英語 + 日本語（phase 2 で日本語 40%） |
| ライセンス | Apache 2.0 |
| instruct | **なし**（事前学習のみ） |
| Ollama | 公式ライブラリなし。GGUF: `yt-koike/plamo-2-1b-gguf` → Modelfile で import |
| 強み | JMMLU 日本語で llm-jp-1.8b / Gemma-2-2b / Qwen2-1.5B を上回る（PFN ブログ） |
| 弱み | Samba（Mamba+Attention）アーキテクチャのため llama.cpp 対応はコミュニティ依存 |

**IME 向けの位置づけ**: 専用モデル `ai-mozc-ime` の **学習用ベース** として最有力。日本語の基礎能力が高く、Apache 2.0 で商用もしやすい。

```bash
# Hugging Face から GGUF を取得して Ollama に登録（例）
ollama create plamo-2-1b -f Modelfile.plamo
```

#### 2. llm-jp-3.1-1.8b-instruct4（`llm-jp/llm-jp-3.1-1.8b-instruct4`）

| 項目 | 内容 |
|------|------|
| 開発 | llm-jp プロジェクト（国立研究開発法人等） |
| サイズ | 1.8B |
| 言語 | 日本語中心 |
| ライセンス | Apache 2.0 |
| instruct | **あり** |
| Ollama | 公式なし。GGUF: `mmnga/llm-jp-3.1-1.8b-instruct4-gguf`（Q4_K_M ≈ 1.16GB） |
| 強み | 日本語トークナイザー最適化、instruct 済みでそのまま試せる |
| 弱み | PLaMo 2 1B よりやや重い |

**IME 向けの位置づけ**: instruct 済みなので **PoC の即試し** と **LoRA の両方** に使える。3.7B 版は `7shi/llm-jp-3-ezo-humanities` が Ollama にあるが IME には重い。

#### 3. EZO-gemma-2-jpn / gemma-2-2b-jpn-it

| 項目 | 内容 |
|------|------|
| ベース | Google `gemma-2-2b-jpn-it`（日本語 instruct） |
| サイズ | 2B |
| Ollama | `7shi/ezo-gemma-2-jpn:2b-instruct-q8_0`、`lucas2024/gemma-2-2b-jpn-it` |
| 強み | **今すぐ `ollama pull` で試せる**。日本語品質は汎用 gemma3:1b より明確に良い |
| 弱み | 日本「製」ではない（Google 日本語版）。Gemma 利用規約 |

```bash
ollama pull 7shi/ezo-gemma-2-jpn:2b-instruct-q8_0
```

`ai_config.json`:

```json
{
  "ollama_model": "7shi/ezo-gemma-2-jpn:2b-instruct-q8_0"
}
```

**IME 向けの位置づけ**: Phase 4 実機テストの **デフォルト候補** として gemma3:1b の代替に最適。専用 LoRA の完成までの暫定モデル。

---

### Tier 2 — 条件付きで検討

#### 4. PLaMo 2.1 2B（`pfnet/plamo-2.1-2b-cpt`）

- 8B からプルーニングした日本語強モデル
- **instruct なし**（`plamo-2.1-2b-vl` は instruct 版ベースだが VL 用）
- PLaMo コミュニティライセンス（商用は要確認）
- 1B より品質は上がるが IME のレイテンシはやや悪化

#### 5. OpenCALM 1B（`cyberagent/open-calm-1b`）

- CyberAgent 製、**日本語のみ** 事前学習
- 1.4B、instruct なし、やや古い（2023）
- ファインチューニング素体としては実績あり。即試しには不向き

#### 6. rinna japanese-gpt-neox-small（203M）

- 超軽量・日本語のみ
- 生成品質が IME 候補補完には弱すぎる可能性大

---

### Tier 3 — 品質は高いが IME には重い

| モデル | サイズ | 備考 |
|--------|--------|------|
| PLaMo 2.1-8B | 8B | 日本語 SOTA 級だが IME の 500ms 目標に厳しい |
| Swallow 8B | 8B | 東工大、Llama 3.1 ベース日本語強化 |
| ELYZA Llama-3-JP-8B | 8B | 日本語 instruct、GGUF あり |
| mitmul/plamo-2-translate | ~9.5B | 翻訳特化。IME には不向き |

---

## Qwen を避ける理由

| モデル | 問題 |
|--------|------|
| `qwen2.5:0.5b` | 日本語の自然さ・漢字変換が破綻しやすい |
| `qwen2.5:1.5b` | ベンチは悪くないが実 IME タスクでは品質不足の報告あり |
| 小型 Qwen 全般 | 多言語向けに圧縮されており、日本語トークン効率が悪い |

PFN のベンチマークでも、同サイズの Qwen2-1.5B は PLaMo 2 1B / llm-jp-3.1-1.8b に JMMLU で劣る。

---

## 推奨ロードマップ（モデル観点）

```
短期  gemma-2-2b-jpn-it（EZO 版）で Ollama 実機テスト
        ↓
中期  DeepSeek で教師データ生成 → PLaMo 2 1B または llm-jp-3.1-1.8b で LoRA
        ↓
長期  GGUF 化して Ollama `ai-mozc-ime` として配布
        ↓
任意  デフォルト ai_config を ai-mozc-ime に切り替え
```

## Ollama 導入例

### 即試し（日本語 instruct）

```bash
ollama pull 7shi/ezo-gemma-2-jpn:2b-instruct-q8_0
```

### llm-jp 3.1 1.8B（Modelfile 例）

```dockerfile
FROM ./llm-jp-3.1-1.8b-instruct4-Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}
{{ end }}<|im_start|>user
{{ .Prompt }}
<|im_start|>assistant
{{ .Response }}
"""

PARAMETER stop 
PARAMETER stop <|im_start|>
```

```bash
# GGUF を mmnga/llm-jp-3.1-1.8b-instruct4-gguf から取得後
ollama create llm-jp-3.1-1.8b -f Modelfile
```

### PLaMo 2 1B（Modelfile 例）

```dockerfile
FROM ./plamo-2-1b-Q4_K_M.gguf

PARAMETER temperature 0.3
PARAMETER num_predict 64
```

> PLaMo は instruct ではないため、プロンプト設計か LoRA 学習が必須。

---

## 参考リンク

- [PLaMo 2 1B (Hugging Face)](https://huggingface.co/pfnet/plamo-2-1b)
- [PLaMo 2 事前検証ブログ (PFN)](https://tech.preferred.jp/ja/blog/plamo-2/)
- [llm-jp-3.1-1.8b-instruct4](https://huggingface.co/llm-jp/llm-jp-3.1-1.8b-instruct4)
- [llm-jp GGUF (mmnga)](https://huggingface.co/mmnga/llm-jp-3.1-1.8b-instruct4-gguf)
- [EZO-gemma-2-jpn (Ollama)](https://ollama.com/7shi/ezo-gemma-2-jpn)
- [gemma-2-2b-jpn-it (Ollama)](https://ollama.com/lucas2024/gemma-2-2b-jpn-it)
- [OpenCALM 1B](https://huggingface.co/cyberagent/open-calm-1b)

関連: [AI_BACKEND_STRATEGY.md](./AI_BACKEND_STRATEGY.md)
