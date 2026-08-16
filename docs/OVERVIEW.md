# Mozc AI Training v1.0.0 — 全体像

最終更新: 2026-08-16

## 目的

Mozcが生成した日本語変換候補を、直前の文脈に合う順へ並べ替える小型モデルを作ります。
AIは候補を新規生成せず、Mozc候補の採点だけを行います。

## v1.0.0で確定した構成

- `sbintuitions/modernbert-ja-30m` ベースの文脈付きクロスエンコーダー
- 公開データトラック `track30m_ctx`
- ONNX fp32、SentencePiece、`max_len=128`
- 候補上限30、直前文脈50文字、マージン `tau=2.5`
- 使用対象ガードと珍字・旧字の昇格防止
- 200msで失敗し、Mozc候補順へ戻るフェイルセーフ
- Windows MSI内のCPUデーモンへ127.0.0.1で接続

## v1.0.0に含めないもの

- DeepSeekや他のクラウドAPIを使う実行時バックエンド
- Ollamaを必要とする実行時バックエンド
- 個人の変換ログを使った `usage30m_v1` fine-tune
- 変換本文の既定ログ保存
- 学習データ、チェックポイント、private artifactのRelease配布

DeepSeek/OpenAI互換APIは、公開コーパスをレビューする任意のデータ作成補助としてのみ
残します。個人データを送る用途では使用しません。

## 成果物の流れ

```text
公開コーパス
  -> 正規化・読み付与
  -> Mozc N-best候補
  -> 文脈付きtrain/eval JSONL
  -> 30Mクロスエンコーダー学習
  -> seen / unseen / fresh評価
  -> fp32 ONNX出力とパリティ検査
  -> Mozc-Ai/runtime/model
  -> MozcAI-1.0.0-x64.msi
```

## 実行環境

- データ処理・テスト: WindowsまたはWSLのCPU
- ローカル学習: CUDAまたはROCm
- クラウド学習: Modal（`data/public/rerank_ctx` だけをアップロード）
- 配布推論: Windows CPU、ネットワーク不要

## セキュリティ境界

公開コードと公開データの経路、個人利用の経路を分離します。private/usage/logを示す
パスはModalランチャーが拒否します。GitHubにはAPIキー、個人ログ、privateモデルを
commitしません。

