# Mozc AI Training v1.0.0

Mozc AIの公開データ版30M文脈リランカーを再現するソース版です。

## 含むもの

- 公開データ取得・正規化・Mozc候補生成
- 文脈付きデータセット組立
- クロスエンコーダー学習・評価
- ONNX fp32出力とPyTorch/ONNXパリティ検査
- 使用対象ガード、文脈clip、マージン評価
- CUDA、ROCm、Modal用の実行コード
- Mozc C++統合の参照実装
- Modal public stagingと敏感パス拒否テスト

## 含まないもの

- 学習データ本体、変換ログ、チャット
- private usage fine-tune、チェックポイント
- APIキー、クラウド認証情報
- MSIおよび出荷ONNX（Mozc-Ai v1.0.0 Releaseで配布）

## バージョン対応

- Training source: `v1.0.0`
- Mozc AI MSI: `v1.0.0`
- Mozc base: `3f235b4eb6fcff7d14ef5f0fb8ee56de7ee4c732`

