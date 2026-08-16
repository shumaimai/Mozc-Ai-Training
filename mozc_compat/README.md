# Mozc compatibility reference

このディレクトリは、学習・評価とMozc実装のパリティを検証するための参照コードです。
v1.0.0の正式な統合・MSI作成は
[Mozc-Ai](https://github.com/shumaimai/Mozc-Ai) の `scripts/integrate_mozc.py` と
`scripts/package_windows.ps1` を使用してください。

## 主なファイル

- `mozc_batch.cc`: 公開データ用Mozc N-best候補抽出
- `context_clip.*`: Python/C++の文脈前処理パリティ
- `rerank_guard.*`: リランク対象ガード
- `rerank_rewriter.*`: loopbackデーモンへ接続するMozc Rewriter
- `rerank_margin.h`: 候補上書きのマージン条件
- `runtime_smoke_client.cc`: 固定合成入力だけを使うIPCテスト

旧AIRewriter、Ollama、DeepSeek実行時バックエンドはv1.0.0の現行ツリーから削除しました。
DeepSeek/OpenAI互換APIは `tools/dataset` の公開データレビューに限って任意利用できます。

