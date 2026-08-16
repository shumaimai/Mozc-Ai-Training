# START HERE

この文書がv1.0.0資料の入口です。

## 現行資料

1. [`OVERVIEW.md`](OVERVIEW.md) — 出荷した構成と境界
2. [`RELEASE_NOTES_V1.0.0.md`](RELEASE_NOTES_V1.0.0.md) — v1.0.0の範囲と検証
3. [`reranker/reports/PHASE3_CTX_REPORT.md`](reranker/reports/PHASE3_CTX_REPORT.md) — 文脈モデル統合の実測
4. [`reranker/plans/PLAN_CONTEXTUAL_RERANKER.md`](reranker/plans/PLAN_CONTEXTUAL_RERANKER.md) — 設計

## フォルダ

- `reranker/plans/`: 設計・判断・旧計画
- `reranker/reports/`: 公開コーパスと合成入力による評価記録
- `reranker/tasks/`: 開発中に使用した時系列の作業指示
- `guides/`: Mozc、Modal、Colab、Windowsの手順
- `background/`: 自由生成AI方式を含む過去資料

`background` と `tasks` は履歴資料です。v1.0.0の仕様判断には `OVERVIEW.md`、
リポジトリ直下の `README.md`、Mozc-Ai側のREADMEを優先してください。

## 非公開データの境界

個人のIME変換ログ、private usage fine-tune、個人チャットはドキュメントや成果物の
リンク対象にしません。ローカルに残っている `PERSONAL_USAGE_*` レポートや
`artifacts/private` はGitHubへアップロードしないでください。

Modalへ渡せるのは `data/public/rerank_ctx` に明示的に置いた公開データだけです。

