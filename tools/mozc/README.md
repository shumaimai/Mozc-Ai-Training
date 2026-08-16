# Local Mozc batch artifacts

このディレクトリには、公開データの候補生成に使うmachine-localのMozc engine dataを
置きます。バイナリとデータはgitignore対象です。

`config/mozc_batch.env.example` を `config/mozc_batch.env` にコピーし、各PCの
`mozc_batch.exe` と `mozc.data` の絶対パスを設定してください。ローカル設定には
APIキーや個人データのパスを記載しないでください。

Mozc更新後は `//data_manager/oss:mozc_dataset_for_oss` から生成された `mozc.data` を
このディレクトリへコピーします。Bazelの出力ルートは環境ごとに異なるため、固定の
ユーザー名や出力パスはリポジトリに記録しません。

