# Mozc AI Training

Mozc AI v1.0.0 の日本語文脈リランカーについて、公開データの生成、Mozc N-best候補の
作成、学習、評価、ONNX出力を再現するためのリポジトリです。

インストール可能なIMEと単一MSIは
[Mozc-Ai](https://github.com/shumaimai/Mozc-Ai) で配布します。このリポジトリの
v1.0.0 Releaseはソース、手順、評価コードの版を固定するもので、モデルや個人データを
Release assetとして配布しません。

## v1.0.0 の出荷構成

- 方式: MozcのN-best候補を文脈付きクロスエンコーダーで再順位付け
- 基盤モデル: `sbintuitions/modernbert-ja-30m`
- 出力: ONNX fp32
- 入力: 読み、直前の文脈、候補
- 既定ポリシー: `tau=2.5`、`max_len=128`、候補上限30、文脈50文字
- 推論: Mozc-Ai MSI内のCPU常駐デーモン（127.0.0.1限定）
- private usage fine-tune: v1.0.0には不採用・不収録

## 最初に読むもの

- [`docs/START_HERE.md`](docs/START_HERE.md): 現行資料の地図
- [`docs/OVERVIEW.md`](docs/OVERVIEW.md): v1.0.0の確定構成
- [`docs/RELEASE_NOTES_V1.0.0.md`](docs/RELEASE_NOTES_V1.0.0.md): 版の範囲と検証
- [`docs/HISTORY.md`](docs/HISTORY.md): 生成方式からリランク方式へ移った経緯

## セットアップ

Python 3.11または3.12を推奨します。

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements-train.txt
```

CPUのみでデータ処理と単体テストを実行できます。学習はCUDA、ROCm、Modalのいずれかを
選べます。APIキーやクラウド認証情報はリポジトリに保存しないでください。

## 主なパイプライン

### 1. 公開データを作る

```bash
python -m tools.dataset.main --help
python -m tools.rerank.prepare --help
python -m tools.rerank.build_ctx_dataset --help
```

DeepSeek/OpenAI互換APIによるレビューは任意です。実行する場合も、公開コーパスだけを
対象にし、変換ログ、チャット、氏名、メール、APIキーなどを送信しないでください。

### 2. クロスエンコーダーを学習する

```bash
python -m tools.rerank.train_cross_encoder --help
```

ROCm補助スクリプトは `scripts/rocm_*.py` と `scripts/run_rerank_gpu_train.sh`、
Modal版は `scripts/modal_train.py` にあります。

### 3. 評価とONNX出力

```bash
python -m tools.rerank.eval_contextual --help
python -m tools.rerank.export_onnx --help
python -m tools.rerank.parity_check --help
```

採用モデルをMozc-Aiへ渡す際は、ONNX、tokenizer、margin policy、基盤モデルの
ライセンス、由来情報を一緒に固定します。

## Modalと個人情報

v1.0.0のModalランチャーがアップロードするデータ元は
`data/public/rerank_ctx` だけです。生成済みの公開JSONLを必要な実行の直前に明示的に
コピーしてください。コンテナ内では `data/rerank_ctx` として見えます。

ランチャーは `private`、`personal`、`usage`、`log` 等を含むパスをローカル側と
リモート側の両方で拒否します。以下はクラウドへ送らず、このPC内だけに保持します。

- IME変換ログと利用履歴
- private usage fine-tuneとチェックポイント
- 個人のチャット・文書・識別情報
- APIキー、トークン、Modal/Hugging Face認証情報

## テスト

外部サービスや実データを必要としないテスト:

```bash
python -m unittest discover -s tools/dataset/tests -p "test_*.py"
python -m unittest discover -s tools/rerank -p "test_*.py"
```

一部のONNX・トークナイザーパリティテストは、対象モデルを明示した場合だけ実行されます。

## リポジトリ構成

- `tools/dataset/`: 公開ソース取得、正規化、候補生成、任意のLLMレビュー
- `tools/dict/`: Mozc辞書バンドル処理
- `tools/rerank/`: データ組立、学習、評価、ONNX、ガード、プライバシー検査
- `tools/train/`: 旧生成モデル実験（v1.0の出荷経路では不使用）
- `mozc_compat/`: Mozc側リランカーと互換テストの参照実装
- `scripts/modal_*.py`: public staging限定のModal GPUジョブ
- `docs/reranker/`: 設計、タスク、評価記録
- `docs/background/`: 旧方式を含む履歴資料

## ライセンス

Mozc由来コードはMozcのBSDライセンスに従います。基盤モデル
`sbintuitions/modernbert-ja-30m` はMITライセンスです。取得する各データソースと
Python依存関係にはそれぞれの利用条件が適用されます。

