# Mozc統合ガイド

AI Mozc IME モジュールを本家 [google/mozc](https://github.com/google/mozc) に統合する手順です。

**最終更新**: 2026年8月1日（Bzlmod / `BUILD.bazel` / `mozc_cc_*` 対応）

---

## 概要

統合後の構成:

```
mozc/src/
├── ai/                    # AIモジュール（Ollama, キャッシュ, ワーカー）
└── rewriter/
    ├── ai_rewriter.*      # Mozc互換 AIRewriter
    ├── BUILD.bazel        # ai_rewriter ターゲット追記
    └── rewriter.cc        # Rewriter Chain 登録
```

---

## 前提条件

1. Mozc ソースを clone 済み（`git clone` のみで可。submodule 不要）
2. Bazelisk がインストール済み
3. Mozc のビルドが単体で成功すること

```bash
git clone https://github.com/google/mozc.git
cd mozc/src
bazelisk build //server:mozc_server
```

---

## 自動統合（推奨）

### Linux / macOS

```bash
cd /path/to/ai_mozc
python3 scripts/integrate_mozc.py --mozc-dir /path/to/mozc/src
```

### Windows

```powershell
cd C:\path\to\ai_mozc
.\scripts\integrate_mozc.ps1 -MozcDir C:\path\to\mozc\src
```

ドライラン:

```bash
python3 scripts/integrate_mozc.py --mozc-dir /path/to/mozc/src --dry-run
```

### スクリプトが行うこと

1. `src/ai/*` → `mozc/src/ai/` にコピー
2. `mozc_compat/ai/BUILD.bazel` → `mozc/src/ai/BUILD.bazel`
3. `mozc_compat/ai_rewriter.*` → `mozc/src/rewriter/`
4. `rewriter/BUILD.bazel` に `ai_rewriter` ターゲットを追記
5. `rewriter/rewriter.cc` に `MOZC_AI_REWRITER` と `AIRewriter` 登録を追記

---

## ビルドとテスト

```bash
cd /path/to/mozc/src

# AIモジュール
bazelisk build //ai:all

# AIRewriter
bazelisk build //rewriter:ai_rewriter

# 統合後の Mozc サーバー
bazelisk build //server:mozc_server

# テスト（5件すべて PASS すること）
bazelisk test //ai:all //rewriter:ai_rewriter_test
```

**2026年8月検証結果**（Bazel 9.0.2, Linux）:

```
//ai:ai_backend_test              PASSED
//ai:ai_candidate_cache_test      PASSED
//ai:ai_config_test                PASSED
//ai:ai_worker_test                PASSED
//rewriter:ai_rewriter_test        PASSED
```

---

## 手動統合

自動スクリプトを使わない場合:

### 1. AIモジュールの配置

```bash
cp -r ai_mozc/src/ai/* mozc/src/ai/
cp ai_mozc/mozc_compat/ai/BUILD.bazel mozc/src/ai/BUILD.bazel
```

### 2. AIRewriter の配置

```bash
cp ai_mozc/mozc_compat/ai_rewriter.{h,cc} mozc/src/rewriter/
cp ai_mozc/mozc_compat/ai_rewriter_test.cc mozc/src/rewriter/
```

### 3. `rewriter/BUILD.bazel` への追記

`mozc_compat/rewriter_build.bazel.patch` の内容を `dice_rewriter` ターゲットの直前に追加し、`rewriter` ターゲットの `deps` に `":ai_rewriter"` を追加します。

### 4. `rewriter/rewriter.cc` への追記

```cpp
#define MOZC_AI_REWRITER

#ifdef MOZC_AI_REWRITER
#include "rewriter/ai_rewriter.h"
#endif

// Rewriter::Rewriter() 内、CorrectionRewriter の後:
#ifdef MOZC_AI_REWRITER
  AddRewriter(std::make_unique<AIRewriter>());
#endif
```

---

## 動作確認

1. Ollama を起動:

   ```bash
   ollama serve
   ollama pull gemma3:1b
   ```

2. 設定ファイルを作成（`~/.mozc/ai_config.json`）:

   ```json
   {
     "enabled": true,
     "backend_type": "ollama",
     "ollama_endpoint": "http://localhost:11434",
     "ollama_model": "gemma3:1b"
   }
   ```

3. Mozc サーバーを起動して入力テスト
4. ログ確認: `tail -f ~/.mozc/ai_log.txt`

---

## トラブルシューティング

### `RewriterInterface::NONE` コンパイルエラー

本家 Mozc では `NOT_AVAILABLE` を使用します。`mozc_compat/` 版を使ってください。

### Bazel 9 の依存警告

Mozc 推奨の Bazel バージョンを使用してください。問題が続く場合は Mozc の `MODULE.bazel` に合わせて Bazelisk のバージョンを固定します。

### AI候補が表示されない

1. 初回変換はキャッシュミスのため Mozc 候補のみ（設計通り）
2. 同じ入力を2回目以降で試す
3. Ollama の起動状態とログを確認

---

## ロールバック

```bash
rm -rf mozc/src/ai
rm mozc/src/rewriter/ai_rewriter.*
# rewriter/BUILD.bazel と rewriter/rewriter.cc の変更を git checkout で戻す
```

---

## mozc_batch（データセット用バッチ変換）

`mozc_batch` は学習データ生成専用のスタンドアロンバイナリで、**AIRewriter を登録しない素の Mozc エンジン**を使い、読み（かなキー）ごとに top-N 候補を出力する。IME 統合（上記の rewriter パッチ）とは独立しており、AIRewriter を登録していない Mozc ツリーでビルドする。

### 統合とビルド

```bash
# ai_rewriter を入れていない、または MOZC_AI_REWRITER 未定義のクリーンな mozc ツリーで
python3 scripts/integrate_mozc.py --mozc-dir /path/to/mozc/src --with-mozc-batch
cd /path/to/mozc/src
bazelisk build //converter:mozc_batch
```

`converter/mozc_batch.cc` と `converter/BUILD.bazel` の `mozc_batch` ターゲットが追加される。Engine 生成・変換呼び出しは同リビジョンの `converter/converter_main.cc` に合わせてあるので、pinned Mozc で API が違う場合は該当数行を converter_main.cc に合わせる。

### 実行（TSV 入出力、JSON 非依存）

```bash
# 入力: 1 行 1 読み(かな) / 出力: "<読み>\t<cand1>\t<cand2>..."
bazel-bin/converter/mozc_batch \
  --engine_data_path=/path/to/mozc.data \
  --input=keys.txt --output=candidates.tsv --max_candidates=50
```

`--engine_data_path` はビルド済みエンジンデータ（例: `bazel-bin/data_manager/oss/mozc.data`）を指す。

### Python 側の前後処理（本リポジトリ）

ローカルパスは `config/mozc_batch.env`（テンプレ: `config/mozc_batch.env.example`）。`MOZC_BATCH_EXE` と `MOZC_ENGINE_DATA_PATH` を設定する。`mozc.data` は `tools/mozc/mozc.data` へのコピーを推奨（`-c opt` 成果物。`bazel-bin` ジャンクションが fastbuild を指すと欠けることがある）。

```powershell
# 一括（推奨）
python -m tools.dataset.main mozc-run `
  --records data/interim/aozora_ruby.jsonl `
  --out data/interim/mozc_batch/aozora/classify_in.jsonl `
  --work-dir data/interim/mozc_batch/aozora
python -m tools.dataset.main classify `
  --input data/interim/mozc_batch/aozora/classify_in.jsonl `
  --out data/interim/mozc_batch/aozora/comparisons.jsonl

# またはステップ分割
python -m tools.dataset.main mozc-keys  --input data/interim/aozora_ruby.jsonl --out keys.txt
#  ↑ keys.txt を mozc_batch.exe に渡し candidates.tsv を得る
python -m tools.dataset.main mozc-merge --records data/interim/aozora_ruby.jsonl --candidates candidates.tsv --out data/interim/aozora_classify_in.jsonl

# PowerShell ラッパ（classify まで）
.\scripts\run_mozc_candidates.ps1 -Records data\interim\aozora_ruby.jsonl
```

`data/benchmark/mozc_batch_keys_sample.txt` に疎通確認用の 6 読みを用意。ビルド時の Abseil `offsetof` は `--copt=-D_CRT_USE_BUILTIN_OFFSETOF`（PowerShell で実行。Git Bash は `//` を壊す）。詳細は `docs/guides/BUILD_ERRORS.md` と `docs/background/TRAINING_HANDOFF.md` §9。

**既知の副作用**: AI パッチ済みツリーでビルドした `mozc_batch` は起動時に AIRewriter を初期化する。素の候補純度が必要なら AIRewriter 未登録ツリーで再ビルドする。

## 関連ドキュメント

- [開発計画書](PLAN.md)
- [テストガイド](TESTING_GUIDE.md)
- [mozc_compat/README.md](../mozc_compat/README.md)
