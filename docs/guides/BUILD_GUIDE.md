# AI Mozc IME ビルドガイド

このドキュメントでは、AI Mozc IMEのビルド方法、成果物の場所、テスト方法を詳しく説明します。

---

## 目次

1. [プロジェクト構造](#プロジェクト構造)
2. [ビルドターゲット一覧](#ビルドターゲット一覧)
3. [成果物の場所](#成果物の場所)
4. [ビルド手順（詳細）](#ビルド手順詳細)
5. [テストの実行](#テストの実行)
6. [デバッグビルドとリリースビルド](#デバッグビルドとリリースビルド)
7. [よくある質問](#よくある質問)

---

## プロジェクト構造

```
ai_mozc/
├── .bazelrc                    # Bazel設定ファイル
├── MODULE.bazel                # 依存関係定義（Bazel 8+用）
├── WORKSPACE                   # 依存関係定義（Bazel 7以前用）
├── README.md                   # プロジェクト概要
│
├── src/                        # ソースコード
│   ├── ai/                     # AIモジュール
│   │   ├── BUILD               # Bazel BUILDファイル
│   │   ├── ai_config.h         # 設定管理ヘッダー
│   │   ├── ai_config.cc        # 設定管理実装
│   │   ├── ai_candidate_cache.h    # キャッシュヘッダー
│   │   ├── ai_candidate_cache.cc   # キャッシュ実装
│   │   ├── ai_worker.h         # ワーカースレッドヘッダー
│   │   ├── ai_worker.cc        # ワーカースレッド実装
│   │   ├── ai_backend.h        # バックエンドインターフェース
│   │   ├── ollama_backend.cc   # Ollamaバックエンド実装
│   │   ├── mock_backend.cc     # モックバックエンド（テスト用）
│   │   ├── ai_logger.h         # ロガーヘッダー
│   │   ├── ai_logger.cc        # ロガー実装
│   │   ├── *_test.cc           # 各種テストファイル
│   │   └── ai_config.proto     # Protocol Buffers定義（将来用）
│   │
│   └── rewriter/               # Rewriterモジュール
│       ├── BUILD               # Bazel BUILDファイル
│       ├── rewriter_interface.h    # Mozcインターフェース定義
│       ├── ai_rewriter.h       # AI Rewriterヘッダー
│       ├── ai_rewriter.cc      # AI Rewriter実装
│       └── ai_rewriter_test.cc # テスト
│
├── scripts/                    # ビルド・インストールスクリプト
│   ├── build.ps1               # Windowsビルドスクリプト
│   ├── build.sh                # Linuxビルドスクリプト
│   ├── install.ps1             # Windowsインストーラー
│   └── install.sh              # Linuxインストーラー
│
├── docs/                       # ドキュメント
│   ├── GETTING_STARTED.md      # 入門ガイド
│   ├── BUILD_GUIDE.md          # このファイル
│   ├── MOZC_INTEGRATION.md     # Mozc統合ガイド
│   └── BUILD_ERRORS.md         # エラー・修正記録
│
└── bazel-bin/                  # ★ビルド成果物（自動生成）
    └── src/
        ├── ai/                 # AIモジュール成果物
        └── rewriter/           # Rewriterモジュール成果物
```

---

## ビルドターゲット一覧

### AIモジュール (`//src/ai:`)

| ターゲット | 種類 | 説明 |
|-----------|------|------|
| `//src/ai:ai` | ライブラリ | 全AIモジュールを含む統合ライブラリ |
| `//src/ai:ai_config` | ライブラリ | 設定管理 |
| `//src/ai:ai_logger` | ライブラリ | ロギング |
| `//src/ai:ai_candidate_cache` | ライブラリ | 変換候補キャッシュ |
| `//src/ai:ai_backend` | ライブラリ | AIバックエンド（Ollama/Mock） |
| `//src/ai:ai_worker` | ライブラリ | 非同期ワーカースレッド |
| `//src/ai:ai_config_test` | テスト | 設定のユニットテスト |
| `//src/ai:ai_candidate_cache_test` | テスト | キャッシュのユニットテスト |
| `//src/ai:ai_backend_test` | テスト | バックエンドのユニットテスト |
| `//src/ai:ai_worker_test` | テスト | ワーカーのユニットテスト |

### Rewriterモジュール (`//src/rewriter:`)

| ターゲット | 種類 | 説明 |
|-----------|------|------|
| `//src/rewriter:ai_rewriter` | ライブラリ | AIリライター（Mozcプラグイン） |
| `//src/rewriter:rewriter_interface` | ライブラリ | Mozcインターフェース定義 |
| `//src/rewriter:ai_rewriter_test` | テスト | リライターのユニットテスト |

### 全ターゲット

| コマンド | 説明 |
|---------|------|
| `//...` | プロジェクト全体 |
| `//src/ai:all` | AIモジュール全て（テスト含む） |
| `//src/rewriter:all` | Rewriterモジュール全て（テスト含む） |

---

## 成果物の場所

### ビルド成果物ディレクトリ

ビルド後、成果物は以下の場所に生成されます：

```
プロジェクトルート/
├── bazel-bin/              # ★メイン成果物
│   └── src/
│       ├── ai/
│       │   ├── libai_config.a          # 設定ライブラリ
│       │   ├── libai_logger.a          # ロガーライブラリ
│       │   ├── libai_candidate_cache.a # キャッシュライブラリ
│       │   ├── libai_backend.a         # バックエンドライブラリ
│       │   ├── libai_worker.a          # ワーカーライブラリ
│       │   ├── libai.a                 # 統合ライブラリ
│       │   ├── ai_config_test          # テスト実行ファイル
│       │   ├── ai_candidate_cache_test
│       │   ├── ai_backend_test
│       │   └── ai_worker_test
│       └── rewriter/
│           ├── libai_rewriter.a        # AIリライターライブラリ
│           ├── librewriter_interface.a # インターフェースライブラリ
│           └── ai_rewriter_test        # テスト実行ファイル
│
├── bazel-out/              # 中間ファイル・キャッシュ
├── bazel-testlogs/         # テストログ
└── bazel-ai_mozc/          # シンボリックリンク
```

### Windows での成果物

Windowsでは、拡張子が異なります：

```
bazel-bin/src/ai/
├── ai_config.lib           # 静的ライブラリ
├── ai_logger.lib
├── ai_candidate_cache.lib
├── ai_backend.lib
├── ai_worker.lib
├── ai.lib                  # 統合ライブラリ
├── ai_config_test.exe      # テスト実行ファイル
└── ...
```

### 成果物の確認コマンド

```bash
# Linux/macOS
ls -la bazel-bin/src/ai/
ls -la bazel-bin/src/rewriter/

# Windows (PowerShell)
Get-ChildItem bazel-bin\src\ai\
Get-ChildItem bazel-bin\src\rewriter\
```

---

## ビルド手順（詳細）

### 前提条件の確認

#### Linux/macOS

```bash
# 1. Bazelisk確認
bazelisk version

# インストールされていない場合
curl -Lo /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x /usr/local/bin/bazel

# 2. コンパイラ確認
g++ --version   # GCC
# または
clang++ --version  # Clang
```

#### Windows

```powershell
# 1. Bazelisk確認
bazelisk version

# インストールされていない場合
choco install bazelisk

# 2. Visual Studio確認
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest

# 3. 環境変数確認（オプション）
echo $env:BAZEL_VC
```

### ビルドスクリプトを使用（推奨）

#### Linux/macOS

```bash
cd /path/to/ai_mozc

# 基本ビルド（デバッグ）
./scripts/build.sh

# リリースビルド（最適化）
./scripts/build.sh --release

# クリーンビルド
./scripts/build.sh --clean

# ビルド + テスト
./scripts/build.sh --test

# 全オプション
./scripts/build.sh --clean --release --test
```

#### Windows

```powershell
cd C:\m\ai_mozc

# 基本ビルド（デバッグ）
.\scripts\build.ps1

# リリースビルド（最適化）
.\scripts\build.ps1 -Release

# クリーンビルド
.\scripts\build.ps1 -Clean

# ビルド + テスト
.\scripts\build.ps1 -Test

# 全オプション
.\scripts\build.ps1 -Clean -Release -Test

# 前提条件のみチェック
.\scripts\build.ps1 -CheckOnly
```

### Bazelコマンドを直接使用

```bash
cd /path/to/ai_mozc

# === ビルド ===

# AIモジュールのみ
bazelisk build //src/ai:ai

# AIリライターのみ
bazelisk build //src/rewriter:ai_rewriter

# 全モジュール
bazelisk build //...

# 特定のライブラリのみ
bazelisk build //src/ai:ai_config
bazelisk build //src/ai:ai_backend

# === デバッグ/リリース ===

# デバッグビルド（デフォルト）
bazelisk build --config=debug //...

# リリースビルド（最適化）
bazelisk build --config=release_build //...

# === クリーン ===

# 通常クリーン
bazelisk clean

# 完全クリーン（キャッシュも削除）
bazelisk clean --expunge
```

---

## テストの実行

### 全テスト実行

```bash
# Linux/macOS
./scripts/build.sh --test

# Windows
.\scripts\build.ps1 -Test

# Bazel直接
bazelisk test //src/ai:all //src/rewriter:all
```

### 特定のテスト実行

```bash
# 個別テスト
bazelisk test //src/ai:ai_config_test
bazelisk test //src/ai:ai_candidate_cache_test
bazelisk test //src/ai:ai_backend_test
bazelisk test //src/ai:ai_worker_test
bazelisk test //src/rewriter:ai_rewriter_test

# テスト出力を表示
bazelisk test --test_output=all //src/ai:ai_config_test

# 失敗したテストの出力を表示
bazelisk test --test_output=errors //...
```

### テスト結果の確認

```bash
# テストログの場所
ls bazel-testlogs/src/ai/

# 特定テストのログ確認
cat bazel-testlogs/src/ai/ai_config_test/test.log
```

---

## デバッグビルドとリリースビルド

### 違い

| 項目 | デバッグビルド | リリースビルド |
|------|--------------|---------------|
| 最適化 | なし (`-O0`) | 最大 (`-O2`/`-O3`) |
| デバッグ情報 | あり (`-g`) | なし |
| アサーション | 有効 | 無効 |
| ファイルサイズ | 大きい | 小さい |
| 実行速度 | 遅い | 速い |
| 用途 | 開発・デバッグ | 本番・配布 |

### 設定ファイル（.bazelrc）

```
# .bazelrc の内容
build:debug --compilation_mode=dbg
build:debug -c dbg

build:release_build --compilation_mode=opt
build:release_build -c opt
```

### 使い分け

```bash
# 開発中 → デバッグビルド
bazelisk build --config=debug //...

# リリース前 → リリースビルド
bazelisk build --config=release_build //...
```

---

## よくある質問

### Q: ビルド成果物はどこにありますか？

**A:** `bazel-bin/` ディレクトリに生成されます。

```bash
# 確認コマンド
ls -la bazel-bin/src/ai/
ls -la bazel-bin/src/rewriter/
```

### Q: ビルドが失敗した場合は？

**A:** 以下を確認してください：

1. **前提条件の確認**
   ```bash
   # Linux
   bazelisk version
   g++ --version

   # Windows
   .\scripts\build.ps1 -CheckOnly
   ```

2. **キャッシュクリア**
   ```bash
   bazelisk clean --expunge
   ```

3. **エラーログ確認**
   - `docs/guides/BUILD_ERRORS.md` を参照
   - 既知のエラーと解決策が記載されています

### Q: Mozc本体に統合するには？

**A:** `docs/guides/MOZC_INTEGRATION.md` を参照してください。

1. `src/ai/` を Mozc の `src/ai/` にコピー
2. `src/rewriter/ai_rewriter.*` を Mozc の `src/rewriter/` にコピー
3. BUILD ファイルを調整
4. Rewriter chain に追加

### Q: テストだけ実行したい

**A:**
```bash
# 全テスト
bazelisk test //...

# 特定のテスト
bazelisk test //src/ai:ai_config_test
```

### Q: ビルドをやり直したい

**A:**
```bash
# 通常クリーン
bazelisk clean

# 完全クリーン（推奨）
bazelisk clean --expunge

# 再ビルド
bazelisk build //...
```

### Q: ビルド時間を短縮したい

**A:**
```bash
# 並列ビルド数を増やす
bazelisk build --jobs=8 //...

# ローカルキャッシュを活用（クリーンしない）
bazelisk build //...  # 2回目以降は高速
```

---

## 次のステップ

1. **ビルド成功後**: テストを実行して動作確認
2. **開発**: `src/` 内のコードを編集、再ビルド
3. **統合**: `docs/guides/MOZC_INTEGRATION.md` を参照してMozcに統合
4. **配布**: `scripts/install.ps1` または `scripts/install.sh` でインストール

---

*最終更新: 2024年*
