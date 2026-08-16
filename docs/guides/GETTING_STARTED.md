# AI統合 Mozc IME - はじめに（Getting Started）

このドキュメントでは、AI統合Mozc IMEの環境構築から実行までを詳しく説明します。

## 目次

1. [前提条件](#前提条件)
2. [環境構築 (Windows)](#環境構築-windows)
3. [環境構築 (Linux)](#環境構築-linux)
4. [Ollamaのセットアップ](#ollamaのセットアップ)
5. [ビルド方法](#ビルド方法)
6. [設定](#設定)
7. [トラブルシューティング](#トラブルシューティング)
8. [開発者向け情報](#開発者向け情報)

---

## 前提条件

### 共通
- Git
- C++17対応コンパイラ
- Bazelisk（推奨）またはBazel 6.0以上
- Ollama（AIバックエンド）

### Windows
- Windows 10/11 (64-bit)
- Visual Studio 2022 Community以上
  - 「C++によるデスクトップ開発」ワークロード
  - Windows SDK (10.0.19041.0以上)
- PowerShell 5.1以上

### Linux
- Ubuntu 20.04+ / Debian 11+ / Fedora 35+
- GCC 9+ または Clang 10+
- 必要なパッケージ: `build-essential`, `git`, `python3`

---

## 環境構築 (Windows)

### ステップ 1: Visual Studio 2022のインストール

1. [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)をダウンロード
2. インストーラーで以下を選択:
   - **ワークロード**: 「C++によるデスクトップ開発」
   - **個別コンポーネント**:
     - Windows SDK (最新版)
     - MSVC v143ビルドツール

### ステップ 2: Bazeliskのインストール

```powershell
# Chocolateyを使用（推奨）
choco install bazelisk

# または手動ダウンロード
# https://github.com/bazelbuild/bazelisk/releases から最新版をダウンロード
# bazelisk-windows-amd64.exe を bazel.exe にリネームしてPATHに追加
```

### ステップ 3: Gitのインストール

```powershell
# Chocolateyを使用
choco install git

# または https://git-scm.com/download/win からダウンロード
```

### ステップ 4: 環境変数の設定

```powershell
# 管理者権限のPowerShellで実行

# 長いパス名を有効化
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f

# BAZEL_SH環境変数を設定（Git bashのパス）
[Environment]::SetEnvironmentVariable("BAZEL_SH", "C:\Program Files\Git\bin\bash.exe", "Machine")
```

### ステップ 5: プロジェクトのクローン

```powershell
# 短いパスにクローン（ビルドエラー防止）
cd C:\
mkdir m
cd m
git clone <repository-url> ai_mozc
cd ai_mozc
```

---

## 環境構築 (Linux)

### ステップ 1: 必要なパッケージのインストール

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y build-essential git python3 python3-pip curl
```

**Fedora:**
```bash
sudo dnf install -y gcc-c++ git python3 python3-pip curl
```

**Arch Linux:**
```bash
sudo pacman -S base-devel git python python-pip curl
```

### ステップ 2: Bazeliskのインストール

```bash
# npmを使用
npm install -g @bazel/bazelisk

# または直接ダウンロード
curl -Lo /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x /usr/local/bin/bazel
```

### ステップ 3: プロジェクトのクローン

```bash
git clone <repository-url> ai_mozc
cd ai_mozc
```

---

## Ollamaのセットアップ

Ollamaは、ローカルでLLM（大規模言語モデル）を実行するためのツールです。

### Windows

1. [Ollama for Windows](https://ollama.ai/download/windows)をダウンロード
2. インストーラーを実行
3. PowerShellで以下を実行:

```powershell
# モデルをダウンロード（約4GB）
ollama pull gemma3:1b

# Ollamaサーバーが自動起動しているか確認
curl http://localhost:11434/api/tags
```

### Linux

```bash
# インストールスクリプトを実行
curl -fsSL https://ollama.ai/install.sh | sh

# モデルをダウンロード
ollama pull gemma3:1b

# サービスを起動（自動起動されない場合）
ollama serve &

# 動作確認
curl http://localhost:11434/api/tags
```

### macOS

```bash
# Homebrewを使用
brew install ollama

# モデルをダウンロード
ollama pull gemma3:1b

# サービスを起動
ollama serve &
```

### 推奨モデル

| モデル | サイズ | 速度 | 品質 | コマンド |
|--------|--------|------|------|----------|
| gemma3:1b | ~4GB | 速い | 良好 | `ollama pull gemma3:1b` |
| llama2:7b | ~4GB | 速い | 良好 | `ollama pull llama2:7b` |
| codellama:7b | ~4GB | 速い | コード特化 | `ollama pull codellama:7b` |
| mixtral:8x7b | ~26GB | 遅い | 高品質 | `ollama pull mixtral:8x7b` |

---

## ビルド方法

### Windows

```powershell
cd C:\m\ai_mozc

# デバッグビルド
.\scripts\build.ps1

# リリースビルド
.\scripts\build.ps1 -Release

# テスト付きビルド
.\scripts\build.ps1 -Test

# クリーン＆リビルド
.\scripts\build.ps1 -Clean -Release
```

### Linux/macOS

```bash
cd ~/ai_mozc

# デバッグビルド
./scripts/build.sh

# リリースビルド
./scripts/build.sh --release

# テスト付きビルド
./scripts/build.sh --test

# クリーン＆リビルド
./scripts/build.sh --clean --release
```

### 手動ビルド（Bazelコマンド）

```bash
# AIモジュールのみビルド
bazelisk build //src/ai:ai

# AIリライターをビルド
bazelisk build //src/rewriter:ai_rewriter

# 全てビルド
bazelisk build //...

# テスト実行
bazelisk test //src/ai:all //src/rewriter:all
```

---

## 設定

### 設定ファイルの場所

- **Windows**: `%LOCALAPPDATA%\Google\Mozc\ai_config.json`
- **Linux/macOS**: `~/.mozc/ai_config.json`

### 設定項目

```json
{
  "enabled": true,
  "backend_type": "deepseek",
  "api_endpoint": "https://api.deepseek.com/v1",
  "api_model": "deepseek-chat",
  "api_key_env": "DEEPSEEK_API_KEY",
  "connect_timeout_ms": 50,
  "request_timeout_ms": 500,
  "max_wait_ms": 600,
  "warmup_timeout_ms": 60000,
  "cache_ttl_seconds": 60,
  "cache_max_entries": 100,
  "cache_include_context": true,
  "history_size": 5,
  "history_expire_min": 5,
  "log_level": "info",
  "log_ai_communication": false,
  "disable_ai": false,
  "use_mock": false
}
```

**DeepSeek API を使う場合**: 環境変数 `DEEPSEEK_API_KEY` に API キーを設定してください（ユーザー環境変数推奨）。設定後 `mozc_server` を再起動します。

Ollama を使う場合は `backend_type` を `"ollama"` に変更し、`ollama_endpoint` / `ollama_model` を指定します。

### 設定項目の説明

| 項目 | 説明 | デフォルト |
|------|------|------------|
| `enabled` | AI機能の有効/無効 | `true` |
| `backend_type` | バックエンド種類 (`deepseek`, `ollama`, `groq`, `disabled`) | `deepseek` |
| `api_endpoint` | OpenAI 互換 API のベース URL | `https://api.deepseek.com/v1` |
| `api_model` | API モデル名 | `deepseek-chat` |
| `api_key_env` | API キーを読む環境変数名 | `DEEPSEEK_API_KEY` |
| `ollama_endpoint` | OllamaサーバーのURL | `http://localhost:11434` |
| `ollama_model` | 使用するモデル名 | `gemma3:1b` |
| `connect_timeout_ms` | 接続タイムアウト（ms） | `50` |
| `request_timeout_ms` | リクエストタイムアウト（ms） | `500` |
| `cache_ttl_seconds` | キャッシュ有効期限（秒） | `60` |
| `log_level` | ログレベル (`trace`, `debug`, `info`, `warn`, `error`) | `info` |
| `use_mock` | モックモード（テスト用） | `false` |

---

## トラブルシューティング

### PowerShellがクラッシュする / 「fatal」エラー

PowerShellでビルドスクリプトを実行中に「fatal」エラーが表示されてクラッシュする場合:

#### 1. 前提条件の確認

```powershell
# まず前提条件のみをチェック
.\scripts\build.ps1 -CheckOnly
```

#### 2. Bazeliskがインストールされているか確認

```powershell
# Bazeliskのバージョン確認
bazelisk version

# インストールされていない場合
# 方法1: Chocolateyを使用
choco install bazelisk

# 方法2: 手動ダウンロード
# https://github.com/bazelbuild/bazelisk/releases から
# bazelisk-windows-amd64.exe をダウンロード
# C:\Windows または PATH内のフォルダに bazel.exe としてコピー
```

#### 3. Visual Studio Build Toolsの確認

```powershell
# Visual Studioが認識されているか確認
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest

# 認識されない場合、Build Toolsをインストール:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### 4. 長いパス名を有効化（Windows）

```powershell
# 管理者権限で実行
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

#### 5. プロジェクトを短いパスに配置

Windowsでは長いパス名が問題になることがあります:
```powershell
# 悪い例: C:\Users\長い名前\Documents\Projects\ai_mozc_ime_project\...
# 良い例: C:\m\ai_mozc
```

### ビルドエラー

#### 「filesystem not found」エラー
```
C++17の`<filesystem>`が見つからない場合:
- GCC 8以下の場合: `-lstdc++fs`をリンクオプションに追加
- 現在のコードはC++17の`<filesystem>`を使用しないように修正済み
```

#### 「cannot find -lwinhttp」エラー (Windows)
```
Windows SDKが正しくインストールされていません。
Visual Studio Installerから「Windows SDK」を再インストールしてください。
```

#### Bazelビルドが失敗する
```bash
# キャッシュをクリアして再ビルド
bazelisk clean --expunge
bazelisk build //...
```

#### Bazel 8で「rules_cc not found」エラー
```
ERROR: Unable to find package for @@[unknown repo 'rules_cc']
WORKSPACE file is disabled by default in Bazel 8
```

Bazel 8以降ではWORKSPACEファイルがデフォルトで無効化されています。
このプロジェクトはMODULE.bazel（bzlmod）に対応済みです。

もし古いBazelを使用する場合は `.bazelrc` に以下を追加：
```
common --enable_workspace
```

#### パッケージが見つからないエラー
```bash
# Bazel registryのキャッシュをクリア
bazelisk clean --expunge
rm -rf ~/.cache/bazel

# 再ビルド
bazelisk build //...
```

### 実行時エラー

#### Ollamaに接続できない
```bash
# Ollamaサービスの状態確認
curl http://localhost:11434/api/tags

# サービスが起動していない場合
ollama serve
```

#### AI候補が表示されない
```
1. Ollamaが起動しているか確認
2. モデルがダウンロード済みか確認: `ollama list`
3. ログを確認: `~/.mozc/ai_log.txt` または `%LOCALAPPDATA%\Google\Mozc\ai_log.txt`
4. 設定ファイルで`enabled: true`になっているか確認
```

### ログの確認

```bash
# Linux/macOS
cat ~/.mozc/ai_log.txt

# Windows (PowerShell)
Get-Content $env:LOCALAPPDATA\Google\Mozc\ai_log.txt
```

デバッグログを有効化:
```json
{
  "log_level": "debug",
  "log_ai_communication": true
}
```

---

## 開発者向け情報

### プロジェクト構成

```
ai_mozc/
├── src/
│   ├── ai/                    # AI関連モジュール
│   │   ├── ai_config.*        # 設定管理
│   │   ├── ai_candidate_cache.* # キャッシュ
│   │   ├── ai_worker.*        # 非同期ワーカー
│   │   ├── ai_backend.*       # バックエンドインターフェース
│   │   ├── ollama_backend.cc  # Ollamaバックエンド
│   │   ├── mock_backend.cc    # テスト用モック
│   │   └── ai_logger.*        # ロギング
│   └── rewriter/
│       ├── rewriter_interface.h # Mozcインターフェース
│       └── ai_rewriter.*      # AIリライター
├── scripts/
│   ├── build.ps1              # Windows用ビルドスクリプト
│   └── build.sh               # Linux用ビルドスクリプト
├── docs/
│   ├── GETTING_STARTED.md     # このファイル
│   └── BUILD_ERRORS.md        # エラー・修正記録
├── MODULE.bazel               # Bazel 8+ 依存関係（bzlmod）
├── WORKSPACE                  # Bazel 7以前用（非推奨）
└── .bazelrc                   # Bazelオプション
```

### 新しいバックエンドの追加

1. `AIBackendInterface`を実装したクラスを作成
2. `ai_backend.h`に宣言を追加
3. `CreateBackend()`ファクトリ関数を更新
4. BUILD fileに追加

### テストの実行

```bash
# 全テスト
bazelisk test //...

# 特定のテスト
bazelisk test //src/ai:ai_config_test

# 詳細出力
bazelisk test --test_output=all //src/ai:ai_config_test
```

### ログ出力の追加

```cpp
// stderrへの出力（ビルド時のデバッグ）
#define AI_LOG(msg) std::cerr << "[AI-Mozc] " << msg << std::endl

// ファイルへのログ出力
#include "ai_logger.h"
ai::AILogger::Info("メッセージ");
ai::AILogger::Error("エラー");
ai::AILogger::Debug("デバッグ情報");
```

---

## 次のステップ

1. [README.md](../README.md) - プロジェクト概要
2. Mozcソースコードへの統合
3. インストーラーの作成

---

## サポート

問題が発生した場合:
1. ログファイルを確認
2. このドキュメントのトラブルシューティングセクションを参照
3. GitHubのIssueを作成

---

*最終更新: 2024年*
