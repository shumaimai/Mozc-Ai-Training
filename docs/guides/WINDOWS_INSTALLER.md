# Windows インストーラー構築ガイド

AI 統合済み Mozc の **Windows MSI インストーラー** (`MozcAI64.msi`) を作る手順です。

## 概要

| 項目 | 内容 |
|------|------|
| インストーラー形式 | MSI（Mozc 公式と同じ WiX Toolset） |
| 出力ファイル | `MozcAI64.msi` |
| AI コード | `mozc_server.exe` に静的リンク（別 DLL 不要） |
| 同梱ファイル | `ai_config.default.json`, `setup_ai_mozc.ps1` |
| ユーザー設定 | 初回インストール時に `%LOCALAPPDATA%\Google\Mozc\ai_config.json` を自動生成 |

**Ollama は MSI に含まれません**（サイズ・ライセンスの都合）。別途インストールが必要です。

---

## 前提条件（ビルド用 PC）

1. Windows 10 1809 以降
2. Visual Studio 2022（「C++ によるデスクトップ開発」）
3. [Bazelisk](https://github.com/bazelbuild/bazelisk)
4. Python 3
5. Git
6. .NET SDK（Mozc の WiX ビルド用）

---

## ワンコマンドで MSI を作る（ローカル）

```powershell
cd C:\path\to\ai_mozc
.\scripts\package_windows.ps1
```

成果物: `dist\MozcAI64.msi`

> **ファイルが見つからない場合**: `scripts\package_windows.ps1` はメインブランチに含まれています。最新の `claude/ai-mozc-ime-integration-01UtNsKb2wmAp6dYJa6c8Hut` を pull してください。

---

## GitHub Release から MSI を入手（ビルド不要）

リリースを作成すると、GitHub Actions が自動で `MozcAI64.msi` をビルドし、Release に添付します。

### 手順

1. GitHub の **Releases** → **Draft a new release**
2. **最新のコミット**からタグを作成（例: `v0.0.2`）
3. **Publish release** をクリック
4. 約 1〜2 時間後、Release ページに `MozcAI64.msi` が表示されます

> **注意**: 古いタグで Release を再実行しても、そのタグ時点のコードが使われます。修正後にビルドする場合は新しいタグを作成してください（`v0.0.1` は UTF-8 修正前のため失敗します）。

手動トリガー（開発者向け）:

- **Actions** → **Windows Release** → **Run workflow**

ローカルで Visual Studio や Bazel を入れなくても、Release から MSI をダウンロードしてインストールできます。

---

### オプション（ローカルビルド）

```powershell
# 既存の Mozc ソースを使う
.\scripts\package_windows.ps1 -MozcDir C:\mozc\src

# 依存関係ダウンロードをスキップ（2回目以降）
.\scripts\package_windows.ps1 -SkipDeps -SkipQt
```

---

## 手動ビルド（詳細）

### 1. AI モジュール統合

```powershell
python scripts\integrate_mozc.py --mozc-dir C:\mozc\src
python scripts\integrate_mozc_installer.py --mozc-dir C:\mozc\src
```

### 2. Mozc 依存関係

```powershell
cd C:\mozc\src
python build_tools\update_deps.py
python build_tools\build_qt.py --release --confirm_license
```

### 3. MSI ビルド

```powershell
bazelisk build package --config release_build
```

出力: `bazel-bin\win32\installer\MozcAI64.msi`

---

## エンドユーザー向けインストール手順

> GUI だけで確認する手順: `installer/windows/VERIFY_INSTALL.txt`（MSI の `documents\` に同梱）

### 1. Mozc AI をインストール（クリーンインストール自動）

`MozcAI64.msi` を管理者として実行するだけで、以下が自動で行われます:

1. 古い Mozc プロセスの停止
2. `C:\Program Files (x86)\Mozc\` の削除（古い x86 版）
3. `C:\Program Files\Mozc\` への 64-bit AI 版インストール
4. `ai_config.json` の配置（毎回テンプレートから上書き）
5. インストール検証結果の書き込み（`install_verify.txt`）

**手動でのファイル削除やサーバー再起動は不要です。**

インストール後に **PC を再起動** してから、下記を確認してください。

**正しいインストール先**: `C:\Program Files\Mozc\`（**(x86) ではない**）

### 2. 日本語 IME を有効化

設定 → 時刻と言語 → 言語 → 日本語 → キーボードに **Mozc** を追加

### 3. DeepSeek API をセットアップ

1. https://platform.deepseek.com/ で API キーを取得
2. **Win + R** → `sysdm.cpl` → **環境変数** → ユーザー環境変数に `DEEPSEEK_API_KEY` を追加
3. PC を再起動

### 4. 動作確認

1. `%LOCALAPPDATA%\Google\Mozc\install_verify.txt` で **RESULT: PASS** を確認
2. 日本語入力で同じ文を2回入力（1回目: Mozc 候補のみ、2回目以降: AI 候補がキャッシュから表示）
3. ログ: `%LOCALAPPDATA%\Google\Mozc\ai_log.txt`（「AIRewriter constructed」が出ていれば OK）

---

## MSI に含まれるもの

| ファイル | 説明 |
|---------|------|
| `mozc_server.exe` | AI 統合済み変換サーバー |
| `mozc_tip32.dll` / `mozc_tip64.dll` | IME 本体 |
| `mozc_tool.exe` | 設定ツール |
| `ai_config.default.json` | デフォルト AI 設定テンプレート |
| `setup_ai_mozc.ps1` | ユーザー設定の初期化スクリプト（クリーンインストール対応） |
| `pre_install_cleanup.ps1` | インストール前クリーンアップ（プロセス停止・x86 削除） |
| `post_install_verify.ps1` | インストール後検証（AIRewriter マーカー確認） |
| `VERIFY_INSTALL.txt` | エンドユーザー向け確認手順 |
| MSVC ランタイム / Qt | Mozc 標準同梱 |

---

## トラブルシューティング

### MSI ビルドが失敗する

- Visual Studio の C++ ワークロードを確認
- `BAZEL_VC` が正しいか確認（`build.ps1 -CheckOnly`）
- Mozc 公式: [build_mozc_in_windows.md](https://github.com/google/mozc/blob/master/docs/build_mozc_in_windows.md)

### AI 候補が出ない / ログがない

1. `%LOCALAPPDATA%\Google\Mozc\install_verify.txt` を確認（PASS かどうか）
2. **タスクマネージャー** (Ctrl+Shift+Esc) → **詳細** → `mozc_server.exe` を右クリック → **ファイルの場所を開く**
   - 正: `C:\Program Files\Mozc\`
   - 誤: `C:\Program Files (x86)\Mozc\` → MSI を再インストール（クリーンアップは自動）
3. 設定: `%LOCALAPPDATA%\Google\Mozc\ai_config.json`（`backend_type: deepseek`）
4. 環境変数 `DEEPSEEK_API_KEY` を設定して PC 再起動
5. ログ: `%LOCALAPPDATA%\Google\Mozc\ai_log.txt`（v0.0.3.2 以降は起動時に作成）

### インストール先が (x86) になる

v0.0.3.1 以前の MSI は Mozc 本家と同じ `ProgramFilesFolder` を使っており、環境によっては `Program Files (x86)` に入ることがあります。**v0.0.3.2 以降** で `ProgramFiles64Folder` に修正済みです。

---

## 関連ドキュメント

- [Mozc統合ガイド](MOZC_INTEGRATION.md)
- [ビルドガイド](BUILD_GUIDE.md)
- [開発計画](PLAN.md)
