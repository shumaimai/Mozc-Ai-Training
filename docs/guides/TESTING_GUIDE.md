# AI Mozc IME テスト・動作確認ガイド

このドキュメントでは、AI Mozc IMEのテスト実行方法と動作確認手順を説明します。

---

## 目次

1. [ユニットテストの実行](#ユニットテストの実行)
2. [手動動作確認](#手動動作確認)
3. [Ollamaとの接続テスト](#ollamaとの接続テスト)
4. [ログの確認方法](#ログの確認方法)
5. [パフォーマンステスト](#パフォーマンステスト)
6. [トラブルシューティング](#トラブルシューティング)

---

## ユニットテストの実行

### 全テスト一括実行

```bash
# スクリプト使用（推奨）
# Linux/macOS
./scripts/build.sh --test

# Windows
.\scripts\build.ps1 -Test
```

```bash
# Bazel直接
bazelisk test //src/ai:all //src/rewriter:all
```

### テスト一覧

| テスト名 | 対象 | 内容 |
|---------|------|------|
| `ai_config_test` | 設定管理 | 設定ファイルの読み書き、デフォルト値 |
| `ai_candidate_cache_test` | キャッシュ | キャッシュの追加・取得・有効期限 |
| `ai_backend_test` | バックエンド | MockBackendの動作、エラーハンドリング |
| `ai_worker_test` | ワーカー | 非同期処理、キュー管理 |
| `ai_rewriter_test` | リライター | 候補追加、重複除去、フリーズ防止 |

### 個別テスト実行

```bash
# 設定テスト
bazelisk test //src/ai:ai_config_test

# キャッシュテスト
bazelisk test //src/ai:ai_candidate_cache_test

# バックエンドテスト
bazelisk test //src/ai:ai_backend_test

# ワーカーテスト
bazelisk test //src/ai:ai_worker_test

# リライターテスト
bazelisk test //src/rewriter:ai_rewriter_test
```

### 詳細出力付きテスト

```bash
# 全ての出力を表示
bazelisk test --test_output=all //src/ai:ai_config_test

# エラーのみ表示
bazelisk test --test_output=errors //src/ai:all

# ストリーミング出力（リアルタイム）
bazelisk test --test_output=streamed //src/ai:ai_worker_test
```

### テスト結果の確認

```bash
# テストログの場所
ls bazel-testlogs/src/ai/

# 特定テストの詳細ログ
cat bazel-testlogs/src/ai/ai_config_test/test.log

# 失敗したテストの確認
cat bazel-testlogs/src/ai/ai_backend_test/test.log
```

---

## 手動動作確認

### テスト実行ファイルを直接実行

ビルド後、テスト実行ファイルを直接実行できます：

```bash
# Linux/macOS
./bazel-bin/src/ai/ai_config_test
./bazel-bin/src/ai/ai_candidate_cache_test
./bazel-bin/src/ai/ai_backend_test
./bazel-bin/src/ai/ai_worker_test
./bazel-bin/src/rewriter/ai_rewriter_test
```

```powershell
# Windows
.\bazel-bin\src\ai\ai_config_test.exe
.\bazel-bin\src\ai\ai_candidate_cache_test.exe
.\bazel-bin\src\ai\ai_backend_test.exe
.\bazel-bin\src\ai\ai_worker_test.exe
.\bazel-bin\src\rewriter\ai_rewriter_test.exe
```

### 期待される出力（成功時）

```
[==========] Running 5 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 5 tests from AIConfigTest
[ RUN      ] AIConfigTest.DefaultValues
[       OK ] AIConfigTest.DefaultValues (0 ms)
[ RUN      ] AIConfigTest.LoadConfig
[       OK ] AIConfigTest.LoadConfig (1 ms)
...
[----------] 5 tests from AIConfigTest (2 ms total)
[----------] Global test environment tear-down
[==========] 5 tests from 1 test suite ran. (2 ms total)
[  PASSED  ] 5 tests.
```

---

## Ollamaとの接続テスト

### 1. Ollamaの起動確認

```bash
# Ollamaサービスの状態確認
curl http://localhost:11434/api/tags

# 期待される応答（モデルがある場合）
# {"models":[{"name":"gemma3:1b",...}]}

# 応答がない場合、Ollamaを起動
ollama serve
```

### 2. モデルの確認

```bash
# インストール済みモデル一覧
ollama list

# 出力例
# NAME           ID           SIZE    MODIFIED
# gemma3:1b     abc123...    4.1 GB  2 days ago

# モデルがない場合、ダウンロード
ollama pull gemma3:1b
```

### 3. 簡単な変換テスト

```bash
# Ollamaに直接リクエスト（テスト用）
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3:1b",
  "prompt": "以下の読みに対して変換候補を3つ提案してください：きょう",
  "stream": false
}'
```

### 4. 接続タイムアウトのテスト

設定ファイルでタイムアウトを短く設定してテスト：

```json
{
  "connect_timeout_ms": 10,
  "request_timeout_ms": 100
}
```

---

## ログの確認方法

### ログファイルの場所

| OS | パス |
|----|------|
| Linux/macOS | `~/.mozc/ai_log.txt` |
| Windows | `%LOCALAPPDATA%\Google\Mozc\ai_log.txt` |

### リアルタイムログ監視

```bash
# Linux/macOS
tail -f ~/.mozc/ai_log.txt

# Windows (PowerShell)
Get-Content -Wait $env:LOCALAPPDATA\Google\Mozc\ai_log.txt
```

### ログレベルの設定

設定ファイル（`ai_config.json`）で調整：

```json
{
  "log_level": "debug",        // trace, debug, info, warn, error
  "log_ai_communication": true  // AI通信の詳細ログ
}
```

### ログ出力例

```
[2024-12-06 10:30:15] [INFO] AIConfigManager initialized
[2024-12-06 10:30:15] [DEBUG] Config loaded from: /home/user/.mozc/ai_config.json
[2024-12-06 10:30:15] [INFO] AI backend: ollama
[2024-12-06 10:30:16] [DEBUG] Request: きょう -> Candidates: 今日, 京, 鏡
[2024-12-06 10:30:16] [INFO] Cache hit for key: きょう
```

### ビルド時のログ

ビルド中のAIモジュールメッセージは `[AI-Mozc]` プレフィックス付きで標準エラー出力に出力されます：

```
[AI-Mozc] Initializing AIConfigManager...
[AI-Mozc] Backend type: ollama
[AI-Mozc] Endpoint: http://localhost:11434
```

---

## パフォーマンステスト

### 応答時間の測定

```bash
# テスト実行と時間測定
time bazelisk test //src/ai:ai_worker_test

# 詳細なタイミング情報
bazelisk test --test_output=all //src/ai:ai_worker_test 2>&1 | grep -E "(ms|elapsed)"
```

### キャッシュ効率の確認

ログでキャッシュヒット率を確認：

```bash
grep -c "Cache hit" ~/.mozc/ai_log.txt
grep -c "Cache miss" ~/.mozc/ai_log.txt
```

### フリーズ防止の確認

AI処理が遅い場合でもIMEがフリーズしないことを確認：

1. 設定でタイムアウトを極端に短く設定
2. Ollamaを停止
3. テストを実行 → タイムアウト後に正常終了することを確認

---

## トラブルシューティング

### テストが失敗する

#### 1. ビルドエラーの場合

```bash
# クリーンビルド
bazelisk clean --expunge
bazelisk build //...
bazelisk test //...
```

#### 2. 依存関係の問題

```bash
# MODULE.bazelの確認
cat MODULE.bazel

# 依存関係の再取得
bazelisk sync
```

### Ollamaに接続できない

```bash
# 1. サービスの確認
curl http://localhost:11434/api/tags

# 2. プロセスの確認
ps aux | grep ollama

# 3. ポートの確認
netstat -tlnp | grep 11434

# 4. 再起動
pkill ollama
ollama serve
```

### ログが出力されない

1. ログディレクトリが存在するか確認：
   ```bash
   ls -la ~/.mozc/
   mkdir -p ~/.mozc/
   ```

2. 書き込み権限を確認：
   ```bash
   touch ~/.mozc/test.txt
   ```

3. 設定でログを有効化：
   ```json
   {
     "log_level": "debug"
   }
   ```

### テストがハングする

```bash
# タイムアウト付きで実行
bazelisk test --test_timeout=60 //src/ai:ai_worker_test

# 特定のテストをスキップ
bazelisk test //src/ai:all --test_filter=-*SlowTest*
```

---

## チェックリスト

### ビルド確認

- [ ] `bazelisk build //...` が成功する
- [ ] `bazel-bin/src/ai/` にライブラリが生成される
- [ ] `bazel-bin/src/rewriter/` にライブラリが生成される

### テスト確認

- [ ] `ai_config_test` パス
- [ ] `ai_candidate_cache_test` パス
- [ ] `ai_backend_test` パス
- [ ] `ai_worker_test` パス
- [ ] `ai_rewriter_test` パス

### Ollama連携確認

- [ ] Ollamaが起動している
- [ ] モデル（gemma3:1b等）がインストール済み
- [ ] `curl http://localhost:11434/api/tags` が応答する

### ログ確認

- [ ] ログファイルが作成される
- [ ] ログレベル設定が反映される
- [ ] エラー時にログが記録される

---

*最終更新: 2024年*
