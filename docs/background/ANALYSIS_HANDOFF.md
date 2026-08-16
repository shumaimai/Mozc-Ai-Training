# 解析班向け引継ぎファイル — Mozc-AI（Windows IME）

最終更新: 2026-08-05  
対象リポジトリ: https://github.com/shumaimai/Mozc-Ai  
ベースブランチ: `claude/ai-mozc-ime-integration-01UtNsKb2wmAp6dYJa6c8Hut`  
ユーザー実機: Windows、ローカルビルド環境あり

---

## 1. このソフトは何か

Google Mozc ベースの **日本語 IME** に、変換候補を AI（現状は DeepSeek API）で補強するモジュールを静的リンクしたもの。

| 項目 | 内容 |
|------|------|
| 成果物 | `MozcAI64.msi` → `C:\Program Files\Mozc\` |
| AI 本体 | `mozc_server.exe` に静的リンク（別 DLL なし） |
| 候補の印 | 候補説明が **`AIが生成`** |
| デフォルト backend | DeepSeek（`deepseek-chat`） |
| ピン留め Mozc | `3f235b4eb6fcff7d14ef5f0fb8ee56de7ee4c732` |

ユーザー実機では **AI 候補は定期的に出ており、DeepSeek API も約 800 回呼ばれている**。  
つまり「AI 推論パスは動いている」。壊れているのは主に **ログ取得** と **MSI によるバイナリ入れ替え**。

---

## 2. 現状サマリ（2026-08-05 時点）

### 動いていること
- IME として Mozc が動作
- AI 候補（説明: `AIが生成`）が出る
- DeepSeek API 呼び出し実績あり（ユーザー報告 ~800 回）
- 設定ファイル `%LOCALAPPDATA%\Google\Mozc\ai_config.json` は存在・読込可能
- Mozc 本体データは `%LOCALAPPDATA%Low\Mozc\` に書き込み継続（例: `history.db` 更新）

### 動いていない / 未確認のこと
- **ファイルログがユーザー環境で一度も確認できていない**
  - 旧パス `%LOCALAPPDATA%\Google\Mozc\ai_log.txt` → 無し
  - Public / Temp / ホーム直下 → 無し
  - 新パス `%LOCALAPPDATA%Low\Mozc\ai_log.txt` → **新バイナリ未インストールのため未検証**
- **DebugView（`SHUMAIN.log`）に `[AI-Mozc]` が一度も出ていない**
  - 入っていたバイナリに `OutputDebugString` が無かった可能性が高い（v0.0.3.3 以前）
- **インストールした `mozc_server.exe` の日付が `2026/08/04 19:52:14` のまま**
  - MSI を何度入れても上書きされない（後述の FileVersion 問題）
- **完全削除後のクリーンインストールで、エラーが一瞬出て即終了**
  - 詳細未取得（ログ必須）。最有力は MSI **エラー 2762**（修正前 MSI）または製品レジストリ残留

### ユーザー実機で確認済みのパス状況
```
C:\Program Files\Mozc\mozc_server.exe
  LastWriteTime: 2026/08/04 19:52:14
  Length:        22674944

%LOCALAPPDATA%\Google\Mozc\   (Medium IL / 設定用)
  ai_config.json
  install_ready.txt
  install_verify.txt
  ※ ai_log.txt / ai_alive.txt なし

%LOCALAPPDATA%Low\Mozc\       (Low IL / Mozc 本体が書く場所)
  boundary.db, cform.db, encrypt_key.db, history.db,
  segment.db, server.lock, session.ipc, renderer.*.ipc
  ※ ai_log.txt なし（新ロガー未導入バイナリのため想定内）
```

---

## 3. 搭載機能・アーキテクチャ

### 3.1 AI パイプライン
```
IME (TIP DLL)
  → IPC
  → mozc_server.exe (Low Integrity sandbox)
       → Converter / Rewriter チェーン
       → AIRewriter (mozc_compat/ai_rewriter.cc を integrate で本家 rewriter にコピー)
            → AIWorker (非同期)
            → OpenAI-compatible backend (DeepSeek) via WinHTTP
            → 候補キャッシュ
            → 次回変換時に候補挿入（説明: "AIが生成"）
```

重要コード:
- `mozc_compat/ai_rewriter.cc` / `.h` … 実際に Mozc へ統合される Rewriter
- `src/ai/*` … config / logger / worker / backends
- `scripts/integrate_mozc.py` … Mozc ツリーへ AI を差し込む
- `scripts/integrate_mozc_installer.py` … WiX / MSI 資産パッチ

### 3.2 設定（デフォルト）
場所（ユーザー）: `%LOCALAPPDATA%\Google\Mozc\ai_config.json`  
テンプレ: `installer/windows/ai_config.default.json`

```json
{
  "enabled": true,
  "backend_type": "deepseek",
  "api_endpoint": "https://api.deepseek.com/v1",
  "api_model": "deepseek-chat",
  "api_key_env": "DEEPSEEK_API_KEY",
  "connect_timeout_ms": 5000,
  "request_timeout_ms": 15000,
  "max_wait_ms": 16000,
  "cache_include_context": false,
  "log_level": "info",
  "log_ai_communication": true
}
```

既知の設定バグ（修正済・PR #10）:
- `cache_include_context: true` だとキャッシュキーが毎回変わり **AI が1回しか出ない**ように見える  
  → デフォルトを `false` に変更済み

### 3.3 Windows インストーラ同梱物
`installer/windows/` 配下（MSI の `documents\` に入る想定）:

| ファイル | 役割 |
|----------|------|
| `ai_config.default.json` | 初期設定テンプレ |
| `setup_ai_mozc.ps1` | ユーザー設定シード / registry fix 呼び出し |
| `pre_install_cleanup.ps1` | プロセス停止・旧 x86 削除 |
| `post_install_verify.ps1` | バイナリ/レジストリ検証 → `install_verify.txt` |
| `fix_mozc_registry.ps1` | TIP CLSID を `Program Files\Mozc` に修正 |
| `VERIFY_INSTALL.txt` | ユーザー向け確認手順 |

インストール先は `ProgramFiles64Folder`（`C:\Program Files\Mozc`）。  
旧版は x86 に入ることがあり、レジストリが x86 のまま残る問題があった（PR #7/#10）。

### 3.4 ビルド・CI
- ローカル: `scripts/package_windows.ps1`（依存: VS2022 C++, Bazelisk, Python, Git, .NET）
- CI: `.github/workflows/windows-release.yml`（~2h、`windows-2025`）
- インストール推奨: `scripts/install_msi.ps1`（`REINSTALLMODE=emus`）

Mozc ピン: `MOZC_PIN_SHA=3f235b4eb6fcff7d14ef5f0fb8ee56de7ee4c732`

---

## 4. ログ設計（過去の変遷と現状コード）

### 4.1 なぜログが出なかったか（確定に近い結論）

Mozc は Windows で `mozc_server` を **Low Integrity** で起動する。

根拠（本家 `server_launcher.cc`）:
```
info.primary_level = USER_NON_ADMIN;
info.integrity_level = INTEGRITY_LEVEL_LOW;
info.creation_flags = CREATE_DEFAULT_ERROR_MODE | CREATE_NO_WINDOW;
```

Windows Mandatory Integrity Control では **Low → Medium への書き込みは拒否**される。

| パス | IL | AI 旧ロガー | 結果 |
|------|-----|-------------|------|
| `%LOCALAPPDATA%\Google\Mozc\` | Medium | 書き込み先だった | **書けない**（設定の読込は可） |
| `%TEMP%\MozcAI\` / Public / ホーム | Medium 相当 | フォールバック候補 | **書けない** |
| `%LOCALAPPDATA%Low\Mozc\` | Low 向け | Mozc 本体のみ使用 | **書ける**（AI は未使用だった） |

これで説明できる矛盾:
- AI は動く（設定読込・HTTP・メモリキャッシュは可能）
- ログファイルがどこにも無い（書き込みだけ失敗、stderr は `CREATE_NO_WINDOW` で捨てられる）

### 4.2 現行ロガー（PR #11 マージ済）

ファイル: `src/ai/ai_logger.cc`

| 出力 | 場所 / 手段 |
|------|-------------|
| メインファイル | **`%LOCALAPPDATA%Low\Mozc\ai_log.txt`** |
| 生存確認 | `%LOCALAPPDATA%Low\Mozc\ai_alive.txt` |
| 場所マーカー | `%LOCALAPPDATA%Low\Mozc\ai_log_location.txt` |
| DebugView | `OutputDebugStringA("[AI-Mozc] ...")` |
| 通信ログ | `log_ai_communication: true` 時、**INFO** レベルで REQUEST/RESPONSE |

※ Public tee は Low IL では書けないため、最終的に LocalLow 一本化へ修正済み。

### 4.3 DebugView について
- `mozc_server` は Session 0 サービスではない → 通常は **Capture Win32** で足りる
- Capture Global はサービス向け（必須ではない）
- **バイナリに ODS 文字列が無いと原理的に出ない**（v0.0.3.3 リリース MSI には未収録、LocalLow+ODS はソース上 PR #11 以降）
- ユーザー提出 `SHUMAIN.log` / `SHUMAIN_*.log`: Brave / iCloud 等のみ、`[AI-Mozc]` **ゼロ件**

### 4.4 ログ確認コマンド（新バイナリ導入後）
```powershell
explorer "$env:USERPROFILE\AppData\LocalLow\Mozc"
Get-Content "$env:USERPROFILE\AppData\LocalLow\Mozc\ai_alive.txt" -ea 0
Get-Content "$env:USERPROFILE\AppData\LocalLow\Mozc\ai_log.txt" -Tail 50 -ea 0
```

---

## 5. インストーラ問題（解析の第二の主戦場）

### 5.1 MSI エラー 2762（修正済・要リビルド）

症状: 「予期しないエラー…エラーコードは 2762」/ 一瞬出て即終了

原因:
- deferred カスタムアクションは `InstallInitialize`〜`InstallFinalize` の間だけ有効
- AI パッチが `PreInstallCleanup` を **deferred + Before=InstallValidate** に置いていた
- → `Cannot write script record. Transaction not started`

修正: PR #12（マージ済）
- SeedAIConfig / PostInstallVerify を **immediate + After=InstallFinalize**
- PreInstallCleanup を execute sequence から外す（ShutdownServer が代替）

**注意:** ユーザーが持っている `dist\MozcAI64.msi` が修正前ビルドだと、クリーンインストールでも 2762 のまま落ちる。

### 5.2 同じ FileVersion で exe が上書きされない（修正手順追加済）

症状: MSI を何度入れても  
`mozc_server.exe LastWriteTime = 2026/08/04 19:52:14` のまま

原因:
- Windows Installer デフォルト `REINSTALLMODE=omus`
- PE の FileVersion が同じだと **既存 exe を残す**
- AI リビルドは Mozc の VERSIONINFO を上げない

対策:
```powershell
msiexec /i MozcAI64.msi REINSTALLMODE=emus /L*v "$env:TEMP\MozcAI_install.log"
# または
.\scripts\install_msi.ps1
.\scripts\install_msi.ps1 -UninstallFirst
```

PR #13 で `scripts/install_msi.ps1` 追加（マージ済）。  
PR #14（OPEN）でフル UI + ログ解析強化。

### 5.3 クリーンインストール直後の「一瞬エラーで終了」

未解決（ログ未取得）。仮説ランキング:

1. **修正前 MSI の 2762**（最有力）
2. 製品レジストリ残留 → `NEWERVERSIONDETECTED`
3. 手動削除でファイルだけ消えて ARP/Upgrade テーブル不整合
4. カスタムアクション DLL / PowerShell 実行ポリシー（可能性低め、Return=ignore 多め）

必須の次アクション:
```powershell
msiexec /i "<path>\MozcAI64.msi" REINSTALLMODE=emus /L*v "$env:TEMP\MozcAI_install.log"
Select-String "$env:TEMP\MozcAI_install.log" -Pattern '2762','1603','Return value 3','NEWERVERSION','Installation success or error status'
```

---

## 6. リリース / PR 履歴（関連）

| Tag / PR | 内容 | 結果 |
|----------|------|------|
| v0.0.3 | DeepSeek backend | - |
| v0.0.3.1 | 早期ログ等 | CI OK |
| v0.0.3.2 | runtime + clean install | CI fail（verify path） |
| v0.0.3.2.1 | path 修正途中 | CI fail（cquery パース） |
| v0.0.3.3 | registry/cache/logger fopen | **CI success**（ODS/LocalLow は未含む） |
| PR #10 | registry TIP / cache / fopen logger | merged |
| PR #11 | LocalLow ログ + ODS + alive | merged |
| PR #12 | MSI 2762 | merged |
| PR #13 | REINSTALLMODE=emus インストール補助 | merged |
| PR #14 | インストール診断強化 | OPEN |

---

## 7. ユーザー実機メモ

- 作業ディレクトリ例: Mozc-Ai のチェックアウト先
- ローカル Bazel キャッシュに 2026-08-05 の `ai_logger` オブジェクトあり → **ソース側ビルドは進んだが Program Files へ未反映**
- PowerShell ExecutionPolicy でスクリプトが止まることあり → `Set-ExecutionPolicy -Scope Process Bypass`
- TIP レジストリ問題（過去）: x86 の `mozc_tip64.dll` を指したまま → MS IME 風 / AI 無し。レジストリ修正で回復
- API キー: ユーザー環境変数 `DEEPSEEK_API_KEY`

---

## 8. 解析班への依頼事項（優先順）

### A. インストール失敗の一次切り分け（最優先）
1. 現在の `dist\MozcAI64.msi` の日時・サイズ
2. `.mozc-build\mozc\src\win32\installer\installer_oss_64bit.wxs` に  
   - `SeedAIConfig" After="InstallFinalize"` があるか  
   - `PreInstallCleanup" Before="InstallValidate"` + deferred が残っていないか
3. `msiexec /L*v` ログで **2762 / 1603 / NEWERVERSION** の有無
4. 残留アンインストール項目（ARP）の有無

### B. 新バイナリ投入の確認
1. 最新 base で `package_windows.ps1` 再ビルド
2. `install_msi.ps1` または `REINSTALLMODE=emus` でインストール
3. `Get-Item ...\mozc_server.exe | Select LastWriteTime` が **今日**になること
4. `findstr /C:"[AI-Mozc]" /C:"LocalLow" mozc_server.exe` で新ロガー痕跡

### C. ログ経路の実証
1. IME 入力後、`%LOCALAPPDATA%Low\Mozc\ai_alive.txt` / `ai_log.txt`
2. DebugView（管理者、Capture Win32）で `[AI-Mozc]`
3. 候補説明が本当に `AIが生成` か（偽陽性排除）

### D. （任意）残課題・設計改善
- AI 設定パス（Medium `Local\Google\Mozc`）とログパス（LocalLow）が分裂している  
  → 将来は設定も LocalLow に寄せるか、読取専用と明示する
- MSI の ProductVersion / FileVersion を AI ビルドごとに上げて、`emus` 無しでも上書きできるようにする
- CI で「インストール後に LocalLow へログが出るか」までの自動検証は未整備（Windows runner + IME は重い）

---

## 9. すぐ使えるコマンド集

```powershell
# 診断
Get-Item "$env:ProgramFiles\Mozc\mozc_server.exe" -ea 0 | Select FullName,Length,LastWriteTime
Get-ChildItem "$env:LOCALAPPDATA\Google\Mozc" -Force -ea 0
Get-ChildItem "$env:USERPROFILE\AppData\LocalLow\Mozc" -Force -ea 0

# MSI ログ付きインストール（管理者）
$msi = "<Mozc-Aiのチェックアウト先>\dist\MozcAI64.msi"
msiexec /i $msi REINSTALLMODE=emus /L*v "$env:TEMP\MozcAI_install.log"

# ログ抜粋
Select-String "$env:TEMP\MozcAI_install.log" -Pattern '2762','1603','Return value 3','NEWERVERSION','Installation success or error status' | Select -Last 40

# リビルド（最新）
cd <Mozc-Aiのチェックアウト先>
git fetch
git checkout claude/ai-mozc-ime-integration-01UtNsKb2wmAp6dYJa6c8Hut
git pull
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\package_windows.ps1 -SkipDeps -SkipQt -MozcRef 3f235b4eb6fcff7d14ef5f0fb8ee56de7ee4c732
.\scripts\install_msi.ps1
```

---

## 10. 関連ドキュメント

- `docs/guides/WINDOWS_INSTALLER.md`
- `docs/guides/MOZC_INTEGRATION.md`
- `docs/background/AI_BACKEND_STRATEGY.md`
- `docs/guides/GETTING_STARTED.md`
- `docs/background/TRAINING_DATA_PIPELINE.md` / `docs/background/JAPANESE_MODELS.md`（自家製 AI は「AI 安定後」の次フェーズ）

---

## 11. 一文で状況を言うと

**AI 変換は実機で動いているが、ログは Low Integrity のせいで旧パスに書けず見えず、さらに MSI が同じ FileVersion の `mozc_server.exe` を置き換えないため、LocalLow 対応ロガー入りバイナリが Program Files に届いていない。クリーンインストール失敗は 2762 修正前 MSI の可能性が高い。解析は「msiexec 詳細ログ」と「mozc_server の更新日時」から入るのが最短。**
