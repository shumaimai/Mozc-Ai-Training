# AI統合 Mozc IME — 開発計画書（改訂版）

**作成日**: 2026年8月1日  
**前版**: `docs/background/PROGRESS_REPORT.md`（2025年12月7日）  
**ステータス**: Phase 3 完了（本家Mozc統合・ビルド・テスト検証済み）、Phase 4 進行中

---

## Phase 3 完了記録（2026年8月1日）

| 項目 | 結果 |
|------|------|
| `integrate_mozc.py` | 現行 Mozc（Bazel 9 / `BUILD.bazel`）向けに刷新 |
| `mozc_compat/` | 本家 API 不整合を修正 |
| 本家ビルド | `//server:mozc_server` 成功 |
| 本家テスト | 5/5 PASS |

---

## 1. プロジェクト目標

Mozc（Google日本語入力）にローカルAI（Ollama）を統合し、文脈に基づいた変換候補を**非同期・非ブロッキング**で追加する。

### 不変の設計原則

```
IMEは絶対にフリーズしない
────────────────────────
• AI処理が遅くても・失敗しても、IMEは即座に応答する
• AI候補は「あれば嬉しい」補助機能
• 1msでもブロックしない。待てないならAI候補は諦める
```

---

## 2. 現状サマリー（2026年8月時点）

### 完了済み

| 領域 | 内容 | 根拠 |
|------|------|------|
| コア実装 | 非同期AIワーカー、Ollamaバックエンド、LRUキャッシュ、JSON設定、ロギング | `src/ai/*` |
| リライター | スタンドアロン版 + Mozc統合版（`mozc_compat/`） | `src/rewriter/`, `mozc_compat/` |
| スタンドアロンビルド | Bazel 8 (bzlmod)、Windows/Linux 対応 | `MODULE.bazel`, `scripts/build.*` |
| ユニットテスト | 5テストすべて PASSED（2025年12月時点） | `docs/background/PROGRESS_REPORT.md` |
| ドキュメント | 入門・ビルド・テスト・統合ガイド | `docs/*` |
| 統合準備 | `integrate_mozc.sh/ps1`、`mozc_compat/` | コミット `a37df96` |

### 未完了・未検証

| 項目 | 問題 |
|------|------|
| **本家Mozcへの実統合** | 一度もビルド・動作確認していない |
| **統合スクリプトの鮮度** | 旧API（`cc_library_mozc`, `BUILD`）を参照。現行Mozcは `mozc_cc_library`, `BUILD.bazel` |
| **エンドツーエンドIME** | `mozc_server` / OS IME としての実機テスト未実施 |
| **`ai_config.proto`** | 定義のみ。実装は手書きJSONパーサー |
| **日本語実入力テスト** | ユニットテストはASCIIキーに限定（Windows `isctype` 問題回避） |

---

## 3. 環境の変化（前版計画からの差分）

前版（2025年12月）以降、**本家Mozcとビルド周りが大きく変わっている**。統合作業はこの前提でやり直す必要がある。

| 変化 | 影響 | 対応方針 |
|------|------|----------|
| Mozcが **Bzlmod専用**（WORKSPACE廃止、2024年11月） | 統合BUILDの記法・依存解決が変わる | `mozc_cc_library` / `mozc_cc_test` を使用 |
| BUILDファイルが **`BUILD.bazel`** に統一 | 統合スクリプトのパス・ファイル名が古い | スクリプト・ドキュメントを更新 |
| Git submodule 廃止（2026年3月） | `git clone` のみでMozc取得可能 | 統合手順を簡素化 |
| Bazel 9 対応議論中（2026年春） | 依存バージョン警告・ビルド破損の可能性 | Mozc推奨Bazelバージョンに合わせる |
| GYP廃止・Bazel本番移行中 | 配布ビルドの正規ルートはBazel | Bazel統合を正とする |
| デフォルトモデル `gemma3:1b` | テスト時間・品質のトレードオフ確定 | 現状維持。品質評価は実機フェーズで実施 |

### 本家Mozcの現行ビルド例（2026年8月 clone 時点）

```python
# mozc/src/rewriter/BUILD.bazel
load("//:build_defs.bzl", "mozc_cc_library", "mozc_cc_test")

mozc_cc_library(name = "dice_rewriter", ...)
```

```cpp
// mozc/src/rewriter/rewriter.cc — コンストラクタ内で AddRewriter
Rewriter::Rewriter(const engine::Modules& modules) {
  AddRewriter(std::make_unique<DiceRewriter>());
  // ...
}
```

---

## 4. 改訂ロードマップ

```
Phase 1–2  ✅ 完了（スタンドアロン開発）
    │
Phase 3    🔄 本家Mozc統合（最優先）
    │
Phase 4       実機検証・品質改善
    │
Phase 5       機能拡張・上流貢献
```

---

### Phase 3: 本家Mozc統合 【最優先】

**ゴール**: `google/mozc` のツリー内で `//ai:*` と `//rewriter:ai_rewriter` がビルド・テスト通過し、`mozc_server` に組み込まれる。

#### 3-1. 統合基盤の更新（このリポジトリ側）

| # | タスク | 詳細 |
|---|--------|------|
| 3-1a | 統合スクリプト刷新 | `integrate_mozc.sh/ps1` を現行Mozc向けに更新：`BUILD.bazel`、`mozc_cc_*`、`//:build_defs.bzl` |
| 3-1b | `mozc_compat/` の検証 | `ai_rewriter.cc/h` が実Mozc API（`converter::Candidate`, `push_back_candidate` 等）と整合しているか確認 |
| 3-1c | AIモジュール BUILD 生成 | `ai/BUILD.bazel` を Mozc 規約で自動生成（`//base:logging`, `//testing:gunit_main` 等） |
| 3-1d | ドキュメント更新 | `docs/guides/MOZC_INTEGRATION.md` を現行手順に合わせて書き換え |
| 3-1e | スタンドアロン `WORKSPACE` 整理 | Bazel 8+ のみサポートと明記。`WORKSPACE` は削除または非推奨化を検討 |

#### 3-2. Mozcツリーへの組み込み

| # | タスク | 詳細 |
|---|--------|------|
| 3-2a | ファイル配置 | `src/ai/*` → `mozc/src/ai/`、`mozc_compat/ai_rewriter.*` → `mozc/src/rewriter/` |
| 3-2b | `rewriter/BUILD.bazel` 追記 | `ai_rewriter` ターゲットと `rewriter` ライブラリへの `deps` 追加 |
| 3-2c | Rewriter Chain 登録 | `rewriter.cc` に `AIRewriter` を追加。推奨位置: `LanguageAwareRewriter` の後、`RemoveRedundantCandidateRewriter` の前 |
| 3-2d | コンパイルフラグ（任意） | `MOZC_AI_REWRITER` でビルド時ON/OFF可能にする（`MOZC_DATE_REWRITER` 等と同パターン） |
| 3-2e | ビルド確認 | `bazelisk build //ai:all //rewriter:ai_rewriter //server:mozc_server` |
| 3-2f | テスト確認 | `bazelisk test //ai:all //rewriter:ai_rewriter_test` |

#### 3-3. 統合の完了条件

- [ ] Mozc ツリー内で AI モジュールがコンパイル・リンク成功
- [ ] 全 AI 関連テストが Mozc 環境で PASS
- [ ] `mozc_server` が AI モジュール込みでビルド成功
- [ ] 統合スクリプトの `--dry-run` と実実行の両方が動作
- [ ] ロールバック手順が文書化・検証済み

---

### Phase 4: 実機検証・配布準備

**ゴール**: 実際の日本語入力で AI 候補が表示され、Windows MSI で配布できること。

#### 4-0. Windows インストーラー ✅ スクリプト完成

| 項目 | 状態 |
|------|------|
| `scripts/package_windows.ps1` | MSI ビルドパイプライン |
| `scripts/integrate_mozc_installer.py` | WiX インストーラーへの AI 設定同梱 |
| `installer/windows/` | デフォルト設定・セットアップスクリプト |
| `docs/guides/WINDOWS_INSTALLER.md` | ビルド・インストール手順 |
| MSI 実ビルド検証 | Windows 環境で要確認 |

#### 4-1. 実機テスト

| # | タスク | 詳細 |
|---|--------|------|
| 4-1a | Linux 実機 | ibus-mozc または直接 `mozc_server` で変換テスト |
| 4-1b | Windows 実機 | MS-IME 互換レイヤー経由で候補表示確認 |
| 4-1c | Ollama 連携 | `ollama serve` + `gemma3:1b` で候補生成・キャッシュヒットを確認 |
| 4-1d | ログ検証 | `~/.mozc/ai_log.txt` にエラーなく記録されること |
| 4-1e | フリーズテスト | Ollama 停止・遅延・タイムアウト時に IME が即応答すること |

#### 4-2. 品質・パフォーマンス

| # | タスク | 詳細 |
|---|--------|------|
| 4-2a | 応答時間計測 | `Rewrite()` 呼び出しが 1ms 未満であること |
| 4-2b | キャッシュ効率 | 2回目以降の同一入力でキャッシュヒット率を計測 |
| 4-2c | 日本語テスト追加 | Mozc 実 API 上で日本語キーの統合テスト（スタンドアロン mock ではなく） |
| 4-2d | モデル評価 | `gemma3:1b` の候補品質をサンプル文で評価。必要なら代替モデル検討 |
| 4-2e | タイムアウト調整 | `connect_timeout_ms` / `request_timeout_ms` の実測ベース調整 |

#### 4-3. 配布準備

| # | タスク | 詳細 |
|---|--------|------|
| 4-3a | インストーラー更新 | `install.sh/ps1` を Mozc 統合ビルド成果物向けに修正 |
| 4-3b | デフォルト設定 | `ai_config.json` の初回生成・サンプル同梱 |
| 4-3c | トラブルシューティング更新 | 実機で起きた問題を `docs/guides/TESTING_GUIDE.md` に反映 |

---

### Phase 5: 機能拡張・上流貢献（Phase 3–4 完了後）

優先度は低め。統合・実機検証が安定してから着手する。

| # | 機能 | 備考 |
|---|------|------|
| 5-1 | Groq / OpenAI バックエンド | `ai_config.proto` に定義済み。`ai_backend.h` 拡張 |
| 5-2 | Proto ベース設定 | JSON 手書きパーサー → Mozc 標準の proto 設定に移行 |
| 5-3 | Mozc 設定 UI 連携 | `mozc_tool` に AI 設定タブ |
| 5-4 | 候補ランキング改善 | AI 候補の表示位置・コスト調整 |
| 5-5 | 公式 Mozc への PR | `MOZC_AI_REWRITER` フラグ付きでオプション機能として提案 |
| 5-6 | 他 IME 移植 | fcitx5-mozc, ibus-mozc 個別対応 |

---

## 5. 技術的リスクと対策

| リスク | 深刻度 | 対策 |
|--------|--------|------|
| Mozc API と `mozc_compat` の不整合 | 高 | Phase 3 で最初にコンパイルエラーを洗い出し、差分を `mozc_compat/` に集約 |
| Bazel 9 / 依存バージョン不一致 | 中 | Mozc `MODULE.bazel` のバージョンに合わせる。`--check_direct_dependencies=off` は最終手段 |
| AI 候補品質が低い | 中 | 品質は Phase 4 で評価。プロンプト改善・モデル変更は後回し可 |
| Ollama 未起動時の UX | 低 | 既にグレースフルデグレード実装済み。実機で再確認 |
| 本家Mozcの破壊的変更 | 中 | 統合スクリプトで Mozc コミット SHA を pin 可能にする |
| テストが ASCII のみ | 低 | Mozc 統合後に UTF-8 対応の統合テストを追加 |

---

## 6. 推奨作業順序（次にやること）

```
1. Mozc を clone し、統合スクリプトを現行 BUILD.bazel 形式に更新
2. --dry-run → 実実行でファイル配置
3. mozc_compat/ai_rewriter.* でコンパイルエラーを修正
4. rewriter.cc に AIRewriter 登録
5. bazelisk test //ai:all //rewriter:ai_rewriter_test
6. bazelisk build //server:mozc_server
7. Ollama 起動 + 実入力テスト
8. 結果を docs/background/PROGRESS_REPORT.md に反映
```

**最初のマイルストーン**: 「Mozc ツリー内で AI テストが全部 PASS」  
**次のマイルストーン**: 「実際に日本語を打って AI 候補が出る」

---

## 7. ディレクトリ役割（改訂）

```
ai_mozc/
├── src/ai/              # AIコア（Mozcにそのままコピー）
├── src/rewriter/        # スタンドアロン用（mock interface）
├── mozc_compat/         # ★本家Mozc統合用（こちらを使う）
├── scripts/
│   ├── integrate_mozc.* # ★Phase 3で刷新が必要
│   ├── build.*          # スタンドアロン開発用
│   └── install.*        # Phase 4で更新
└── docs/
    ├── PLAN.md          # ★本ドキュメント
    └── PROGRESS_REPORT.md  # 履歴・完了報告（Phase 1–2）
```

---

## 8. 成功指標

| 段階 | 指標 |
|------|------|
| Phase 3 完了 | Mozc 内ビルド・テスト全 PASS、`mozc_server` ビルド成功 |
| Phase 4 完了 | Linux/Windows で日本語入力時に AI 候補表示、IME フリーズゼロ |
| Phase 5 着手条件 | Phase 4 の実機テストが安定（主要シナリオで再現性あり） |

---

## 関連ドキュメント

- [進捗報告（2025年12月）](PROGRESS_REPORT.md) — Phase 1–2 の完了記録
- [Mozc統合ガイド](MOZC_INTEGRATION.md) — ※Phase 3-1d で更新予定
- [ビルドガイド](BUILD_GUIDE.md)
- [テストガイド](TESTING_GUIDE.md)

---

*最終更新: 2026年8月1日*
