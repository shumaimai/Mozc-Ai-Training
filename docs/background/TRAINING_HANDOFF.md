# AI Training 引継ぎ

最終更新: 2026-08-07（Step 4 拡大完了: aozora / japanpost / wikidata の mozc_batch+classify。Step 5 DeepSeek は aozora generation_gap の小規模 execute 済み）

## 1. リポジトリと目的

- GitHub: `shumaimai/Mozc-Ai-Training`（Private）
- ローカル: リポジトリのルート
- 既定ブランチ: `main`
- 目的: Mozc-AI 本体とは分離して、IME 専用 AI のデータ収集、品質管理、評価、LoRA/QLoRA 学習を管理する。

`Mozc-Ai` 本体は IME、C++ Rewriter、Windows MSI、インストーラだけを管理する。生データ、学習データ、DeepSeek 審査結果、モデル成果物は本リポジトリでのみ扱う。

## 2. Git 状態

初期移行済みコミット:

```text
dd94731 Exclude generated Python bytecode
79fb84c Add private training data pipeline
```

`main` は `origin/main` を追跡する。元の長い `claude/...` ブランチも remote に残っているが、既定ブランチは `main`。

## 3. 公開・保持ポリシー

このリポジトリは Private だが、以下を Git に入れない。

```text
data/raw/
data/interim/
data/train/
data/review/
__pycache__/
*.pyc
```

追跡対象:

```text
tools/dataset/
data/sources.json
data/benchmark/v1.jsonl
docs/
AGENTS.md
```

各入力レコードには少なくとも次を持たせる。

```text
source_id
source_url
license_id
retrieved_at
source_version
reading_source
reading_confidence
```

ユーザー入力、個人の変換履歴、選択イベントは外部 API へ送信しない。将来集約する場合は opt-in、匿名化、削除手段、送信前フィルタが必須。

## 4. 実装済みのパイプライン

`tools/dataset/` は Python 3.12 と標準ライブラリだけで動く。

| ファイル | 役割 |
|---|---|
| `records.py` | 来歴付き `TermRecord` と Mozc 比較結果のデータ構造 |
| `normalize.py` | NFKC、カタカナ→ひらがな、読み/表記の検証 |
| `japanpost.py` | 日本郵便郵便番号 ZIP の取得・読み付き住所語彙化 |
| `aozora.py` | 青空文庫の GitHub ミラー索引取得・公有作品フィルタ・定型ヘッダ除去・ルビ抽出 |
| `classify.py` | Mozc top K / top 50 から `abstain`、`rerank`、`dictionary_gap`、`generation_gap`、`reject` を分類 |
| `jsonl.py` | JSONL 入出力 |
| `wikidata.py` | Wikidata Query Service から日本の地名/施設＋かな読み(P1814)を取得（CC0、unverified、バックオフ付き） |
| `mozc_batch.py` | mozc_batch 前後処理 + C++ バイナリ起動（`mozc-run`） |
| `deepseek_review.py` | 公開データ限定の DeepSeek JSON 審査、予算カウンタ |
| `main.py` | CLI |

ローカルパス: `config/mozc_batch.env.example` → `config/mozc_batch.env`（gitignore）。エンジンデータは `tools/mozc/mozc.data` に安定コピー推奨（`bazel-bin` が fastbuild/opt で切り替わるため）。

CLI:

```powershell
python -m tools.dataset.main --help
python -m tools.dataset.main japanpost --download --archive data/raw/japanpost/ken_all.zip --out data/interim/japanpost_terms.jsonl
python -m tools.dataset.main aozora-ruby --input <text> --out <records.jsonl> --source-id <id> --source-url <url>
python -m tools.dataset.main aozora-index --out data/interim/aozora_index.jsonl
python -m tools.dataset.main aozora-fetch --index data/interim/aozora_index.jsonl --out data/interim/aozora_ruby.jsonl --work-ids <id1> <id2> ...
python -m tools.dataset.main wikidata-places --out data/interim/wikidata_places.jsonl
python -m tools.dataset.main mozc-keys --input <records.jsonl> --out keys.txt
python -m tools.dataset.main mozc-merge --records <records.jsonl> --candidates candidates.tsv --out <classify_input.jsonl>
# 一括: keys → C++ mozc_batch → merge（config/mozc_batch.env）
python -m tools.dataset.main mozc-run --records data/interim/aozora_ruby.jsonl --out data/interim/mozc_batch/aozora/classify_in.jsonl --work-dir data/interim/mozc_batch/aozora
python -m tools.dataset.main classify --input <classify_input.jsonl> --out <comparisons.jsonl>
python -m tools.dataset.main deepseek-review --input <comparisons.jsonl> --out <reviews.jsonl> --model deepseek-chat --input-price-per-million <price> --output-price-per-million <price>
.\scripts\run_mozc_candidates.ps1 -Records data\interim\aozora_ruby.jsonl
```

`deepseek-review` は既定では dry run。実 API 呼び出しには `--execute` が必要。

## 5. データソース台帳

`data/sources.json` に初期候補を登録済み。

| source_id | 用途 | 読みの信頼度 | 配布状況 |
|---|---|---|---|
| `japanpost_zipcode` | 辞書・評価 | 公式カナ | 利用条件の精査が完了するまで再配布不可 |
| `gsi_place_names` | 地名・公共施設・評価 | 公式 | 出典表示が必要 |
| `aozora_public_domain` | 学習・評価 | ルビ | 作品単位で権利確認が必要 |
| `wikidata` | 表記・別名・分類の発見 | 未検証 | CC0 の構造化データのみ |
| `ndla` | 人名・団体・作品名の発見 | 未検証 | 出典表示と個人情報配慮が必要 |

一般のニュース本文、NHK、時事通信、共同通信、ユーザー入力は学習データに入れない。Wikipedia 本文は CC BY-SA 方針を別途決めるまで保留。

## 6. 取得済みデータの実態

以前、元の `Mozc-Ai` 側で日本郵便の小書きカナ版全国 ZIP を取得し、以下を実行した。

```text
生成レコード数: 201,331
カテゴリ: address
読みの信頼度: official
都道府県数: 47
```

プレースホルダー、町域範囲指定、非かなの読みは除外済みだった。

2026-08-06 に Training 側で再取得・再生成済み（いずれも Git 追跡外）:

```text
data/raw/japanpost/ken_all.zip            日本郵便 小書きカナ版全国 ZIP（約1.7MB）
data/interim/japanpost_terms.jsonl        201,331 件（前回実績と一致、住所・official）
data/interim/aozora_index.jsonl           18,560 件 青空文庫 公有作品カタログ
data/interim/aozora_ruby.jsonl            3,689 件 公有10作品からのルビ語彙（文脈付き）
data/interim/wikidata_places.jsonl        53,384 件 日本の地名/施設＋かな読み（CC0・unverified）
data/interim/mozc_batch/aozora/           2026-08-07: keys 2,604 / classify 3,689
data/interim/mozc_batch/japanpost/        2026-08-07: keys 195,777 / classify 201,331
data/interim/mozc_batch/wikidata/         2026-08-07: keys 49,118 / classify 53,384
data/review/aozora/                       DeepSeek generation_gap サンプル審査（gitignore）
tools/mozc/mozc.data                      mozc_batch 用エンジンデータ安定コピー（約18MB、gitignore）
config/mozc_batch.env                     ローカル exe/data パス（gitignore）
```

classify 分布（2026-08-07）:

| corpus | abstain | rerank | dictionary_gap | generation_gap | reject |
| --- | ---: | ---: | ---: | ---: | ---: |
| aozora | 1720 | 548 | 0 | 980 | 441 |
| japanpost | 56382 | 3290 | 141475 | 0 | 184 |
| wikidata | 23902 | 248 | 0 | 28597 | 637 |


青空文庫の公有作品選別は `list_person_all_extended_utf8.csv` の `作品著作権フラグ==なし` かつ `人物著作権フラグ==なし` で機械判定する。取得した10作品: 芥川竜之介『羅生門』『蜘蛛の糸』『鼻』、夏目漱石『こころ』、太宰治『走れメロス』、宮沢賢治『銀河鉄道の夜』『注文の多い料理店』、中島敦『山月記』、森鴎外『高瀬舟』、梶井基次郎『檸檬』。`底本：` 以降と記号説明ブロックは除去済み。

再取得が必要なら次を実行する。

```powershell
python -m tools.dataset.main japanpost --download --archive data/raw/japanpost/ken_all.zip --out data/interim/japanpost_terms.jsonl
python -m tools.dataset.main aozora-index --out data/interim/aozora_index.jsonl
python -m tools.dataset.main aozora-fetch --index data/interim/aozora_index.jsonl --out data/interim/aozora_ruby.jsonl --work-ids 000127 000092 000042 000773 001567 043737 043754 000624 045245 000424
python -m tools.dataset.main wikidata-places --out data/interim/wikidata_places.jsonl
```

**Wikidata の注意**: 既定エンドポイントは公式 WQS (`https://query.wikidata.org/sparql`)。本作業時は WQS が障害で 429（1req/分）だったため、`--endpoint https://qlever.cs.uni-freiburg.de/api/wikidata`（Wikidata の CC0 ミラー）で生成した。WQS 復旧後は既定エンドポイントで再生成し、権威データで置き換えること。読みは P1814 由来の `unverified` で、後段の Mozc top50 / DeepSeek で検証してから辞書採用する。

## 7. DeepSeek

- 環境変数: `DEEPSEEK_API_KEY`
- API: `https://api.deepseek.com/chat/completions`
- 推奨モデル: `deepseek-v4-flash`（旧 alias `deepseek-chat` は 2026-07-24 退役）
- 公式単価（2026-08 時点、cache miss）: input `$0.14` / output `$0.28` per 1M tokens
- ユーザー承認済み上限: 10 USD

公開の固定テストレコード1件での初期疎通に加え、2026-08-07 に青空 `generation_gap` を段階 execute。

```text
model: deepseek-v4-flash
prices: --input-price-per-million 0.14 --output-price-per-million 0.28
20件:  約455秒 / $0.0128 / accept14 reject_bad_reading2 dictionary_preferred2
       reject_not_conversion_unit1 reject_noisy_text1
       -> data/review/aozora/reviews_sample20.jsonl
100件: 約2357秒 / $0.0664 / accept73 dictionary_preferred12
       reject_not_conversion_unit11 reject_bad_reading2
       review_ambiguous1 reject_noisy_text1
       -> data/review/aozora/reviews_sample100.jsonl
101-400: 拡大バッチ実行中（推定 ~$0.20 / ~2h）
全980件: 下記コマンド（推定 ~$0.65 / ~6h @ 同レート、予算上限 $10）
```

注意: `deepseek-v4-flash` の既定 thinking mode により completion tokens が大きくなりやすい（例: input≈430 / output≈2700）。予算カウンタは指定単価と実 usage で積算する。大量実行前に dry run と小規模バッチで件数・コストを検証する。

DeepSeek は正解・読み・ライセンスを創作させる用途ではなく、規則ベースで残った曖昧行のノイズ分類にのみ使う。

全件コマンド例:

```powershell
python -m tools.dataset.main deepseek-review `
  --input data/interim/mozc_batch/aozora/comparisons.jsonl `
  --out data/review/aozora/reviews_generation_gap_all.jsonl `
  --model deepseek-v4-flash `
  --input-price-per-million 0.14 `
  --output-price-per-million 0.28 `
  --actions generation_gap `
  --max-cost-usd 10 `
  --execute
```


## 8. 評価とテスト

固定 fixture は `data/benchmark/v1.jsonl` に5件のみ。カテゴリは住所、地名、専門語、同音異義語、複合句。

実行済み:

```powershell
python -m unittest discover -s tools/dataset/tests -v
python -m compileall -q tools/dataset
```

結果: 13 tests passed（2026-08-07 時点）。

テスト対象:

- 読み正規化
- 青空ルビ抽出／定型ヘッダ除去／文脈捕捉
- 青空 公有フィルタ（作品・人物の両著作権フラグ）
- Wikidata バインディング解析・読み検証・重複排除
- mozc_batch グルー（読み抽出・候補TSV結合・classify 疎通）
- top 5 内の abstain
- top 50 内の rerank
- 固定語の dictionary gap
- DeepSeek JSON request 形式
- 予算計算

## 9. 既知の制約

### Mozc 候補バッチ

`//converter:converter_main` を候補抽出の参考として調査したが、現在の Windows SDK `10.0.26100.0` と bundled clang の組合せでは、テスト専用ターゲットの依存ビルドが Abseil の `offsetof` 定数式エラーで失敗する。

```text
static assertion expression is not an integral constant expression
```

MSI の `release_build` 成功とは別の問題。コンパイラ、SDK、Bazel、依存関係の安全設定を回避する変更は、明示承認なしに行わない。

解決方針は、AI Rewriter を登録しない専用の本番構成 C++ `mozc_batch` を追加すること。まず小規模 fixture で Converter API の呼び出しを確立する。

**2026-08-06〜07 実装・実行状況**: `mozc_batch` を実装し、AI パッチ済み Mozc ツリーでビルド・青空コーパスへの候補付与まで完了。

- **C++（`mozc_compat/mozc_batch.cc`）**: JSON 非依存の TSV 入出力。`DataManager::CreateFromFile` → `Engine::CreateEngine` → `converter->StartConversion`。ビルドは PowerShell + `offsetof` copt で成功。バイナリ例: `Mozc-Ai\.mozc-build\mozc\src\bazel-bin\converter\mozc_batch.exe`。
- **Python（`mozc-keys` / `mozc-merge` / `mozc-run`）**: 設定は `config/mozc_batch.env`。青空 3,689 件（一意読み 2,604）に top50 を付与し `classify` 済み（約 8 秒）。
- **注意（品質）**: 現行バイナリは AI パッチ済みツリー由来で起動時に **AIRewriter / DeepSeek バックエンドを初期化**する。非同期キャッシュのログが出る。学習用「素の Mozc top-N」純度を厳密に揃えるなら、AIRewriter 未登録ツリーでの再ビルドが望ましい（ユーザー承認済みで現状は AI ツリー継続）。

### 実行時 AI の文脈

IME 側の `AIRewriter` は現在先頭変換セグメントを使い、非同期で候補をキャッシュして次回変換時に出す。学習データの key/context は最終的にこの実行時仕様と一致させる必要がある。

### 選択イベント

Mozc の `RewriterInterface::Finish` は確定処理後に呼ばれる。将来的に AI 候補の採用率・無視率を端末内で記録できる。ただし Mozc 本来の履歴学習と重複しないよう、初期用途は AI 品質評価とローカル適応に限定する。

## 10. 次の実行順

1. ~~Training 側で日本郵便データを再取得し、`data/interim/japanpost_terms.jsonl` を再生成する。~~ **完了（2026-08-06、201,331 件）**
2. ~~国土地理院の地名/公共施設データ用コネクタを追加する。~~ **方針変更・完了（2026-08-06）**: GSI の無償データは公式かな読みを持たない（住居表示住所は住居番号のみ、公共施設 P02 も漢字名のみ）ことを一次情報で確認。代替として無償 CC0 の Wikidata から地名/施設＋かな(P1814) を取得する `wikidata-places` を実装（53,384 件、unverified）。GSI 公式かなが必要になった場合は「数値地図（国土基本情報）地名情報」を別途入手して専用コネクタを追加する。
3. ~~著作権切れを確認した青空文庫10作品だけを対象に、ルビ抽出と文脈付きサンプル生成を行う。~~ **完了（2026-08-06、`aozora-index`/`aozora-fetch` を実装、公有10作品3,689件）**
4. ~~専用 `mozc_batch` C++ バイナリを実装し、各サンプルに Mozc top 50 を付与する。~~ **完了（2026-08-07）**: aozora / japanpost / wikidata へ top50 付与。出力は `data/interim/mozc_batch/{aozora,japanpost,wikidata}/`（gitignore）。
5. ~~`classify` で分離する。~~ **3コーパス実行済み（2026-08-07）**。上表参照。japanpost は dictionary_gap が多い（住所語彙）。wikidata は generation_gap が多い（unverified 読み）。
6. `generation_gap`（および必要なら `rerank`）だけを DeepSeek へ送る。**青空 20/100件 execute 済み、101–400 拡大中**; 全980件コマンドは §7。wikidata generation_gap（約2.9万）は人手サンプル監査後に段階拡大。
7. 辞書のみ、辞書+ranker、LoRA 生成モデルを同じベンチマークで比較する。
8. （任意）AIRewriter なしの素 Mozc ツリーで `mozc_batch` を再ビルドし、候補純度を揃える。**現行 AI ツリーでは mozc_batch 起動時に AIRewriter が DeepSeek を初期化し、変換中に AI 候補を非同期キャッシュする**（ラベル純度に影響しうる）。

## 11. 作業上の注意

- 生データ・中間データ・レビュー結果・モデル重みを Git に追加しない。
- DeepSeek に送る前に `source_id`、`license_id`、`retrieved_at`、読みの信頼度を確認する。
- 日本郵便派生データを GitHub、MSI、Hugging Face に公開する前に利用条件と加工物の再配布可否を確認する。
- 学習済みモデルを公開する前に、全ソースとベースモデルの条件を再確認する。
- `Mozc-Ai` 本体の変更はこのリポジトリで行わない。IME 統合が必要な変更は本体リポジトリで別タスクとして実施する。
