# 次タスク指示: WSL↔Windows 境界を解いて N-best 生成を通す

対象: リポジトリ作業エージェント
前提: `docs/reranker/plans/PLAN_CONTEXTUAL_RERANKER.md`（文脈付きデータ計画）。本タスクはその §2「データ構築」の**前段ブロッカー解消**。

---

## 背景（確定事項）

文脈付きデータセット（`train.jsonl` / `attached.jsonl`）が**未生成で止まっている**。原因は WSL↔Windows の境界:

- **`/mnt/c` 経由で mmap すると失敗する**（DrvFs/9p は mmap 非対応）。
- **Windows の exe を WSL から Windows パスで起動しようとして詰まる**。

持っているもの（実体）:

| 項目 | パス |
|--|--|
| 変換器 | `<Mozc-Ai>\.mozc-build\mozc\src\bazel-bin\converter\mozc_batch.exe`（Windows PE。Linux ビルドは無い） |
| 辞書 | `tools\mozc\mozc.data` |
| 設定 | `config/mozc_batch.env`（`MOZC_BATCH_EXE` / `MOZC_ENGINE_DATA_PATH`） |
| 既存Python | `tools/dataset/mozc_batch.py`, `build_ctx_dataset.py` の `attach_mozc`（TSV を join） |

現行の呼び出し（Windows 形式）:
```
mozc_batch.exe --engine_data_path=...\mozc.data --input=keys.txt --output=candidates.tsv --max_candidates=80
# 出力 TSV:  key<TAB>cand1<TAB>cand2<TAB>...
```

**重要な前提（設計として正しい）**: Mozc には**文脈を渡さない**。キー単位の無文脈 N-best を出すのが正しい（Mozc=候補の列挙、文脈での選択は後段の CE の仕事）。したがって Mozc 側は現状のままでよい。**変えるのは"どこで exe を動かすか"だけ。**

---

## 直しの原則

**mmap が要るのは `mozc.data` だけで、それを読むのは `mozc_batch.exe`。exe を Windows プロセスとして動かせば NTFS からネイティブ mmap するので問題ゼロ。** mmap が壊れるのは「WSL プロセスが /mnt/c を mmap する」時だけ。

→ **パイプラインを分割する:**

```
[WSL/ext4] Sudachi 抽出 → 読みをユニーク化した keys.txt ＋ 文脈/gold レコード
      │ (keys.txt を渡す)
      ▼
[Windows ネイティブ] mozc_batch.exe 実行（全引数 Windows パス）→ candidates.tsv
      │ (candidates.tsv を受け取る)
      ▼
[WSL/ext4] attach_mozc で join → train.jsonl / attached.jsonl
      ▼
[WSL/ext4] 学習（モデル・データセットの mmap は ext4 なのでOK）
```

境界をまたぐのは **keys.txt を渡す／candidates.tsv を受け取る のファイル 2 枚だけ**。両方ただのテキストで mmap しないので置き場所は自由。

---

## タスク

### 1. 作業領域を ext4 に固定
- 生成・学習の作業ディレクトリを **ext4 上**（例 `~/work/mozc-ai-training/` またはコンテナ `/work/...`）にする。`/mnt/c/...` を mmap 対象にしない。
- `df -T <作業dir>` で `ext4` を確認（`9p`/`drvfs` なら退避）。
- mmap する可能性のあるもの（HF datasets キャッシュ、モデル重み、tokenizer）はすべて ext4。`HF_HOME` も ext4 に向ける。

### 2. Mozc 呼び出しを「Windows ネイティブ実行」に切り出す
`mozc_batch.py` に、境界をまたぐ変換ヘルパーを実装/修正する。**2 モードを用意**:

- **モード A（推奨・確実）: 手動 Windows 実行**
  - Python は「ユニーク読みの `keys.txt`」と「そのまま貼って実行できる PowerShell コマンド文字列」を出力して一旦停止。
  - ユーザ（または CI）が Windows 側で PowerShell を実行し `candidates.tsv` を生成。
  - Python 再開で `candidates.tsv` を join。
- **モード B（自動・WSL interop）**: WSL から exe を叩く。
  - exe は Linux パスで参照（`/mnt/c/.../mozc_batch.exe`）。
  - **exe に渡すファイル引数は全て `wslpath -w` で Windows 形式へ変換**（`--engine_data_path` `--input` `--output`）。
  - exe は Windows プロセスとして mozc.data を NTFS mmap するので mmap 問題は起きない。
  - 失敗時（interop 不可）は自動でモード A の手順を表示してフォールバック。

`config/mozc_batch.env` の `MOZC_BATCH_EXE` / `MOZC_ENGINE_DATA_PATH` は Windows パスのまま使い、引数生成時に上記変換を通す。

### 3. 読みのユニーク化（必須の最適化）
- レコードは読みを共有するので、**keys.txt はユニークな読みだけ**にする（重複排除）。
- 変換後、`reading → candidates` のマップを作り、各レコードへ join。これで Windows 側の変換量が激減する。

### 4. 正規化の一致
- keys.txt に書く読みは **ひらがな・NFKC 正規化**（Mozc が期待する形）に統一。抽出側（Sudachi 出力はカタカナ）と Mozc 入力の形式ズレで空 N-best にならないよう、**数件で往復確認**（`はし` → 箸/橋/端 が返るか）。過去に「読み形式ズレで gold_in_nbest=0」の前科があるので必ず確認。

### 5. 小スケール検証 → 本生成
- まず読み 200 件で全経路（抽出→keys→mozc_batch→join）を通し、`candidates.tsv` が `key<TAB>cand...` で来て join できることを確認。
- `gold_in_nbest` 率を出す（極端に低ければ正規化 or 抽出の不整合を疑う）。
- 問題なければ本スケールで `train.jsonl` / `attached.jsonl` を生成。

---

## 成功基準
- `df -T` で作業領域が ext4。mmap 由来のエラーが消えている。
- `mozc_batch.exe` が Windows ネイティブ実行（モード A か B）で回り、`candidates.tsv` が生成される。
- ユニーク読み経由で join され、`train.jsonl` / `attached.jsonl` が生成される。
- 小スケールで `はし`→箸/橋/端 等が N-best に入り、`gold_in_nbest` が妥当な値。

## 制約・注意
- **WSL プロセスから /mnt/c を mmap しない。** mmap 対象は全て ext4。
- **Mozc には文脈を渡さない**（無文脈 N-best が正しい。keys.txt は読みのみ）。
- exe に渡すパスは Windows 形式（モード B は `wslpath -w` 必須）。exe をシェルに素の `C:\...` で渡さない。
- 読み正規化（ひらがな/NFKC）を抽出側と一致させる。ズレたら空 N-best になる。
- keys.txt/candidates.tsv はテキストなので /mnt/c でも可。**mmap する大物だけ ext4** を徹底。
- 旧成果物は上書きせず残す。

## 成果物
- 修正した `tools/dataset/mozc_batch.py`（モード A/B、ユニーク化、正規化、join）
- （モード A を使う場合）生成される `keys.txt` と実行用 PowerShell コマンド、受け取り手順
- `data/rerank_ctx/train.jsonl` / `attached.jsonl`（小スケール→本スケール）
- `docs/` に「境界解消の構成と検証結果（gold_in_nbest 率含む）」を追記
