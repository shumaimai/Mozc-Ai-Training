# WSL↔Windows 境界解消（Mozc N-best）と検証結果

対象: 文脈付きリランカデータ構築（`docs/reranker/plans/PLAN_CONTEXTUAL_RERANKER.md` §2 前段）  
最終更新: 2026-08-12

## 構成

```
[WSL/ext4] Sudachi 抽出 → ユニーク読み keys.txt ＋ 文脈/gold レコード
      │ (keys.txt のみ受け渡し)
      ▼
[Windows ネイティブ] mozc_batch.exe（全引数 Windows パス）→ candidates.tsv
      │ (candidates.tsv のみ受け取り)
      ▼
[WSL/ext4] attach_mozc join → attached.jsonl → ambiguous → assemble-small
      → train.jsonl / eval_*.jsonl
```

- Mozc には**文脈を渡さない**（無文脈 N-best。選択は後段 CE）。
- `mozc.data` の mmap は **Windows プロセス + NTFS** のみ。WSL から `/mnt/c` を mmap しない。
- 作業/HF/モデル用 mmap は **ext4**（`~/work/mozc-ai-training`）。`df -T` で確認。

## 実装

| 部品 | 役割 |
|--|--|
| `tools/dataset/mozc_batch.py` | Mode A/B、キー正規化（ひらがな+NFKC）、ユニーク化、join |
| `tools/rerank/build_ctx_dataset.py` `attach_mozc` | 上記ヘルパー経由（生の `C:\...` を WSL argv に渡さない） |
| `scripts/run_mozc_batch_mode_a.ps1` | Mode A: Windows PowerShell から exe 実行 |
| `scripts/_run_ctx_mozc_smoke200.sh` | ~200 キー Mode B スモーク |
| `scripts/_run_ctx_mozc_assemble.sh` | 本スケール join / assemble |
| `config/mozc_batch.env` | Windows パスのまま（`MOZC_BATCH_EXE` / `MOZC_ENGINE_DATA_PATH`） |

### Mode A（推奨・確実）

1. Python が `keys.txt` を書き、貼り付け用 PowerShell を表示して停止（`--mode a`）。
2. Windows で `scripts/run_mozc_batch_mode_a.ps1`（または表示コマンド）を実行 → `candidates.tsv`。
3. `--mode join`（または `auto` で既存 TSV 再利用）で join。

### Mode B（WSL interop）

- exe: `/mnt/c/.../mozc_batch.exe`
- `--engine_data_path` / `--input` / `--output` はすべて `wslpath -w`
- 失敗時は Mode A 手順を表示してフォールバック

## 検証結果（2026-08-12）

### ファイルシステム

| パス | `df -T` |
|--|--|
| `$HOME/work/mozc-ai-training` | **ext4** |
| `/mnt/c/.../Mozc-Ai-Training/data/rerank_ctx` | 9p（テキスト I/O のみ・mmap しない） |

### スモーク（Mode **B**, ~200 unique keys）

- Work: `data/rerank_ctx/work/mozc_smoke200/`
- `はし` N-best 先頭付近: **橋 / 箸 / はし / 端** …（箸・橋・端 すべて含む）
- `gold_in_nbest` ≈ **0.849**（219 rows, empty N-best = 0）

### 本スケール（既存 `candidates.tsv` を Mode A resume = `--mode join`）

| 成果物 | パス |
|--|--|
| keys.txt | `data/rerank_ctx/work/mozc/keys.txt`（unique ≈ **30568**） |
| candidates.tsv | `data/rerank_ctx/work/mozc/candidates.tsv` |
| attached.jsonl | `data/rerank_ctx/work/mozc/attached.jsonl`（**303830** rows） |
| train.jsonl | `data/rerank_ctx/train.jsonl`（20000） |
| eval_*.jsonl | `data/rerank_ctx/eval_{seen,unseen,fresh}.jsonl` |
| assemble_summary | `data/rerank_ctx/assemble_summary.json` |

- assemble 後の train/eval は `gold_in_nbest` フィルタ済みのため各セット **frac=1.0**
- 本スケール **attached 全体** `gold_in_nbest` ≈ **0.970**（303830 rows, empty N-best = 0, hit1 ≈ 0.757）
- train `mozc_hit1` ≈ 0.78（リランク余地あり）
- 本スケール exe 実行は事前の Windows/interop 生成 TSV を再利用（`--mode join`）

ext4 同期先: `$HOME/work/mozc-ai-training/data/rerank_ctx/`（学習用）

## 学習ステップへの残課題（本タスク外）

データ構築ブロッカーは解消済み。`PLAN_CONTEXTUAL_RERANKER` の学習側は別途:

- ROCm / Docker 上で ext4 の `train.jsonl` を読むこと（`/mnt/c` mmap 禁止は継続）
- `mozc_batch` が AIRewriter を起動し API 401 を出しても N-best 自体は出るが、将来的に AIRewriter 無効化ビルドがあると静か
- Colab は使わない（計画どおり）
