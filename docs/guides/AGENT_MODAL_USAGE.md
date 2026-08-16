# エージェント運用手順: Modal で学習・評価を回す

対象: リポジトリ作業エージェント
前提: **Modal の登録・認証（`modal setup`）はユーザが済ませてある。** 実行スクリプトは `scripts/modal_train.py`。
原則: **必ずリポジトリのルートから実行**（`tools/` と `data/rerank_ctx/` が見える所）。Modal はファイルを読むだけで mmap しないので `/mnt/c` でも可。

---

## 0. Modal を使う理由（運用上の約束）

- 学習・評価は**ローカル GPU ではなく Modal 上**で回す。ROCm/WSL の不安定さを避ける。
- ジョブは**デタッチで投げる**（ユーザが離席していても完走する）。
- 成果物は **Modal Volume `mozc-artifacts`** に残し、必要時に取り出す。**大きな重みを git に入れない。**

---

## GPU 割り当てルール（VRAM から逆算・余裕込み）

Modal で選べる GPU（このアカウントで実在）と、モデル規模からの選び方。**限界ギリギリは狙わず 2 倍程度の余裕**を持たせる。

**学習の VRAM 概算**: おおよそ `params × 18 byte`（重み fp32＋勾配＋AdamW 状態）＋ 活性化（grad-checkpointing で小さい）。
- 70M ≈ 2〜4GB / 130M ≈ 4〜6GB / 310M ≈ 8〜12GB / ~1B ≈ 16〜24GB

**割り当て表**

| モデル規模 | 学習VRAM目安 | 既定GPU | VRAM | 参考価格 |
|--|--|--|--|--|
| **≤150M（30m/70m/130m）** | <6GB | **`L4`（既定）** / 安くは `T4` | 24 / 16GB | $0.80 / $0.59 |
| ≤400M（310m） | <12GB | `L4` / `A10` | 24GB | $0.80 / $1.10 |
| ≤1B | <24GB | `A100`（40GB） | 40GB | $2.10 |
| 1〜3B | <40GB | `A100-80GB` | 80GB | $2.50 |
| >3B / 大バッチ | 大 | `H100` | 80GB | $3.95 |

- **本プロジェクトのリランカーは全部 ≤130M** → 実質 **`L4` 既定でよい**（このPCのローカル16GBで回せている＝T4でも足りるが、L4は少し余裕＋高速で価格差わずか）。
- **推論・評価は学習の 1/3 以下**しか要らないので **`T4` で十分**（`modal_eval.py` は `gpu="T4"` でよい）。
- OOM が出たら `--batch-size` を下げるのが先。それでもダメなら 1 段上の GPU。
- GPU 文字列は Modal の受理名を使う（`T4`,`L4`,`A100`,`A100-80GB`,`H100` は安定。`A10`/`L40S` は名称が受理されるか要確認）。変更は `scripts/modal_train.py` / `modal_eval.py` の `gpu=` を編集。

---

## 1. 学習を投げる（Track A: 素体から）

```bash
# デタッチ実行（投げっぱなしで完走する。ログは Modal ダッシュボード or `modal app logs` で追える）
modal run --detach scripts/modal_train.py \
  --train-path data/rerank_ctx/train_v2.jsonl \
  --eval-path  data/rerank_ctx/eval_unseen_v2.jsonl \
  --model cl-nagoya/ruri-v3-pt-70m \
  --out /artifacts/trackA_ruri70m \
  --epochs 2 --batch-size 256
```

- 初回はイメージビルドで数分。以後キャッシュ。
- **データ（`*_v2.jsonl`）が生成済みであること**を先に確認（無ければ `NEXT_TASK_CTX_SUBSET_FIX` を先に実施）。
- OOM なら `--batch-size` を 128 に。

進捗確認:
```bash
modal app list            # 実行中アプリ
modal app logs mozc-reranker
```

## 2. 成果物を回収

```bash
modal volume ls  mozc-artifacts
modal volume get mozc-artifacts /trackA_ruri70m ./artifacts/rerank_ctx/trackA_ruri70m
```
`cross_encoder.pt` / `train_meta.json` / `train.log` を回収してリポの `artifacts/` に置く（git には .pt を含めない。`.gitignore` 確認）。

---

## 3. 評価を Modal で回す（`scripts/modal_eval.py` を作る）

評価も GPU が要るので、`modal_train.py` と同型の `scripts/modal_eval.py` を作成する。要件:

- 同じ `image`（依存）・同じ Volume を使う。
- コード＋データ＋**評価対象チェックポイント**をコンテナに載せる（ckpt は Volume `mozc-artifacts` からマウントして読む）。
- 関数内で `tools.rerank.eval_cross_encoder` を各評価セットに対して実行し、**文脈あり/なしの両方**で走らせ、**モデル別に τ をスイープ**（`tools/rerank/margin.py`）して JSON を `/artifacts/eval/` に出す。

実行イメージ:
```bash
modal run --detach scripts/modal_eval.py \
  --ckpt /artifacts/trackA_ruri70m/cross_encoder.pt \
  --sets data/rerank_ctx/eval_seen_v2.jsonl,data/rerank_ctx/eval_unseen_v2.jsonl,data/rerank_ctx/eval_fresh_v2.jsonl
```
出力（Volume `/artifacts/eval/`）: `eval_<model>_<set>_{ctx,noctx}.json`＋要約。
- **主要指標**: `context_sensitive` サブセットでの `hit@1(文脈あり) − hit@1(文脈なし)` デルタ。
- 併せて hit@1/hit@3/MRR/回復率/退行率/gold_in_nbest を出す。

回収:
```bash
modal volume get mozc-artifacts /eval ./artifacts/rerank_ctx/eval
```

---

## 4. Track B（現行モデルから継続学習）

Track A が通ってから。2 点の準備が要る:

1. **学習スクリプトに `--init-from <pt>` を追加**（HF 名からではなく、既存 `cross_encoder.pt` の重みを読んで継続できるように `train_cross_encoder.py` を拡張）。
2. **現行チェックポイントを Volume に置く**:
   ```bash
   modal volume put mozc-artifacts ./artifacts/rerank/<current_ce>/cross_encoder.pt /current_ce/cross_encoder.pt
   ```
3. 実行:
   ```bash
   modal run --detach scripts/modal_train.py \
     --init-from /current_ce/cross_encoder.pt \
     --out /artifacts/trackB_continue \
     --train-path data/rerank_ctx/train_v2.jsonl --eval-path data/rerank_ctx/eval_unseen_v2.jsonl
   ```
   （`modal_train.py` の `train()`・`main()` に `init_from` 引数を通す修正も必要。）

**A/B は初期重み以外を完全に一致**（同一データ・同一ハイパラ・同一アーキ）。

---

## 5. 反復ワークフロー（まとめ）

```
[ローカル/WSL] データ生成/選別（*_v2.jsonl）      ← Mozc は Windows ネイティブ、mmap 物は ext4
        ↓ (modal run が data/ をアップロード)
[Modal GPU]  Track A 学習 → Volume
[Modal GPU]  Track B 学習 → Volume
[Modal GPU]  評価（3セット×文脈あり/なし×τスイープ）→ Volume
        ↓ (modal volume get)
[ローカル]   summary.md / PHASE_CTX_REPORT.md に比較表と結論
```

---

## 6. ガードレール
- **必ずリポルートから `modal run`**（`ModuleNotFoundError: tools` はこれが原因）。
- **`--detach` で投げる**（ユーザ離席前提）。長時間ジョブをフォアグラウンドで待たない。
- **`modal run --detach` のローカルプロセスを pkill しない**（クラウド job がキャンセルされる）。詳細は `docs/guides/AGENT_BG_OPS.md`。
- **成果物は Volume。git に .pt を入れない**（`.gitignore` 確認）。
- **途中チェックポイント**: `modal_train.py` 既定 `save_every=200`。`checkpoint_latest.pt` を Volume に commit。止まったら**同じ `--out`** で再投入（`auto_resume`）。現行ジョブより**次の run から有効**。
- **データが実在するパス**を渡す（`--train-path`）。無ければデータ生成タスクを先に。
- **τ はモデルごとに再スイープ**（使い回さない）。
- **評価は必ず文脈あり/なし両方**＋ `context_sensitive` サブセット分離集計（これが本命指標）。
- GPU は既定 `L4`（割当ルールは上の §GPU）。学習は `L4`、評価は `T4` でよい。変えるなら `scripts/modal_train.py` / `modal_eval.py` の `gpu=` を編集。
- 依存は Image に固定（`transformers>=4.48,<5` 等）。むやみに変えない（イメージ再ビルド＆再現性）。
- 状態確認: `scripts/bg_status.sh`

## 7. 成果物
- `scripts/modal_eval.py`（新規）／必要なら `train_cross_encoder.py` の `--init-from` 対応
- Volume `mozc-artifacts` 上の trackA / trackB / eval
- `docs/reranker/reports/PHASE_CTX_REPORT.md` に、Modal 実行での学習・評価結果（A vs B、文脈デルタ）を追記
- 最後に「文脈は効いたか／A・B どちらを採るか」を §5.1（`PLAN_CONTEXTUAL_RERANKER`）の決定ルールで結論
