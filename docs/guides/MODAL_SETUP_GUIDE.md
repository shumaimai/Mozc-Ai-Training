# Modal セットアップ＆学習手順（投げて離席できる）

狙い: CLI から `modal run` でジョブを投げ、GPU で学習が走って成果物が Volume に残る。**離席してOK**。Colab のように張り付かなくていい。

前提: Modal のアカウント登録は済ませてある。実行スクリプトは `scripts/modal_train.py`。

---

## 1. 一度だけのセットアップ

ローカル（Windows でも WSL でもよいが、**リポジトリのファイルが見える環境**）で:

```bash
pip install modal
modal setup          # ブラウザが開いて認証。トークンが ~/.modal.toml に保存される
```

> `modal setup` が使えない環境なら `modal token new` でも可。CI 等では環境変数 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` でも認証できる。

確認:
```bash
modal profile current      # ログインできているか
```

---

## 2. どこから実行するか

**リポジトリのルート**（`tools/` と `data/rerank_ctx/` が見える場所）で `modal run` する。
スクリプトはそのカレントから `tools/` と `data/rerank_ctx/` を**コンテナにアップロードして**実行する。

- Windows で回すなら: リポジトリのルートで PowerShell/cmd から `modal run ...`
- WSL で回すなら: リポジトリのルート（`/mnt/c/...` でも可。Modal は**ファイルを読むだけで mmap しない**ので /mnt/c 問題は無関係）

> データが `train_v2.jsonl` などまだ無い場合は、先に `NEXT_TASK_CTX_SUBSET_FIX` のデータ生成を済ませる。スクリプトの `--train-path` は実在するファイルに合わせる。

---

## 3. 学習を投げる

```bash
# 既定（Track A: 素体 ruri-v3-pt-70m から）
modal run scripts/modal_train.py

# パラメータを変える
modal run scripts/modal_train.py --epochs 3 --batch-size 512
modal run scripts/modal_train.py --model cl-nagoya/ruri-v3-pt-30m --out /artifacts/track30m
modal run scripts/modal_train.py --train-path data/rerank_ctx/train_v2.jsonl --eval-path data/rerank_ctx/eval_unseen_v2.jsonl
```

- **初回はコンテナイメージのビルドで数分**かかる（依存の DL）。2 回目以降はキャッシュされて速い。
- 実行中はログが流れる。**ここで Ctrl-C してもジョブは走り続ける**（`modal run` はデタッチ可能）。進捗は Modal のダッシュボード（web）でも見られる。
- 離席するなら、投げてブラウザを閉じてよい。終われば Volume に残る。

---

## 4. 成果物を取り出す

学習は `/artifacts/...`（Modal Volume `mozc-artifacts`）に保存される。

```bash
modal volume ls  mozc-artifacts                      # 中身を見る
modal volume get mozc-artifacts /trackA_ruri70m ./artifacts_dl   # ローカルへDL
```

`cross_encoder.pt` / `train_meta.json` / `train.log` が入っている。

---

## 5. GPU とコスト感

70m は小さいので **A10G（既定）で十分**。学習1回は分〜十数分オーダー。

| GPU | 用途 | 目安 |
|--|--|--|
| `T4` | 一番安い | 遅め |
| `A10G`（既定） | バランス | 推奨 |
| `L4` | やや速い/安い | 良い代替 |
| `A100` | 速い | 大きめ実験用 |

Modal は**秒課金・scale-to-zero**なので、走った分だけ。小さい学習なら1回あたり数十円〜。GPU を変えるなら `scripts/modal_train.py` の `gpu="A10G"` を書き換える。

---

## 6. Track B（現行モデルから継続学習）について

`modal_train.py` は **Track A（素体から）** 用。Track B（現行 CE から継続）は、現行チェックポイント（`.pt`）をコンテナに載せ、学習スクリプト側で「HF名ではなくローカル重みから再開」する対応が要る。まず Track A を Modal で通してから、Track B 用に:
- 現行チェックポイントを Volume にアップ（`modal volume put mozc-artifacts ./artifacts/rerank/<ckpt> /current_ce`）
- `train_cross_encoder.py` に「`--init-from <pt>` で重みを読んで継続」する引数を足す

この2点が要るので、Track B に進むときに別途対応する。

---

## 7. ハマりどころ

| 症状 | 対処 |
|--|--|
| `modal: command not found` | `pip install modal` した Python の PATH を確認。venv 推奨 |
| 認証エラー | `modal setup`（または `modal token new`）をやり直す |
| `FileNotFoundError: data/rerank_ctx/train_v2.jsonl` | データ未生成。パスを実在ファイルに合わせる（`--train-path`） |
| `ModuleNotFoundError: tools` | **リポジトリのルートから** `modal run` しているか確認（`tools/` が見える所） |
| イメージビルドが毎回遅い | 依存を変えなければ 2 回目以降キャッシュされる。頻繁に変えない |
| CUDA OOM | `--batch-size` を下げる（256→128）。A10G なら 256 は通るはず |
| 成果物が見つからない | `/artifacts/...` に出しているか、`artifacts.commit()` 後に `modal volume get` |

---

## 8. まとめ（最短フロー）

```bash
pip install modal && modal setup            # 初回だけ
cd <リポジトリのルート>
modal run scripts/modal_train.py            # 投げて離席
modal volume get mozc-artifacts /trackA_ruri70m ./artifacts_dl   # 後で回収
```
