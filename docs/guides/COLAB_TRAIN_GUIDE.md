# Colab でリランカー学習を確実に動かす手順書

対象: `tools/rerank/train_cross_encoder.py` を Colab GPU で回す。  
**本番モデル（Phase 1/2）: `sbintuitions/modernbert-ja-70m`**（手順書初期案の ruri よりこちらが実学習済み・継続）。  
Phase 2 データ: `data/rerank_v3/`（holdout は v2 固定）。ノート: `docs/guides/COLAB_PHASE2_TRAIN.ipynb`。  
狙い: 「依存バージョンで動かない」を潰す。**セルは上から順にコピペ**すれば通る構成。

---

## 0. なぜ Colab で動かなくなるか（原因の9割）

1. **torch を pip で入れ直してしまう** → Colab 既設の torch/CUDA と食い違って壊れる。**torch は絶対に触らない**（Colab のを使う）。
2. **transformers が古い** → ModernBERT-Ja は `transformers>=4.48` が必須。古いと `KeyError: modernbert` 等で落ちる。
3. **pip 後にランタイム未再起動** → 入れ替えたパッケージが反映されず古いまま動く。**依存を入れたら一度だけ再起動**。
4. **バージョンを記録していない** → 次回また別バージョンが来て再発。→ **動いた構成を lock ファイルに固める**（本手順書の最後）。

原則: **「torch は既設を使う／その上に transformers 系だけ最小追加／入れたら再起動／動いたら freeze」**。これだけで再発はほぼ消える。

---

## 1. 事前準備

- メニュー **ランタイム → ランタイムのタイプを変更 → GPU（T4）** を選択。
- 成果物を消さないため、**Google Drive にリポjspan を置く**のを推奨（セッション切断で `/content` は消える）。

---

## 2. セル構成（この順にコピペ）

### Cell 0 — GPU と既設バージョンの確認（何も入れない）
```python
!nvidia-smi -L
import torch, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("cuda", torch.version.cuda)
```
`torch.cuda.is_available()` が `True` であること。ここで torch のバージョンを**メモ**（後で lock に使う）。torch は以降**入れ直さない**。

### Cell 1 — 上乗せ依存だけ最小インストール（torch を巻き込まない）
```python
# torch/torchvision/torchaudio は触らない。transformers 系だけ。
!pip install -q "transformers>=4.48,<5" "tokenizers>=0.21" sentencepiece accelerate
# 直後に「本当に動くか」だけ確認（赤い警告が出ても、これが通れば OK）
import transformers, importlib.metadata as m
print("transformers", transformers.__version__, "/ hub", m.version("huggingface_hub"))
```
- `sentencepiece` はトークナイザに必須。
- `transformers>=4.48` で ModernBERT-Ja がネイティブ対応（`trust_remote_code` 有無どちらでも可）。
- `flash-attn` は**入れない**（コンパイルで沼る。無くても eager で動く）。
- **赤い `ERROR: pip's dependency resolver ...` が出ても止まらない。** これは Colab 既設の別パッケージ（`gradio` など、学習に無関係）との相性を pip が愚痴っているだけ。上の確認行で transformers 版が表示されれば問題なし（詳細は §2.5）。

### Cell 2 — ランタイム再起動（重要）
```python
# 依存を入れた後、一度だけ再起動して反映させる。
import os
os.kill(os.getpid(), 9)
```
再起動後、**Cell 0 だけ再実行**して torch が生きているのを確認 → Cell 1・2 は飛ばして Cell 3 へ。

### Cell 3 — リポとデータの取り込み
必要なのは `tools/`（パッケージ）と **`data/rerank_v3/{train,holdout}.jsonl`**（無ければ暫定で `data/rerank_v2/`）。

**方法A: Google Drive（推奨・成果物も残る）**
```python
from google.colab import drive
drive.mount('/content/drive')
# 例: Drive に Mozc-Ai-Training フォルダを置いた場合
%cd /content/drive/MyDrive/Mozc-Ai-Training
!ls tools/rerank data/rerank_v3
!wc -l data/rerank_v3/train.jsonl data/rerank_v3/holdout.jsonl
```

**方法B: zip をアップロード（手早い・毎回消える）**
ローカルで `tools/` と `data/rerank_v3/` を含む zip を作ってアップロード:
```python
from google.colab import files
up = files.upload()            # ここで repo.zip を選択
!unzip -q repo.zip -d /content/repo
%cd /content/repo/Mozc-Ai-Training   # zip の中の実際のルートに合わせる
!ls tools/rerank data/rerank_v3
```
> `python -m tools.rerank.train_cross_encoder` はリポ**ルートで実行**する必要がある（`from tools.dataset.jsonl import read_jsonl` のため）。`%cd` でルートに居ることを必ず確認。  
> Phase 1 比較用に v2 を置く場合のみ `data/rerank_v2/` も同梱（本番学習パスは **v3**）。

### Cell 4 — スモークテスト（本番の前に1回だけ）
長い学習を回す前に、**モデルとトークナイザが実際にロードできるか**だけ確認する。ここで落ちれば依存問題、通れば本番へ。
```python
import torch
from transformers import AutoModel, AutoTokenizer
name = "sbintuitions/modernbert-ja-70m"
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
mdl = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float32).cuda().eval()
enc = tok("読み: とうきょうと [SEP] 候補: 東京都", return_tensors="pt", truncation=True, max_length=128).to("cuda")
with torch.no_grad():
    out = mdl(**enc)
print("OK hidden:", out.last_hidden_state.shape)
```
> もし `flash_attn` 関連でエラーが出たら、`AutoModel.from_pretrained(..., attn_implementation="sdpa")` を付ける（または `"eager"`）。

### Cell 5 — 学習実行（T4向けに batch を調整）
```python
!python -m tools.rerank.train_cross_encoder train \
  --train data/rerank_v3/train.jsonl \
  --eval  data/rerank_v3/holdout.jsonl \
  --model sbintuitions/modernbert-ja-70m \
  --out   artifacts/rerank/modernbert70m_ce_v3 \
  --epochs 2 --batch-size 64 --max-len 128 --fp16 --grad-checkpointing \
  --require-cuda --require-gold-in-nbest
```
> **スクリプト既定は `--batch-size 512` だが T4(16GB) では OOM する。** 64 から始める（下の「メモリ調整」参照）。`--fp16` と `--grad-checkpointing` は既定ON。実効バッチを上げたいなら batch を保ったままエポックを増やすか、A100(Colab Pro)を使う。  
> Phase 1 ローカル実績は ROCm で batch 1280 級だが、**Colab T4 では 64 が初手**。

### Cell 6 — 成果物の保存
```python
# Drive を使っていれば artifacts/ はそのまま残る。zip 派はダウンロード:
!zip -qr /content/modernbert70m_ce_v3.zip artifacts/rerank/modernbert70m_ce_v3
from google.colab import files
files.download("/content/modernbert70m_ce_v3.zip")
```

### Cell 7 — 動いた構成を固める（再発防止・最重要）
学習が通った**そのセッションで**実行し、出力をリポに保存:
```python
!pip freeze | grep -Ei "^(torch|transformers|tokenizers|sentencepiece|accelerate|numpy)==" 
```
出た行を `docs/colab_requirements.lock.txt` に貼っておく。次回は Cell 1 の代わりに、この lock の**transformers 系だけ**を同じバージョンで入れれば同じ環境が再現できる（torch は毎回 Colab 既設を使う点は変えない）。

---

## 2.5 出て当たり前の警告（無視してよい）

Colab は**赤い文字＝エラー、ではない**。多くは黄/赤の「警告」で、処理は続いている。止まって当たり前かどうかの見分け方:

- **セルの実行が止まったか？** → 次のセルに進めるなら警告。Traceback が出て `...Error:` で終わっていれば本物のエラー。
- **後続の確認行/print が出たか？** → 出ていれば成功。

実際に本デモで出る「無視してよい」警告:

| 表示 | 意味 | 対応 |
|--|--|--|
| `ERROR: pip's dependency resolver ... gradio ... requires huggingface-hub<2.0,>=1.2.0, but you have huggingface-hub 0.36.2` | 学習に無関係な `gradio` が新しい hub を欲しがっているだけ。transformers は 0.36.2 で正常動作 | **無視。** Cell 1 末尾の確認行で transformers 版が出れば OK |
| `The secret HF_TOKEN does not exist in your Colab secrets ... authentication is recommended but still optional to access public models` | HF トークン未設定。だが本番の `modernbert-ja-70m` は**公開モデルなのでトークン不要**（DEMO の ruri も同様） | **無視。** ゲート付きモデルを使う時だけ設定（§6 参照） |
| `UserWarning: ... deprecated ...` / `FutureWarning` | 将来の非推奨予告 | 無視 |
| `Some weights of ... were not initialized ... you should probably TRAIN this model` | ヘッドが新規初期化された合図 | **正常**（まさにこれから学習する） |

> 迷ったら「**Traceback があるか**」で判断。無ければ警告、あれば §3 の早見表へ。

---

## 3. ハマりどころ早見表（本物のエラー）

| 症状 | 原因 | 対処 |
|--|--|--|
| `torch.cuda.is_available()` が False | ランタイムが GPU でない | ランタイムのタイプを GPU に変更 |
| `KeyError: 'modernbert'` / モデル未対応 | transformers が古い | `transformers>=4.48` を入れて**再起動** |
| pip 後もエラーが直らない | 再起動していない | Cell 2 で再起動、Cell 0 で再確認 |
| torch 周りが突然壊れた | torch を入れ直した | torch は触らない。ランタイムを factory reset |
| `flash_attn` ImportError | flash-attn 前提で読もうとした | `attn_implementation="sdpa"` を付ける |
| `sentencepiece` 無い系エラー | 未インストール | Cell 1 に含む（入れて再起動） |
| CUDA out of memory | batch が大きい | `--batch-size` を 64→32 に。`--max-len 96` も可 |
| `ModuleNotFoundError: tools` | ルート外で実行 | `%cd` でリポルートへ。`!ls tools` で確認 |
| `ModuleNotFoundError: <pkg>` | 依存の入れ忘れ | Cell 1 に足して**再起動** |
| `ImportError: numpy.core... ` / numpy 2 系の衝突 | 何かが numpy を上げ/下げした | `!pip install -q "numpy<2"` して再起動（torch は触らない） |
| `NameError`／変数が無い | セルを飛ばした/順序ミス | 上から順に再実行。再起動後は Cell 0 からやり直す |
| `You are not connected to a GPU` / 割当不可 | 無料枠の GPU 在庫切れ | 時間を置く、または Colab Pro。`nvidia-smi` で確認 |
| `Your session crashed after using all available RAM` | システムRAM超過（VRAMではない） | データを小さく/逐次読み。ランタイム再起動 |
| ダウンロードが毎回遅い | モデルを毎セッション取得 | HF キャッシュを Drive に置く（§6） |
| `401/403` gated model | ゲート付きモデル | HF ログイン＋利用許諾（§6）。`modernbert-ja-70m` / DEMO の `ruri` は不要 |
| 学習途中で切断 | 無料枠のアイドル/時間制限 | Drive に保存。短時間なので通常は問題なし |

---

## 4. T4 メモリ調整の目安（70m・max_len 128・fp16・grad-checkpointing ON）

| batch-size | 目安 |
|--|--|
| 32 | 安全 |
| 64 | 推奨の初手 |
| 128 | ギリ（OOM なら 64 へ） |
| 256+ | T4 では非現実的。A100(Pro) 向け |

（任意・DEMO/速度比較）30m モデル（`cl-nagoya/ruri-v3-pt-30m`）に替えるとメモリは半分以下、batch をもっと上げられる。本番パスは 70m 継続。差し替えは `--model` のみ。

---

## 5. 補足: ROCm 用の分岐について

学習スクリプトには ROCm/WSL 向けのコメント（fused AdamW 回避・早期ログ等）が入っているが、**Colab は CUDA なので特別な対応は不要**。`--require-cuda`（既定ON）のまま動く。ROCm 特有のパッチ（`causal_conv1d`/`mamba_ssm` stub 等）はこのリランカーには無関係なので触らない。

---

## 6. Colab 全般で先に潰しておく所（「変なエラー」対策）

「原因がわからず止まる」の多くは環境・セッション由来。以下を先に押さえると激減する。

**環境まわり**
- **torch は絶対に pip で入れ替えない。** Colab の torch↔CUDA↔ドライバは噛み合わせ済み。入れ替えた瞬間に一番厄介な CUDA エラーが出る。torchvision/torchaudio も同様。
- **依存を入れたら一度だけ再起動 → Cell 0 から。** 中途半端に古い版が残る事故を防ぐ。
- **numpy は 2 系と 1 系で衝突しやすい。** 何かが numpy を動かしてエラーになったら `numpy<2` に固定して再起動。
- **Colab の基盤イメージは時々更新される。** 先週動いたのに今日動かない＝基盤更新の可能性。だから**動いた構成を lock**（Cell 7）しておくのが効く。

**セッションまわり**
- **無料枠は「アイドル約90分／連続最大約12時間／GPU在庫」で切れる。** 長時間回すなら Drive に保存必須。学習が短時間なら実害は少ない。
- **`/content` はセッションで消える。** 残したいものは Drive か手元にダウンロード。
- **GPU が割り当たらない時間帯がある。** `nvidia-smi` で確認、ダメなら時間を置くか Pro。

**モデル取得まわり**
- **本番 `sbintuitions/modernbert-ja-70m` は公開モデルなので HF トークン不要**（DEMO の `ruri-v3` も同様）。`HF_TOKEN` 警告は無視。ゲート付き（利用許諾が要る）モデルを使う時だけ、左の鍵アイコン(Secrets)に `HF_TOKEN` を入れて `from huggingface_hub import login; login()`。
- **毎回のダウンロードが重いなら Drive にキャッシュ**を逃がすと速い＆安定:
  ```python
  import os
  os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"   # drive.mount 後
  ```

**再現性まわり**
- **エラーを貼るときは "Traceback の最後の1行" が最重要。** そこに本当の原因（`XxxError:`）が出る。上の警告群と混ぜない。
- **動いたら必ず Cell 7 で freeze。** 「マジで変なエラー」の再発は、ほぼこれで止まる。

**判断の一言**: 赤くても Traceback が無ければ進んでよい。止まるのは `Error:` で終わる Traceback が出た時だけ。
