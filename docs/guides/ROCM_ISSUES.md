# ROCm 利用時に起きた問題と原因

日付: 2026-08-22
対象環境: Windows + WSL2 + Docker `rocm-torch` + **AMD Radeon RX 7800 XT**
目的: 当時何が壊れたか、なぜ壊れたか、どう回避したかを後から読めるように残す。
関連: `docs/HISTORY.md` 壁⑨、`docs/background/STATUS_FOR_PLANNING.md`、`tools/train/lora_sft.py`

このマシンでは NVIDIA CUDA は使えない。PyTorch は ROCm ビルドを「CUDA API」として見せる（`torch.cuda.is_available()` が True でも中身は HIP）。

当時の実測スタック（2026-08-09 時点）:

| 項目 | 値 |
|--|--|
| ホスト | Windows + WSL2（ユーザー `shuhei`） |
| GPU | RX 7800 XT（RDNA3 / gfx1101）、WSL 経由 `/dev/dxg` |
| コンテナ | `rocm-torch`（当初 `rocm/pytorch:latest`、後に snap） |
| torch | `2.13.0+rocm7.14.0` |
| 作業ディレクトリ | WSL `$HOME/work/mozc-ai-training` → コンテナ `/work/mozc-ai-training` |

結論だけ先に書くと、**「GPU が見えない」より「見えるが、NVIDIA 前提の部品と WSL 経由の寿命が足りない」** が本体だった。学習を完走させるために stub・再起動ループ・増分再開を積み、最終的に本学習は **Modal（CUDA）へ移した**。

---

## 1. 土台: Windows 上の AMD では公式 CUDA が無い

**現象:** 最初は Windows ネイティブの PyTorch では GPU 学習ができず、CPU の `rinna/japanese-gpt2-medium` で PoC していた。

**原因:**

- CUDA / bitsandbytes の QLoRA は NVIDIA 専用。
- ROCm の公式ルートはほぼ Linux。Windows ネイティブの実用パスが弱い。
- `torch-directml` は動くことがあるが、Transformers / PEFT の LoRA 学習向きではない。

**対処:** WSL2 上に ROCm 対応 Docker（`rocm-torch`）を立て、コンテナ内の PyTorch ROCm ビルドを使った。GPU は WSL の **DirectX アダプタ `/dev/dxg`** 経由でコンテナに渡す。

コンテナ再作成時に必要だったデバイス／マウント（欠けると GPU が消える）:

- `--device /dev/dxg`
- `-e HSA_ENABLE_DXG_DETECTION=1`
- `/usr/lib/wsl/lib/libdxcore.so`
- `/opt/rocm/lib/librocdxg.so` と `dids.conf`
- `--shm-size 8g`、`seccomp=unconfined`

対話の `docker start -ai rocm-torch` はエージェントから使いにくい。実運用は **`docker exec`**。

---

## 2. bitsandbytes / QLoRA が使えない

**現象:** `pip` 上は bitsandbytes があるのに import が失敗する。

```
libbitsandbytes_rocm84.so not found
```

**原因:**

- bitsandbytes の量子化カーネルは NVIDIA CUDA 向けが本体。
- 入っていた wheel は **ROCm 8.4 用の `.so` 名**（`libbitsandbytes_rocm84.so`）を探していた。
- 実際の PyTorch は **ROCm 7.14**。ABI / ファイル名が一致しない。
- ネイティブ拡張が無いので、Python パッケージだけあっても QLoRA（4bit）は動かない。

**対処:**

- 4bit を捨て、**通常 LoRA（fp16/bf16）+ peft** に変更。
- 壊れたパッケージは `scripts/rocm_fix_bnb.sh` で uninstall（残骸が peft を巻き込むのを防ぐ）。

これは「インストール失敗」ではなく、**この GPU / ROCm 版に対応したネイティブライブラリが無い**。

---

## 3. PLaMo の CUDA 専用カーネル（いわゆる ROCm 地獄）

**現象:** `pfnet/plamo-2-1b` の LoRA が、モデル読み込みや forward で落ちる。`causal_conv1d` / `mamba_ssm` が無い、または CUDA extension がロードできない。

**原因:**

- PLaMo-2 は Transformer だけでなく **Mamba/SSM + causal conv1d** を使う。
- それらの高速カーネルは **NVIDIA CUDA 拡張**として配布されている。
- ROCm/HIP 向けの同等カーネルは、当時この環境には無かった。
- `trust_remote_code` の `modeling_plamo` が、無い拡張を直接呼ぶ。

**対処:** `tools/train/lora_sft.py` の

- `_install_causal_conv1d_torch_stub()` — 純 PyTorch で causal conv を再実装し、モジュールとして差し込む
- `patch_plamo_rocm_kernel_fallbacks()` — `modeling_plamo` のカーネル呼び出しを stub / naive SSM scan に差し替え
- `patch_plamo_tied_weights_compat()` — 後述の transformers 非互換

推論側 `tools/train/infer.py` も同じ stub を入れる。

**副作用:** 動くが **遅い**。GPU カーネルではなく Python / 素の tensor 演算になる。学習は完走できたが、IME 推論は CPU で約 **291 秒/回** と実用外。これが「生成モデルをリアルタイム経路から外し、リランカーへ転換する」直接の理由の一つ。

`docs/reranker/plans/PLAN_RERANKER.md` が ModernBERT-Ja を選んだ理由の先頭もこれ（「PLaMo の ROCm 地獄が消える」）。普通の Transformer エンコーダなら CUDA 専用 SSM カーネルが要らない。

---

## 4. Transformers のバージョン食い違い

**現象:** コンテナ既定の Transformers 5.x で PLaMo がロードできない。

**原因:**

- PLaMo の remote code は当時 **Transformers 4.48 系** 前提。
- 5.x は `_tied_weights_keys` の型（list → mapping）などが変わり、`get_expanded_tied_weights_keys` が落ちる。

**対処:**

- コンテナ内を `transformers==4.48.3`、`tokenizers<0.22` に下げた。
- 加えて `patch_plamo_tied_weights_compat()` で list の tied keys を空 dict 扱いにした。

ROCm そのもののバグではないが、**ROCm 公式イメージの Python スタックが新しすぎて、学習したいモデルの remote code と衝突する**典型例。

---

## 5. `expandable_segments` で最初の確保が即死する

**現象:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を付けると、最初の `torch.zeros(..., device="cuda")` で落ちる。

```
hipErrorInvalidValue
```

**原因:**

- このフラグは CUDA アロケータの拡張セグメント用。
- 当時の **ROCm 7.14 + この PyTorch ビルド** では HIP アロケータが値を拒否する。
- NVIDIA 向けのメモリ節約 tips をそのまま持ち込むと、確保の時点で死ぬ。

**対処:**

- **設定しない。** `_train_rerank_gpu.py` / `_train_ctx_v2_rocm.py` / `_train_rerank_gpu_v3.py` に禁止コメントがある。
- Phase 2 手順書も「この ROCm では invalid」と明記。
- 一部の古い起動スクリプト（`run_rerank_gpu_train.sh` 等）にはまだ残っているので、それを使うと再発する。

VRAM 不足は expandable ではなく **batch を大きくして OOM したら下げる backoff** で吸収した（7800 XT なら ModernBERT-Ja 70m で batch 512〜1280 級）。

---

## 6. 最初の CUDA 確保がフラッキー（WSL / dxg）

**現象:** さっきまで GPU 計算できていたのに、次のプロセスで `torch.cuda.is_available()` が False、または小さな `zeros` が例外。コンテナや WSL を殺した直後に特に多い。

**原因:**

- GPU はベアメタル ROCm ではなく **WSL の `/dev/dxg`（DirectX 経由）**。
- プロセスを `pkill -9` したりコンテナを急停止すると、dxg / KFD 相当の状態が壊れたまま残る。
- 重いライブラリ（transformers）を import してから初めて CUDA を触ると、初期化順で失敗しやすい。

**対処:**

- 学習プロセスの**最初**に小さな CUDA alloc + `synchronize`（ウォーム）。transformers より前。
- 失敗したらコンテナ restart、最大 30 回プローブ（`scripts/_run_train_sameproc.sh`、`_cuda_probe.py`）。
- 「同一プロセスでウォームしてから train」する `_train_rerank_gpu.py`。

これはカーネル不足というより **WSL2 + AMD のデバイスノード寿命**。

---

## 7. fused AdamW が不安定

**現象:** オプティマイザ初期化や step で落ちる／妙な HIP エラー。

**原因:** fused AdamW は CUDA 向けカーネル。ROCm ビルドに同名 API があっても、実装が欠けているか壊れている。

**対処:** `train_cross_encoder.py` では **標準 `torch.optim.AdamW` 固定**（「fused AdamW has been flaky on this ROCm build」）。

---

## 8. ONNX Runtime に ROCm Execution Provider が無い

**現象:** ablation で `onnx+cuda` を指定しても GPU に乗らない。

**原因:** このコンテナの ORT が公開しているのは **`CPUExecutionProvider` だけ**。HIP / ROCm EP がリンクされていない。

**対処:**

- フラグ上は cuda でも実際は CPU ORT、`actual_device=cpu_ort` と記録（`ablation_bench.py`）。
- 出荷推論は **WSL CPU の ORT fp32**（後のデーモン経路）。ローカル AMD GPU で ONNX を加速する経路は作っていない。

PyTorch の HIP と ORT の EP は別物。片方動いてももう片方は動かない。

---

## 9. WSL / コンテナが学習の途中で死ぬ

**現象:** 長時間ジョブが突然切れる。コンテナ Exit、WSL ごと落ちる、GPU が復帰しない。watchdog 前提の運用になった。

**原因（複合）:**

- WSL2 の GPU パラ仮想化（dxg）は長時間・高負荷に弱い。
- ホスト Windows のスリープ、他プロセス、IME/デスクトップ負荷が乗ると更に不安定。
- 前述の不正 alloc・9 キルのあとにデバイスが戻らない。
- エージェントやシェルからコンテナを止めると、中の学習も死ぬ。

**対処:**

- `bench_incremental.py` など **example 単位 JSON チェックポイント**（落ちても再開）。
- `host_plamo_watchdog` / `run_plamo_full_watchdog.sh` で監視して再起動。
- それでも「PC の前に張り付いて再起動する」必要がある。離席学習ができない。

**最終判断:** 本学習・評価は **Modal（クラウド CUDA）** へ。ROCm の苦労と離席不可を同時に消す。ローカル `rocm-torch` はプローブや軽い実験用に残した。

---

## 10. 隣接だが ROCm 作業を邪魔した問題

ROCm 本体ではないが、同じ WSL 経路で必ず踏んだ。

### 10.1 `/mnt/c` の mmap

WSL プロセスが NTFS（`/mnt/c`）を mmap すると Mozc の `mozc.data` が壊れる。学習データをコンテナから `/mnt/c` で読むのも避ける。データは **ext4（`$HOME/work/...`）** に置き、Mozc 変換だけ Windows ネイティブ exe。

### 10.2 CRLF の bash

Windows 側リポの `.sh` を WSL でそのまま `source` すると `\r` で `pipefail` 等が壊れる。Linux FS にコピーして `sed -i 's/\r$//'` してから実行。PLaMo スモークの初回失敗はこれ。

### 10.3 対話コンテナ vs exec

ユーザー案内は `docker start -ai rocm-torch`。エージェントは TTY を持てないので `docker exec`。コンテナが interactive 専用だと、抜けた瞬間にプロセスが全部死ぬ。後に `sleep infinity` の detached コンテナへ作り直した。

---

## 11. 問題と層の対応

```
[Windows]
   └── WSL2
         └── Docker rocm-torch
               ├── PyTorch ROCm  ← HIP を CUDA API に見せる
               ├── bitsandbytes  ← ネイティブ .so 不一致（QLoRA 不可）
               ├── PLaMo remote code ← CUDA SSM カーネル必須
               └── ORT           ← CPU EP のみ
         └── /dev/dxg          ← 確保失敗・突然死の温床
```

| # | 層 | 症状 | 根本原因 |
|--|--|--|--|
| 1 | OS | Windows で GPU 学習できない | ROCm が Linux 前提 |
| 2 | 量子化 | QLoRA 不可 | bnb の ROCm .so が版不一致／未提供 |
| 3 | モデル | PLaMo が落ちる／極端に遅い | CUDA 専用 Mamba/conv カーネル |
| 4 | 依存 | PLaMo ロード失敗 | Transformers 5 vs 4.48 remote code |
| 5 | アロケータ | 起動直後に HIP invalid | `expandable_segments` 非対応 |
| 6 | デバイス | 直後の CUDA が死ぬ | WSL dxg の寿命・初期化順 |
| 7 | 最適化 | Adam 初期化が不安定 | fused AdamW が ROCm で壊れる |
| 8 | 推論ランタイム | ONNX が GPU に乗らない | ORT に ROCm EP が無い |
| 9 | 運用 | 長時間ジョブが落ちる | WSL+dxg+ホスト負荷 |
| 10 | 周辺 | mmap / CRLF / TTY | WSL↔Windows 境界 |

---

## 12. 何が残って、何を捨てたか

**残したもの（動いた）**

- RX 7800 XT 上での matmul・ModernBERT-Ja 系の LoRA / クロスエンコーダ学習（普通の Transformer）
- PLaMo LoRA 1 epoch 完走（stub 付き、遅い）
- 大きな batch + OOM backoff

**捨てた／移したもの**

- QLoRA
- PLaMo をリアルタイム IME に載せる案（遅すぎ・脆すぎ）
- ローカル ROCm での本学習（Modal CUDA へ）
- コンテナ内 ORT GPU

**いま触るなら守ること**

1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` を付けない
2. CUDA ウォームを transformers import より前にやる
3. fused AdamW を使わない
4. PLaMo を載せるなら stub 必須。リランカーには不要
5. 長時間学習は Modal。ローカルは落ちる前提
6. データと mmap 対象は ext4。`/mnt/c` を ROCm プロセスから読まない

---

## 13. 参照コード

| ファイル | 何を避けるか |
|--|--|
| `tools/train/lora_sft.py` | causal_conv1d / mamba stub、tied weights |
| `tools/train/infer.py` | 同上（推論） |
| `scripts/_train_rerank_gpu.py` | expandable_segments 禁止、同一プロセスウォーム |
| `tools/rerank/train_cross_encoder.py` | 早期 CUDA init、標準 AdamW |
| `scripts/_run_train_sameproc.sh` | dxg 復活待ちループ |
| `scripts/rocm_fix_bnb.sh` | 壊れた bitsandbytes の除去 |
| `tools/rerank/ablation_bench.py` | ORT は CPU のみ、と明記 |
| `scripts/bench_incremental.py` | 落ちても再開 |
| `docs/guides/AGENT_MODAL_USAGE.md` | 本学習は Modal、と運用ルール化 |

---

## 14. 2026-08-22 現行環境での再検証

この文書の 1〜13 は当時の WSL2 + Docker 環境の事故記録として残す。
ただし、現在の能力や推奨構成としては次の点が変わっている。

再検証環境:

- Windows 11 ネイティブ（WSL / Docker / `/dev/dxg` なし）
- RX 7800 XT (`gfx1101`)
- ROCm 7.14.0 / HIP 7.14.60850
- PyTorch 2.12.0+rocm7.14.0
- bitsandbytes 0.50.1
- transformers 4.57.1 / PEFT 0.20.0

### 解決したもの

- Windows ネイティブ PyTorch ROCm が公式配布され、GPU学習が可能になった。
- WSL、`/dev/dxg`、Docker、`/mnt/c`、CRLF、TTYの障害層を学習経路から除去した。
- bitsandbytes 0.50.1のROCm 7.14 Windows backendでNF4 4bit forward/backwardが成功した。
- 小型GPT-2でTransformers + PEFT + Trainerを通したNF4 QLoRA 1 stepとadapter保存が成功した。
- `pfnet/plamo-2-1b`でもNF4 QLoRA 1 step、backward、optimizer step、adapter保存が成功した。
- fused AdamWは現行環境のFP16/BF16試験で成功した。ROCm全体で使用禁止という旧説明は正しくない。
- native Windows向けにDataLoader pinned memoryを無効化した。
- 古いWSLランチャーに残っていた`expandable_segments:True`を削除した。

検証結果:

- bitsandbytes probe: `artifacts/bnb-rocm-probe.json`
- 小型QLoRA adapter: `artifacts/qlora_native_smoke/adapter`
- PLaMo QLoRA adapter: `artifacts/plamo_native_qlora_smoke/adapter`

### まだ残るもの

- `expandable_segments`は現在のROCmでもcrash/hang報告があるため使わない。
- PLaMoのpure-PyTorch SSM fallbackは正しく動くが遅い。今回の長さ64、batch 1では約31秒/stepだった。
- PLaMoが要求するMamba/causal-conv1d高速拡張はLinux ROCm対応を持つが、native Windows向けの対応済みwheelはない。
- PyTorch 2.12 native WindowsにはTriton/Inductorの通常経路とRCCLがない。
- ONNX Runtimeの旧ROCm EPはORT 1.23で削除済み。Linux/WSLはMIGraphX、WindowsはWinML/DirectMLを別途評価する。

### 現在の起動方法

```powershell
.\scripts\setup_native_rocm.ps1

# ModernBERT reranker（学習データが存在する場合）
.\scripts\run_native_rocm_reranker.ps1

# PLaMo QLoRA smoke/full training
.\scripts\run_native_rocm_plamo.ps1 `
  -Data .\train_mixed.jsonl `
  -QLoRA
```

PLaMo remote codeは検証済みrevision
`92c75fd6eea9018bcb9c33ee8921589febe071fa`へ固定している。

現在の結論は「ROCmではQLoRA/PLaMo学習が不可能」ではない。
**native WindowsでQLoRAと学習の正しさは回復したが、PLaMo高速SSMカーネルのWindows移植が性能上の残課題**である。

---

## 15. PLaMo SSM高速化とメモリoffload再検証

過去のリポジトリとGit履歴を再調査したが、Accelerate CPU/disk offload、
DeepSpeed、ZeRO、FSDPを実際に設定して失敗した記録は見つからなかった。
記録が残るメモリ関連の失敗は次の二つだった。

- bitsandbytes native backendのROCm版不一致
- `expandable_segments`によるHIP allocator初期化失敗

前者はbitsandbytes 0.50.1で解決済み、後者は現在も無効化が必要。

### PLaMo fallbackの性能問題

旧fallbackには二つの大きな無駄があった。

1. 長さ64の系列もMamba chunk size 256へ左paddingし、Python SSM recurrenceを256回実行していた。
2. gradient checkpointing時のcausal convolutionをtokenごとのPython loopと小さな`conv1d` launchで処理していた。

現在は次のように変更した。

- sequential fallbackではchunk paddingを行わず、実際のsequence lengthだけscanする。
- 幅4のcausal convolutionをshifted tensor 4本の演算へvectorizeする。
- `seq_idx`境界maskを維持し、packed sequence間で状態を混ぜない。
- SSM decayがheadごとのscalarである構造を使い、`[H,P,N]`への不要な展開を除去する。
- forward/backwardを旧recurrenceと比較するparity testを追加する。

同じPLaMo-2-1B NF4 QLoRA、batch 1、max length 64、1 stepでの実測:

- 旧fallback: 約31.61秒/step
- 改善後: 約1.90〜2.00秒/step
- 高速化: 約16倍
- peak VRAM: 約1.90GB

`scripts/test_plamo_fallback_math.py`でcausal convolutionとSSD scanの
forward/backward一致を検査できる。

### 学習用offload

Accelerateの`device_map="auto"`、CPU offload、disk offloadは公式には
big-model inference用であり、一般的なtraining weight offloadではない。
これをLoRA学習へ流用すると、autograd graph切断やdevice mismatchを起こし得る。

native Windows ROCmで検証した構成:

- NF4 QLoRAで凍結base weightを4bit化
- non-reentrant gradient checkpointing
- `torch.autograd.graph.save_on_cpu`によるautograd saved activationのCPU RAM offload
- bitsandbytes `PagedAdamW8bit`を任意選択

今回の短いlength 64ではcheckpointingが既にactivation保存を抑えているため、
activation offloadのpeak VRAM差は約0.7MBだけだった。長いsequence、
checkpoint対象外演算、大きなbatchで効果が増える。paged optimizerは2 stepの
forward/backward/update試験を通過したが、LoRAのoptimizer state自体は小さいため
通常は`adamw_torch`でよい。

```powershell
.\scripts\run_native_rocm_plamo.ps1 `
  -Data .\train_mixed.jsonl `
  -QLoRA `
  -ActivationOffload `
  -PagedOptimizer
```

量子化base weightそのものがVRAMへ収まらないモデルは、このWindows構成では
training-awareなweight streaming手段がない。`scripts/plan_model_memory.py`で
NF4 baseの概算と安全なGPU予算を比較し、未対応のAccelerate disk offloadへ
誤って進まないようにする。その範囲を超える場合は、Linux DeepSpeed ZeRO-3、
大容量GPU、または小さいモデルが必要。

### 専用HIP SSD forward/backward

さらに、PLaMoの選択的SSM scanをnative Windows HIP拡張として実装した。

- `extensions/plamo_ssd.cpp`
- `extensions/plamo_ssd.cu`
- `tools/train/plamo_ssd_hip.py`
- `scripts/build_plamo_ssd.ps1`

実装内容:

- 各`(batch, head, channel)` work-itemが最大64要素のFP32 stateを保持してsequenceを走査
- `softplus(dt+bias)`、decay、state update、C reduction、D skip、SiLU gateを1 kernelへ融合
- packed sequenceの`seq_idx`変化時にstateをreset
- reverse recurrenceによる専用HIP backward
- `dx`, `ddt`, `dA`, `dB`, `dC`, `dD`, `dz`, `ddt_bias`を計算
- channel共有gradientはFP32 atomic reduction
- backward用state historyをPyTorch saved tensorとして登録し、`save_on_cpu`でRAM offload可能
- unsupported shape、ビルド失敗時はpure-PyTorch fallbackを維持

`scripts/test_plamo_fallback_math.py`でnative HIP forward/backwardを参照数式と比較し、
packed resetを含む全activation gradientの一致を確認した。

最終実モデル試験:

- PLaMo-2-1B NF4 QLoRA
- BF16 compute
- activation CPU offload
- PagedAdamW8bit
- native HIP SSD forward/backward
- 8,421,376 trainable parameters
- peak VRAM約1.94GB
- 約1.94秒/step
- adapter保存成功

`scripts/setup_native_rocm.ps1`はbitsandbytes、paged optimizer、HIP extension build、
SSD forward/backward parityをすべて必須検証する。

この時点でPLaMo固有の「CUDA extensionがなければ学習不能」という差は解消した。
残るCUDAとの差は、native WindowsにRCCLと対応Triton/Inductorが配布されていないこと、
および量子化base自体がVRAMを超える場合のtraining-aware weight streamingがないこと。

HIP kernelは評価を受けて明示opt-inへ変更した。通常実行は検証しやすいTorch fallbackを
使用し、次の指定がある場合だけcustom HIP kernelを使う。

```powershell
.\scripts\run_native_rocm_plamo.ps1 `
  -Data .\train_mixed.jsonl `
  -QLoRA `
  -UseHipKernels
```

`train_meta.json`には`ssd`、`causal_conv`、GPU architecture、source build hash、
実際にロードしたbinary pathを保存する。extension名とbuild directoryは
`.cpp/.cu`、PyTorch、HIP、GPU architectureのSHA-256から生成されるため、
古い`.pyd`をソース変更後に誤ロードしない。

ROCm Pythonは`ROCM_PYTHON`環境変数で指定でき、未指定時だけ同一workspace内の
`ROCmforwindows/.venv`を探索する。GPU architectureは実機から取得し、`gfx1101`
固定を廃止した。ビルド時に変更する`ROCM_HOME/CC/CXX/PATH`等は終了時に復元する。

検証も強化した。

- 独立した2-token/1-state閉形式でFP16/BF16 forward/backwardを比較
- PagedAdamW8bitを3 step連続実行と2 step→state_dict保存→復元→3 step目で比較
- optimizer resume後のparameterとlossが連続実行と一致

### HIP kernel長系列ベンチ

PLaMo実形状`H=32, P=128, N=64`でSSD forward+backwardを直接比較した。

- length 64: Torch 0.1111秒 / HIP 0.0629秒、258.6MB / 223.0MB
- length 256: Torch 0.4696秒 / HIP 0.2396秒、578.5MB / 436.0MB
- length 512: Torch 0.9605秒 / HIP 0.5304秒、1005.0MB / 720.0MB

HIP SSDは約1.8〜2.0倍高速で、peak VRAMを約28%削減した。
結果は`artifacts/plamo-ssd-benchmark.json`へ保存する。

causal convolutionも専用HIP forward/backwardへ置換した。現行hash buildでは入力gradientを
atomic scatterではなく入力要素ごとのgatherで計算し、凍結weightではweight-gradient
kernelを省略する。

- length 64: Torch 0.00304秒 / HIP 0.00144秒
- length 256: Torch 0.00375秒 / HIP 0.00242秒
- length 512: Torch 0.00428秒 / HIP 0.00339秒
- peak VRAMは各長さで約30%削減

結果は`artifacts/plamo-conv-benchmark.json`へ保存する。

全token state履歴を8-token chunk境界へ圧縮する試作も行ったが、packed sequence reset
を含むgradient parityを満たさなかったため採用していない。現在の既定は全token stateを
保存する検証済み実装で、activation offload時にはRAMへ退避される。未検証の省メモリ化を
速度目的で残さない。

### 現行運用の要点

この節を1〜15の履歴・実測より優先する。

- 入力JSONLは読み取り専用。adapter/checkpoint/cache/reportは`--out`、`HF_HOME`、
  `artifacts/`へ別途書く。
- HIP kernelは自動有効化しない。`-UseHipKernels` / `--use-hip-kernels`で明示する。
- 複数step検証でSSD HIPはNaN gradientを発生させたため、`-UseHipKernels`はcausal-conv
  HIPだけを有効化しSSDはTorchを使う。SSD HIPは`-UseExperimentalHipSsd`のみ。
- 明示指定中の緊急無効化は`MOZC_DISABLE_HIP_SSD=1`、
  `MOZC_DISABLE_HIP_CONV=1`。
- extensionは`.cpp/.cu`、Torch、HIP、GPU archのsource hashで識別し、旧`vN`
  module名は使用しない。
- 標準GPU試験入口は`scripts/run_native_rocm_tests.ps1`。通常のCPU unittestとは別。
- `scripts/audit_artifacts.ps1`は容量を読むだけで、artifactを削除・変更しない。

### このPC向け完成受入条件

`scripts/run_completion_gate.ps1 -IncludeModel`が全項目を通過することを、このPCで
達成可能な完成条件とする。

- Windows native ROCm device allocation
- bitsandbytes NF4 forward/backward
- PagedAdamW8bit state_dict保存・復元・継続一致
- SSD/causal-conv FP32全勾配・packed reset parity
- 独立閉形式によるFP16/BF16 forward/backward
- hidden 4096、tokens 512のQLoRA memory stress
- source、Torch、HIP、GPU arch hashに対応したbundle生成とSHA-256検証
- PLaMo-2-1B NF4 QLoRAを3 step実行
- checkpoint-3から再起動しcheckpoint-4とadapterを生成
- 固定seed、有限loss/gradient norm、LoRA parameter fingerprint変化を検査
- checkpoint-3 final fingerprintと再開時initial fingerprintの一致を検査
- bundle copyを意図的に改ざんし、verifierが失敗する負テスト
- 現行hash buildを退避後、ゼロから再ビルドして同一identity・parityを確認

2026-08-22に全項目を通過し、結果は
`artifacts/native-rocm-completion-report.json`へ保存した。

次はこのPC単体では完成判定できない外部制約として、レポートに明示する。

- 1 GPUしかないためmulti-GPU通信性能を検証できない
- native Windows RCCLが提供されていない
- PyTorch 2.12 native Windows向け対応Triton/Inductorが提供されていない
- AMD Windows PAL/HIP runtime内部は非公開

### Vast.ai異種AMD検証

2026-08-22にVast.ai CLI 1.5.5でread-only認証を確認した。

- API key認証: 成功
- 残高: $5.00
- AMD offer検索: 成功
- `gpu_arch=amd` offer: 0件
- `rocm/pytorch` template: 164件
- instance作成: なし
- 課金操作: なし

`scripts/check_vastai_amd.ps1`はAPI keyを表示・artifactへ保存せず、認証と在庫だけを
確認する。サニタイズ済み結果は`artifacts/vastai-amd-availability.json`へ保存する。
検索は公式案内どおり`vastai search offers -n 'gpu_arch=amd rented=any'`を使う。
`-n`で暗黙の`external=false rentable=true verified=true`を無効化し、Unverified、
非rentable、externalを含むAMD machine recordを検索する。
現時点ではRadeon/Instinct在庫がないため異種GPU実行は未実施。在庫が出た時点で
source-hash bundleを転送し、同じcompletion gateを対象archで実行する。

7B級hidden shapeを模したQLoRA memory stressも追加した。

- hidden 4096、tokens 512、checkpoint反復8
- offloadなし: 0.664秒、peak 662.5MB
- activation offload: 0.820秒、peak 634.5MB
- loss一致、forward/backward/PagedAdamW8bit成功

`scripts/stress_qlora_memory.py`で再実行できる。gradient checkpointing併用時は既に
activation保存量が小さいため、CPU offloadはOOM時だけ有効化するのがよい。
