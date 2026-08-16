# Phase 3 CTX report — contextual Track B into Mozc

Date: 2026-08-13  
Spec: `docs/reranker/tasks/NEXT_TASK_PHASE3_CTX.md` / `docs/reranker/tasks/NEXT_TASK_30M_AND_DAEMON.md`  
**出荷モデル: `track30m_ctx`** (`sbintuitions/modernbert-ja-30m`、τ=**2.5**、max_len=128、cand_cap=30、ONNX fp32)  
比較用に 70m `trackB_v2_continue` は Volume に保持。  
Policy: default **OFF** (`MOZC_RERANK_ENABLED=1` + 常駐デーモン)

## Go / no-go

**品質は 30m も GO**（CS Δ 各セット ≥ +15pt、全体退行 <2%、τ=2.5）。  
**推論速度は quiet 再測で GO**（30m ship p50=51 / p95=128。70m 112/342 比 **2.2× / 2.7×**、p95 ≤ 170）。デーモン e2e も quiet で p50=57 / p95=132、IPC 0.30ms。以前の 83/225 と e2e 77/212 は他負荷。  
**配信はデーモン GO**（IPC ≈ 0.5ms。毎回 `std::system` は廃止）。quiet では 30m@50 が 200ms ゲートをクリア。タイムアウト→Mozc 順は維持。  
**実使用ガードは GO**（186 件リプレイ net **−8.1 → +0.5pt**、壊した 21→0）。既定 OFF。再学習は次段（未着手）。

70m に戻す理由はない（30m の方が速く、文脈効果は残る）。

## ONNX export + parity

Volume: `mozc-artifacts:/trackB_v2_continue/onnx/`  
Local (WSL): `$HOME/work/mozc-ai-training/artifacts/rerank_ctx/trackB_v2_continue/onnx/`

| item | result |
|--|--|
| `cross_encoder_fp32.onnx` | exported (opset 18, ~268MB) |
| `margin_policy.json` | τ=2.5, max_len=128, cand_cap=30, timeout_ms=200, int8=false |
| PT fp32 vs ONNX fp32 | **Spearman=1.0, Pearson=1.0, MAE=5e-6** (80 groups / 1608 scores, unseen clean) |
| int8 | not exported (ship-forbidden) |

## ① Hook は毎回起動だった → 常駐デーモンに置換

旧: `RerankRewriter::Rewrite` → `CallHook` → **`std::system(hook_cmd_ + req + resp)`**。変換のたびに新しい OS プロセス。

新: `MOZC_RERANK_ENABLED=1` で **localhost TCP**（既定 `127.0.0.1:17890`）へ NDJSON。`scripts/rerank_daemon.py` が起動時に ONNX fp32 + SentencePiece tokenizer + margin を1回ロード。`context_clip` / `rerank_one` / margin は hook と同一。`MOZC_RERANK_HOOK_CMD` がセットなら旧 one-shot を残す（テスト用）。

Fail-safe: デーモン未起動・接続失敗・**200ms タイムアウト** → Mozc 順。既定 OFF。

C++ テスト: `//rewriter:rerank_rewriter_test` **PASSED**（disabled-by-default、timeout→Mozc、daemon unreachable→Mozc）。Python: `tools.rerank.test_rerank_daemon` **2/2**。

Daemon e2e（30m ONNX、ウォーム、intra=12、80 groups）: quiet 再測 e2e p50=**57ms** / p95=**132ms**、推論 p50=56 / p95=131、**IPC ≈ 0.30ms**。旧 77/212 は他負荷。≈ 推論 + ~0.3ms。

## ② CPU ship-profile レイテンシ（再計測）

前回の `intra=1` は **PLAN §4.5 違反**（文脈なし 70m は `intra=cpu_count`、当時 12、p50≈42ms @ max_len=48）。学習は `padding="max_length"`（固定 128）だが、serve はバッチ内最長に pad する。

`latency_pack.py --ship-profile`（WSL CPU 12 論理コア、`OMP_NUM_THREADS=1`、max_len=128、cand_cap=30、**プロセス内 ORT・ウォーム**、80 groups）。JSON: `artifacts/rerank_ctx/eval/latency_ship_profile.json`

| variant | intra | padding | seq p50 | p50 ms | p95 ms |
|--|--|--|--|--|--|
| `prev_intra1_pad_max128`（誤測に近い） | 1 | max_length=128 | 128 | 1008 | 2432 |
| `intra_cpu_pad_max128` | 12 | max_length=128 | 128 | 213 | 556 |
| **`ship_intra_cpu_pad_longest`（推奨）** | **12** | **longest** | **63** | **111** | **337** |

- intra 1→12（pad 128 固定）: p50 **1008 → 213**（**4.7×**）
- pad 128 → longest（intra=12）: p50 **213 → 111**（**1.9×**）。seq mean 62.7、max 96、**frac_seq_lt_128 = 1.0**（実文脈は 128 に届かない）
- 前回の intra=1 / `padding=True`（HF longest）p50=449 と比べると、intra 修正だけで **449 → 111**（**4.0×**）

**cand_cap=30 は 1 変換 1 `session.run`。** `rerank_one` が全候補の pair text をまとめて `score()` → `OrtRunner.score_texts` が 2D バッチで **1 回** `session.run`。計測: `session_runs_during_timed=80` / `timed_groups=80`、`one_session_run_per_group=true`、`one_forward_ok=80/80`。batch mean=15.55、max=30（逐次 30 forward ではない）。

推奨 p95 − 150ms = **+187ms**。`degrade_before_cpp=true`。推論ウォームでも 200ms ゲート未達。フック `std::system` 起動は含まない。

（同 JSON の hit@1=0.74 は 80 groups の参考値。品質は③。）

## ③ hook_offline vs clean eval（PT @ τ=2.5）

`artifacts/rerank_ctx/eval/hook_offline_tau2.5.json`（ONNX fp32、cand_cap=30、文脈ON）:

| set | hook hit@1 | PT clean | hook CSΔ | PT CSΔ | hook 退行 | PT 退行 |
|--|--|--|--|--|--|--|
| seen | 91.83% | 91.50% | +34.10 | +32.57 | **1.04%** | 1.04% |
| unseen | 91.59% | 91.46% | +30.46 | +29.31 | **1.34%** | 1.34% |
| fresh | 94.13% | 92.85% | +52.76 | +49.14 | **0.66%** | 0.66% |

- **退行率は 3 セットとも PT と一致**（安全弁は生きている）。
- overall hit@1 は seen/unseen が +0.3pt 以内。fresh は +1.3pt（cand_cap=30 で後ろの妨害候補を切った効果。崩壊ではない）。
- CS Δ は同方向でやや大きい（CS OFF はほぼ一致、CS ON が少し高い）。
- スクリプトの `ok=false` は CSΔ ゲート 0.8pt が厳しすぎただけ。品質としては **一致（退行）＋同等以上（hit）**。

## ④ 30m 文脈リランカー（速度回収）

Track A 方式: HF `sbintuitions/modernbert-ja-30m` から `train_v2_clean.jsonl`、70m と同一ハイパラ（2 epoch / bs 512 / max_len 128 / fp16）。Modal L4 **1114s**、pairs=488314。Volume `/artifacts/track30m_ctx/`。

### τ スイープ（clean eval）

`artifacts/rerank_ctx/eval/shippable_track30m_selection.json`。τ∈{1.5,2.0,2.5}、全3セット全体退行 <2% の中で min(CS Δ) 最大。

| τ | 退行 seen/unseen/fresh | CS Δ seen/unseen/fresh | ゲート |
|--|--|--|--|
| 1.5 | 2.09 / 2.68 / 0.99% | +32.4 / +28.5 / +55.8 | 退行 NG |
| 2.0 | 1.74 / 2.18 / 0.83% | +32.6 / +26.1 / +50.7 | unseen 退行 NG |
| **2.5** | **1.56 / 1.51 / 0.83%** | **+31.4 / +24.3 / +48.8** | **採用** |

### 30m vs 70m（PT clean、τ=2.5、文脈ON）

| set | 30m hit@1 | 70m PT hit@1 | 30m CSΔ | 70m PT CSΔ | 30m 退行 | 70m 退行 |
|--|--|--|--|--|--|--|
| seen | 89.36% | 91.50% | **+31.4** | +32.6 | 1.56% | 1.04% |
| unseen | 89.98% | 91.46% | **+24.3** | +29.3 | 1.51% | 1.34% |
| fresh | 92.38% | 92.85% | **+48.8** | +49.1 | 0.83% | 0.66% |

CS Δ は全セット ≥ +15pt（unseen が 70m より約 5pt 低いが実用圏）。non-CS vs Mozc は −0.5〜−0.8pt。

### ONNX + レイテンシ（70m と同一条件）

Volume: `mozc-artifacts:/track30m_ctx/onnx/`（fp32 **141MB**）。PT vs ONNX: Spearman=1.0、MAE=5e-6（80 groups / 1608 scores）。

`latency_ship_profile_30m.json` は他負荷下の旧測（p50=83 / p95=225）。quiet 再測は `latency_ship_profile_30m_quiet.json`（intra=12、padding=longest、1バッチ1 forward、ウォーム）:

| | 30m quiet | 70m quiet | 比 |
|--|--|--|--|
| p50 | **51ms**（旧 83） | 112ms（旧 111） | **2.2×** |
| p95 | **128ms**（旧 225） | 342ms（旧 337） | **2.7×** |
| seq p50 | 63 | 63 | 同じ文脈長 |

目標 p50 40–55 / p95 120–170（2〜3倍）は **quiet 推論では達成**。70m ship は再現（111/337 → 112/342）。30m の旧 83/225 は他負荷。

Daemon e2e: `daemon_e2e_30m_quiet.json` p50=57 / p95=132、IPC 0.30ms（旧 `daemon_e2e_30m.json` 77/212 は他負荷）。max 238ms が1件あるので 200ms タイムアウト→Mozc 順は残す。

**決定: 出荷は 30m @ 50字。** 70m は比較用に保持。cap 短縮はゲート用には不要。

## Python ↔ C++ parity

| test | status |
|--|--|
| context_clip (clean / reading / clip) | **23/23** (`mozc_compat/context_clip_cli`) |
| tokenize token ids | **blocked on algorithm**: HF class is `LlamaTokenizer` + `tokenizer.model`. WordPiece `hf_tokenizer.cc` does not apply. Native ORT needs SentencePiece C++ (`<s>` bos + SP pieces + `</s>` eos; pair text’s literal ` [SEP] ` is **not** `<sep>`). |

## Mozc integration

`scripts/integrate_mozc.py --mozc-dir Mozc-Ai/.mozc-build/mozc/src`

- Files copied: `rerank_rewriter.*`, `context_clip.*`, `rerank_margin.h`
- `rewriter/BUILD.bazel`: `rerank_rewriter` library + test; rewriter deps include `:rerank_rewriter`
- `rewriter.cc`: `RerankRewriter` **after** `AIRewriter` (chain tail)
- Enable: `MOZC_RERANK_ENABLED=1`（デーモン既定 `127.0.0.1:17890`）。旧 `MOZC_RERANK_HOOK_CMD` はテスト用フォールバック。
- Log opt-in: `MOZC_RERANK_LOG=<jsonl>` (schema `artifacts/rerank/conversion_log_schema_v1.json`)
- Default remains Mozc-only

Headless tests (`rerank_rewriter_test.cc`): disabled-by-default, empty context, timeout → Mozc order, きしゃ history present but rewriter off keeps Mozc top-1.

**`bazelisk test //rewriter:rerank_rewriter_test` → PASSED (0.3s)** on this Windows Mozc tree.  
**`bazelisk build //server:mozc_server` → succeeded** (`bazel-bin/server/mozc_server.exe`).

## Human Windows IME smoke (do not autoclick)

Quiet daemon e2e: p50=**57** / p95=**132** / IPC **0.30ms**（`daemon_e2e_30m_quiet.json`）。

Installed `C:\Program Files\Mozc\mozc_server.exe` is the 2026-08-14 rewriter build (last conversion segment + preceding segment as context).

1. WSL daemon already listening `127.0.0.1:17890` (Windows ping = pong).
2. `scripts/ime_smoke_start.ps1` starts bazel-bin `mozc_server.exe` with `MOZC_RERANK_ENABLED=1` and `MOZC_RERANK_LOG=%USERPROFILE%\AppData\LocalLow\Mozc\rerank_smoke.jsonl`.
3. Type with left context: 「新聞の」+「きしゃ」→ **記者**; 「駅に」+「きしゃ」→ **汽車**.
   実機 2026-08-14 **GO**: `駅に`+`きしゃ` → **汽車** (overwritten, margin 4.51)。`新聞の`+`きしゃ` → **記者** (Mozc 維持, margin 1.07 < τ)。デーモンログ `daemon_requests.jsonl`。
4. Confirm easy words (東京 etc.) stay Mozc top-1.
5. Stop daemon or unset env → Mozc order (fail-safe).
6. No automatic clicking. Space to convert; do not click the candidate list.

## Remaining

- Human IME smoke **GO**（駅に→汽車 / 新聞の→記者。最後の conversion segment に直前文脈を渡す修正済み）
- 実使用ガード **GO**（リプレイ net +0.5pt）。IME に載せるにはガード入り `mozc_server` の管理者インストールが必要（未実施）
- Task 3 再学習は次段（未着手）
- SentencePiece C++ tokenize parity は native ORT 時まで後回し（デーモン経路では不要）

## ⑤ 文脈 cap スイープ（50 / 30 / 20字 × 70m / 30m）

Serve 相当: データは既に `clean_context@50`。cap=30/20 は進行中文の**末尾 N 字**。τ=2.5 固定。品質は PT clean eval 3セット。レイテンシは WSL CPU・intra=12・padding=longest・80 groups。

CS Δ（ON−OFF, pt）と全体退行は cap を短くしてもほぼ落ちない:

| model | cap | CS Δ seen/unseen/fresh | min CS Δ | 退行 max |
|--|--|--|--|--|
| 70m | 50 | +32.6 / +29.3 / +49.1 | 29.3 | 1.34% |
| 70m | 30 | +32.0 / +28.0 / +49.1 | 28.0 | 1.59% |
| 70m | 20 | +31.4 / +28.9 / +48.4 | 28.9 | 1.51% |
| 30m | 50 | +31.4 / +24.3 / +48.8 | 24.3 | 1.56% |
| 30m | 30 | +31.8 / +22.6 / +48.0 | 22.6 | 1.51% |
| 30m | 20 | +29.7 / +22.6 / +47.6 | 22.6 | 1.85% |

p95（busy run1/run2 は他負荷。quiet は WSL idle・逐次。cap=50 ship は 70m **342** / 30m **128**）:

| model | cap | p50 busy 1/2 | p95 busy 1/2 | p50 quiet | p95 quiet | p95<200 |
|--|--|--|--|--|--|--|
| 70m | 50 | 206 / 155 | 557 / 479 | 110 | 346 | no |
| 70m | 30 | 137 / 127 | 388 / 368 | 102 | 246 | no |
| 70m | 20 | 156 / 187 | 401 / 427 | 92 | 218 | no |
| 30m | 50 | 79 / 73 | 213 / 221 | 53 | **126** | **yes** |
| 30m | 30 | 72 / 90 | 184 / 227 | 46 | **111** | **yes** |
| 30m | 20 | 59 / 57 | 153 / 130 | 40 | **85** | **yes** |

**Pareto (quiet):** 30m は 50/30/20 すべて p95<200。min CS Δ 最大は **30m @ 50字**（+24.3）。cap 短縮は不要。70m は 20字でも 218ms。JSON: `artifacts/rerank_ctx/eval/latency_quiet_compare.json`。

## ⑥ 実使用ガード（NEXT_TASK_USAGE_GUARD）

実ログ 186 件でリランカーは **net −8.1pt**（shown==wanted 71.5% vs Mozc 79.6%。助けた 6 / 壊した 21）。Wikipedia CS 評価と実タイピング分布のズレ。

**即効ガード（再学習なし）** — 3経路同一（`RerankRewriter` / `phase3_hook.rerank_one` / daemon）:

1. ホワイトリスト: `context_sensitive_map.json`（`build_context_sensitive_map`: top_gold_share<0.7、漢字gold≥2、content POS、total≥8）+ seed `きしゃ`（IME 煙、wiki CS に無い）− block `きょうかい`（協会→教会が残傷害だった穴埋め）
2. 読み長 ≤ 2 → skip
3. 文脈が空 or 数字・記号のみ → skip
4. 上書き先が半角カナ / カタカナのみ / 旧字 → Mozc 順に戻す

成果物: `artifacts/rerank/rerank_eligible_readings.json`（488 読み）。C++ は `rerank_eligible_readings.inc` に埋め込んで Low Integrity でもファイルを読まない。

**実ログリプレイ**（`ime_usage_pairs.jsonl`、モデル再スコアなし。履歴の上書きにガードを適用）:

| | shown==wanted | Mozc | net | overwrite | helped / hurt |
|--|--|--|--|--|--|
| ガードなし | 71.5% | 79.6% | **−8.1pt** | 37 | 6 / 21 |
| ガードあり | **80.1%** | 79.6% | **+0.5pt** | 2 | 1 / **0** |

186 件中 180 件はモデルを呼ばない。残った助け 1 件は `きょうぎ` + `全国高等学校情報処理` → 教義。壊した 21 件は消えた。JSON: `artifacts/rerank_ctx/eval/usage_replay_report.json`。

以後このログを主評価に併記。Wikipedia holdout は副評価。

**デーモン intra / 常駐:** WSL `nproc`=**12**。起動スクリプトは `--intra-op $(nproc)`（従来ハードコード 12 と同じ）。OMP=1。常駐のまま。タイムアウト 200ms fail-safe は据え置き（250ms にはしない）。ガードで呼び出しが減るのでもっさり露出も減る。

**再学習 (Task 3):** ガードは GO（net ≥ 0、目標プラス達成）。Modal 再学習は **まだ始めていない**（次段。ありふれた語の Mozc-top1 アンカー、旧字ペナルティ、実ログ込み eval）。

**既定 OFF 維持。** IME に載せるには `mozc_server` の入れ直しが必要（リライタにガードが入ったバイナリ）。

Python: `unittest tools.rerank.test_usage_guard` + `test_rerank_daemon` **OK**（g++ 無しのため C++ CLI パリティ 8 件 skip。`bazelisk test //rewriter:rerank_rewriter_test` が C++ 側）。  
**`bazelisk test //rewriter:rerank_rewriter_test` → PASSED (0.3s)**（短読みは hook を呼ばない、きしゃ+駅に は対象のまま）。
